"""Train/resume one experiment in persistent storage; dry run unless --execute."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import yaml

from app.model_registry import validate_names
from .train import training_plan


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class CheckpointSnapshots:
    """Keep two resumable copies while native last.pt is being overwritten."""
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def __call__(self, trainer):
        source = Path(trainer.last)
        folder = self.run_dir / "weights"
        folder.mkdir(parents=True, exist_ok=True)
        latest, previous = folder / "resume.pt", folder / "resume_previous.pt"
        temporary = folder / "resume.pt.tmp"
        # Interruption during this copy leaves the previous snapshots intact.
        shutil.copyfile(source, temporary)
        if latest.exists():
            latest.replace(previous)
        temporary.replace(latest)
        write_json(self.run_dir / "progress.json", {
            "phase": "running", "completed_epochs": trainer.epoch + 1,
            "target_epochs": trainer.epochs, "checkpoint": str(latest),
            "updated_at": datetime.now().astimezone().isoformat(),
        })


def select_checkpoint(run_dir: Path, loader):
    """Choose the newest readable resumable checkpoint; never silently discard corrupt files."""
    folder = Path(run_dir) / "weights"
    available, failures, candidates = [], [], []
    for name in ("resume.pt", "resume_previous.pt", "last.pt", "best.pt"):
        path = folder / name
        if not path.is_file():
            continue
        available.append(path)
        try:
            model = loader(path)
            validate_names(model.names)
            checkpoint = model.ckpt or {}
            epoch = int(checkpoint.get("epoch", -1))
            # Ultralytics strips optimizer/epoch from last.pt after normal completion.
            if name == "last.pt" and epoch == -1 and checkpoint.get("optimizer") is None:
                return "completed", model, path
            if epoch >= 0 and checkpoint.get("optimizer") is not None:
                target = int(checkpoint.get("train_args", {}).get("epochs", 0))
                if target and epoch + 1 >= target:
                    return "completed", model, path
                candidates.append((epoch, model, path))
            else:
                failures.append(f"{name}: optimizer/epoch missing")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if candidates:
        # Stable sorting favors resume.pt for equal epoch indices.
        _, model, path = max(candidates, key=lambda item: item[0])
        for message in failures:
            print(f"Checkpoint skipped: {message}")
        return "resume", model, path
    if available:
        raise RuntimeError("No readable resumable checkpoint. Existing files were retained. " + "; ".join(failures))
    return "start", None, None


def latest_run(project: Path):
    runs = sorted((p for p in Path(project).glob("pcb_*") if (p / "experiment.json").is_file()),
                  key=lambda p: p.name)
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--project", type=Path, required=True, help="Persistent experiment folder on mounted Drive")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--new-run", action="store_true", help="Explicitly start a separate experiment")
    parser.add_argument("--leakage-note", default="")
    args = parser.parse_args()
    project = args.project.resolve()
    run_dir = None if args.new_run else latest_run(project)
    if run_dir and (run_dir / "progress.json").is_file():
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        if progress.get("phase") == "completed":
            print(f"This experiment has finished. Best model: {run_dir / 'weights/best.pt'}")
            print("Use --new-run only when you want another experiment.")
            return 0
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "training_started": False, "project": str(project),
                          "existing_run": str(run_dir) if run_dir else None,
                          "action": "inspect checkpoints and resume" if run_dir else "start a new experiment"}, indent=2))
        return 0

    plan, audit = training_plan(args.config)
    blockers = plan["blockers"]
    if plan["leakage_review_required"] and not args.leakage_note.strip():
        blockers.append("Review dataset duplication and record --leakage-note")
    if blockers:
        parser.error("\n".join(blockers))
    settings = dict(plan["settings"])
    tensorboard_enabled = settings.pop("tensorboard")
    weights = settings.pop("model")
    settings.pop("name", None)
    settings.pop("project", None)
    settings["save"] = True
    settings["save_period"] = -1  # Two rolling snapshots avoid retaining every epoch forever.
    if run_dir:
        original = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
        if original["dataset_manifest_sha256"] != audit["manifest_sha256"]:
            parser.error("Dataset differs from the original experiment; restore the same data or use --new-run")
        # The saved epoch target/settings govern a resumed run.
    else:
        run_dir = project / ("pcb_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        run_dir.mkdir(parents=True, exist_ok=False)
        plan.update(training_started=True, mode="execute", leakage_note=args.leakage_note,
                    tensorboard_log_dir=str(run_dir))
        write_json(run_dir / "experiment.json", plan)
        write_json(run_dir / "dataset_audit.json", audit)
    resolved_yaml = run_dir / "dataset.resolved.yaml"
    resolved_yaml.write_text(yaml.safe_dump(audit["resolved_dataset"], sort_keys=False), encoding="utf-8")

    from app.runtime import prepare_ultralytics
    prepare_ultralytics()
    from ultralytics import YOLO
    from .tensorboard_logging import configure_tensorboard, close_tensorboard, flush_tensorboard

    print(f"Persistent run directory: {run_dir}")
    status, model, checkpoint = select_checkpoint(run_dir, lambda path: YOLO(str(path), task="detect"))
    if status == "completed":
        write_json(run_dir / "progress.json", {"phase": "completed", "checkpoint": str(checkpoint),
                    "updated_at": datetime.now().astimezone().isoformat()})
        print(f"Training already completed. Best model: {run_dir / 'weights/best.pt'}")
        return 0
    if model is None:
        print("Starting from pretrained weights; no completed-epoch checkpoint exists yet.")
        model = YOLO(weights, task="detect")
        overrides = dict(settings, data=str(resolved_yaml), project=str(project),
                         name=run_dir.name, exist_ok=True)
    else:
        validate_names(model.names)
        print(f"Resuming from {checkpoint}; completed epochs: {model.ckpt['epoch'] + 1}")
        overrides = {"resume": True, "data": str(resolved_yaml), "device": settings["device"],
                     "batch": settings["batch"], "workers": settings["workers"], "cache": False,
                     "save_dir": str(run_dir)}
    callbacks = configure_tensorboard(tensorboard_enabled)
    model.add_callback("on_model_save", CheckpointSnapshots(run_dir))
    if tensorboard_enabled:
        model.add_callback("on_train_epoch_start", lambda trainer: flush_tensorboard(callbacks))
    try:
        model.train(**overrides)
    except BaseException as exc:
        # Hard runtime deletion cannot execute this handler; epoch snapshots are already on Drive.
        write_json(run_dir / "interruption.json", {"error": repr(exc),
                   "updated_at": datetime.now().astimezone().isoformat()})
        raise
    else:
        write_json(run_dir / "progress.json", {"phase": "completed",
                   "best_model": str(run_dir / "weights/best.pt"),
                   "updated_at": datetime.now().astimezone().isoformat()})
        print(f"Finished. Model and logs remain in {run_dir}")
    finally:
        close_tensorboard(callbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
