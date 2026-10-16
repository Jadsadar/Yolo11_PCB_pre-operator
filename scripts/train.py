"""Prepare a plan by default. Training requires the explicit --execute flag."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform

import yaml

from .dataset import audit_dataset


def training_plan(config_path: Path):
    config_path = Path(config_path).resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings.setdefault("tensorboard", True)
    if not isinstance(settings["tensorboard"], bool):
        raise ValueError("tensorboard must be true or false")
    for key in ("data", "model", "project"):
        settings[key] = str((config_path.parent / settings[key]).resolve())
    if not Path(settings["model"]).name.startswith("yolo11") or Path(settings["model"]).suffix != ".pt":
        raise ValueError("Use a YOLO11 pretrained .pt file for transfer learning")
    if int(settings["imgsz"]) <= 0 or int(settings["imgsz"]) % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    audit = audit_dataset(Path(settings["data"]))
    versions = {}
    for package in ("ultralytics", "torch", "torchvision", "numpy", "Pillow", "PyYAML", "tensorboard"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    blockers = list(audit["errors"])
    if not Path(settings["model"]).is_file():
        blockers.append(f"Pretrained weights not installed: {settings['model']}")
    if versions["ultralytics"] is None or versions["torch"] is None:
        blockers.append("Install requirements-train.txt before training")
    if settings["tensorboard"] and versions["tensorboard"] is None:
        blockers.append("TensorBoard is enabled; install requirements-train.txt before training")
    leakage = bool(audit["exact_cross_split_duplicates"] or audit["suspected_cross_split_source_groups"])
    return {"mode": "dry-run", "training_started": False,
            "created_at": datetime.now().astimezone().isoformat(), "settings": settings,
            "versions": versions, "python": platform.python_version(), "platform": platform.platform(),
            "dataset_manifest_sha256": audit["manifest_sha256"], "counts": audit["counts"],
            "blockers": blockers, "leakage_review_required": leakage,
            "dataset_warnings": audit["warnings"]}, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--plan-output", type=Path, default=Path("reports/training_plan.json"))
    parser.add_argument("--execute", action="store_true", help="Actually start training; otherwise only write a plan")
    parser.add_argument("--leakage-note", default="", help="Record provenance review/limitations when cross-split groups exist")
    args = parser.parse_args()
    plan, audit = training_plan(args.config)
    plan["leakage_note"] = args.leakage_note
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("DRY RUN ONLY: no model loaded, no download, no training started.")
        return 0
    blockers = list(plan["blockers"])
    if plan["leakage_review_required"] and not args.leakage_note.strip():
        blockers.append("Review cross-split groups and supply --leakage-note, or correct the split first")
    if blockers:
        parser.error("Training blocked:\n" + "\n".join(blockers))
    from app.runtime import prepare_ultralytics
    prepare_ultralytics()
    from ultralytics import YOLO
    from app.model_registry import sha256_file
    from .tensorboard_logging import configure_tensorboard, flush_tensorboard, close_tensorboard, PerClassMetricsLogger

    settings = dict(plan["settings"])
    tensorboard_enabled = settings.pop("tensorboard")
    tensorboard_callbacks = configure_tensorboard(tensorboard_enabled)
    weights = settings.pop("model")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(settings["project"]) / f'{settings.pop("name")}_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_yaml = run_dir / "dataset.resolved.yaml"
    resolved_yaml.write_text(yaml.safe_dump(audit["resolved_dataset"], sort_keys=False), encoding="utf-8")
    (run_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    settings.update(data=str(resolved_yaml), project=str(run_dir.parent), name=run_dir.name, exist_ok=True)
    plan.update(mode="execute", training_started=True, pretrained_sha256=sha256_file(Path(weights)))
    plan["tensorboard_log_dir"] = str(run_dir) if tensorboard_enabled else None
    (run_dir / "experiment.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    model = YOLO(weights, task="detect")
    model.add_callback("on_fit_epoch_end", PerClassMetricsLogger(tensorboard_callbacks))
    if tensorboard_enabled:
        # This event follows on_fit_epoch_end, after native metrics have been written.
        model.add_callback("on_train_epoch_start", lambda trainer: flush_tensorboard(tensorboard_callbacks))
        print(f"TensorBoard logs for this run: {run_dir}")
        print(f'Compare runs: python -m tensorboard.main --logdir "{run_dir.parent}" --host localhost --port 6006')
    try:
        model.train(**settings)
    finally:
        # Also preserve the final points on early stop, Ctrl+C, or training errors.
        close_tensorboard(tensorboard_callbacks)
    print(f"Training finished. Review validation results: {run_dir}")
    print("Test evaluation and model export are separate explicit commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
