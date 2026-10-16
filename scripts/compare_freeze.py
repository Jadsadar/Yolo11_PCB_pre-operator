"""Compare freeze/no-freeze for up to 60 epochs with early-stopping patience 15."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEAKAGE_NOTE = "Learning comparison; duplicate images exist across splits, evaluation is not independent"


def comparison_configs(base, config_dir, output_dir):
    settings = dict(base)
    for key in ("data", "model"):
        settings[key] = str((config_dir / settings[key]).resolve())
    settings.update(epochs=60, batch=8, patience=15, val=True, tensorboard=True,
                    project=str(output_dir.resolve()), resume=False)
    # Both runs start from the SAME pretrained weights and seed, never each other's checkpoint.
    return {name: dict(settings, name=name, freeze=freeze)
            for name, freeze in (("freeze2", 2), ("no_freeze", 0))}


def summarize_run(run_dir, variant, wall_seconds):
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key.strip(): float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if not 1 <= len(rows) <= 60 or [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"Expected 1-60 consecutive completed epochs: {run_dir}")
    # Installed Ultralytics detection fitness selects best.pt by mAP50-95.
    best = max(rows, key=lambda row: (row["metrics/mAP50-95(B)"], row["epoch"]))
    for filename in ("weights/best.pt", "weights/last.pt", "per_class_metrics.csv"):
        if not (run_dir / filename).is_file():
            raise RuntimeError(f"Missing output: {run_dir / filename}")
    return dict(variant=variant, epochs=len(rows), best_epoch=int(best["epoch"]),
                best_precision=best["metrics/precision(B)"],
                best_recall=best["metrics/recall(B)"],
                best_mAP50=best["metrics/mAP50(B)"],
                best_mAP50_95=best["metrics/mAP50-95(B)"],
                final_mAP50_95=rows[-1]["metrics/mAP50-95(B)"],
                training_seconds=rows[-1]["time"],
                average_epoch_seconds=rows[-1]["time"] / len(rows),
                wall_seconds=round(wall_seconds, 2), run_dir=str(run_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Start both training runs; default only checks plans")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--leakage-note", default=LEAKAGE_NOTE)
    args = parser.parse_args()
    base_path = args.config.resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    folder = ROOT / "experiments" / f"freeze_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir(parents=True, exist_ok=False)
    configs = comparison_configs(base, base_path.parent, folder / "runs")
    summaries = []
    print(f"Comparison folder: {folder}", flush=True)
    for variant, settings in configs.items():
        config_path = folder / f"{variant}.yaml"
        config_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "-u", "-m", "scripts.train", "--config", str(config_path),
                   "--plan-output", str(folder / f"{variant}_plan.json"),
                   "--leakage-note", args.leakage_note]
        if args.execute:
            command.append("--execute")
        print(f"\nStarting {variant}: freeze={settings['freeze']}, epochs<=60, batch=8, patience=15", flush=True)
        started = time.perf_counter()
        # Each process releases GPU memory before the next run starts; abort on failure.
        subprocess.run(command, cwd=ROOT, check=True)
        if args.execute:
            runs = list((folder / "runs").glob(f"{variant}_*"))
            if len(runs) != 1:
                raise RuntimeError(f"Expected one output run for {variant}, found {len(runs)}")
            summaries.append(summarize_run(runs[0], variant, time.perf_counter() - started))
            with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
    if args.execute:
        print(f"\nComparison saved: {folder / 'comparison.csv'}")
        for row in summaries:
            print(f"{row['variant']}: epochs={row['epochs']}, best mAP50-95={row['best_mAP50_95']:.5f}, "
                  f"average epoch={row['average_epoch_seconds']:.2f}s")
        print("Compare validation accuracy and speed separately; one seed is exploratory.")
        print("Cross-split duplicates limit evaluation independence.")
    else:
        print("Plans only. Add --execute to train both runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
