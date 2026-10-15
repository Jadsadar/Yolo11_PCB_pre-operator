"""python -m scripts.validate_dataset --data data.yaml"""
import argparse
import json
from pathlib import Path

from .dataset import audit_dataset


def main():
    parser = argparse.ArgumentParser(description="Read-only PCB YOLO dataset audit; no training")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_audit.json"))
    args = parser.parse_args()
    report = audit_dataset(args.data, progress=print)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: report[key] for key in ("counts", "total_images", "manifest_sha256", "errors", "warnings")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Full report: {args.output.resolve()}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
