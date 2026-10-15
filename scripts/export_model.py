"""Bundle a trained YOLO11 model and verified metadata for the desktop app."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil

from app import config
from app.model_registry import sha256_file, validate_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True, help="experiment.json saved by scripts.train")
    parser.add_argument("--output", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=config.DEFAULT_IOU)
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.experiment.is_file():
        parser.error("Both trained weights and experiment.json must exist")
    if not args.model_id.strip() or not (0 <= args.conf <= 1 and 0 <= args.iou <= 1) or args.imgsz <= 0 or args.imgsz % 32:
        parser.error("Invalid model ID or detection settings")
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    if not Path(experiment["settings"]["model"]).name.startswith("yolo11"):
        parser.error("Experiment is not YOLO11")
    if (args.output / "best.pt").exists() or (args.output / "metadata.json").exists():
        parser.error("Output bundle exists; choose a new --output directory to preserve it")
    from app.runtime import prepare_ultralytics
    prepare_ultralytics()
    from ultralytics import YOLO
    model = YOLO(str(args.weights.resolve()), task="detect")
    if model.task != "detect":
        parser.error("Expected a detection model")
    validate_names(model.names)
    architecture_file = str(model.model.yaml.get("yaml_file", ""))
    if not Path(architecture_file).name.startswith("yolo11"):
        parser.error(f"Model architecture is not identifiable as YOLO11: {architecture_file}")
    metadata = {"model_id": args.model_id, "architecture": "YOLO11", "task": "detect",
        "created_at": datetime.now().astimezone().isoformat(), "sha256": sha256_file(args.weights),
        "class_names": list(config.CLASS_NAMES), "imgsz": args.imgsz,
        "confidence": args.conf, "nms_iou": args.iou,
        "experiment": str(args.experiment.resolve()),
        "dataset_manifest_sha256": experiment["dataset_manifest_sha256"],
        "leakage_note": experiment.get("leakage_note", ""),
        "versions": {name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "Pillow")}}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "best.pt"
    metadata_path = args.output / "metadata.json"
    created_weights = created_metadata = False
    try:
        with target.open("xb") as destination, args.weights.open("rb") as source:
            created_weights = True
            shutil.copyfileobj(source, destination)
        with metadata_path.open("x", encoding="utf-8") as stream:
            created_metadata = True
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
    except Exception:
        if created_weights:
            target.unlink(missing_ok=True)
        if created_metadata:
            metadata_path.unlink(missing_ok=True)
        raise
    print(f"Model bundle ready: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
