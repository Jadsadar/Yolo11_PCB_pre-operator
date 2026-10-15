"""Prepare evaluation by default; --execute is required to run a trained model."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter

import yaml

from app.image_service import read_image
from app.model_registry import sha256_file, validate_names
from .dataset import audit_dataset


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def error_examples(model, image_dir, output_dir, conf, nms_iou, imgsz, device, limit):
    """Diagnostic one-to-one matching, distinct from Ultralytics AP calculations."""
    from PIL import ImageDraw
    examples = []
    for path in sorted(Path(image_dir).iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = read_image(path)
        width, height = image.size
        label = path.parent.parent / "labels" / (path.stem + ".txt")
        truth = []
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            cid, x, y, w, h = map(float, line.split())
            truth.append((int(cid), ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)))
        pred = model.predict(image, conf=conf, iou=nms_iou, imgsz=imgsz, device=device, verbose=False)[0]
        matched, false_positives = set(), []
        for box in sorted(pred.boxes, key=lambda item: -float(item.conf[0])):
            coords = tuple(float(v) for v in box.xyxy[0].tolist())
            cid = int(box.cls[0])
            candidates = [(iou(coords, gt), index) for index, (gt_cid, gt) in enumerate(truth)
                          if gt_cid == cid and index not in matched]
            best = max(candidates, default=(0, -1))
            if best[0] >= .5:
                matched.add(best[1])
            else:
                false_positives.append({"class_id": cid, "xyxy": coords, "confidence": float(box.conf[0])})
        false_negatives = [{"class_id": cid, "xyxy": coords} for index, (cid, coords) in enumerate(truth)
                           if index not in matched]
        if false_positives or false_negatives:
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            for item in false_positives:
                draw.rectangle(item["xyxy"], outline="red", width=3)
            for item in false_negatives:
                draw.rectangle(item["xyxy"], outline="blue", width=3)
            destination = output_dir / f"{len(examples)+1:02d}_{path.stem}.png"
            canvas.save(destination)
            examples.append({"image": str(path), "preview": str(destination),
                             "false_positives": false_positives, "false_negatives": false_negatives})
            if len(examples) >= limit:
                break
    return {"matching_iou": .5, "confidence": conf, "nms_iou": nms_iou,
            "method": "Greedy confidence-order same-class one-to-one matching; red FP, blue FN",
            "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=.25, help="Diagnostic/benchmark operating confidence")
    parser.add_argument("--iou", type=float, default=.45, help="NMS IoU, not ground-truth matching IoU")
    parser.add_argument("--benchmark-images", type=int, default=20)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--leakage-note", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.conf <= 1 and 0 <= args.iou <= 1 and args.imgsz > 0 and args.imgsz % 32 == 0):
        parser.error("Invalid confidence, IoU, or image size")
    if args.benchmark_images < 1 or args.examples < 0:
        parser.error("Use at least one benchmark image and a nonnegative example count")
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "evaluation_started": False,
                          "model": str(args.model.resolve()), "split": args.split,
                          "model_exists": args.model.is_file(), "data": str(args.data.resolve())}, indent=2))
        return 0
    if not args.model.is_file():
        parser.error("Trained weights do not exist; no download is performed")
    audit = audit_dataset(args.data)
    if audit["errors"]:
        parser.error("Fix dataset errors before evaluation")
    if (audit["exact_cross_split_duplicates"] or audit["suspected_cross_split_source_groups"]) and not args.leakage_note.strip():
        parser.error("Review leakage and record --leakage-note before evaluating")
    from app.runtime import prepare_ultralytics
    prepare_ultralytics()
    from ultralytics import YOLO
    import torch
    from app.config import CLASS_NAMES

    model = YOLO(str(args.model.resolve()), task="detect")
    validate_names(model.names)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = Path("experiments/evaluations") / f"{args.split}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    resolved = folder / "dataset.resolved.yaml"
    resolved.write_text(yaml.safe_dump(audit["resolved_dataset"], sort_keys=False), encoding="utf-8")
    # Low confidence for PR/AP integration. Operating-point diagnostics use args.conf separately.
    metrics = model.val(data=str(resolved.resolve()), split=args.split, imgsz=args.imgsz,
                        conf=.001, iou=args.iou, device=args.device, workers=0,
                        plots=True, project=str(folder.parent.resolve()), name=folder.name, exist_ok=True)
    by_class = {name: None for name in CLASS_NAMES}
    for index, cid in enumerate(metrics.box.ap_class_index):
        p, r, map50, map_all = map(float, metrics.box.class_result(index))
        by_class[CLASS_NAMES[int(cid)]] = {"precision": p, "recall": r,
            "f1": float(metrics.box.f1[index]), "mAP50": map50, "mAP50_95": map_all}
    paths = [p for p in sorted(Path(audit["resolved_dataset"][args.split]).iterdir())
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:args.benchmark_images]
    model.predict(read_image(paths[0]), conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                  device=args.device, verbose=False)  # One excluded warm-up.
    durations = []
    for path in paths:
        start = perf_counter()
        model.predict(read_image(path), conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                      device=args.device, verbose=False)
        if torch.cuda.is_available() and str(args.device) != "cpu":
            torch.cuda.synchronize()
        durations.append((perf_counter()-start)*1000)
    report = {"created_at": datetime.now().astimezone().isoformat(), "split": args.split,
        "model": str(args.model.resolve()), "model_sha256": sha256_file(args.model),
        "dataset_manifest_sha256": audit["manifest_sha256"], "leakage_note": args.leakage_note,
        "dataset_warnings": audit["warnings"], "imgsz": args.imgsz,
        "device": args.device, "platform": platform.platform(), "processor": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {name: importlib.metadata.version(name) for name in ("ultralytics", "torch")},
        "metrics": {str(k): float(v) for k,v in metrics.results_dict.items()},
        "macro_f1_at_validator_operating_point": float(metrics.box.f1.mean()) if len(metrics.box.f1) else None,
        "per_class": by_class, "validator_stage_ms_per_image": metrics.speed,
        "operating_confidence": args.conf, "nms_iou": args.iou,
        "benchmark": {"images": len(durations), "warmup_excluded": 1,
                      "scope": "decode + model.predict, excludes UI, annotation and evidence storage",
                      "mean_ms": mean(durations), "median_ms": median(durations)},
        "acceptance_thresholds": None}
    (folder / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.examples:
        example_dir = folder / "error_examples"
        example_dir.mkdir()
        examples = error_examples(model, audit["resolved_dataset"][args.split], example_dir,
                                  args.conf, args.iou, args.imgsz, args.device, args.examples)
        (example_dir / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"Evaluation saved to {folder.resolve()}. No accuracy/speed pass-fail threshold applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
