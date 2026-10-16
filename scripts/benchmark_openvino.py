"""Compare warmed CPU inspection latency and sampled detections; not a validation accuracy score."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter


def overlap(a, b):
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
    return intersection / union if union else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--openvino", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("valid/images"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/openvino_benchmark.json"))
    args = parser.parse_args()
    if args.limit < 1 or args.repeats < 1:
        parser.error("limit and repeats must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from app.detector import YoloBackend, run_detection
    from app.models import DetectionRequest
    from app.config import IMAGE_SUFFIXES

    available = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not available:
        parser.error("No images found")
    count = min(args.limit, len(available))
    images = [available[i * (len(available) - 1) // max(count - 1, 1)] for i in range(count)]
    reference, candidate = YoloBackend(args.reference), YoloBackend(args.openvino)
    reference.model.overrides["device"] = "cpu"
    backends = {"pytorch_cpu": reference, "openvino_cpu": candidate}
    times = {name: [] for name in backends}
    inference = {name: [] for name in backends}
    samples = []
    for backend in backends.values():
        run_detection(backend, DetectionRequest(images[0]))
    for image in images:
        results = {}
        for _ in range(args.repeats):
            for name, backend in backends.items():
                started = perf_counter()
                result = run_detection(backend, DetectionRequest(image))
                times[name].append((perf_counter() - started) * 1000)
                inference[name].append(result.inference_ms)
                results[name] = result.detections
        original, converted = results["pytorch_cpu"], results["openvino_cpu"]
        remaining = list(converted)
        matches, differences = [], []
        for detection in original:
            options = [(overlap(detection, d), i) for i, d in enumerate(remaining)
                       if d.class_id == detection.class_id]
            if options:
                iou, index = max(options)
                if iou >= .99:
                    matches.append(iou)
                    differences.append(abs(detection.confidence - remaining.pop(index).confidence))
        samples.append(dict(image=str(image), reference_count=len(original), openvino_count=len(converted),
                            matched_at_iou_099=len(matches),
                            max_confidence_difference=max(differences, default=0),
                            all_matched=len(matches) == len(original) == len(converted)))
    report = dict(device="CPU", imgsz=640, conf=.25, iou=.45, repeats=args.repeats,
                  timing="Warm runs; read/decode, inference, postprocess and annotation; excludes UI/storage",
                  models={name: dict(median_total_ms=median(times[name]), median_inference_ms=median(inference[name]))
                          for name in backends}, samples=samples,
                  note="Sample agreement is not a dataset-wide accuracy evaluation; concurrent training affects timing.")
    report["total_speedup"] = median(times["pytorch_cpu"]) / median(times["openvino_cpu"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
