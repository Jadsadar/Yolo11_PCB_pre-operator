"""Audit existing YOLO detection splits without changing images or annotations."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import math
from pathlib import Path

from PIL import Image
import yaml

from app.config import CLASS_NAMES
from app.model_registry import validate_names

SPLITS = ("train", "val", "test")
SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_dataset(yaml_path: Path) -> dict:
    yaml_path = Path(yaml_path).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    validate_names(data["names"])
    if data.get("nc", len(CLASS_NAMES)) != len(CLASS_NAMES):
        raise ValueError("Dataset nc must be 6")
    root = (yaml_path.parent / data.get("path", ".")).resolve()
    resolved = {"path": str(root), "names": list(CLASS_NAMES), "nc": len(CLASS_NAMES)}
    for split in SPLITS:
        value = data.get(split)
        if not isinstance(value, str):
            raise ValueError(f"{split} must name an images directory")
        image_dir = (root / value).resolve()
        if image_dir.name != "images" or not image_dir.is_dir():
            raise ValueError(f"Expected existing {split}/images directory: {image_dir}")
        resolved[split] = str(image_dir)
    return resolved


def source_group(path: Path) -> str:
    # Roboflow export: <original_name>_jpg.rf.<variant_hash>.jpg.
    # This is a filename heuristic, not proof of source identity.
    return path.stem.split(".rf.", 1)[0]


def audit_dataset(yaml_path: Path, progress=None) -> dict:
    resolved = load_dataset(yaml_path)
    errors, warnings, records, counts = [], [], [], {}
    hashes, groups = defaultdict(list), defaultdict(list)
    for split in SPLITS:
        image_dir = Path(resolved[split])
        label_dir = image_dir.parent / "labels"
        if not label_dir.is_dir():
            errors.append(f"Missing labels directory: {label_dir}")
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
        labels = list(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        image_stems = Counter(p.stem for p in images)
        class_counts, backgrounds = Counter(), 0
        if not images:
            errors.append(f"Empty image split: {split}")
        for stem, count in image_stems.items():
            if count > 1:
                errors.append(f"Ambiguous label basename in {split}: {stem}")
        for label in labels:
            if label.stem not in image_stems:
                errors.append(f"Orphan label: {label}")
        for number, image_path in enumerate(images):
            if progress and number % 500 == 0:
                progress(f"Checking {split}: {number}/{len(images)}")
            record = {"split": split, "image": str(image_path.relative_to(Path(resolved["path"]))),
                      "source_group": source_group(image_path)}
            try:
                raw = image_path.read_bytes()
                record["sha256"] = hashlib.sha256(raw).hexdigest()
                with Image.open(image_path) as image:
                    image.load()
                    if image.format not in {"JPEG", "PNG"}:
                        raise ValueError("Image contents must be JPEG or PNG")
                    record["width"], record["height"] = image.size
                hashes[record["sha256"]].append({"split": split, "image": record["image"]})
                groups[record["source_group"]].append({"split": split, "image": record["image"]})
            except Exception as exc:
                errors.append(f"Unreadable image {image_path}: {exc}")
            label = label_dir / (image_path.stem + ".txt")
            record["label"] = str(label.relative_to(Path(resolved["path"])))
            found = Counter()
            if not label.is_file():
                errors.append(f"Missing label: {label}")
            else:
                try:
                    raw_label = label.read_bytes()
                    record["label_sha256"] = hashlib.sha256(raw_label).hexdigest()
                    lines = raw_label.decode("utf-8-sig").splitlines()
                    nonempty = [line for line in lines if line.strip()]
                    if not nonempty:
                        backgrounds += 1
                    for line_no, line in enumerate(lines, 1):
                        if not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            raise ValueError(f"line {line_no}: expected class x_center y_center width height")
                        cid = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        if cid not in range(6) or not all(math.isfinite(v) for v in (x, y, w, h)):
                            raise ValueError(f"line {line_no}: invalid class or non-finite number")
                        epsilon = 1e-6
                        if not (0 < w <= 1 and 0 < h <= 1 and
                                -epsilon <= x-w/2 <= x+w/2 <= 1+epsilon and
                                -epsilon <= y-h/2 <= y+h/2 <= 1+epsilon):
                            raise ValueError(f"line {line_no}: normalized box lies outside image")
                        found[cid] += 1
                    class_counts.update(found)
                except Exception as exc:
                    errors.append(f"Invalid label {label}: {exc}")
            record["class_counts"] = dict(found)
            records.append(record)
        missing_classes = [CLASS_NAMES[i] for i in range(6) if not class_counts[i]]
        if missing_classes:
            warnings.append(f"{split} has no labeled boxes for: {', '.join(missing_classes)}")
        counts[split] = {"images": len(images), "labels": len(labels), "background_images": backgrounds,
                         "boxes_by_class": {CLASS_NAMES[i]: class_counts[i] for i in range(6)}}
    exact = [items for items in hashes.values() if len({item["split"] for item in items}) > 1]
    related = [items for items in groups.values() if len({item["split"] for item in items}) > 1]
    if exact:
        warnings.append(f"{len(exact)} exact image hash groups cross dataset splits")
    if related:
        warnings.append(f"{len(related)} filename source groups cross splits; review provenance before training")
    actual_total = sum(counts[s]["images"] for s in SPLITS)
    if actual_total != 6652:
        warnings.append(f"Local image count {actual_total} differs from report/export count 6652")
    manifest_hash = hashlib.sha256()
    for record in records:
        manifest_hash.update((record["split"] + record["image"] + record.get("sha256", "") +
                              record.get("label_sha256", "")).encode("utf-8"))
    return {"created_at": datetime.now().astimezone().isoformat(), "data_yaml": str(Path(yaml_path).resolve()),
            "resolved_dataset": resolved, "counts": counts, "total_images": actual_total,
            "manifest_sha256": manifest_hash.hexdigest(), "errors": errors, "warnings": warnings,
            "exact_cross_split_duplicates": exact, "suspected_cross_split_source_groups": related,
            "source_group_method": "Roboflow filename prefix before .rf.; heuristic only",
            "limitation": "No perceptual duplicate detection or verified original-PCB identity mapping",
            "files": records}
