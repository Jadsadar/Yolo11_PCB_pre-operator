"""Model bundle metadata, shared by the desktop app and export script."""
import hashlib
import json
from pathlib import Path

from . import config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_names(names) -> tuple[str, ...]:
    if isinstance(names, dict):
        normalized = {int(k): v for k, v in names.items()}
        if set(normalized) != set(range(len(config.CLASS_NAMES))):
            raise ValueError("Class IDs must be exactly 0 through 5")
        return tuple(normalized[i] for i in range(len(normalized)))
    return tuple(names)


def validate_names(names) -> None:
    if ordered_names(names) != config.CLASS_NAMES:
        raise ValueError("โมเดลไม่ตรงกับชุดคลาส PCB ที่กำหนด")


def read_metadata(weights: Path) -> dict:
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"ยังไม่มีโมเดลที่ฝึกแล้ว: {weights}")
    metadata_path = weights.with_name("metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ไม่พบ metadata โมเดล: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_names(metadata["class_names"])
    if metadata.get("architecture") != "YOLO11" or metadata.get("task") != "detect":
        raise ValueError("ต้องใช้ YOLO11 สำหรับงานตรวจจับวัตถุ")
    if not isinstance(metadata.get("model_id"), str) or not metadata["model_id"].strip():
        raise ValueError("Missing model ID")
    if metadata.get("sha256") != sha256_file(weights):
        raise ValueError("Model checksum does not match metadata")
    return metadata
