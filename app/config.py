"""Paths and the canonical PCB class mapping."""
from pathlib import Path

APP_NAME = "ระบบตรวจจับตำหนิ PCB ด้วย YOLO11"
APP_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "pcb"
MODEL_PATH = MODELS_DIR / "best.pt"
OUTPUT_DIR = BASE_DIR / "outputs" / "pcb"
DB_PATH = BASE_DIR / "data" / "pcb_detections.db"
DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ = 0.25, 0.45, 640
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGE_FILTER = "ภาพ PCB (*.jpg *.jpeg *.png)"
DAMAGE_CLASSES = {
    0: ("missing_hole", "รูเจาะหาย", (220, 55, 65)),
    1: ("mouse_bite", "รอยแหว่ง", (0, 125, 160)),
    2: ("open_circuit", "วงจรขาด", (170, 85, 200)),
    3: ("short", "วงจรลัด", (205, 120, 0)),
    4: ("spur", "ส่วนทองแดงยื่น", (0, 130, 90)),
    5: ("spurious_copper", "ทองแดงส่วนเกิน", (55, 90, 210)),
}
CLASS_NAMES = tuple(info[0] for info in DAMAGE_CLASSES.values())


def class_name(class_id: int) -> str:
    return DAMAGE_CLASSES[class_id][0]


def class_label(class_id: int) -> str:
    return DAMAGE_CLASSES[class_id][1]


def class_color(class_id: int) -> tuple[int, int, int]:
    return DAMAGE_CLASSES[class_id][2]
