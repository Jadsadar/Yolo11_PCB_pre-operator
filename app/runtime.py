"""Keep model runtime settings local and disable implicit network/install work."""
import os
from pathlib import Path

from .config import BASE_DIR


def prepare_ultralytics():
    directory = BASE_DIR / ".runtime"
    directory.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(directory))
    # Create the parent before import so Ultralytics does not choose a global fallback.
    (Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
