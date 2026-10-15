"""Thread-independent values shared by inference, storage, and UI."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from . import config


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def class_name(self) -> str:
        return config.class_name(self.class_id)

    @property
    def class_label(self) -> str:
        return config.class_label(self.class_id)

    def validate(self, width: int, height: int) -> None:
        if self.class_id not in config.DAMAGE_CLASSES:
            raise ValueError(f"Unknown PCB class ID: {self.class_id}")
        if not all(math.isfinite(v) for v in (self.confidence, self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("Detection contains non-finite numbers")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0, 1]")
        if not (0 <= self.x1 < self.x2 <= width and 0 <= self.y1 < self.y2 <= height):
            raise ValueError("Bounding box lies outside the image")


@dataclass(frozen=True)
class DetectionRequest:
    source_path: Path
    conf: float = config.DEFAULT_CONF
    iou: float = config.DEFAULT_IOU
    imgsz: int = config.DEFAULT_IMGSZ

    def __post_init__(self):
        if not (0 <= self.conf <= 1 and 0 <= self.iou <= 1):
            raise ValueError("Confidence and IoU must be in [0, 1]")
        if self.imgsz <= 0 or self.imgsz % 32:
            raise ValueError("Image size must be a positive multiple of 32")


@dataclass
class DetectionResult:
    request: DetectionRequest
    original: Image.Image
    annotated: Image.Image
    detections: tuple[Detection, ...]
    inference_ms: float
    total_ms: float
    model_id: str
    model_sha256: str
    is_mock: bool
    detected_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    event_id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def counts(self) -> Counter:
        return Counter(d.class_id for d in self.detections)

    def validate(self) -> None:
        width, height = self.original.size
        if self.annotated.size != (width, height):
            raise ValueError("Annotated image size mismatch")
        for detection in self.detections:
            detection.validate(width, height)


@dataclass(frozen=True)
class SaveResult:
    status: str  # saved, failed, not_required, mock
    event_id: str
    result_path: str = ""
    error: str = ""
