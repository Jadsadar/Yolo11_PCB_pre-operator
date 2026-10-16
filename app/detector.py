"""Persistent worker: model loading, inference, and storage stay off the UI thread."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from . import config
from .image_service import annotate, read_image
from .model_registry import read_metadata, validate_names
from .models import Detection, DetectionRequest, DetectionResult
from .storage import EventStorage

LOG = logging.getLogger(__name__)


class MockBackend:
    """Deterministic UI fixture; never used as an automatic fallback."""
    model_id, sha256, is_mock = "DEMO — ผลจำลอง", "", True

    def predict(self, image: Image.Image, request: DetectionRequest):
        width, height = image.size
        boxes = (Detection(0, .92, .1*width, .15*height, .25*width, .3*height),
                 Detection(3, .79, .55*width, .5*height, .75*width, .7*height))
        return tuple(d for d in boxes if d.confidence >= request.conf)


class YoloBackend:
    is_mock = False

    def __init__(self, weights: Path):
        weights = Path(weights)
        metadata = read_metadata(weights)
        from .runtime import prepare_ultralytics
        prepare_ultralytics()
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.is_openvino = weights.suffix.lower() == ".xml"
        if self.is_openvino:
            # Pin Intel CPU before lazy backend initialization, including names lookup.
            self.model.overrides["device"] = "intel:cpu"
            # Compile and warm up in the loading worker before accepting the first image.
            size = int(metadata["imgsz"])
            self.model.predict(source=Image.new("RGB", (size, size)), imgsz=size,
                               device="intel:cpu", verbose=False)
        if self.model.task != "detect":
            raise ValueError("Model task is not object detection")
        validate_names(self.model.names)
        self.model_id, self.sha256 = metadata["model_id"], metadata["sha256"]

    def predict(self, image: Image.Image, request: DetectionRequest):
        # PIL RGB is supported directly; Ultralytics returns xyxy in original image pixels.
        prediction = self.model.predict(source=image, conf=request.conf, iou=request.iou,
                                        imgsz=request.imgsz, verbose=False,
                                        **({"device": "intel:cpu"} if self.is_openvino else {}))[0]
        boxes = []
        for box in prediction.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            detection = Detection(int(box.cls[0].item()), float(box.conf[0].item()), x1, y1, x2, y2)
            detection.validate(*image.size)
            boxes.append(detection)
        return tuple(boxes)


def run_detection(backend, request: DetectionRequest) -> DetectionResult:
    start = perf_counter()
    image = read_image(request.source_path)
    inference_start = perf_counter()
    detections = tuple(backend.predict(image, request))
    inference_ms = (perf_counter() - inference_start) * 1000
    annotated = annotate(image, detections)
    result = DetectionResult(request, image, annotated, detections, inference_ms,
                             (perf_counter() - start) * 1000, backend.model_id,
                             backend.sha256, backend.is_mock)
    result.validate()
    return result


class DetectionWorker(QObject):
    model_loaded = pyqtSignal(bool, str)
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(object, object)
    failed = pyqtSignal(str)
    done = pyqtSignal()
    consistency = pyqtSignal(object)

    def __init__(self, weights, mock=False, db_path=config.DB_PATH, output_dir=config.OUTPUT_DIR):
        super().__init__()
        self.weights, self.mock = Path(weights), mock
        self.backend = None
        self.storage = EventStorage(db_path, output_dir)

    @pyqtSlot()
    def initialize(self):
        try:
            self.consistency.emit(self.storage.consistency_report())
        except Exception as exc:
            LOG.exception("Startup evidence check failed")
            self.consistency.emit({"error": str(exc)})
        self.load_model(self.weights)

    @pyqtSlot(object)
    def load_model(self, weights):
        self.backend = None
        try:
            self.progress.emit("กำลังโหลดโมเดล…")
            self.backend = MockBackend() if self.mock else YoloBackend(Path(weights))
            text = ("โหมดจำลอง: กล่องตัวอย่างสำหรับทดสอบหน้าจอ ไม่บันทึกในประวัติจริง"
                    if self.mock else f"โมเดลพร้อม: {self.backend.model_id}")
            self.model_loaded.emit(True, text)
        except Exception as exc:
            LOG.exception("Model unavailable")
            self.model_loaded.emit(False, f"โมเดลไม่พร้อม — {exc}")
        finally:
            self.done.emit()

    @pyqtSlot(object)
    def detect(self, request):
        try:
            if self.backend is None:
                raise RuntimeError("ยังไม่มีโมเดลพร้อมใช้งาน")
            self.progress.emit("กำลังตรวจจับ…")
            result = run_detection(self.backend, request)
            self.progress.emit("กำลังบันทึกหลักฐาน…")
            saved = self.storage.save(result)
            self.result_ready.emit(result, saved)
        except Exception as exc:
            LOG.exception("Detection failed")
            self.failed.emit(f"ตรวจจับไม่สำเร็จ: {exc}")
        finally:
            self.done.emit()

    @pyqtSlot(object)
    def retry_save(self, result):
        try:
            self.progress.emit("กำลังบันทึกผลเดิมซ้ำ…")
            self.result_ready.emit(result, self.storage.save(result))
        except Exception as exc:
            LOG.exception("Save retry failed")
            self.failed.emit(f"บันทึกไม่สำเร็จ: {exc}")
        finally:
            self.done.emit()
