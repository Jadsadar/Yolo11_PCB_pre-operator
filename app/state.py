"""Controller owns the worker thread and snapshots every inspection request."""
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from . import config
from .detector import DetectionWorker
from .models import DetectionRequest


class AppState(QObject):
    busy_changed = pyqtSignal(bool)
    model_changed = pyqtSignal(bool, str)
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(object, object)
    failed = pyqtSignal(str)
    consistency = pyqtSignal(object)
    request_detection = pyqtSignal(object)
    request_model = pyqtSignal(object)
    request_save = pyqtSignal(object)

    def __init__(self, weights=config.MODEL_PATH, mock=False,
                 db_path=config.DB_PATH, output_dir=config.OUTPUT_DIR):
        super().__init__()
        self.weights, self.mock, self.db_path = Path(weights), mock, Path(db_path)
        self.busy, self.model_ready = True, False
        self.result, self.saved = None, None
        self.thread = QThread(self)
        self.worker = DetectionWorker(weights, mock, db_path, output_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.initialize)
        self.thread.finished.connect(self.worker.deleteLater)
        self.request_detection.connect(self.worker.detect)
        self.request_model.connect(self.worker.load_model)
        self.request_save.connect(self.worker.retry_save)
        self.worker.model_loaded.connect(self._model_loaded)
        self.worker.result_ready.connect(self._result_ready)
        self.worker.failed.connect(self.failed)
        self.worker.progress.connect(self.progress)
        self.worker.consistency.connect(self.consistency)
        self.worker.done.connect(self._done)

    def start(self):
        self.thread.start()

    def _set_busy(self, value):
        self.busy = value
        self.busy_changed.emit(value)

    @pyqtSlot(bool, str)
    def _model_loaded(self, ready, message):
        self.model_ready = ready
        self.model_changed.emit(ready, message)

    @pyqtSlot(object, object)
    def _result_ready(self, result, saved):
        self.result, self.saved = result, saved
        self.result_ready.emit(result, saved)

    @pyqtSlot()
    def _done(self):
        self._set_busy(False)

    def detect(self, path, conf, iou, imgsz):
        if self.busy or not self.model_ready:
            return False
        request = DetectionRequest(Path(path).resolve(), conf, iou, imgsz)
        self.result = self.saved = None
        self._set_busy(True)
        self.request_detection.emit(request)
        return True

    def reload_model(self, weights):
        if self.busy:
            return False
        self.weights = Path(weights)
        self.model_ready = False
        self._set_busy(True)
        self.request_model.emit(self.weights)
        return True

    def retry_save(self):
        if self.busy or self.result is None or self.saved.status != "failed":
            return False
        self._set_busy(True)
        self.request_save.emit(self.result)
        return True

    def close(self):
        if self.busy:
            return False
        self.thread.quit()
        self.thread.wait()
        return True
