"""Single-image inspection, model readiness, and one alert per result."""
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                            QMessageBox, QProgressBar, QPushButton, QSpinBox,
                            QVBoxLayout, QWidget)

from .. import config
from ..image_service import read_image
from .result_page import ResultPage, class_summary


class UploadPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state, self.path = state, None
        self._last_alert_event = None
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_browse = QPushButton("เลือกภาพ PCB…")
        self.btn_browse.clicked.connect(self.browse)
        self.file_label = QLabel("รองรับ JPG / JPEG / PNG ครั้งละหนึ่งภาพ")
        self.file_label.setTextFormat(Qt.TextFormat.PlainText)
        self.file_label.setWordWrap(True)
        self.btn_run = QPushButton("ตรวจจับ")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.start_detection)
        self.btn_run.setEnabled(False)
        top.addWidget(self.btn_browse)
        top.addWidget(self.file_label, 1)
        top.addWidget(self.btn_run)
        layout.addLayout(top)
        params = QHBoxLayout()
        self.conf, self.iou = QDoubleSpinBox(), QDoubleSpinBox()
        for control, value in [(self.conf, config.DEFAULT_CONF), (self.iou, config.DEFAULT_IOU)]:
            control.setRange(0, 1)
            control.setDecimals(2)
            control.setSingleStep(.05)
            control.setValue(value)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 2048)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(config.DEFAULT_IMGSZ)
        for text, control in [("Confidence", self.conf), ("NMS IoU", self.iou), ("Image size", self.imgsz)]:
            params.addWidget(QLabel(text))
            params.addWidget(control)
        params.addStretch(1)
        self.btn_retry = QPushButton("ลองบันทึกผลเดิมซ้ำ")
        self.btn_retry.clicked.connect(self.state.retry_save)
        self.btn_retry.hide()
        params.addWidget(self.btn_retry)
        layout.addLayout(params)
        self.result_view = ResultPage()
        layout.addWidget(self.result_view, 1)
        self.status = QLabel("กำลังเตรียมระบบ…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        state.busy_changed.connect(self.refresh_controls)
        state.model_changed.connect(lambda *_: self.refresh_controls(state.busy))
        state.progress.connect(self.status.setText)
        state.failed.connect(self.on_error)
        state.result_ready.connect(self.on_result)
        self.refresh_controls(state.busy)

    def refresh_controls(self, busy):
        self.btn_browse.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and self.path is not None and self.state.model_ready)
        for control in (self.conf, self.iou, self.imgsz, self.btn_retry):
            control.setEnabled(not busy)
        self.progress.setVisible(busy)
        if not busy:
            self.status.setText("พร้อมตรวจภาพ" if self.state.model_ready else "รอโมเดล PCB ที่ฝึกแล้ว — ใช้ --mock เพื่อทดสอบหน้าจอ")

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกภาพ PCB", "", config.IMAGE_FILTER)
        if path:
            self.select_image(Path(path))

    def select_image(self, path):
        if self.state.busy:
            return False
        self.path = None
        self.result_view.clear()
        self.btn_retry.hide()
        self.state.result = self.state.saved = None
        try:
            image = read_image(Path(path))
            self.path = Path(path).resolve()
            self.file_label.setText(self.path.name)
            self.file_label.setToolTip(str(self.path))
            self.result_view.original.set_image(image)
            self.refresh_controls(False)
            return True
        except Exception as exc:
            self.file_label.setText("กรุณาเลือกภาพที่ถูกต้อง")
            self.refresh_controls(False)
            self.on_error(str(exc))
            return False

    def start_detection(self):
        if self.path is None:
            return
        try:
            if self.state.detect(self.path, self.conf.value(), self.iou.value(), self.imgsz.value()):
                self.result_view.annotated.clear()
                self.result_view.table.setRowCount(0)
                self.result_view.summary.setText("กำลังตรวจภาพ…")
                self.result_view.save_status.clear()
                self.btn_retry.hide()
        except Exception as exc:
            self.on_error(str(exc))

    def on_error(self, message):
        self.status.setText(message)
        self.result_view.summary.setText(message)
        QMessageBox.warning(self, "ทำรายการไม่สำเร็จ", message)

    def on_result(self, result, saved):
        self.result_view.show_result(result, saved)
        self.btn_retry.setVisible(saved.status == "failed")
        if result.detections and self._last_alert_event != result.event_id:
            self._last_alert_event = result.event_id
            title = "ผลจำลอง — ทดสอบหน้าจอ" if result.is_mock else "พบความเสียหายบน PCB"
            QMessageBox.information(self, title,
                f"ภาพ: {result.request.source_path.name}\n"
                f"พบ {len(result.detections)} ตำแหน่ง\n{class_summary(result.detections)}\n\n"
                f"{self.result_view.save_status.text()}")
