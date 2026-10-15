"""Search saved defect events; never substitute a missing evidence image."""
from datetime import date
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView,
                            QLabel, QLineEdit, QPushButton, QTableWidget,
                            QTableWidgetItem, QVBoxLayout, QWidget)

from .. import config
from ..db import Repository
from .result_page import ResultPage, class_summary


class HistoryPage(QWidget):
    def __init__(self, db_path=config.DB_PATH):
        super().__init__()
        self.db_path, self.rows = Path(db_path), []
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.keyword = QLineEdit()
        self.keyword.setPlaceholderText("ค้นหาชื่อภาพ")
        self.class_filter = QComboBox()
        self.class_filter.addItem("ทุกคลาส", None)
        for cid, info in config.DAMAGE_CLASSES.items():
            self.class_filter.addItem(info[1], cid)
        self.date_from, self.date_to = QLineEdit(), QLineEdit()
        self.date_from.setPlaceholderText("ตั้งแต่ YYYY-MM-DD")
        self.date_to.setPlaceholderText("ถึง YYYY-MM-DD")
        self.refresh_button = QPushButton("ค้นหา / รีเฟรช")
        self.refresh_button.clicked.connect(self.refresh)
        for widget in (self.keyword, self.class_filter, self.date_from, self.date_to, self.refresh_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.note = QLabel("แสดงเฉพาะเหตุการณ์ที่พบตำหนิและบันทึกสำเร็จ")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["วันเวลา", "ไฟล์ต้นฉบับ", "คลาส", "ตำแหน่ง", "Confidence สูงสุด", "โมเดล"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.table.setMaximumHeight(200)
        layout.addWidget(self.table)
        self.details = ResultPage()
        layout.addWidget(self.details, 1)

    def refresh(self):
        try:
            start, end = self.date_from.text().strip(), self.date_to.text().strip()
            for value in (start, end):
                if value:
                    date.fromisoformat(value)
            if start and end and start > end:
                raise ValueError("วันที่เริ่มต้องไม่เกินวันที่สิ้นสุด")
            with Repository(self.db_path) as repo:
                self.rows = repo.history(self.keyword.text().strip(), self.class_filter.currentData(), start, end)
            self.table.blockSignals(True)
            self.table.setRowCount(len(self.rows))
            for row, event in enumerate(self.rows):
                values = [event["detected_at"], event["original_name"], event["classes"],
                          str(event["detection_count"]), f'{event["max_confidence"]:.3f}', event["model_id"]]
                for col, value in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(value))
            self.table.clearSelection()
            self.table.blockSignals(False)
            self.details.clear()
            self.note.setText(f"พบ {len(self.rows)} เหตุการณ์ที่บันทึกสำเร็จ — ไม่รวมผลไม่พบตำหนิหรือผลจำลอง")
        except Exception as exc:
            self.note.setText(f"อ่านประวัติไม่สำเร็จ: {exc}")

    def show_selected(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.rows):
            return
        event = self.rows[row]
        try:
            with Repository(self.db_path) as repo:
                detections = repo.detections_of(event["id"])
            self.details.clear()
            self.details.fill_detections(detections)
            self.details.summary.setText(
                f'{event["original_name"]} | {event["detected_at"]}\n'
                f'{event["detection_count"]} ตำแหน่ง — {class_summary(detections)}\n'
                f'โมเดล {event["model_id"]} | inference {event["inference_ms"]:.1f} ms | '
                f'Confidence {event["conf_threshold"]} | NMS IoU {event["iou_threshold"]}')
            path = Path(event["result_path"])
            if path.is_file():
                self.details.annotated.set_image(str(path))
                self.details.save_status.setText(f"หลักฐาน: {path}")
            else:
                self.details.save_status.setText(f"ไม่พบไฟล์หลักฐาน: {path}")
        except Exception as exc:
            self.details.save_status.setText(f"อ่านรายละเอียดไม่สำเร็จ: {exc}")
