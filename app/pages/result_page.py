"""Reusable original/result comparison and detection details."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QGroupBox, QHBoxLayout, QHeaderView,
                            QLabel, QSplitter, QTableWidget, QTableWidgetItem,
                            QVBoxLayout, QWidget)

from .. import config
from ..image_view import ImageView


def class_summary(detections):
    counts = {}
    for d in detections:
        counts[d.class_id] = counts.get(d.class_id, 0) + 1
    return ", ".join(f"{config.class_label(cid)} ({config.class_name(cid)}): {n}"
                     for cid, n in sorted(counts.items()))


class ResultPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original = ImageView("ภาพต้นฉบับ")
        self.annotated = ImageView("ภาพผลลัพธ์")
        for title, view in [("ภาพต้นฉบับ", self.original), ("ผลตรวจจับ", self.annotated)]:
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            group_layout.addWidget(view)
            splitter.addWidget(group)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, 3)
        self.summary = QLabel("เลือกภาพ PCB เพื่อเริ่มต้น")
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["คลาส", "Confidence", "x1", "y1", "x2", "y2", "ประเภท"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setMaximumHeight(190)
        layout.addWidget(self.table, 1)
        self.save_status = QLabel("")
        self.save_status.setWordWrap(True)
        self.save_status.setTextFormat(Qt.TextFormat.PlainText)
        self.save_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.save_status)

    def clear(self):
        self.original.clear()
        self.annotated.clear()
        self.table.setRowCount(0)
        self.summary.setText("ยังไม่มีผลตรวจจับ")
        self.save_status.clear()

    def fill_detections(self, detections):
        self.table.setRowCount(len(detections))
        for row, d in enumerate(sorted(detections, key=lambda item: -item.confidence)):
            values = [d.class_label, f"{d.confidence:.3f}", f"{d.x1:.1f}", f"{d.y1:.1f}",
                      f"{d.x2:.1f}", f"{d.y2:.1f}", d.class_name]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def show_result(self, result, saved):
        self.original.set_image(result.original)
        self.annotated.set_image(result.annotated)
        self.fill_detections(result.detections)
        prefix = "[ผลจำลองสำหรับทดสอบหน้าจอ] " if result.is_mock else ""
        message = (f"พบความเสียหาย {len(result.detections)} ตำแหน่ง — {class_summary(result.detections)}"
                   if result.detections else "ไม่พบความเสียหายตามเกณฑ์ที่ตั้งไว้")
        self.summary.setText(f"{prefix}{message}\nโมเดล: {result.model_id} | "
                             f"inference {result.inference_ms:.1f} ms | อ่านและตรวจภาพ {result.total_ms:.1f} ms\n"
                             f"Confidence ≥ {result.request.conf:.2f} | NMS IoU {result.request.iou:.2f} | "
                             f"ขนาดเข้าโมเดล {result.request.imgsz}")
        messages = {"saved": f"บันทึกสำเร็จ: {saved.result_path}",
                    "failed": f"ตรวจพบความเสียหาย แต่บันทึกหลักฐานไม่สำเร็จ: {saved.error}",
                    "not_required": "ไม่สร้างเหตุการณ์ เนื่องจากไม่พบกรอบผ่านเกณฑ์",
                    "mock": "โหมดจำลอง — ไม่บันทึกภาพและไม่เพิ่มประวัติจริง"}
        self.save_status.setText(messages[saved.status])
