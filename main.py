"""Run the PCB desktop app. Training is available only in scripts/train.py."""
import argparse
import logging
import os
from pathlib import Path
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                            QMainWindow, QPushButton, QTabWidget, QVBoxLayout, QWidget)

from app import config
from app.pages.history_page import HistoryPage
from app.pages.upload_page import UploadPage
from app.state import AppState


def configure_application(application):
    """Register Windows Thai fonts explicitly, including in offscreen Qt sessions."""
    application.setStyle("Fusion")
    font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for filename in ("tahoma.ttf", "tahomabd.ttf", "segoeui.ttf", "segoeuib.ttf"):
        path = font_dir / filename
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    application.setFont(QFont("Tahoma", 10))


class MainWindow(QMainWindow):
    def __init__(self, weights=config.MODEL_PATH, mock=False,
                 db_path=config.DB_PATH, output_dir=config.OUTPUT_DIR):
        super().__init__()
        self.setWindowTitle(config.APP_NAME + (" — โหมดจำลอง" if mock else ""))
        self.resize(1220, 900)
        self.setMinimumSize(850, 650)
        self.state = AppState(weights, mock, db_path, output_dir)
        widget = QWidget()
        self.setCentralWidget(widget)
        root = QVBoxLayout(widget)
        heading = QLabel(config.APP_NAME)
        heading.setStyleSheet("font-size:22px;font-weight:600;color:#123c56;padding:8px")
        root.addWidget(heading)
        model_row = QHBoxLayout()
        self.model_label = QLabel("กำลังโหลดโมเดล…")
        self.model_label.setTextFormat(Qt.TextFormat.PlainText)
        self.model_label.setWordWrap(True)
        model_row.addWidget(self.model_label, 1)
        self.reload_button = QPushButton("เลือกโมเดล / โหลดใหม่…")
        self.reload_button.setEnabled(False)
        self.reload_button.clicked.connect(self.choose_model)
        model_row.addWidget(self.reload_button)
        root.addLayout(model_row)
        self.consistency_label = QLabel("")
        self.consistency_label.setWordWrap(True)
        self.consistency_label.hide()
        root.addWidget(self.consistency_label)
        self.tabs = QTabWidget()
        self.inspection = UploadPage(self.state)
        self.history = HistoryPage(db_path)
        self.tabs.addTab(self.inspection, "ตรวจภาพ")
        self.tabs.addTab(self.history, "ประวัติการตรวจพบ")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        root.addWidget(self.tabs, 1)
        self.state.model_changed.connect(lambda ready, text: self.model_label.setText(text))
        self.state.busy_changed.connect(self.on_busy)
        self.state.consistency.connect(self.on_consistency)
        self.state.start()

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือก best.pt พร้อม metadata.json ในโฟลเดอร์เดียวกัน",
                                             str(self.state.weights.parent), "YOLO weights (*.pt)")
        if path:
            self.state.reload_model(Path(path))

    def on_busy(self, busy):
        self.reload_button.setEnabled(not busy and not self.state.mock)
        self.tabs.setTabEnabled(1, not busy)

    def on_tab_changed(self, index):
        if index == 1:
            self.history.refresh()

    def on_consistency(self, report):
        count = sum(len(report.get(key, [])) for key in ("missing", "orphan", "temporary"))
        if count or report.get("error"):
            text = (f"ตรวจหลักฐานพบ {count} รายการที่ต้องตรวจสอบ — ดู logs/pcb.log"
                    if count else f'ตรวจหลักฐานไม่สำเร็จ: {report["error"]}')
            self.consistency_label.setText(text)
            self.consistency_label.show()
            logging.getLogger(__name__).warning("Evidence consistency: %s", report)

    def closeEvent(self, event):
        if self.state.busy:
            self.statusBar().showMessage("กำลังทำงาน กรุณารอให้เสร็จก่อนปิดแอป", 5000)
            event.ignore()
        else:
            self.state.close()
            event.accept()


def main():
    parser = argparse.ArgumentParser(description="Offline PCB defect inspection with YOLO11")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--mock", action="store_true", help="Explicit UI demo; no real event storage")
    args = parser.parse_args()
    log_dir = config.BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(filename=log_dir / "pcb.log", encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    application = QApplication(sys.argv[:1])
    configure_application(application)
    application.setStyleSheet("""
        QWidget { font-family: 'Segoe UI', 'Tahoma'; font-size: 12px; }
        QMainWindow { background: #f4f7fa; }
        QPushButton { padding: 7px 12px; border: 1px solid #b8c8d4; border-radius: 5px; background: white; }
        QPushButton#primary { background: #126e86; color: white; font-weight: bold; }
        QPushButton:disabled { color: #87929b; background: #edf0f3; }
        QGroupBox { font-weight: bold; border: 1px solid #ced9e1; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        QTableWidget, QGraphicsView { background: white; border: 1px solid #d8e0e6; }
        QTabWidget::pane { border: 1px solid #d5dfe7; }
        QTabBar::tab { padding: 10px 22px; }
    """)
    window = MainWindow(args.model, args.mock)
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
