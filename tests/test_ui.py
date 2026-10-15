import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.detector import MockBackend
from app.models import DetectionRequest
from main import MainWindow, configure_application


class FakeRealBackend(MockBackend):
    is_mock = False
    model_id, sha256 = "test-real-pipeline-fixture", "a" * 64

    def __init__(self, weights):
        pass


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_application(cls.app)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "ภาพทดสอบ.png"
        Image.new("RGB", (200, 160), "white").save(self.path)
        self.dialogs = patch.object(QMessageBox, "information", return_value=QMessageBox.StandardButton.Ok)
        self.errors = patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok)
        self.dialog_mock = self.dialogs.start()
        self.errors.start()
        self.window = None

    def wait_for(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.002)
        self.app.processEvents()
        self.assertTrue(predicate(), "Timed out waiting for Qt worker")

    def open_window(self, mock=True):
        self.window = MainWindow(self.root / "missing.pt", mock, self.root / "db.sqlite", self.root / "outputs")
        self.window.show()
        self.wait_for(lambda: not self.window.state.busy)
        return self.window

    def tearDown(self):
        if self.window:
            self.wait_for(lambda: not self.window.state.busy)
            self.window.close()
            self.app.processEvents()
            self.assertFalse(self.window.state.thread.isRunning())
        self.dialogs.stop()
        self.errors.stop()
        self.temp.cleanup()

    def test_missing_model_does_not_silently_start_mock(self):
        window = self.open_window(mock=False)
        self.assertFalse(window.state.model_ready)
        window.inspection.select_image(self.path)
        self.assertFalse(window.inspection.btn_run.isEnabled())
        self.assertFalse(window.state.mock)

    def test_mock_multi_class_dialog_no_persistence_and_reset(self):
        window = self.open_window()
        page = window.inspection
        page.select_image(self.path)
        page.start_detection()
        self.assertFalse(page.btn_browse.isEnabled())
        self.assertFalse(window.state.detect(self.path, .25, .45, 640))
        self.wait_for(lambda: not window.state.busy)
        self.assertEqual(page.result_view.table.rowCount(), 2)
        self.assertEqual(self.dialog_mock.call_count, 1)
        self.assertEqual(window.state.saved.status, "mock")
        self.assertFalse((self.root / "outputs").exists())
        page.select_image(self.path)
        self.assertEqual(page.result_view.table.rowCount(), 0)
        self.assertEqual(page.result_view.annotated.item, None)

    def test_no_detection_creates_no_alert(self):
        window = self.open_window()
        window.inspection.select_image(self.path)
        window.inspection.conf.setValue(.99)
        window.inspection.start_detection()
        self.wait_for(lambda: not window.state.busy)
        self.assertEqual(window.inspection.result_view.table.rowCount(), 0)
        self.assertIn("ไม่พบความเสียหายตามเกณฑ์", window.inspection.result_view.summary.text())
        self.dialog_mock.assert_not_called()

    def test_real_pipeline_fixture_history_and_missing_evidence(self):
        # Exercises integration with a deterministic fixture, not model accuracy.
        with patch("app.detector.YoloBackend", FakeRealBackend):
            window = self.open_window(mock=False)
        window.inspection.select_image(self.path)
        window.inspection.start_detection()
        self.wait_for(lambda: not window.state.busy)
        self.assertEqual(window.state.saved.status, "saved")
        window.tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(window.history.table.rowCount(), 1)
        window.history.table.selectRow(0)
        self.app.processEvents()
        self.assertEqual(window.history.details.table.rowCount(), 2)
        Path(window.state.saved.result_path).unlink()
        window.history.show_selected()
        self.assertIn("ไม่พบไฟล์หลักฐาน", window.history.details.save_status.text())
        self.assertIsNone(window.history.details.annotated.item)

    def test_invalid_image_clears_previous_selection(self):
        window = self.open_window()
        window.inspection.select_image(self.path)
        invalid = self.root / "bad.png"
        invalid.write_text("not image")
        self.assertFalse(window.inspection.select_image(invalid))
        self.assertIsNone(window.inspection.path)
        self.assertFalse(window.inspection.btn_run.isEnabled())

    def test_single_class_and_close_during_active_job(self):
        window = self.open_window()
        window.inspection.select_image(self.path)
        window.inspection.conf.setValue(.85)
        window.inspection.start_detection()
        self.assertFalse(window.close())
        self.wait_for(lambda: not window.state.busy)
        self.assertEqual(window.inspection.result_view.table.rowCount(), 1)
        self.assertEqual(self.dialog_mock.call_count, 1)

    def test_retry_save_does_not_duplicate_alert_or_event(self):
        from app.db import Repository
        with patch("app.detector.YoloBackend", FakeRealBackend):
            window = self.open_window(mock=False)
        window.inspection.select_image(self.path)
        with patch.object(Repository, "save_event", side_effect=OSError("test disk full")):
            window.inspection.start_detection()
            self.wait_for(lambda: not window.state.busy)
        self.assertEqual(window.state.saved.status, "failed")
        self.assertTrue(window.inspection.btn_retry.isVisible())
        self.assertTrue(window.state.retry_save())
        self.wait_for(lambda: not window.state.busy)
        self.assertEqual(window.state.saved.status, "saved")
        self.assertEqual(self.dialog_mock.call_count, 1)
        with Repository(window.state.db_path) as repo:
            self.assertEqual(len(repo.history()), 1)


if __name__ == "__main__":
    unittest.main()
