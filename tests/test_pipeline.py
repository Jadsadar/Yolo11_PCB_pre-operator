import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from app import config
from app.db import Repository
from app.detector import MockBackend, run_detection
from app.image_service import read_image
from app.model_registry import read_metadata, sha256_file, validate_names
from app.models import Detection, DetectionRequest, DetectionResult
from app.storage import EventStorage
from scripts.dataset import audit_dataset, load_dataset
from scripts.train import training_plan


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "ภาพทดสอบ.png"
        Image.new("RGB", (120, 100), "white").save(self.source)
        self.db = self.root / "data/pcb.db"
        self.outputs = self.root / "outputs"
        self.storage = EventStorage(self.db, self.outputs)

    def tearDown(self):
        self.temp.cleanup()

    def result(self, detections=None, mock=False):
        image = read_image(self.source)
        if detections is None:
            detections = (Detection(0, .9, 10, 10, 25, 30), Detection(3, .8, 50, 50, 75, 70))
        return DetectionResult(DetectionRequest(self.source), image, image.copy(), tuple(detections),
                               12.5, 15., "pcb-test", "a" * 64, mock,
                               detected_at="2026-10-15T00:30:00+07:00")

    def test_store_reopen_filter_and_counts(self):
        result = self.result()
        saved = self.storage.save(result)
        self.assertEqual(saved.status, "saved")
        self.assertTrue(Path(saved.result_path).is_file())
        self.assertIn("2026/10/15", Path(saved.result_path).as_posix())
        with Repository(self.db) as repo:
            history = repo.history(class_id=0, date_from="2026-10-15", date_to="2026-10-15")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["detection_count"], 2)
            self.assertEqual(history[0]["max_confidence"], .9)
            self.assertEqual(len(repo.detections_of(history[0]["id"])), 2)
            self.assertEqual(repo.history(class_id=2), [])

    def test_same_event_retry_is_idempotent(self):
        result = self.result()
        first, second = self.storage.save(result), self.storage.save(result)
        self.assertEqual(first, second)
        with Repository(self.db) as repo:
            self.assertEqual(len(repo.history()), 1)

    def test_same_second_different_events_do_not_overwrite(self):
        first, second = self.storage.save(self.result()), self.storage.save(self.result())
        self.assertEqual(first.status, "saved")
        self.assertNotEqual(first.result_path, second.result_path)
        self.assertTrue(Path(first.result_path).exists())

    def test_mock_and_no_detection_do_not_create_evidence(self):
        self.assertEqual(self.storage.save(self.result(mock=True)).status, "mock")
        self.assertEqual(self.storage.save(self.result(detections=())).status, "not_required")
        self.assertFalse(self.db.exists())
        self.assertFalse(self.outputs.exists())

    def test_transaction_rollback_removes_partial_rows(self):
        with Repository(self.db) as repo:
            repo._conn.execute("CREATE TRIGGER fail_detection BEFORE INSERT ON detections BEGIN SELECT RAISE(ABORT, 'test failure'); END")
            repo._conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                repo.save_event(self.result(), self.root / "x.png")
            self.assertEqual(repo._conn.execute("SELECT count(*) FROM images").fetchone()[0], 0)
            self.assertEqual(repo.history(), [])

    def test_storage_compensates_failed_database_write_and_can_retry(self):
        result = self.result()
        with patch.object(Repository, "save_event", side_effect=sqlite3.OperationalError("disk full")):
            saved = self.storage.save(result)
        self.assertEqual(saved.status, "failed")
        self.assertFalse(list(self.outputs.rglob("*.png")))
        self.assertFalse(list(self.outputs.rglob("*.tmp")))
        self.assertEqual(self.storage.save(result).status, "saved")

    def test_cannot_write_evidence_directory(self):
        self.outputs.write_text("not a directory")
        self.assertEqual(self.storage.save(self.result()).status, "failed")
        with Repository(self.db) as repo:
            self.assertEqual(repo.history(), [])

    def test_consistency_reports_missing_and_orphan_without_deleting(self):
        saved = self.storage.save(self.result())
        Path(saved.result_path).unlink()
        orphan = self.outputs / "2026/10/15/orphan_annotated.png"
        Image.new("RGB", (10, 10)).save(orphan)
        report = self.storage.consistency_report()
        self.assertEqual(report["missing"], [saved.result_path])
        self.assertEqual(report["orphan"], [str(orphan)])
        self.assertTrue(orphan.exists())

    def test_validation_rejects_invalid_boxes_and_thresholds(self):
        for detection in (Detection(6, .9, 1, 1, 10, 10), Detection(0, float("nan"), 1, 1, 10, 10),
                          Detection(0, .9, 1, 1, 999, 10)):
            with self.assertRaises(ValueError):
                detection.validate(120, 100)
        with self.assertRaises(ValueError):
            DetectionRequest(self.source, imgsz=641)

    def test_image_validation_and_exif_orientation(self):
        bad = self.root / "bad.png"
        bad.write_text("not an image")
        with self.assertRaises(ValueError):
            read_image(bad)
        disguised = self.root / "disguised.png"
        Image.new("RGB", (20, 30)).save(disguised, format="BMP")
        with self.assertRaises(ValueError):
            read_image(disguised)
        path = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (20, 30)).save(path, exif=exif)
        self.assertEqual(read_image(path).size, (30, 20))

    def test_model_mapping_checksum_and_missing_model(self):
        weights = self.root / "best.pt"
        with self.assertRaises(FileNotFoundError):
            read_metadata(weights)
        weights.write_bytes(b"fixture-only; not a trained model")
        metadata = {"model_id": "test", "architecture": "YOLO11", "task": "detect",
                    "class_names": list(config.CLASS_NAMES), "sha256": sha256_file(weights)}
        path = self.root / "metadata.json"
        path.write_text(json.dumps(metadata))
        self.assertEqual(read_metadata(weights)["model_id"], "test")
        weights.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            read_metadata(weights)
        with self.assertRaises(ValueError):
            validate_names(["pothole"])

    def test_failed_optional_codec_import_is_reported_as_invalid_image(self):
        with patch("app.image_service.Image.open", side_effect=ModuleNotFoundError("pi_heif")):
            with self.assertRaises(ValueError):
                read_image(self.source)

    def test_mock_deterministic_multi_class_and_no_detection(self):
        backend = MockBackend()
        first = run_detection(backend, DetectionRequest(self.source))
        second = run_detection(backend, DetectionRequest(self.source))
        self.assertEqual(first.detections, second.detections)
        self.assertEqual(set(first.counts), {0, 3})
        self.assertTrue(first.is_mock)
        high = run_detection(backend, DetectionRequest(self.source, conf=.99))
        self.assertEqual(high.detections, ())

    def make_dataset(self):
        for split in ("train", "valid", "test"):
            images, labels = self.root / split / "images", self.root / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            Image.new("RGB", (20, 20), "white").save(images / f"{split}.png")
            (labels / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        path = self.root / "data.yaml"
        path.write_text(yaml.safe_dump({"path": ".", "train": "train/images", "val": "valid/images",
                                       "test": "test/images", "nc": 6, "names": list(config.CLASS_NAMES)}))
        return path

    def test_dataset_audit_detects_cross_split_duplicates_and_invalid_label(self):
        data = self.make_dataset()
        report = audit_dataset(data)
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["exact_cross_split_duplicates"]), 1)
        self.assertEqual(Path(load_dataset(data)["train"]), self.root / "train/images")
        (self.root / "test/labels/test.txt").write_text("6 0.5 0.5 2 2")
        self.assertTrue(audit_dataset(data)["errors"])

    def test_training_plan_does_not_load_or_train_model(self):
        data = self.make_dataset()
        settings = self.root / "train.yaml"
        settings.write_text(yaml.safe_dump({"data": str(data), "model": "yolo11n.pt", "project": "runs",
                                           "name": "test", "imgsz": 640}))
        with patch.dict("sys.modules", {"ultralytics": None}):
            plan, _ = training_plan(settings)
        self.assertFalse(plan["training_started"])
        self.assertEqual(plan["mode"], "dry-run")
        self.assertTrue(plan["settings"]["tensorboard"])
        self.assertFalse((self.root / "runs").exists())
        self.assertTrue(plan["leakage_review_required"])
        self.assertTrue(plan["blockers"])


if __name__ == "__main__":
    unittest.main()
