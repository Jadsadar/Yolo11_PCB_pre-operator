"""Exercise the installed YOLO API with random test weights; never train a model."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from app import config
from app.detector import YoloBackend, run_detection
from app.model_registry import sha256_file
from app.models import DetectionRequest
from app.runtime import prepare_ultralytics
from scripts.export_model import main as export_main


class YoloAdapterTests(unittest.TestCase):
    def test_real_api_roundtrip_without_training_or_download(self):
        prepare_ultralytics()
        from ultralytics import YOLO

        package_dir = Path(importlib.util.find_spec("ultralytics").submodule_search_locations[0])
        specification = yaml.safe_load((package_dir / "cfg/models/11/yolo11.yaml").read_text(encoding="utf-8"))
        specification["nc"] = 6
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            architecture = root / "yolo11n.yaml"
            architecture.write_text(yaml.safe_dump(specification), encoding="utf-8")
            # Architecture initialization plus inference is not training. No train() call here.
            model = YOLO(str(architecture), task="detect")
            model.model.names = dict(enumerate(config.CLASS_NAMES))
            weights = root / "best.pt"
            model.save(weights)
            experiment = root / "experiment.json"
            experiment.write_text(json.dumps({"settings": {"model": "yolo11n.pt"},
                                               "dataset_manifest_sha256": "test-fixture"}))
            bundle = root / "bundle"
            argv = ["export_model", "--weights", str(weights), "--experiment", str(experiment),
                    "--output", str(bundle), "--model-id", "random-test-fixture"]
            with patch("sys.argv", argv):
                self.assertEqual(export_main(), 0)
            self.assertEqual(sha256_file(bundle / "best.pt"), sha256_file(weights))
            image = root / "input.png"
            Image.new("RGB", (64, 64), "white").save(image)
            backend = YoloBackend(bundle / "best.pt")
            result = run_detection(backend, DetectionRequest(image, conf=.99, imgsz=64))
            self.assertFalse(result.is_mock)
            self.assertEqual(result.model_id, "random-test-fixture")
            self.assertEqual(result.original.size, (64, 64))
            self.assertEqual(result.detections, ())


if __name__ == "__main__":
    unittest.main()
