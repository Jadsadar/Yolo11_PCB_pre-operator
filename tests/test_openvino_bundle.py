import json
from pathlib import Path
import tempfile
import unittest

from app.config import CLASS_NAMES
from app.model_registry import read_metadata, sha256_file


class OpenVinoBundleTests(unittest.TestCase):
    def test_all_runtime_files_are_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xml = root / "best.xml"
            for name in ("best.xml", "best.bin", "metadata.yaml"):
                (root / name).write_text("fixture", encoding="utf-8")
            metadata = dict(model_id="fixture-openvino", architecture="YOLO11", task="detect",
                            format="openvino", class_names=list(CLASS_NAMES), sha256=sha256_file(xml),
                            files={name: sha256_file(root / name)
                                   for name in ("best.xml", "best.bin", "metadata.yaml")})
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(read_metadata(xml)["model_id"], "fixture-openvino")
            for name in ("best.bin", "metadata.yaml"):
                with self.subTest(file=name):
                    (root / name).write_text("tampered", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        read_metadata(xml)
                    (root / name).write_text("fixture", encoding="utf-8")
            metadata["files"].pop("best.bin")
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "XML, BIN"):
                read_metadata(xml)


if __name__ == "__main__":
    unittest.main()
