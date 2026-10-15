"""Simulate epoch checkpoints and abrupt writes without starting model training."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.config import CLASS_NAMES
from scripts.colab_train import CheckpointSnapshots, latest_run, main, select_checkpoint, write_json


def fixture_model(path):
    return SimpleNamespace(ckpt=json.loads(Path(path).read_text()), names=list(CLASS_NAMES))


class DriveCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name)
        self.run = self.project / "pcb_20261015_000001"
        (self.run / "weights").mkdir(parents=True)
        write_json(self.run / "experiment.json", {"dataset_manifest_sha256": "fixture"})

    def tearDown(self):
        self.temp.cleanup()

    def save_epoch(self, epoch):
        last = self.run / "weights/last.pt"
        last.write_text(json.dumps({"epoch": epoch, "optimizer": {"fixture": True},
                                    "train_args": {"epochs": 100}}))
        CheckpointSnapshots(self.run)(SimpleNamespace(last=last, epoch=epoch, epochs=100))

    def test_snapshots_rotate_and_resume_latest_completed_epoch(self):
        self.save_epoch(0)
        self.save_epoch(1)
        progress = json.loads((self.run / "progress.json").read_text())
        self.assertEqual(progress["completed_epochs"], 2)
        state, model, path = select_checkpoint(self.run, fixture_model)
        self.assertEqual(state, "resume")
        self.assertEqual(model.ckpt["epoch"], 1)
        self.assertEqual(path.name, "resume.pt")
        self.assertEqual(fixture_model(self.run / "weights/resume_previous.pt").ckpt["epoch"], 0)

    def test_abrupt_write_can_fall_back_to_previous_snapshot(self):
        self.save_epoch(0)
        self.save_epoch(1)
        (self.run / "weights/last.pt").write_text("truncated")
        (self.run / "weights/resume.pt").write_text("truncated")
        state, model, path = select_checkpoint(self.run, fixture_model)
        self.assertEqual(state, "resume")
        self.assertEqual(path.name, "resume_previous.pt")
        self.assertEqual(model.ckpt["epoch"], 0)

    def test_snapshot_copy_failure_preserves_previous_good_copy(self):
        self.save_epoch(3)
        before = (self.run / "weights/resume.pt").read_bytes()
        with patch("scripts.colab_train.shutil.copyfile", side_effect=OSError("Drive disconnected")):
            with self.assertRaises(OSError):
                self.save_epoch(4)
        self.assertEqual((self.run / "weights/resume.pt").read_bytes(), before)

    def test_corrupt_only_checkpoint_does_not_silently_restart(self):
        (self.run / "weights/last.pt").write_text("truncated")
        with self.assertRaises(RuntimeError):
            select_checkpoint(self.run, fixture_model)
        self.assertTrue((self.run / "weights/last.pt").exists())

    def test_finished_native_checkpoint_is_detected(self):
        self.save_epoch(50)
        (self.run / "weights/last.pt").write_text(json.dumps({"epoch": -1, "optimizer": None}))
        self.assertEqual(select_checkpoint(self.run, fixture_model)[0], "completed")

    def test_final_epoch_snapshot_is_detected_before_native_cleanup(self):
        self.save_epoch(99)
        self.assertEqual(select_checkpoint(self.run, fixture_model)[0], "completed")

    def test_no_checkpoint_first_epoch_can_start_from_pretrained(self):
        self.assertEqual(select_checkpoint(self.run, fixture_model), ("start", None, None))

    def test_dry_run_and_completed_run_never_load_model(self):
        self.assertEqual(latest_run(self.project), self.run)
        with patch.dict("sys.modules", {"ultralytics": None}), patch("sys.argv", [
                "colab_train", "--project", str(self.project)]):
            self.assertEqual(main(), 0)
        write_json(self.run / "progress.json", {"phase": "completed"})
        with patch.dict("sys.modules", {"ultralytics": None}), patch("sys.argv", [
                "colab_train", "--project", str(self.project), "--execute"]):
            self.assertEqual(main(), 0)

    def test_resume_wiring_and_interruption_keep_saved_progress(self):
        from app.runtime import prepare_ultralytics
        prepare_ultralytics()
        self.save_epoch(0)
        captured = {}
        run = self.run

        class FakeModel:
            def __init__(self, path, task):
                self.ckpt = fixture_model(path).ckpt
                self.names = list(CLASS_NAMES)
                self.callbacks = {}

            def add_callback(self, event, function):
                self.callbacks[event] = function

            def train(self, **kwargs):
                # Simulated callback flow only; no real model training occurs.
                captured.update(kwargs)
                checkpoint = run / "weights/last.pt"
                checkpoint.write_text(json.dumps({"epoch": 1, "optimizer": {"fixture": True},
                                                   "train_args": {"epochs": 100}}))
                self.callbacks["on_model_save"](SimpleNamespace(last=checkpoint, epoch=1, epochs=100))
                raise KeyboardInterrupt("simulated runtime stop")

        plan = {"settings": {"tensorboard": False, "model": "yolo11n.pt", "epochs": 100,
                             "device": "cpu", "batch": 8, "workers": 0},
                "blockers": [], "leakage_review_required": False, "dataset_manifest_sha256": "fixture"}
        audit = {"manifest_sha256": "fixture", "resolved_dataset": {"path": "fixture"}}
        with patch("scripts.colab_train.training_plan", return_value=(plan, audit)), \
             patch("ultralytics.YOLO", FakeModel), \
             patch("scripts.tensorboard_logging.configure_tensorboard", return_value=SimpleNamespace(WRITER=None)), \
             patch("sys.argv", ["colab_train", "--project", str(self.project), "--execute"]):
            with self.assertRaises(KeyboardInterrupt):
                main()
        self.assertTrue(captured["resume"])
        self.assertEqual(captured["save_dir"], str(run))
        self.assertEqual(json.loads((run / "progress.json").read_text())["completed_epochs"], 2)
        self.assertEqual(json.loads((run / "progress.json").read_text())["phase"], "running")
        self.assertTrue((run / "interruption.json").is_file())

    def test_changed_dataset_cannot_resume_into_old_experiment(self):
        plan = {"settings": {"tensorboard": False, "model": "yolo11n.pt"},
                "blockers": [], "leakage_review_required": False}
        audit = {"manifest_sha256": "different"}
        with patch("scripts.colab_train.training_plan", return_value=(plan, audit)), \
             patch("sys.argv", ["colab_train", "--project", str(self.project), "--execute"]):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
