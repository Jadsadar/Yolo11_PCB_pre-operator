"""Check real event files using synthetic epoch values; never run training."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from app.runtime import prepare_ultralytics
from scripts.tensorboard_logging import configure_tensorboard, flush_tensorboard, close_tensorboard


class TensorBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_ultralytics()
        from ultralytics import settings
        cls.previous = settings["tensorboard"]

    def tearDown(self):
        from ultralytics import settings
        settings.update({"tensorboard": self.previous})

    def test_separate_runs_contain_losses_metrics_and_learning_rate(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        callbacks = configure_tensorboard(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, loss in [("run_a", .8), ("run_b", .4)]:
                folder = root / name
                trainer = SimpleNamespace(save_dir=folder, epoch=0, tloss=loss,
                    lr={"lr/pg0": .001}, metrics={"metrics/mAP50(B)": .7, "val/box_loss": .6},
                    label_loss_items=lambda value, prefix: {f"{prefix}/box_loss": value})
                try:
                    callbacks.on_pretrain_routine_start(trainer)
                    self.assertIsNotNone(callbacks.WRITER)
                    callbacks.on_train_epoch_end(trainer)
                    callbacks.on_fit_epoch_end(trainer)
                    flush_tensorboard(callbacks)
                finally:
                    close_tensorboard(callbacks)
                self.assertIsNone(callbacks.WRITER)
                self.assertTrue(list(folder.glob("events.out.tfevents.*")))
                events = EventAccumulator(str(folder)).Reload()
                self.assertEqual(events.Scalars("train/box_loss")[0].step, 1)
                self.assertAlmostEqual(events.Scalars("train/box_loss")[0].value, loss)
                self.assertAlmostEqual(events.Scalars("metrics/mAP50(B)")[0].value, .7)
                self.assertIn("lr/pg0", events.Tags()["scalars"])

    def test_can_disable_then_reenable_without_stale_callback_state(self):
        disabled = configure_tensorboard(False)
        self.assertEqual(disabled.callbacks, {})
        enabled = configure_tensorboard(True)
        self.assertIn("on_fit_epoch_end", enabled.callbacks)
        close_tensorboard(enabled)


if __name__ == "__main__":
    unittest.main()
