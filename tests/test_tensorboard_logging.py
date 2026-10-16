"""Check real event files using synthetic epoch values; never run training."""
from pathlib import Path
import csv
import tempfile
from types import SimpleNamespace
import unittest

from app.runtime import prepare_ultralytics
from scripts.tensorboard_logging import configure_tensorboard, flush_tensorboard, close_tensorboard, PerClassMetricsLogger


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

    def test_per_class_history_uses_class_ids_and_ignores_final_revalidation(self):
        import numpy as np
        from ultralytics.utils.metrics import DetMetrics
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        metrics = DetMetrics(names={0: "missing_hole", 1: "absent", 2: "open_circuit"})
        metrics.box.ap_class_index = np.array([0, 2])
        metrics.box.p = np.array([.8, .6])
        metrics.box.r = np.array([.5, .3])
        metrics.box.f1 = np.array([.615, .4])
        metrics.box.all_ap = np.array([[.7] * 10, [.2] * 10])
        callbacks = configure_tensorboard(True)
        with tempfile.TemporaryDirectory() as temporary:
            trainer = SimpleNamespace(save_dir=Path(temporary), epoch=0,
                                      validator=SimpleNamespace(metrics=metrics))
            logger = PerClassMetricsLogger(callbacks)
            try:
                callbacks.on_pretrain_routine_start(trainer)
                logger(trainer)
                trainer.epoch = 1
                metrics.box.p[1] = .9
                logger(trainer)
                # Ultralytics calls on_fit_epoch_end again after validating best.pt.
                metrics.box.p[1] = .1
                logger(trainer)
            finally:
                close_tensorboard(callbacks)
            with (Path(temporary) / "per_class_metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["class_id"] for row in rows], ["0", "2", "0", "2"])
            self.assertEqual(rows[-1]["class_name"], "open_circuit")
            self.assertAlmostEqual(float(rows[-1]["precision"]), .9)
            self.assertAlmostEqual(float(rows[-1]["AP50_95"]), .2)
            events = EventAccumulator(temporary).Reload()
            points = events.Scalars("per_class/open_circuit/precision")
            self.assertEqual([point.step for point in points], [1, 2])
            self.assertAlmostEqual(points[-1].value, .9)
            self.assertNotIn("per_class/absent/precision", events.Tags()["scalars"])
            self.assertAlmostEqual(events.Scalars("per_class/missing_hole/f1")[0].value, .615)

    def test_per_class_csv_is_saved_when_tensorboard_is_disabled(self):
        box = SimpleNamespace(ap_class_index=[2], f1=[0.0],
                              class_result=lambda index: (0.0, 0.0, 0.0, 0.0))
        with tempfile.TemporaryDirectory() as temporary:
            trainer = SimpleNamespace(save_dir=Path(temporary), epoch=0,
                validator=SimpleNamespace(metrics=SimpleNamespace(box=box, names={2: "open_circuit"})))
            PerClassMetricsLogger(SimpleNamespace(WRITER=None))(trainer)
            with (Path(temporary) / "per_class_metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["recall"]), 0.0)


if __name__ == "__main__":
    unittest.main()
