import csv
from pathlib import Path
import tempfile
import unittest

from scripts.compare_freeze import comparison_configs, summarize_run


class FreezeComparisonTests(unittest.TestCase):
    def test_comparison_controls_settings_and_resolves_same_initial_weights(self):
        base = dict(data="../data.yaml", model="../models/pretrained/yolo11n.pt",
                    epochs=100, batch=16, patience=20, seed=42, lr0=.001, resume=True)
        configs = comparison_configs(base, Path("configs").resolve(), Path("experiments/test"))
        frozen, full = configs["freeze2"], configs["no_freeze"]
        self.assertEqual({k: v for k, v in frozen.items() if k not in ("name", "freeze")},
                         {k: v for k, v in full.items() if k not in ("name", "freeze")})
        self.assertEqual((frozen["freeze"], full["freeze"]), (2, 0))
        self.assertEqual((frozen["epochs"], frozen["batch"], frozen["patience"]), (60, 8, 15))
        self.assertFalse(frozen["resume"])
        self.assertTrue(Path(frozen["model"]).is_absolute())
        self.assertEqual(base["epochs"], 100)

    def test_summary_reports_best_metrics_for_full_and_early_stopped_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "weights").mkdir()
            for name in ("weights/best.pt", "weights/last.pt", "per_class_metrics.csv"):
                (run / name).touch()
            rows = [dict(epoch=i, time=i * 2, **{"metrics/precision(B)": i / 100,
                    "metrics/recall(B)": .5, "metrics/mAP50(B)": .7,
                    "metrics/mAP50-95(B)": .8 if i == 20 else .2}) for i in range(1, 61)]
            for epochs in (35, 60):
                with self.subTest(epochs=epochs):
                    with (run / "results.csv").open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows[:epochs])
                    summary = summarize_run(run, "freeze2", 125)
                    self.assertEqual(summary["epochs"], epochs)
                    self.assertEqual(summary["training_seconds"], epochs * 2)
                    self.assertEqual(summary["best_epoch"], 20)
                    self.assertEqual(summary["best_precision"], .2)
                    self.assertEqual(summary["best_mAP50_95"], .8)
                    self.assertEqual(summary["final_mAP50_95"], .2)
                    self.assertEqual(summary["average_epoch_seconds"], 2)
            (run / "results.csv").write_text("epoch,time\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "1-60 consecutive completed epochs"):
                summarize_run(run, "freeze2", 3)


if __name__ == "__main__":
    unittest.main()
