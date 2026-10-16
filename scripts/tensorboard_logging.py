"""Enable Ultralytics' scalar logger and always flush its run-specific writer."""
from importlib import import_module, reload
import csv
from pathlib import Path


class PerClassMetricsLogger:
    """Save validation metrics per epoch; ignore the final best-weight revalidation."""

    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.last_epoch = -1

    def __call__(self, trainer):
        epoch = trainer.epoch + 1
        if epoch <= self.last_epoch:
            return
        metrics = trainer.validator.metrics
        box = metrics.box
        rows = []
        writer = getattr(self.callbacks, "WRITER", None)
        # Rows in metric arrays correspond to ap_class_index, not every dataset ID.
        # Classes absent from validation are omitted rather than assigned fake scores.
        for index, class_id in enumerate(box.ap_class_index):
            class_id = int(class_id)
            precision, recall, ap50, ap50_95 = box.class_result(index)
            values = dict(precision=float(precision), recall=float(recall),
                          f1=float(box.f1[index]), AP50=float(ap50), AP50_95=float(ap50_95))
            name = metrics.names[class_id]
            rows.append(dict(epoch=epoch, class_id=class_id, class_name=name, **values))
            if writer is not None:
                for key, value in values.items():
                    writer.add_scalar(f"per_class/{name}/{key}", value, epoch)
        if rows:
            path = Path(trainer.save_dir) / "per_class_metrics.csv"
            needs_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                if needs_header:
                    csv_writer.writeheader()
                csv_writer.writerows(rows)
        if writer is not None:
            writer.flush()
        self.last_epoch = epoch


def configure_tensorboard(enabled: bool):
    # Called only on --execute, never while preparing a dry-run plan.
    if enabled:
        from torch.utils.tensorboard import SummaryWriter  # Verify dependency before training.
    from ultralytics import settings

    settings.update({"tensorboard": enabled})
    callbacks = import_module("ultralytics.utils.callbacks.tensorboard")
    # Integration may have been imported earlier while disabled in this process.
    if enabled and (callbacks.SummaryWriter is None or not callbacks.callbacks):
        callbacks = reload(callbacks)
    if enabled and callbacks.SummaryWriter is None:
        raise RuntimeError("TensorBoard logger could not be initialized; install requirements-train.txt")
    if not enabled:
        callbacks.callbacks = {}
    return callbacks


def flush_tensorboard(callbacks):
    writer = getattr(callbacks, "WRITER", None)
    if writer is not None:
        writer.flush()


def close_tensorboard(callbacks):
    writer = getattr(callbacks, "WRITER", None)
    if writer is not None:
        try:
            writer.flush()
        finally:
            writer.close()
            callbacks.WRITER = None
