"""Enable Ultralytics' scalar logger and always flush its run-specific writer."""
from importlib import import_module, reload


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
