# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.utils import RANK
from ultralytics.utils.logger import ConsoleLogger, DEFAULT_LOG_PATH

def on_pretrain_routine_start(trainer):
    """Initialize and start console logging immediately at the very beginning."""
    if RANK in {-1, 0}:
        # Create and start logger immediately before any other output
        trainer.platform_logger = ConsoleLogger(DEFAULT_LOG_PATH)
        trainer.platform_logger.start_capture()


def on_pretrain_routine_end(trainer):
    """Console capture already started in on_pretrain_routine_start."""
    pass


def on_fit_epoch_end(trainer):
    """Log epoch completion."""
    pass


def on_model_save(trainer):
    """Log model save events."""
    pass


def on_train_end(trainer):
    """Stop console capture and finalize logs."""
    if logger := getattr(trainer, "platform_logger", None):
        logger.stop_capture()


def on_train_start(trainer):
    """Log training start."""
    pass


def on_val_start(validator):
    """Disabled - only log training."""
    pass


def on_predict_start(predictor):
    """Disabled - only log training."""
    pass


def on_export_start(exporter):
    """Disabled - only log training."""
    pass


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_model_save": on_model_save,
    "on_train_end": on_train_end,
    "on_train_start": on_train_start,
    "on_val_start": on_val_start,
    "on_predict_start": on_predict_start,
    "on_export_start": on_export_start,
}  # always register callbacks, check platform_source in each callback
