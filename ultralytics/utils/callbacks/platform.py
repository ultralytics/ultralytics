# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import RANK, SETTINGS
from ultralytics.utils.logger import DEFAULT_LOG_PATH, ConsoleLogger


def on_pretrain_routine_start(trainer):
    """Initialize and start console logging immediately at the very beginning."""
    if RANK in {-1, 0}:
        # Create and start logger immediately before any other output
        trainer.platform_logger = ConsoleLogger(DEFAULT_LOG_PATH)
        trainer.platform_logger.start_capture()


def on_pretrain_routine_end(trainer):
    """Handle pre-training routine completion event."""
    pass


def on_fit_epoch_end(trainer):
    """Handle end of training epoch event."""
    pass


def on_model_save(trainer):
    """Handle model checkpoint save event."""
    pass


def on_train_end(trainer):
    """Stop console capture and finalize logs."""
    if logger := getattr(trainer, "platform_logger", None):
        logger.stop_capture()


def on_train_start(trainer):
    """Handle training start event."""
    pass


def on_val_start(validator):
    """Handle validation start event."""
    pass


def on_predict_start(predictor):
    """Handle prediction start event."""
    pass


def on_export_start(exporter):
    """Handle model export start event."""
    pass


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
        "on_train_start": on_train_start,
        "on_val_start": on_val_start,
        "on_predict_start": on_predict_start,
        "on_export_start": on_export_start,
    }
    if SETTINGS.get("platform", False) is True  # disabled for debugging
    else {}
)
