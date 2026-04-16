# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from time import time

from ultralytics.hub import HUB_WEB_ROOT, PREFIX, HUBTrainingSession
from ultralytics.utils import LOGGER, RANK, SETTINGS
from ultralytics.utils.events import events


def on_pretrain_routine_start(trainer):
    """Create a remote Ultralytics HUB session to log local model training."""
    if RANK in {-1, 0} and SETTINGS["hub"] is True and SETTINGS["api_key"] and trainer.hub_session is None:
        trainer.hub_session = HUBTrainingSession.create_session(trainer.args.model, trainer.args)


def on_pretrain_routine_end(trainer):
    """Initialize timers for upload rate limiting before training begins."""
    if session := getattr(trainer, "hub_session", None):
        # Start timer for upload rate limit
        session.timers = {"metrics": time(), "ckpt": time()}  # start timer for session rate limiting


def on_fit_epoch_end(trainer):
    """Upload training progress metrics to Ultralytics HUB at the end of each epoch."""
    if session := getattr(trainer, "hub_session", None):
        # Upload metrics after validation ends
        all_plots = {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
        }
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            all_plots = {**all_plots, **model_info_for_loggers(trainer)}

        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)

        # If any metrics failed to upload previously, add them to the queue to attempt uploading again
        if session.metrics_upload_failed_queue:
            session.metrics_queue.update(session.metrics_upload_failed_queue)

        if time() - session.timers["metrics"] > session.rate_limits["metrics"]:
            session.upload_metrics()
            session.timers["metrics"] = time()  # reset timer
            session.metrics_queue = {}  # reset queue


def on_model_save(trainer):
    """Upload model checkpoints to Ultralytics HUB with rate limiting."""
    if session := getattr(trainer, "hub_session", None):
        # Upload checkpoints with rate limiting
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.timers["ckpt"] > session.rate_limits["ckpt"]:
            LOGGER.info(f"{PREFIX}Uploading checkpoint {HUB_WEB_ROOT}/models/{session.model.id}")
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.timers["ckpt"] = time()  # reset timer


def on_train_end(trainer):
    """Upload final model and metrics to Ultralytics HUB at the end of training."""
    if session := getattr(trainer, "hub_session", None):
        # Upload final model and metrics with exponential standoff
        LOGGER.info(f"{PREFIX}Syncing final model...")
        session.upload_model(
            trainer.epoch,
            trainer.best,
            map=trainer.metrics.get("metrics/mAP50-95(B)", 0),
            final=True,
        )
        session.alive = False  # stop heartbeats
        LOGGER.info(f"{PREFIX}Done ✅\n{PREFIX}View model at {session.model_url} 🚀")


def on_train_start(trainer):
    """Run events on train start."""
    events(trainer.args, trainer.device)


def on_val_start(validator):
    """Run events on validation start."""
    if not validator.training:
        events(validator.args, validator.device)


def on_predict_batch_end(predictor):
    """Queue/update a predict event with the latest batch metadata.

    Events.__call__ upserts by mode name, so repeated calls from a long-running stream overwrite the queued entry rather
    than appending, keeping memory bounded. The rate limiter inside Events.__call__ ensures at most one POST per
    rate_limit seconds during the run, and on_predict_end flushes any remaining pending event after the final batch.
    """
    model = getattr(predictor, "model", None)
    backend = getattr(model, "backend", None)

    # Image size from args
    imgsz = getattr(predictor.args, "imgsz", None)

    # Model parameter count (PyTorch models only)
    try:
        model_params = sum(p.numel() for p in model.parameters()) if model and hasattr(model, "parameters") else None
    except Exception:
        model_params = None

    # Per-image speed from the last processed batch
    speed = None
    results = getattr(predictor, "results", None)
    if results:
        try:
            first = results[0]
        except (TypeError, KeyError, IndexError):
            first = next(iter(results), None)
        speed = getattr(first, "speed", None) if first is not None else None

    # Use the backend's actual inference device (e.g. "npu", "metis", "cpu", "cuda:0") rather than
    # predictor.device, which may reflect the torch data-movement device instead of the real hardware.
    infer_device = getattr(backend, "infer_device", None) or str(predictor.device)
    events(predictor.args, infer_device, backend=backend, imgsz=imgsz, model_params=model_params, speed=speed)


def on_predict_end(_predictor):
    """Flush any pending predict event so it is delivered even if the rate limit never elapsed."""
    events.flush()


def on_export_start(exporter):
    """Run events on export start."""
    events(exporter.args, exporter.device)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
        "on_train_start": on_train_start,
        "on_val_start": on_val_start,
        "on_predict_batch_end": on_predict_batch_end,
        "on_predict_end": on_predict_end,
        "on_export_start": on_export_start,
    }
    if SETTINGS["hub"] is True
    else {}
)
