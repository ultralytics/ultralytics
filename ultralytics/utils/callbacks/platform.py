# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path
from threading import Thread
from time import time

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING

_last_upload = 0  # Rate limit model uploads

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS.get("platform", False) is True or os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    _api_key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    assert _api_key  # verify API key is present

    import requests
    from ultralytics.utils.logger import SystemLogger
    from ultralytics.utils.torch_utils import model_info_for_loggers

except AssertionError:
    _api_key = None


def _send(event, data, project, name):
    """Send event to Platform endpoint."""
    try:
        requests.post(
            "https://alpha.ultralytics.com/api/training/webhook",
            json={"event": event, "project": project, "name": name, "data": data},
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        LOGGER.debug(f"Platform: Failed to send {event}: {e}")


def _send_async(event, data, project, name):
    """Send event asynchronously in a background thread."""
    Thread(target=_send, args=(event, data, project, name), daemon=True).start()


def _upload_model(model_path, project, name):
    """Upload model checkpoint to Platform via signed URL."""
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            return None

        # Get signed upload URL
        response = requests.post(
            "https://alpha.ultralytics.com/api/training/upload-url",
            json={"project": project, "name": name, "filename": model_path.name},
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        # Upload to GCS
        with open(model_path, "rb") as f:
            requests.put(
                data["uploadUrl"],
                data=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=600,  # 10 min timeout for large models
            ).raise_for_status()

        LOGGER.info(f"Platform: Model uploaded to '{project}'")
        return data.get("gcsPath")

    except Exception as e:
        LOGGER.debug(f"Platform: Failed to upload model: {e}")
        return None


def _upload_model_async(model_path, project, name):
    """Upload model asynchronously in a background thread."""
    Thread(target=_upload_model, args=(model_path, project, name), daemon=True).start()


def on_pretrain_routine_start(trainer):
    """Initialize Platform logging at training start."""
    if RANK not in {-1, 0} or not trainer.args.project:
        return

    project, name = str(trainer.args.project), str(trainer.args.name or "train")
    LOGGER.info(f"Platform: Streaming to project '{project}' as '{name}'")
    _send_async(
        "training_started",
        {
            "trainArgs": {k: str(v) for k, v in vars(trainer.args).items()},
            "epochs": trainer.epochs,
            "device": str(trainer.device),
        },
        project,
        name,
    )


def on_fit_epoch_end(trainer):
    """Log training and system metrics at epoch end."""
    if RANK not in {-1, 0} or not trainer.args.project:
        return

    project, name = str(trainer.args.project), str(trainer.args.name or "train")
    metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics}

    if trainer.optimizer and trainer.optimizer.param_groups:
        metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]
    if trainer.epoch == 0:
        try:
            metrics.update(model_info_for_loggers(trainer))
        except Exception:
            pass

    # Get system metrics
    system = {}
    try:
        system = SystemLogger().get_metrics(rates=True)
    except Exception:
        pass

    _send_async(
        "epoch_end",
        {
            "epoch": trainer.epoch,
            "metrics": metrics,
            "system": system,
            "fitness": trainer.fitness,
            "best_fitness": trainer.best_fitness,
        },
        project,
        name,
    )


def on_model_save(trainer):
    """Upload model checkpoint (rate limited to every 15 min)."""
    global _last_upload

    if RANK not in {-1, 0} or not trainer.args.project:
        return

    # Rate limit to every 15 minutes (900 seconds)
    if time() - _last_upload < 900:
        return

    model_path = trainer.best if trainer.best and Path(trainer.best).exists() else trainer.last
    if not model_path:
        return

    project, name = str(trainer.args.project), str(trainer.args.name or "train")
    _upload_model_async(model_path, project, name)
    _last_upload = time()


def on_train_end(trainer):
    """Log final results and upload best model."""
    if RANK not in {-1, 0} or not trainer.args.project:
        return

    project, name = str(trainer.args.project), str(trainer.args.name or "train")

    # Upload best model (blocking to ensure it completes)
    model_path = None
    if trainer.best and Path(trainer.best).exists():
        model_path = _upload_model(trainer.best, project, name)

    # Send training complete
    _send(
        "training_complete",
        {
            "results": {
                "metrics": {**trainer.metrics, "fitness": trainer.fitness},
                "bestEpoch": getattr(trainer, "best_epoch", trainer.epoch),
                "bestFitness": trainer.best_fitness,
                "modelPath": model_path or str(trainer.best) if trainer.best else None,
            }
        },
        project,
        name,
    )
    LOGGER.info(f"Platform: Training complete, results uploaded to '{project}'")


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
    }
    if _api_key
    else {}
)
