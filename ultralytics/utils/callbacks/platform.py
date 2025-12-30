# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import platform
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time

from ultralytics.utils import ENVIRONMENT, GIT, LOGGER, PYTHON_VERSION, RANK, SETTINGS, TESTS_RUNNING

_last_upload = 0  # Rate limit model uploads
_console_logger = None  # Global console logger instance
_system_logger = None  # Cached system logger instance

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS.get("platform", False) is True or os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    _api_key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    assert _api_key  # verify API key is present

    import requests

    from ultralytics.utils.logger import ConsoleLogger, SystemLogger
    from ultralytics.utils.torch_utils import model_info_for_loggers

    _executor = ThreadPoolExecutor(max_workers=10)  # Bounded thread pool for async operations

except (AssertionError, ImportError):
    _api_key = None


def _send(event, data, project, name):
    """Send event to Platform endpoint."""
    try:
        requests.post(
            "https://alpha.ultralytics.com/api/webhooks/training/metrics",
            json={"event": event, "project": project, "name": name, "data": data},
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        LOGGER.debug(f"Platform: Failed to send {event}: {e}")


def _send_async(event, data, project, name):
    """Send event asynchronously using bounded thread pool."""
    _executor.submit(_send, event, data, project, name)


def _upload_model(model_path, project, name):
    """Upload model checkpoint to Platform via signed URL."""
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            return None

        # Get signed upload URL
        response = requests.post(
            "https://alpha.ultralytics.com/api/webhooks/models/upload",
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
    """Upload model asynchronously using bounded thread pool."""
    _executor.submit(_upload_model, model_path, project, name)


def _get_environment_info():
    """Collect comprehensive environment info using existing ultralytics utilities."""
    import torch

    from ultralytics import __version__
    from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info

    env = {
        "ultralyticsVersion": __version__,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "environment": ENVIRONMENT,
        "pythonVersion": PYTHON_VERSION,
        "pythonExecutable": sys.executable,
        "cpuCount": os.cpu_count() or 0,
        "cpu": get_cpu_info(),
        "command": " ".join(sys.argv),
    }

    # Git info using cached GIT singleton (no subprocess calls)
    try:
        if GIT.is_repo:
            if GIT.origin:
                env["gitRepository"] = GIT.origin
            if GIT.branch:
                env["gitBranch"] = GIT.branch
            if GIT.commit:
                env["gitCommit"] = GIT.commit[:12]  # Short hash
    except Exception:
        pass

    # GPU info
    try:
        if torch.cuda.is_available():
            env["gpuCount"] = torch.cuda.device_count()
            env["gpuType"] = get_gpu_info(0) if torch.cuda.device_count() > 0 else None
    except Exception:
        pass

    return env


def on_pretrain_routine_start(trainer):
    """Initialize Platform logging at training start."""
    global _console_logger, _last_upload

    if RANK not in {-1, 0} or not trainer.args.project:
        return

    # Initialize upload timer to now so first checkpoint waits 15 min from training start
    _last_upload = time()

    project, name = str(trainer.args.project), str(trainer.args.name or "train")
    LOGGER.info(f"Platform: Streaming to project '{project}' as '{name}'")

    # Create callback to send console output to Platform
    def send_console_output(content, line_count, chunk_id):
        """Send batched console output to Platform webhook."""
        _send_async("console_output", {"chunkId": chunk_id, "content": content, "lineCount": line_count}, project, name)

    # Start console capture with batching (5 lines or 5 seconds)
    _console_logger = ConsoleLogger(batch_size=5, flush_interval=5.0, on_flush=send_console_output)
    _console_logger.start_capture()

    # Gather model info for richer metadata
    model_info = {}
    try:
        info = model_info_for_loggers(trainer)
        model_info = {
            "parameters": info.get("model/parameters", 0),
            "gflops": info.get("model/GFLOPs", 0),
            "classes": getattr(trainer.model, "yaml", {}).get("nc", 0),  # number of classes
        }
    except Exception:
        pass

    # Collect environment info (W&B-style metadata)
    environment = _get_environment_info()

    _send_async(
        "training_started",
        {
            "trainArgs": {k: str(v) for k, v in vars(trainer.args).items()},
            "epochs": trainer.epochs,
            "device": str(trainer.device),
            "modelInfo": model_info,
            "environment": environment,
        },
        project,
        name,
    )


def on_fit_epoch_end(trainer):
    """Log training and system metrics at epoch end."""
    global _system_logger

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

    # Get system metrics (cache SystemLogger for efficiency)
    system = {}
    try:
        if _system_logger is None:
            _system_logger = SystemLogger()
        system = _system_logger.get_metrics(rates=True)
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
    global _console_logger

    if RANK not in {-1, 0} or not trainer.args.project:
        return

    project, name = str(trainer.args.project), str(trainer.args.name or "train")

    # Stop console capture and flush remaining output
    if _console_logger:
        _console_logger.stop_capture()
        _console_logger = None

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
