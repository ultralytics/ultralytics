# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import platform
import re
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time

from ultralytics.utils import ENVIRONMENT, GIT, LOGGER, PYTHON_VERSION, RANK, SETTINGS, TESTS_RUNNING, Retry, colorstr

PREFIX = colorstr("Platform: ")

# Configurable platform URL for debugging (e.g. ULTRALYTICS_PLATFORM_URL=http://localhost:3000)
PLATFORM_URL = os.getenv("ULTRALYTICS_PLATFORM_URL", "https://platform.ultralytics.com").rstrip("/")
PLATFORM_API_URL = f"{PLATFORM_URL}/api/webhooks"


def slugify(text):
    """Convert text to URL-safe slug (e.g., 'My Project 1' -> 'my-project-1')."""
    if not text:
        return text
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9\s-]", "", str(text).lower()).replace(" ", "-")).strip("-")[:128]


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


def resolve_platform_uri(uri, hard=True):
    """Resolve ul:// URIs to signed URLs by authenticating with Ultralytics Platform.

    Formats:
        ul://username/datasets/slug  -> Returns signed URL to NDJSON file
        ul://username/project/model  -> Returns signed URL to .pt file

    Args:
        uri (str): Platform URI starting with "ul://".
        hard (bool): Whether to raise an error if resolution fails.

    Returns:
        (str | None): Signed URL on success, None if not found and hard=False.

    Raises:
        ValueError: If API key is missing/invalid or URI format is wrong.
        PermissionError: If access is denied.
        RuntimeError: If resource is not ready (e.g., dataset still processing).
        FileNotFoundError: If resource not found and hard=True.
        ConnectionError: If network request fails and hard=True.
    """
    import requests

    path = uri[5:]  # Remove "ul://"
    parts = path.split("/")

    api_key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    if not api_key:
        raise ValueError(f"ULTRALYTICS_API_KEY required for '{uri}'. Get key at {PLATFORM_URL}/settings")

    base = PLATFORM_API_URL
    headers = {"Authorization": f"Bearer {api_key}"}

    # ul://username/datasets/slug
    if len(parts) == 3 and parts[1] == "datasets":
        username, _, slug = parts
        url = f"{base}/datasets/{username}/{slug}/export"

    # ul://username/project/model
    elif len(parts) == 3:
        username, project, model = parts
        url = f"{base}/models/{username}/{project}/{model}/download"

    else:
        raise ValueError(f"Invalid platform URI: {uri}. Use ul://user/datasets/name or ul://user/project/model")

    try:
        timeout = 3600 if "/datasets/" in url else 90  # NDJSON generation can be slow for large datasets
        r = requests.head(url, headers=headers, allow_redirects=False, timeout=timeout)

        # Handle redirect responses (301, 302, 303, 307, 308)
        if 300 <= r.status_code < 400 and "location" in r.headers:
            return r.headers["location"]  # Return signed URL

        # Handle error responses
        if r.status_code == 401:
            raise ValueError(f"Invalid ULTRALYTICS_API_KEY for '{uri}'")
        if r.status_code == 403:
            raise PermissionError(f"Access denied for '{uri}'. Check dataset/model visibility settings.")
        if r.status_code == 404:
            if hard:
                raise FileNotFoundError(f"Not found on platform: {uri}")
            LOGGER.warning(f"Not found on platform: {uri}")
            return None
        if r.status_code == 409:
            raise RuntimeError(f"Resource not ready: {uri}. Dataset may still be processing.")

        # Unexpected response
        r.raise_for_status()
        raise RuntimeError(f"Unexpected response from platform for '{uri}': {r.status_code}")

    except requests.exceptions.RequestException as e:
        if hard:
            raise ConnectionError(f"Failed to resolve {uri}: {e}") from e
        LOGGER.warning(f"Failed to resolve {uri}: {e}")
        return None


def _interp_plot(plot, n=101):
    """Interpolate plot curve data to n points to reduce storage size."""
    import numpy as np

    if not plot.get("x") or not plot.get("y"):
        return plot  # No interpolation needed (e.g., confusion_matrix)

    x, y = np.array(plot["x"]), np.array(plot["y"])
    if len(x) <= n:
        return plot  # Already small enough

    # New x values (101 points gives clean 0.01 increments: 0, 0.01, 0.02, ..., 1.0)
    x_new = np.linspace(x[0], x[-1], n)

    # Interpolate y values (handle both 1D and 2D arrays)
    if y.ndim == 1:
        y_new = np.interp(x_new, x, y)
    else:
        y_new = np.array([np.interp(x_new, x, yi) for yi in y])

    # Also interpolate ap if present (for PR curves)
    result = {**plot, "x": x_new.tolist(), "y": y_new.tolist()}
    if "ap" in plot:
        result["ap"] = plot["ap"]  # Keep AP values as-is (per-class scalars)

    return result


def _send(event, data, project, name, model_id=None, retry=2):
    """Send event to Platform endpoint with retry logic."""
    payload = {"event": event, "project": project, "name": name, "data": data}
    if model_id:
        payload["modelId"] = model_id

    @Retry(times=retry, delay=1)
    def post():
        r = requests.post(
            f"{PLATFORM_API_URL}/training/metrics",
            json=payload,
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=30,
        )
        if 400 <= r.status_code < 500 and r.status_code not in {408, 429}:
            LOGGER.warning(f"{PREFIX}Failed to send {event}: {r.status_code} {r.reason}")
            return None  # Don't retry client errors (except 408 timeout, 429 rate limit)
        r.raise_for_status()
        return r.json()

    try:
        return post()
    except Exception as e:
        LOGGER.debug(f"{PREFIX}Failed to send {event}: {e}")
        return None


def _send_async(event, data, project, name, model_id=None):
    """Send event asynchronously using bounded thread pool."""
    _executor.submit(_send, event, data, project, name, model_id)


def _upload_model(model_path, project, name, progress=False, retry=1):
    """Upload model checkpoint to Platform via signed URL."""
    from ultralytics.utils.uploads import safe_upload

    model_path = Path(model_path)
    if not model_path.exists():
        LOGGER.warning(f"{PREFIX}Model file not found: {model_path}")
        return None

    # Get signed upload URL from Platform (server sanitizes filename for storage safety)
    @Retry(times=3, delay=2)
    def get_signed_url():
        r = requests.post(
            f"{PLATFORM_API_URL}/models/upload",
            json={"project": project, "name": name, "filename": model_path.name},
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    try:
        data = get_signed_url()
    except Exception as e:
        LOGGER.warning(f"{PREFIX}Failed to get upload URL: {e}")
        return None

    # Upload to GCS using safe_upload with retry logic and optional progress bar
    if safe_upload(file=model_path, url=data["uploadUrl"], retry=retry, progress=progress):
        return data.get("gcsPath")
    return None


def _upload_model_async(model_path, project, name):
    """Upload model asynchronously using bounded thread pool."""
    _executor.submit(_upload_model, model_path, project, name)


def _get_environment_info():
    """Collect comprehensive environment info using existing ultralytics utilities."""
    import shutil

    import psutil
    import torch

    from ultralytics import __version__
    from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info

    # Get RAM and disk totals
    memory = psutil.virtual_memory()
    disk_usage = shutil.disk_usage("/")

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
        "totalRamGb": round(memory.total / (1 << 30), 1),  # Total RAM in GB
        "totalDiskGb": round(disk_usage.total / (1 << 30), 1),  # Total disk in GB
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


def _get_project_name(trainer):
    """Get slugified project and name from trainer args."""
    raw = str(trainer.args.project)
    parts = raw.split("/", 1)
    project = f"{parts[0]}/{slugify(parts[1])}" if len(parts) == 2 else slugify(raw)
    return project, slugify(str(trainer.args.name or "train"))


def on_pretrain_routine_start(trainer):
    """Initialize Platform logging at training start."""
    if RANK not in {-1, 0} or not trainer.args.project:
        return

    # Per-trainer state to isolate concurrent training runs
    trainer._platform_model_id = None
    trainer._platform_last_upload = time()

    project, name = _get_project_name(trainer)
    url = f"{PLATFORM_URL}/{project}/{name}"
    LOGGER.info(f"{PREFIX}Streaming to {url}")

    # Create callback to send console output to Platform
    def send_console_output(content, line_count, chunk_id):
        """Send batched console output to Platform webhook."""
        _send_async(
            "console_output",
            {"chunkId": chunk_id, "content": content, "lineCount": line_count},
            project,
            name,
            getattr(trainer, "_platform_model_id", None),
        )

    # Start console capture with batching (5 lines or 5 seconds)
    trainer._platform_console_logger = ConsoleLogger(batch_size=5, flush_interval=5.0, on_flush=send_console_output)
    trainer._platform_console_logger.start_capture()

    # Collect environment info (W&B-style metadata)
    environment = _get_environment_info()

    # Build trainArgs - callback runs before get_dataset() so args.data is still original (e.g., ul:// URIs)
    # Note: model_info is sent later in on_fit_epoch_end (epoch 0) when the model is actually loaded
    train_args = {k: str(v) for k, v in vars(trainer.args).items()}

    # Send synchronously to get modelId for subsequent webhooks (critical, more retries)
    response = _send(
        "training_started",
        {
            "trainArgs": train_args,
            "epochs": trainer.epochs,
            "device": str(trainer.device),
            "environment": environment,
        },
        project,
        name,
        retry=4,
    )
    if response and response.get("modelId"):
        trainer._platform_model_id = response["modelId"]
        # Check for immediate cancellation (cancelled before training started)
        # Note: trainer.stop is set in on_pretrain_routine_end (after _setup_train resets it)
        if response.get("cancelled"):
            trainer._platform_cancelled = True
    else:
        LOGGER.warning(f"{PREFIX}Failed to register training session - metrics may not sync to Platform")


def on_pretrain_routine_end(trainer):
    """Apply pre-start cancellation after _setup_train resets trainer.stop."""
    if getattr(trainer, "_platform_cancelled", False):
        LOGGER.info(f"{PREFIX}Training cancelled from Platform before starting ✅")
        trainer.stop = True


def on_fit_epoch_end(trainer):
    """Log training and system metrics at epoch end."""
    if RANK not in {-1, 0} or not trainer.args.project:
        return

    project, name = _get_project_name(trainer)
    metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics}

    if trainer.optimizer and trainer.optimizer.param_groups:
        metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]

    # Extract model info at epoch 0 (sent as separate field, not in metrics)
    model_info = None
    if trainer.epoch == 0:
        try:
            info = model_info_for_loggers(trainer)
            model_info = {
                "parameters": info.get("model/parameters", 0),
                "gflops": info.get("model/GFLOPs", 0),
                "speedMs": info.get("model/speed_PyTorch(ms)", 0),
            }
        except Exception:
            pass

    # Get system metrics (cache SystemLogger on trainer for efficiency)
    system = {}
    try:
        if not hasattr(trainer, "_platform_system_logger"):
            trainer._platform_system_logger = SystemLogger()
        system = trainer._platform_system_logger.get_metrics(rates=True)
    except Exception:
        pass

    payload = {
        "epoch": trainer.epoch,
        "metrics": metrics,
        "system": system,
        "fitness": trainer.fitness,
        "best_fitness": trainer.best_fitness,
    }
    if model_info:
        payload["modelInfo"] = model_info

    def _send_and_check_cancel():
        """Send epoch_end and check response for cancellation (runs in background thread)."""
        response = _send("epoch_end", payload, project, name, getattr(trainer, "_platform_model_id", None), retry=1)
        if response and response.get("cancelled"):
            LOGGER.info(f"{PREFIX}Training cancelled from Platform ✅")
            trainer.stop = True
            trainer._platform_cancelled = True

    _executor.submit(_send_and_check_cancel)


def on_model_save(trainer):
    """Upload model checkpoint (rate limited to every 15 min)."""
    if RANK not in {-1, 0} or not trainer.args.project:
        return

    # Rate limit to every 15 minutes (900 seconds)
    if time() - getattr(trainer, "_platform_last_upload", 0) < 900:
        return

    model_path = trainer.best if trainer.best and Path(trainer.best).exists() else trainer.last
    if not model_path:
        return

    project, name = _get_project_name(trainer)
    _upload_model_async(model_path, project, name)
    trainer._platform_last_upload = time()


def on_train_end(trainer):
    """Log final results, upload best model, and send validation plot data."""
    if RANK not in {-1, 0} or not trainer.args.project:
        return

    project, name = _get_project_name(trainer)

    if getattr(trainer, "_platform_cancelled", False):
        LOGGER.info(f"{PREFIX}Uploading partial results for cancelled training")

    # Stop console capture
    if hasattr(trainer, "_platform_console_logger") and trainer._platform_console_logger:
        trainer._platform_console_logger.stop_capture()
        trainer._platform_console_logger = None

    # Upload best model (blocking with progress bar to ensure it completes)
    gcs_path = None
    model_size = None
    if trainer.best and Path(trainer.best).exists():
        model_size = Path(trainer.best).stat().st_size
        gcs_path = _upload_model(trainer.best, project, name, progress=True, retry=3)
        if not gcs_path:
            LOGGER.warning(f"{PREFIX}Model will not be available for download on Platform (upload failed)")

    # Collect plots from trainer and validator, deduplicating by type
    plots_by_type = {}
    for info in getattr(trainer, "plots", {}).values():
        if info.get("data") and info["data"].get("type"):
            plots_by_type[info["data"]["type"]] = info["data"]
    for info in getattr(getattr(trainer, "validator", None), "plots", {}).values():
        if info.get("data") and info["data"].get("type"):
            plots_by_type.setdefault(info["data"]["type"], info["data"])  # Don't overwrite trainer plots
    plots = [_interp_plot(p) for p in plots_by_type.values()]  # Interpolate curves to reduce size

    # Get class names
    names = getattr(getattr(trainer, "validator", None), "names", None) or (trainer.data or {}).get("names")
    class_names = list(names.values()) if isinstance(names, dict) else list(names) if names else None

    _send(
        "training_complete",
        {
            "results": {
                "metrics": {**trainer.metrics, "fitness": trainer.fitness},
                "bestEpoch": getattr(trainer, "best_epoch", trainer.epoch),
                "bestFitness": trainer.best_fitness,
                "modelPath": gcs_path,  # Only send GCS path, not local path
                "modelSize": model_size,
            },
            "classNames": class_names,
            "plots": plots,
        },
        project,
        name,
        getattr(trainer, "_platform_model_id", None),
        retry=4,  # Critical, more retries
    )
    url = f"{PLATFORM_URL}/{project}/{name}"
    LOGGER.info(f"{PREFIX}View results at {url}")


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
    }
    if _api_key
    else {}
)
