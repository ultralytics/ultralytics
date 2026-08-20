# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import platform
import re
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from pathlib import Path
from time import time

from ultralytics.utils import (
    ENVIRONMENT,
    GIT,
    LOGGER,
    PLATFORM_API_URL,
    PLATFORM_URL,
    PYTHON_VERSION,
    SETTINGS,
    TESTS_RUNNING,
    Retry,
    colorstr,
)

PREFIX = colorstr("Platform: ")
_api_key = None
_executor = ThreadPoolExecutor(max_workers=10)


def slugify(text):
    """Convert text to URL-safe slug (e.g., 'My Project 1' -> 'my-project-1')."""
    if not text:
        return text
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9\s-]", "", str(text).lower()).replace(" ", "-")).strip("-")[:128]


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


def _validation_payload(image_metrics, sample_limit=5_000, extremes_limit=100):
    """Return exact F1 extremes and an evenly ranked sample for correlation analysis."""
    ranked = sorted(image_metrics.items(), key=lambda item: (item[1]["f1"], item[0]))
    if len(ranked) > sample_limit:
        sample = [ranked[round(i * (len(ranked) - 1) / (sample_limit - 1))] for i in range(sample_limit)]
    else:
        sample = ranked

    def rows(items):
        return [[Path(name).stem.split("_", 1)[0], metric["tp"], metric["fp"], metric["fn"]] for name, metric in items]

    return {
        "population": len(ranked),
        "sampling": "f1_rank",
        "rows": rows(sample),
        "extremes": {"worst": rows(ranked[:extremes_limit]), "best": rows(reversed(ranked[-extremes_limit:]))},
    }


def _sanitize_json_value(value):
    """Replace non-finite floats in payloads with None so requests JSON encoding succeeds."""
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, float):
        return value if isfinite(value) else None  # avoid "Out of range float values are not JSON compliant" warnings
    return value


def _send(event, data, project, name, model_id=None, retry=2, timeout=30):
    """Send event to Platform endpoint with retry logic."""
    if not _api_key:
        return None
    import requests  # scoped as slow import

    payload = {"event": event, "data": _sanitize_json_value(data)}
    if model_id:
        payload["modelId"] = model_id
    else:
        payload.update(project=project, name=name)

    def send_once():
        global _api_key
        r = requests.post(
            f"{PLATFORM_API_URL}/training/metrics",
            json=payload,
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=timeout,
        )
        if 400 <= r.status_code < 500 and r.status_code not in {408, 429}:
            try:
                msg = r.json().get("error", r.reason)
            except Exception:
                msg = r.reason
            # Only 401 is credential-scoped; 403/404 concern one run and must not disable the process.
            if r.status_code == 401:
                _api_key = None
            # A console_output failure must not be logged: ConsoleLogger flushes the warning back as the
            # next chunk, which fails again. 401 is safe — the cleared key short-circuits _send.
            if event != "console_output" or r.status_code == 401:
                LOGGER.warning(f"{PREFIX}{msg}")
            return None  # Don't retry client errors (except 408 timeout, 429 rate limit)
        r.raise_for_status()
        return r.json()

    # Same loop as above, so a console_output send stays silent at every level — including Retry's
    # per-attempt warning. It must still retry: _flush_buffer clears the buffer before calling us.
    quiet = event == "console_output"
    try:
        return Retry(times=retry, delay=1, verbose=not quiet)(send_once)()
    except Exception as e:
        if not quiet:
            LOGGER.debug(f"{PREFIX}Failed to send {event}: {e}")
        return None


def _handle_control_response(trainer, ctx, response):
    """Apply centralized stop signals returned by Platform webhook responses.

    Notes:
        ``ctx["cancelled"]`` is the durable cancellation signal. During startup, trainer setup later resets
        ``trainer.stop``, so early stop requests still rely on ``on_pretrain_routine_end()`` to reapply the flag after
        setup completes.
    """
    if response and response.get("cancelled"):
        ctx["cancelled"] = True
        trainer.stop = True
        LOGGER.info(f"{PREFIX}Training cancelled from Platform ⚠️")


def _upload_model(model_path, project, name, progress=False, retry=1, model_id=None, run_id=None):
    """Publish a model checkpoint to its configured Platform storage location."""
    from ultralytics.utils.uploads import safe_upload

    if not _api_key:
        return None
    model_path = Path(model_path)
    if not model_path.exists():
        LOGGER.warning(f"{PREFIX}Model file not found: {model_path}")
        return None
    model_size = model_path.stat().st_size
    if os.getenv("PLATFORM_API_URL"):
        return {"modelPath": str(model_path.resolve()), "modelSize": model_size}
    import requests  # scoped as slow import

    # Get signed upload URL from Platform (server sanitizes filename for storage safety)
    @Retry(times=3, delay=2)
    def get_signed_url():
        payload = {"filename": model_path.name}
        if model_id:
            payload["modelId"] = model_id  # Direct lookup avoids slug mismatch from auto-increment
        else:
            payload.update(project=project, name=name)
        if run_id:
            payload["runId"] = run_id
        r = requests.post(
            f"{PLATFORM_API_URL}/models/upload",
            json=payload,
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
        gcs_path = data.get("gcsPath")
        if gcs_path and run_id:
            saved = _send(
                "checkpoint_saved",
                {
                    "modelPath": gcs_path,
                    "runId": run_id,
                    "uploadPath": data.get("uploadPath"),
                },
                project,
                name,
                model_id,
                timeout=90,
            )
            return {"modelPath": gcs_path, "modelSize": model_size} if saved else None
        return {"modelPath": gcs_path, "modelSize": model_size}
    return None


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
            if GIT.message:
                env["gitCommitMessage"] = GIT.message
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
    global _api_key
    if TESTS_RUNNING or not trainer.args.project:
        return
    _api_key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    if not _api_key:
        return

    project, name = _get_project_name(trainer)
    LOGGER.info(f"{PREFIX}Streaming training metrics to Platform")

    from ultralytics.utils.logger import ConsoleLogger

    # Single dict for all platform callback state
    ctx = {
        "model_id": None,
        "run_id": None,
        "last_upload": time(),
        "checkpoint_upload": None,
        "cancelled": False,
        "console_logger": None,
        "system_logger": None,
    }
    trainer.platform = ctx

    # Create callback to send console output to Platform
    def send_console_output(content, line_count, chunk_id):
        """Send batched console output to Platform webhook."""
        _executor.submit(
            _send,
            "console_output",
            {"chunkId": chunk_id, "content": content, "lineCount": line_count},
            project,
            name,
            ctx["model_id"],
        )

    # Console capture with batching (5 lines or 5 seconds). Built here, but not started until Platform
    # has accepted the run below: capturing first left the user's stdout redirected through a dead
    # integration whenever training_started failed, and its final flush would post a console chunk
    # carrying no model_id.
    ctx["console_logger"] = ConsoleLogger(batch_size=5, flush_interval=5.0, on_flush=send_console_output)

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
        ctx["model_id"] = response["modelId"]
        ctx["run_id"] = response.get("runId")
        # Server returns actual slug (may differ from requested name due to auto-increment, e.g. "train" → "train-2")
        if response.get("modelSlug"):
            ctx["model_slug"] = response["modelSlug"]
            url = f"{PLATFORM_URL}/{project}/{ctx['model_slug']}"
            LOGGER.info(f"{PREFIX}View model at {url}")
        ctx["console_logger"].start_capture()  # only now: the run is tracked and model_id is known
        # Note: trainer.stop is set in on_pretrain_routine_end (after _setup_train resets it)
        _handle_control_response(trainer, ctx, response)
    else:
        LOGGER.warning(f"{PREFIX}Training will not be tracked on Platform")
        trainer.platform = None  # Disable further callbacks


def on_pretrain_routine_end(trainer):
    """Apply pre-start cancellation after _setup_train resets trainer.stop."""
    ctx = getattr(trainer, "platform", None)
    if ctx and ctx["cancelled"]:
        LOGGER.info(f"{PREFIX}Training cancelled from Platform before starting ✅")
        trainer.stop = True


def on_fit_epoch_end(trainer):
    """Log training and system metrics at epoch end."""
    ctx = getattr(trainer, "platform", None)
    if not ctx:
        return

    project, name = _get_project_name(trainer)
    metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics}

    if trainer.optimizer and trainer.optimizer.param_groups:
        metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]

    # Extract model info at epoch 0 (sent as separate field, not in metrics)
    model_info = None
    if trainer.epoch == 0:
        try:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            info = model_info_for_loggers(trainer)
            model_info = {
                "parameters": info.get("model/parameters", 0),
                "gflops": info.get("model/GFLOPs", 0),
                "speedMs": info.get("model/speed_PyTorch(ms)", 0),
            }
        except Exception:
            pass

    # Get system metrics (cache SystemLogger in platform context for efficiency)
    system = {}
    try:
        if not ctx["system_logger"]:
            from ultralytics.utils.logger import SystemLogger

            ctx["system_logger"] = SystemLogger(all_drives=True)
        system = ctx["system_logger"].get_metrics(rates=True)
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
        response = _send("epoch_end", payload, project, name, ctx["model_id"], retry=1)
        _handle_control_response(trainer, ctx, response)

    _executor.submit(_send_and_check_cancel)


def on_model_save(trainer):
    """Upload model checkpoint (rate limited to every 15 min)."""
    ctx = getattr(trainer, "platform", None)
    if not ctx:
        return
    # Rate limit to every 15 minutes (900 seconds)
    if time() - ctx["last_upload"] < 900:
        return
    if ctx["checkpoint_upload"] and not ctx["checkpoint_upload"].done():
        return

    model_path = trainer.best if trainer.best and Path(trainer.best).exists() else trainer.last
    if not model_path:
        return

    project, name = _get_project_name(trainer)
    ctx["checkpoint_upload"] = _executor.submit(
        _upload_model, model_path, project, name, model_id=ctx["model_id"], run_id=ctx["run_id"]
    )
    ctx["last_upload"] = time()


def on_train_end(trainer):
    """Log final training results and upload the best model to Platform."""
    ctx = getattr(trainer, "platform", None)  # set only by on_pretrain_routine_start, so unset without an API key
    if not ctx:
        return

    project, name = _get_project_name(trainer)

    if ctx["cancelled"]:
        LOGGER.info(f"{PREFIX}Uploading partial results for cancelled training")

    # Stop console capture
    if ctx["console_logger"]:
        ctx["console_logger"].stop_capture()
        ctx["console_logger"] = None

    # Upload best model (blocking with progress bar to ensure it completes)
    artifact = None
    if trainer.best and Path(trainer.best).exists():
        if ctx["checkpoint_upload"]:
            ctx["checkpoint_upload"].result()
        artifact = _upload_model(
            trainer.best,
            project,
            name,
            progress=True,
            retry=3,
            model_id=ctx["model_id"],
            run_id=ctx["run_id"],
        )
        if not artifact:
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

    # stopper.best_epoch is 1-indexed; -1 aligns with the 0-indexed `epoch` field
    best_epoch = max(0, getattr(getattr(trainer, "stopper", None), "best_epoch", trainer.epoch + 1) - 1)

    image_metrics = trainer.validator.metrics.box.image_metrics if trainer.args.task == "detect" else {}
    validation = _validation_payload(image_metrics)
    _send(
        "training_complete",
        {
            "results": {
                "metrics": {**trainer.metrics, "fitness": trainer.fitness},
                "bestEpoch": best_epoch,
                "bestFitness": trainer.best_fitness,
                **({"calibration": c} if (c := getattr(trainer, "depth_calibration", None)) else {}),
                **({"validation": validation} if validation["rows"] else {}),
                **(artifact or {}),
            },
            "classNames": class_names,
            "plots": plots,
            "runId": ctx["run_id"],
        },
        project,
        name,
        ctx["model_id"],
        retry=4,  # Critical, more retries
    )
    url = f"{PLATFORM_URL}/{project}/{ctx.get('model_slug', name)}"
    LOGGER.info(f"{PREFIX}View results at {url}")


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_model_save": on_model_save,
    "on_train_end": on_train_end,
}
