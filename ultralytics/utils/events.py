# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import functools
import json
import random
import re
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

from ultralytics import SETTINGS, __version__
from ultralytics.utils import (
    ARGV,
    ENVIRONMENT,
    GIT,
    IS_PIP_PACKAGE,
    ONLINE,
    PYTHON_VERSION,
    RANK,
    ROOT,
    TESTS_RUNNING,
    TORCH_VERSION,
)
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info


def _post(url: str, data: dict, timeout: float = 5.0) -> None:
    """Send a one-shot JSON POST request."""
    try:
        body = json.dumps(data, separators=(",", ":")).encode()  # compact JSON
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        urlopen(req, timeout=timeout).close()
    except Exception:
        pass


@functools.lru_cache(maxsize=1)
def _shipped_archs() -> frozenset[str]:
    """Return the stems of the model configs shipped with the package, i.e. {'yolo26', 'yolo11-seg', ...}."""
    return frozenset(p.stem for p in (ROOT / "cfg" / "models").rglob("*.yaml"))


def _arch(model) -> str:
    """Return the shipped architecture a model is built from, i.e. 'yolo11n-seg'.

    Fine-tuned checkpoints report their true architecture even though their filename is anonymized, since the
    source config travels inside the checkpoint. User-authored architectures collapse to 'custom' so that private
    config filenames are never transmitted.

    Args:
        model (AutoBackend): The loaded inference model.

    Returns:
        (str): The scaled architecture name, or 'custom' if it is not one shipped with the package.
    """
    stem = Path((getattr(getattr(model, "model", None), "yaml", None) or {}).get("yaml_file", "")).stem
    if not stem:  # exported models carry the name in their metadata description rather than a source config
        match = re.search(r"Ultralytics (\S+) model", getattr(model, "description", "") or "")
        stem = match.group(1).lower() if match else ""
    unified = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", stem)  # configs ship unscaled, i.e. yolo11n-seg -> yolo11-seg
    return stem if unified in _shipped_archs() else "custom"


def _predict_params(cfg, device, model, speed, n) -> dict:
    """Return the deployment and timing fields describing a completed prediction run.

    Args:
        cfg (IterableSimpleNamespace): The configuration the run used.
        device (torch.device | str): The device inference ran on.
        model (AutoBackend): The loaded inference model.
        speed (dict[str, float] | None): Per-image preprocess, inference and postprocess times in milliseconds.
        n (int): Number of images processed.

    Returns:
        (dict): Fields to attach to the predict event.
    """
    names = getattr(model, "names", None)
    session = getattr(model, "session", None)  # ONNX Runtime: the provider drives latency as much as the format
    params = {
        "format": getattr(model, "format", None),
        "provider": session.get_providers()[0] if session is not None else None,
        "arch": _arch(model),
        "precision": 16 if getattr(model, "fp16", False) else cfg.quantize,
        "imgsz": cfg.imgsz if isinstance(cfg.imgsz, int) else max(cfg.imgsz),
        "batch": cfg.batch,
        "nc": len(names) if names else None,  # class count drives head width and NMS cost
        "n": n,
        "torch": TORCH_VERSION,
        **{f"{k}_ms": round(v, 3) for k, v in (speed or {}).items()},
    }
    if getattr(device, "type", None) == "cuda":  # CUDA is already initialized here, so this adds no context or cost
        params["GPU"] = get_gpu_info(device.index or 0)
    return params


class Events:
    """Collect and send anonymous usage analytics with rate-limiting.

    Event collection and transmission are enabled when sync is enabled in settings, the current process is rank -1 or 0,
    tests are not running, the environment is online, and the installation source is either pip or the official
    Ultralytics GitHub repository.

    Attributes:
        url (str): Measurement Protocol endpoint for receiving anonymous events.
        events (list[dict]): In-memory queue of event payloads awaiting transmission.
        rate_limit (float): Minimum time in seconds between POST requests.
        t (float): Timestamp of the last transmission in seconds since the epoch.
        metadata (dict): Static metadata describing runtime, installation source, and environment.
        enabled (bool): Flag indicating whether analytics collection is active.

    Methods:
        __init__: Initialize the event queue, rate limiter, and runtime metadata.
        __call__: Queue an event and trigger a non-blocking send when the rate limit elapses.
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self) -> None:
        """Initialize the Events instance with queue, rate limiter, and environment metadata."""
        self.events = []  # pending events
        self.rate_limit = 30.0  # rate limit (seconds)
        self.t = 0.0  # last send timestamp (seconds)
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "python": PYTHON_VERSION.rsplit(".", 1)[0],  # i.e. 3.13
            "CPU": get_cpu_info(),
            "version": __version__,
            "env": ENVIRONMENT,
            "session_id": round(random.random() * 1e15),
            "engagement_time_msec": 1000,
        }
        self.enabled = (
            SETTINGS["sync"]
            and RANK in {-1, 0}
            and not TESTS_RUNNING
            and ONLINE
            and (IS_PIP_PACKAGE or GIT.origin == "https://github.com/ultralytics/ultralytics.git")
        )

    def __call__(self, cfg, device=None, model=None, speed=None, n=0) -> None:
        """Queue an event and flush the queue asynchronously when the rate limit elapses.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
            device (torch.device | str, optional): The device type (e.g., 'cpu', 'cuda').
            model (AutoBackend, optional): The loaded inference model, for prediction runs.
            speed (dict[str, float], optional): Per-image speeds in milliseconds, for prediction runs.
            n (int, optional): Number of images processed, for prediction runs.
        """
        if not self.enabled:
            # Events disabled, do nothing
            return

        # Attempt to enqueue a new event
        if len(self.events) < 25:  # Queue limited to 25 events to bound memory and traffic
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
                "device": str(device),
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            elif cfg.mode == "predict":
                try:
                    params.update(_predict_params(cfg, device, model, speed, n))
                except Exception:
                    pass  # analytics must never raise, warn or delay inside a user's prediction run
            self.events.append({"name": cfg.mode, "params": params})

        # Check rate limit and return early if under limit
        t = time.time()
        if (t - self.t) < self.rate_limit:
            return

        # Overrate limit: send a snapshot of queued events in a background thread
        payload_events = list(self.events)  # snapshot to avoid race with queue reset
        Thread(
            target=_post,
            args=(self.url, {"client_id": SETTINGS["uuid"], "events": payload_events}),  # SHA-256 anonymized
            daemon=True,
        ).start()

        # Reset queue and rate limit timer
        self.events = []
        self.t = t


events = Events()
