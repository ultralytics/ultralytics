# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

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


def _arch(model) -> str:
    """Return the shipped architecture a model is built from, i.e. 'yolo11n-seg', else 'custom'."""
    desc = getattr(model, "description", "").split()  # exported models name the arch here instead of a YAML
    stem = Path((getattr(model.model, "yaml", None) or {}).get("yaml_file", "")).stem
    stem = stem or (desc[1].lower() if len(desc) > 1 else "")
    unified = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", stem)  # configs ship unscaled, i.e. yolo11n-seg -> yolo11-seg
    return stem if any((ROOT / "cfg" / "models").rglob(f"{unified}.yaml")) else "custom"


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
            model (AutoBackend, optional): The loaded model, for prediction runs.
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
                try:  # never raise, warn or delay inside a user's prediction run
                    session = getattr(model, "session", None)  # provider drives latency as much as format does
                    params["format"] = model.format
                    params["provider"] = session.get_providers()[0] if session else None
                    params["arch"] = _arch(model)
                    params["quantize"] = 16 if model.fp16 else cfg.quantize  # as validator.py records it
                    params["imgsz"] = cfg.imgsz if isinstance(cfg.imgsz, int) else max(cfg.imgsz)
                    params["batch"] = cfg.batch
                    params["nc"] = len(model.names)  # class count drives head width and NMS cost
                    params["n"] = n
                    params["torch"] = TORCH_VERSION
                    for k, v in (speed or {}).items():
                        params[f"{k}_ms"] = round(v, 3)
                    if device.type == "cuda":  # CUDA is already initialized here, so this costs nothing
                        params["GPU"] = get_gpu_info(device.index or 0)
                except Exception:
                    pass
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
