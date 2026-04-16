# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import random
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

import torch

from ultralytics import SETTINGS, __version__
from ultralytics.utils import ARGV, ENVIRONMENT, GIT, IS_PIP_PACKAGE, ONLINE, PYTHON_VERSION, RANK, TESTS_RUNNING
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
        self._thread = None  # reference to the last background send thread
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "python": PYTHON_VERSION.rsplit(".", 1)[0],  # i.e. 3.13
            "CPU": get_cpu_info(),
            "GPU": " | ".join(get_gpu_info(i) for i in range(torch.cuda.device_count())) or None,
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

    def __call__(self, cfg, device=None, backend=None, imgsz=None, model_params=None, speed=None) -> None:
        """Queue an event and flush the queue asynchronously when the rate limit elapses.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
            device (torch.device | str, optional): The device type (e.g., 'cpu', 'cuda').
            backend (object | None, optional): The inference backend instance used during prediction.
            imgsz (int | list | None, optional): Input image size used during prediction.
            model_params (int | None, optional): Total number of model parameters.
            speed (dict | None, optional): Per-image inference speed dict with keys 'preprocess', 'inference', and
                'postprocess' (all in milliseconds).
        """
        if not self.enabled:
            # Events disabled, do nothing
            return

        # Build the event payload
        params = {
            **self.metadata,
            "task": cfg.task,
            "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
            "device": str(device),
        }
        if cfg.mode == "export":
            params["format"] = cfg.format
        if cfg.mode == "predict":
            params["backend"] = type(backend).__name__ if backend is not None else None
            if imgsz is not None:
                params["imgsz"] = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
            if model_params is not None:
                params["model_params"] = model_params
            if speed is not None:
                params["speed_preprocess_ms"] = round(speed.get("preprocess") or 0, 2)
                params["speed_inference_ms"] = round(speed.get("inference") or 0, 2)
                params["speed_postprocess_ms"] = round(speed.get("postprocess") or 0, 2)

        # Upsert: replace an existing event with the same name, or append if none.
        # This keeps the queue from growing during long-running streams (e.g. predict over many batches)
        # while still preserving distinct events for different modes (train, val, export, predict).
        event = {"name": cfg.mode, "params": params}
        for i, e in enumerate(self.events):
            if e["name"] == cfg.mode:
                self.events[i] = event
                break
        else:
            self.events.append(event)

        # Check rate limit and return early if under limit
        t = time.time()
        if (t - self.t) < self.rate_limit:
            return

        # Overrate limit: send a snapshot of queued events in a background thread
        payload_events = list(self.events)  # snapshot to avoid race with queue reset
        self._thread = Thread(
            target=_post,
            args=(self.url, {"client_id": SETTINGS["uuid"], "events": payload_events}),  # SHA-256 anonymized
            daemon=True,
        )
        self._thread.start()

        # Reset queue and rate limit timer
        self.events = []
        self.t = t

    def flush(self, timeout: float = 5.0) -> None:
        """Send any pending queued events synchronously and wait for an in-flight background thread.

        Call this at the end of short-lived processes (e.g. single-image predict) to ensure all
        queued events are delivered even when the rate limit has not yet elapsed.

        Args:
            timeout (float): Maximum seconds to wait for the background thread to finish.
        """
        # Wait for any in-flight background thread first
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        # Synchronously send any events still sitting in the queue (rate limit not yet elapsed)
        if self.events:
            _post(self.url, {"client_id": SETTINGS["uuid"], "events": list(self.events)})
            self.events = []


events = Events()
