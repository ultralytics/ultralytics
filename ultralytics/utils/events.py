import random
import time
from pathlib import Path
import json
from urllib.request import Request, urlopen
from threading import Thread

from ultralytics import __version__, SETTINGS
from ultralytics.utils import ARGV, GIT, IS_PIP_PACKAGE, PYTHON_VERSION, ENVIRONMENT, RANK, TESTS_RUNNING, ONLINE
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
from ultralytics.utils.torch_utils import get_cpu_info


class Events:
    """
    A class for collecting anonymous event analytics.

    Event analytics are enabled when sync=True in settings and disabled when sync=False. Run 'yolo settings' to see and
    update settings.

    Attributes:
        url (str): The URL to send anonymous events.
        events (list): List of collected events to be sent.
        rate_limit (float): The rate limit in seconds for sending events.
        t (float): Rate limit timer in seconds.
        metadata (dict): A dictionary containing metadata about the environment.
        enabled (bool): A flag to enable or disable Events based on certain conditions.
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self):
        """Initialize the Events object with default values for events, rate_limit, and metadata."""
        self.events = []  # events list
        self.rate_limit = 30.0  # rate limit (seconds)
        self.t = 0.0  # rate limit timer (seconds)
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "python": PYTHON_VERSION.rsplit(".", 1)[0],  # i.e. 3.13
            "CPU": get_cpu_info(),
            # "GPU": get_gpu_info(index=0) if cuda else None,
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

    def __call__(self, cfg, device=None):
        """
        Attempt to add a new event to the events list and send events if the rate limit is reached.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
            device (torch.device | str, optional): The device type (e.g., 'cpu', 'cuda').
        """
        if not self.enabled:
            # Events disabled, do nothing
            return

        # Attempt to add to events
        if len(self.events) < 25:  # Events list limited to 25 events (drop any events past this)
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
                "device": str(device),
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            self.events.append({"name": cfg.mode, "params": params})

        # Check rate limit
        t = time.time()
        if (t - self.t) < self.rate_limit:
            # Time is under rate limiter, wait to send
            return

        # Time is over rate limiter, send now
        data = {"client_id": SETTINGS["uuid"], "events": self.events}  # SHA-256 anonymized UUID hash and events list

        def _post(url, data, timeout=5):
            try:
                body = json.dumps(data, separators=(",", ":")).encode()  # compact JSON
                req = Request(url, data=body, headers={"Content-Type": "application/json"})
                urlopen(req, timeout=timeout).close()
            except Exception:
                pass

        Thread(target=_post, args=(self.url, data), daemon=True).start()

        # Reset events and rate limit timer
        self.events = []
        self.t = t


events = Events()
