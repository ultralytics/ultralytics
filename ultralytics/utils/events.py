# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import random
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

from ultralytics import SETTINGS, __version__
from ultralytics.cfg import MODES, TASKS
from ultralytics.utils import (
    ARGV,
    ENVIRONMENT,
    GIT,
    IS_PIP_PACKAGE,
    ONLINE,
    PYTHON_VERSION,
    RANK,
    TESTS_RUNNING,
    TORCH_VERSION,
)
from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info, unwrap_model


def _post(url: str, data: dict, timeout: float = 5.0) -> None:
    """Send a one-shot JSON POST request."""
    try:
        body = json.dumps(data, separators=(",", ":")).encode()  # compact JSON
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        urlopen(req, timeout=timeout).close()
    except Exception:
        pass


def _arch(model):
    """Return the architecture a model is built from, i.e. 'yolo11n-seg', or None if it cannot be determined.

    The config travels inside a checkpoint, so fine-tuned models report the architecture they descend from however many
    generations back.
    """
    desc = f"{getattr(model, 'description', '')}".split()  # exported models name the arch here instead of a YAML
    # a trainer holds the config on the model itself, a predictor one level down on the backend; SAM has no .model
    yaml = getattr(model, "yaml", None) or getattr(getattr(model, "model", None), "yaml", None) or {}
    stem = Path(yaml.get("yaml_file", "")).stem or (desc[1] if len(desc) > 1 else "")
    return stem.lower()[:100] or None  # lowercased so one arch cannot split into two cells; 100 is the GA4 limit


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
            "torch": TORCH_VERSION,
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

    def __call__(self, cfg, device=None, run=None) -> None:
        """Queue an event and flush the queue asynchronously when the rate limit elapses.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
            device (torch.device | str, optional): The device type (e.g., 'cpu', 'cuda').
            run (BasePredictor | BaseTrainer, optional): The completed run, read for the mode's result fields.
        """
        # An event is named for its mode, so an arbitrary mode becomes an arbitrary event name, and GA4 drops every new
        # name once a property reaches 500 of them - eventually the real ones.
        if not self.enabled or cfg.mode not in MODES or cfg.task not in TASKS:
            return

        # Attempt to enqueue a new event
        if len(self.events) < 25:  # Queue limited to 25 events to bound memory and traffic
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": Path(str(cfg.model)).name[:100] if cfg.model else None,  # basename, never a path
                "device": str(device),
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            elif cfg.mode == "train":
                # Guarded and ordered exactly as predict below, for the same reasons
                try:
                    # grouping key; stem unifies YAML and directory, and isinstance keeps a dict repr's path out
                    params["data"] = Path(cfg.data).stem[:100] if isinstance(cfg.data, (str, Path)) else None
                    params["imgsz"] = cfg.imgsz
                    # epochs this session, to stay consistent with hours; a resumed run restores an absolute epoch
                    params["epochs_done"] = run.epoch + 1 - run.start_epoch
                    params["batch"] = run.batch_size  # resolved, since autobatch and OOM retries both move it
                    params["hours"] = round((time.time() - run.train_time_start) / 3600, 4)
                    params["n"] = len(run.train_loader.dataset)  # train split size, matching predict's n
                    if run.best_fitness is not None:  # None when a run never validated
                        # a per-task composite: mAP50-95 for detect, box+mask for segment, so compare within a task
                        params["fitness"] = round(float(run.best_fitness), 5)
                    # both resolved: the default 'auto' fits its own optimizer and lr0, ignoring cfg.lr0
                    params["optimizer"] = type(run.optimizer).__name__
                    # min, since MuSGD splits every group in two and puts the finetuning lr*3 half first
                    params["lr0"] = min(g["initial_lr"] for g in run.optimizer.param_groups)
                    flags = {
                        "pretrained": bool(cfg.pretrained),
                        "cos_lr": cfg.cos_lr,
                        "amp": run.amp,  # as applied: check_amp() turns a requested True off on unsupported hardware
                        "rect": cfg.rect,
                        "multi_scale": bool(cfg.multi_scale),
                        "freeze": bool(cfg.freeze),  # freeze=0 and freeze=[] both freeze nothing
                        "dropout": cfg.dropout > 0,
                        "early_stop": run.epoch + 1 < run.epochs,  # .stop is also set on the last planned epoch
                        "resume": bool(cfg.resume),  # fitness carries over, epochs and hours do not
                    }
                    params["flags"] = ",".join(k for k, v in flags.items() if v) or None
                    params["arch"] = _arch(unwrap_model(run.model))  # DDP and EMA both wrap away the .yaml
                    params["ngpu"] = run.world_size if run.world_size > 1 else None  # only a count above one informs
                    if device.type == "cuda":  # makes hours comparable
                        params["GPU"] = get_gpu_info(device.index or 0).rsplit(", ", 1)[0]
                except Exception:
                    pass
            elif cfg.mode in {"predict", "track"}:  # track runs the predictor too, and is most of the video inference
                # Reads inside the guard so nothing can raise into a user's run, cheapest first; order is drop order
                try:
                    params["n"] = run.seen  # predictor state this file's own owner sets, so it cannot raise
                    params["pixels"] = run.pixels  # mean inference area, which FLOPs scale with; sqrt for a side
                    for k, v in (run.speed or {}).items():  # absent when a run processed no images
                        params[f"{k}_ms"] = round(v, 3)
                    params["batch"] = min(getattr(run.dataset, "bs", 0), run.seen) or None
                    model = run.model
                    params["format"] = model.format
                    params["nc"] = len(getattr(model, "names", None) or ()) or None  # drives head width and NMS
                    # toggles that move inference time, as applied: compile replaces .model, end2end reflects the backend output
                    flags = {
                        "compile": hasattr(model, "_orig_mod"),
                        "end2end": getattr(model, "end2end", False),
                        "augment": cfg.augment,
                    }
                    params["flags"] = ",".join(k for k, v in flags.items() if v) or None
                    params["arch"] = _arch(model)  # reads into model.description and model.model.yaml
                    meta = getattr(model, "metadata", None) or {}
                    params["quantize"] = str(meta.get("args", {}).get("quantize") or cfg.quantize or 32)
                    if device.type == "cuda":  # CUDA is already initialized here, so this costs nothing
                        params["GPU"] = get_gpu_info(device.index or 0).rsplit(", ", 1)[0]
                    session = getattr(model, "session", None)  # ONNX Runtime provider, else OpenVINO device
                    ov = getattr(model, "ov_compiled_model", None)  # an Arc GPU run must not look like CPU
                    devices = session.get_providers() if session else ov.get_property("EXECUTION_DEVICES") if ov else []
                    params["provider"] = devices[0] if devices else None  # last: least reliable read
                except Exception:
                    pass
            # nulls are dropped anyway, and an event over 25 params is rejected outright, so cap rather than lose it
            params = dict([(k, v) for k, v in params.items() if v is not None][:25])
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


def on_train_end(trainer):
    """Record an anonymous training event after final metrics are available."""
    events(trainer.args, trainer.device, trainer)


def on_val_start(validator):
    """Record an anonymous standalone validation event.

    A trainer's final validation retains mode=train, so the guard prevents a duplicate train event.
    """
    if validator.args.mode == "val":
        events(validator.args, validator.device)


def on_predict_end(predictor):
    """Record an anonymous prediction event after per-image speeds are available."""
    events(predictor.args, predictor.device, predictor)


def on_export_start(exporter):
    """Record an anonymous export event."""
    events(exporter.args, exporter.device)


callbacks = {
    "on_train_end": on_train_end,
    "on_val_start": on_val_start,
    "on_predict_end": on_predict_end,
    "on_export_start": on_export_start,
}
