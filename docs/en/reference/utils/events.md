---
description: Reference for utilities supporting telemetry, analytics, and event handling with lightweight background requests.
keywords: Ultralytics, YOLO, utils, telemetry, analytics, events, anonymization, background, JSON, POST, Python
---

# Reference for `ultralytics/utils/events.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/events.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/events.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`Events`](#ultralytics.utils.events.Events)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`Events.__call__`](#ultralytics.utils.events.Events.__call__)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`_post`](#ultralytics.utils.events._post)


## Class `ultralytics.utils.events.Events` {#ultralytics.utils.events.Events}

```python
Events(self) -> None
```

Collect and send anonymous usage analytics with rate-limiting.

Event collection and transmission are enabled when sync is enabled in settings, the current process is rank -1 or 0, tests are not running, the environment is online, and the installation source is either pip or the official Ultralytics GitHub repository.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `url` | `str` | Measurement Protocol endpoint for receiving anonymous events. |
| `events` | `list[dict]` | In-memory queue of event payloads awaiting transmission. |
| `rate_limit` | `float` | Minimum time in seconds between POST requests. |
| `t` | `float` | Timestamp of the last transmission in seconds since the epoch. |
| `metadata` | `dict` | Static metadata describing runtime, installation source, and environment. |
| `enabled` | `bool` | Flag indicating whether analytics collection is active. |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.events.Events.__call__) | Queue an event and flush the queue asynchronously when the rate limit elapses. |

<details>
<summary>Source code in <code>ultralytics/utils/events.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/events.py#L26-L110"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.utils.events.Events.__call__` {#ultralytics.utils.events.Events.\_\_call\_\_}

```python
def __call__(self, cfg, device = None) -> None
```

Queue an event and flush the queue asynchronously when the rate limit elapses.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `IterableSimpleNamespace` | The configuration object containing mode and task information. | *required* |
| `device` | `torch.device | str, optional` | The device type (e.g., 'cpu', 'cuda'). | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/events.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/events.py#L72-L110"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, cfg, device=None) -> None:
    """Queue an event and flush the queue asynchronously when the rate limit elapses.

    Args:
        cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
        device (torch.device | str, optional): The device type (e.g., 'cpu', 'cuda').
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.events._post` {#ultralytics.utils.events.\_post}

```python
def _post(url: str, data: dict, timeout: float = 5.0) -> None
```

Send a one-shot JSON POST request.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `url` | `str` |  | *required* |
| `data` | `dict` |  | *required* |
| `timeout` | `float` |  | `5.0` |

<details>
<summary>Source code in <code>ultralytics/utils/events.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/events.py#L16-L23"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _post(url: str, data: dict, timeout: float = 5.0) -> None:
    """Send a one-shot JSON POST request."""
    try:
        body = json.dumps(data, separators=(",", ":")).encode()  # compact JSON
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        urlopen(req, timeout=timeout).close()
    except Exception:
        pass
```
</details>

<br><br>
