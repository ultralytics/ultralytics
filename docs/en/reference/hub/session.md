---
description: Explore the HUBTrainingSession class for managing Ultralytics YOLO model training, heartbeats, and checkpointing.
keywords: Ultralytics, YOLO, HUBTrainingSession, model training, heartbeats, checkpointing, Python
---

# Reference for `ultralytics/hub/session.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`HUBTrainingSession`](#ultralytics.hub.session.HUBTrainingSession)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`HUBTrainingSession.create_session`](#ultralytics.hub.session.HUBTrainingSession.create_session)
        - [`HUBTrainingSession.load_model`](#ultralytics.hub.session.HUBTrainingSession.load_model)
        - [`HUBTrainingSession.create_model`](#ultralytics.hub.session.HUBTrainingSession.create_model)
        - [`HUBTrainingSession._parse_identifier`](#ultralytics.hub.session.HUBTrainingSession._parse_identifier)
        - [`HUBTrainingSession._set_train_args`](#ultralytics.hub.session.HUBTrainingSession._set_train_args)
        - [`HUBTrainingSession.request_queue`](#ultralytics.hub.session.HUBTrainingSession.request_queue)
        - [`HUBTrainingSession._should_retry`](#ultralytics.hub.session.HUBTrainingSession._should_retry)
        - [`HUBTrainingSession._get_failure_message`](#ultralytics.hub.session.HUBTrainingSession._get_failure_message)
        - [`HUBTrainingSession.upload_metrics`](#ultralytics.hub.session.HUBTrainingSession.upload_metrics)
        - [`HUBTrainingSession.upload_model`](#ultralytics.hub.session.HUBTrainingSession.upload_model)
        - [`HUBTrainingSession._show_upload_progress`](#ultralytics.hub.session.HUBTrainingSession._show_upload_progress)
        - [`HUBTrainingSession._iterate_content`](#ultralytics.hub.session.HUBTrainingSession._iterate_content)


## Class `ultralytics.hub.session.HUBTrainingSession` {#ultralytics.hub.session.HUBTrainingSession}

```python
HUBTrainingSession(self, identifier: str)
```

HUB training session for Ultralytics HUB YOLO models.

This class encapsulates the functionality for interacting with Ultralytics HUB during model training, including model creation, metrics tracking, and checkpoint uploading.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `identifier` | `str` | Model identifier used to initialize the HUB training session. It can be a URL string or a<br>    model key with specific format. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model_id` | `str` | Identifier for the YOLO model being trained. |
| `model_url` | `str` | URL for the model in Ultralytics HUB. |
| `rate_limits` | `dict[str, int]` | Rate limits for different API calls in seconds. |
| `timers` | `dict[str, Any]` | Timers for rate limiting. |
| `metrics_queue` | `dict[str, Any]` | Queue for the model's metrics. |
| `metrics_upload_failed_queue` | `dict[str, Any]` | Queue for metrics that failed to upload. |
| `model` | `Any` | Model data fetched from Ultralytics HUB. |
| `model_file` | `str` | Path to the model file. |
| `train_args` | `dict[str, Any]` | Arguments for training the model. |
| `client` | `Any` | Client for interacting with Ultralytics HUB. |
| `filename` | `str` | Filename of the model. |

**Methods**

| Name | Description |
| --- | --- |
| [`_get_failure_message`](#ultralytics.hub.session.HUBTrainingSession._get_failure_message) | Generate a retry message based on the response status code. |
| [`_iterate_content`](#ultralytics.hub.session.HUBTrainingSession._iterate_content) | Process the streamed HTTP response data. |
| [`_parse_identifier`](#ultralytics.hub.session.HUBTrainingSession._parse_identifier) | Parse the given identifier to determine the type and extract relevant components. |
| [`_set_train_args`](#ultralytics.hub.session.HUBTrainingSession._set_train_args) | Initialize training arguments and create a model entry on the Ultralytics HUB. |
| [`_should_retry`](#ultralytics.hub.session.HUBTrainingSession._should_retry) | Determine if a request should be retried based on the HTTP status code. |
| [`_show_upload_progress`](#ultralytics.hub.session.HUBTrainingSession._show_upload_progress) | Display a progress bar to track the upload progress of a file download. |
| [`create_model`](#ultralytics.hub.session.HUBTrainingSession.create_model) | Initialize a HUB training session with the specified model arguments. |
| [`create_session`](#ultralytics.hub.session.HUBTrainingSession.create_session) | Create an authenticated HUBTrainingSession or return None. |
| [`load_model`](#ultralytics.hub.session.HUBTrainingSession.load_model) | Load an existing model from Ultralytics HUB using the provided model identifier. |
| [`request_queue`](#ultralytics.hub.session.HUBTrainingSession.request_queue) | Execute request_func with retries, timeout handling, optional threading, and progress tracking. |
| [`upload_metrics`](#ultralytics.hub.session.HUBTrainingSession.upload_metrics) | Upload model metrics to Ultralytics HUB. |
| [`upload_model`](#ultralytics.hub.session.HUBTrainingSession.upload_model) | Upload a model checkpoint to Ultralytics HUB. |

**Examples**

```python
Create a training session with a model URL
>>> session = HUBTrainingSession("https://hub.ultralytics.com/models/example-model")
>>> session.upload_metrics()
```

**Raises**

| Type | Description |
| --- | --- |
| `ValueError` | If the provided model identifier is invalid. |
| `ConnectionError` | If connecting with global API key is not supported. |
| `ModuleNotFoundError` | If hub-sdk package is not installed. |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L21-L420"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class HUBTrainingSession:
    """HUB training session for Ultralytics HUB YOLO models.

    This class encapsulates the functionality for interacting with Ultralytics HUB during model training, including
    model creation, metrics tracking, and checkpoint uploading.

    Attributes:
        model_id (str): Identifier for the YOLO model being trained.
        model_url (str): URL for the model in Ultralytics HUB.
        rate_limits (dict[str, int]): Rate limits for different API calls in seconds.
        timers (dict[str, Any]): Timers for rate limiting.
        metrics_queue (dict[str, Any]): Queue for the model's metrics.
        metrics_upload_failed_queue (dict[str, Any]): Queue for metrics that failed to upload.
        model (Any): Model data fetched from Ultralytics HUB.
        model_file (str): Path to the model file.
        train_args (dict[str, Any]): Arguments for training the model.
        client (Any): Client for interacting with Ultralytics HUB.
        filename (str): Filename of the model.

    Examples:
        Create a training session with a model URL
        >>> session = HUBTrainingSession("https://hub.ultralytics.com/models/example-model")
        >>> session.upload_metrics()
    """

    def __init__(self, identifier: str):
        """Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session. It can be a URL string or a
                model key with specific format.

        Raises:
            ValueError: If the provided model identifier is invalid.
            ConnectionError: If connecting with global API key is not supported.
            ModuleNotFoundError: If hub-sdk package is not installed.
        """
        from hub_sdk import HUBClient

        self.rate_limits = {"metrics": 3, "ckpt": 900, "heartbeat": 300}  # rate limits (seconds)
        self.metrics_queue = {}  # holds metrics for each epoch until upload
        self.metrics_upload_failed_queue = {}  # holds metrics for each epoch if upload failed
        self.timers = {}  # holds timers in ultralytics/utils/callbacks/hub.py
        self.model = None
        self.model_url = None
        self.model_file = None
        self.train_args = None

        # Parse input
        api_key, model_id, self.filename = self._parse_identifier(identifier)

        # Get credentials
        active_key = api_key or SETTINGS.get("api_key")
        credentials = {"api_key": active_key} if active_key else None  # set credentials

        # Initialize client
        self.client = HUBClient(credentials)

        # Load models
        try:
            if model_id:
                self.load_model(model_id)  # load existing model
            else:
                self.model = self.client.model()  # load empty model
        except Exception:
            if identifier.startswith(f"{HUB_WEB_ROOT}/models/") and not self.client.authenticated:
                LOGGER.warning(
                    f"{PREFIX}Please log in using 'yolo login API_KEY'. "
                    "You can find your API Key at: https://hub.ultralytics.com/settings?tab=api+keys."
                )
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession._get_failure_message` {#ultralytics.hub.session.HUBTrainingSession.\_get\_failure\_message}

```python
def _get_failure_message(self, response, retry: int, timeout: int) -> str
```

Generate a retry message based on the response status code.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `response` | `requests.Response` | The HTTP response object. | *required* |
| `retry` | `int` | The number of retry attempts allowed. | *required* |
| `timeout` | `int` | The maximum timeout duration. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | The retry message. |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L334-L357"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_failure_message(self, response, retry: int, timeout: int) -> str:
    """Generate a retry message based on the response status code.

    Args:
        response (requests.Response): The HTTP response object.
        retry (int): The number of retry attempts allowed.
        timeout (int): The maximum timeout duration.

    Returns:
        (str): The retry message.
    """
    if self._should_retry(response.status_code):
        return f"Retrying {retry}x for {timeout}s." if retry else ""
    elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:  # rate limit
        headers = response.headers
        return (
            f"Rate limit reached ({headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}). "
            f"Please retry after {headers['Retry-After']}s."
        )
    else:
        try:
            return response.json().get("message", "No JSON message.")
        except AttributeError:
            return "Unable to read JSON."
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession._iterate_content` {#ultralytics.hub.session.HUBTrainingSession.\_iterate\_content}

```python
def _iterate_content(response) -> None
```

Process the streamed HTTP response data.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `response` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L417-L420"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _iterate_content(response) -> None:
    """Process the streamed HTTP response data."""
    for _ in response.iter_content(chunk_size=1024):
        pass  # Do nothing with data chunks
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession._parse_identifier` {#ultralytics.hub.session.HUBTrainingSession.\_parse\_identifier}

```python
def _parse_identifier(identifier: str)
```

Parse the given identifier to determine the type and extract relevant components.

The method supports different identifier formats:
    - A HUB model URL https://hub.ultralytics.com/models/MODEL
    - A HUB model URL with API Key https://hub.ultralytics.com/models/MODEL?api_key=APIKEY
    - A local filename that ends with '.pt' or '.yaml'

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `identifier` | `str` | The identifier string to be parsed. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `api_key (str | None)` | Extracted API key if present. |
| `model_id (str | None)` | Extracted model ID if present. |
| `filename (str | None)` | Extracted filename if present. |

**Raises**

| Type | Description |
| --- | --- |
| `HUBModelError` | If the identifier format is not recognized. |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L181-L210"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _parse_identifier(identifier: str):
    """Parse the given identifier to determine the type and extract relevant components.

    The method supports different identifier formats:
        - A HUB model URL https://hub.ultralytics.com/models/MODEL
        - A HUB model URL with API Key https://hub.ultralytics.com/models/MODEL?api_key=APIKEY
        - A local filename that ends with '.pt' or '.yaml'

    Args:
        identifier (str): The identifier string to be parsed.

    Returns:
        api_key (str | None): Extracted API key if present.
        model_id (str | None): Extracted model ID if present.
        filename (str | None): Extracted filename if present.

    Raises:
        HUBModelError: If the identifier format is not recognized.
    """
    api_key, model_id, filename = None, None, None
    if identifier.endswith((".pt", ".yaml")):
        filename = identifier
    elif identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
        parsed_url = urlparse(identifier)
        model_id = Path(parsed_url.path).stem  # handle possible final backslash robustly
        query_params = parse_qs(parsed_url.query)  # dictionary, i.e. {"api_key": ["API_KEY_HERE"]}
        api_key = query_params.get("api_key", [None])[0]
    else:
        raise HUBModelError(f"model='{identifier} invalid, correct format is {HUB_WEB_ROOT}/models/MODEL_ID")
    return api_key, model_id, filename
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession._set_train_args` {#ultralytics.hub.session.HUBTrainingSession.\_set\_train\_args}

```python
def _set_train_args(self)
```

Initialize training arguments and create a model entry on the Ultralytics HUB.

This method sets up training arguments based on the model's state and updates them with any additional arguments provided. It handles different states of the model, such as whether it's resumable, pretrained, or requires specific file setup.

**Raises**

| Type | Description |
| --- | --- |
| `ValueError` | If the model is already trained, if required dataset information is missing, or if there are<br>    issues with the provided training arguments. |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L212-L241"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _set_train_args(self):
    """Initialize training arguments and create a model entry on the Ultralytics HUB.

    This method sets up training arguments based on the model's state and updates them with any additional arguments
    provided. It handles different states of the model, such as whether it's resumable, pretrained, or requires
    specific file setup.

    Raises:
        ValueError: If the model is already trained, if required dataset information is missing, or if there are
            issues with the provided training arguments.
    """
    if self.model.is_resumable():
        # Model has saved weights
        self.train_args = {"data": self.model.get_dataset_url(), "resume": True}
        self.model_file = self.model.get_weights_url("last")
    else:
        # Model has no saved weights
        self.train_args = self.model.data.get("train_args")  # new response

        # Set the model file as either a *.pt or *.yaml file
        self.model_file = (
            self.model.get_weights_url("parent") if self.model.is_pretrained() else self.model.get_architecture()
        )

    if "data" not in self.train_args:
        # RF bug - datasets are sometimes not exported
        raise ValueError("Dataset may still be processing. Please wait a minute and try again.")

    self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)  # YOLOv5->YOLOv5u
    self.model_id = self.model.id
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession._should_retry` {#ultralytics.hub.session.HUBTrainingSession.\_should\_retry}

```python
def _should_retry(status_code: int) -> bool
```

Determine if a request should be retried based on the HTTP status code.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `status_code` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L325-L332"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _should_retry(status_code: int) -> bool:
    """Determine if a request should be retried based on the HTTP status code."""
    retry_codes = {
        HTTPStatus.REQUEST_TIMEOUT,
        HTTPStatus.BAD_GATEWAY,
        HTTPStatus.GATEWAY_TIMEOUT,
    }
    return status_code in retry_codes
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession._show_upload_progress` {#ultralytics.hub.session.HUBTrainingSession.\_show\_upload\_progress}

```python
def _show_upload_progress(content_length: int, response) -> None
```

Display a progress bar to track the upload progress of a file download.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `content_length` | `int` |  | *required* |
| `response` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L410-L414"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _show_upload_progress(content_length: int, response) -> None:
    """Display a progress bar to track the upload progress of a file download."""
    with TQDM(total=content_length, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession.create_model` {#ultralytics.hub.session.HUBTrainingSession.create\_model}

```python
def create_model(self, model_args: dict[str, Any])
```

Initialize a HUB training session with the specified model arguments.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model_args` | `dict[str, Any]` | Arguments for creating the model, including batch size, epochs, image size,<br>    etc. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `None` | If the model could not be created. |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L138-L178"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def create_model(self, model_args: dict[str, Any]):
    """Initialize a HUB training session with the specified model arguments.

    Args:
        model_args (dict[str, Any]): Arguments for creating the model, including batch size, epochs, image size,
            etc.

    Returns:
        (None): If the model could not be created.
    """
    payload = {
        "config": {
            "batchSize": model_args.get("batch", -1),
            "epochs": model_args.get("epochs", 300),
            "imageSize": model_args.get("imgsz", 640),
            "patience": model_args.get("patience", 100),
            "device": str(model_args.get("device", "")),  # convert None to string
            "cache": str(model_args.get("cache", "ram")),  # convert True, False, None to string
        },
        "dataset": {"name": model_args.get("data")},
        "lineage": {
            "architecture": {"name": self.filename.replace(".pt", "").replace(".yaml", "")},
            "parent": {},
        },
        "meta": {"name": self.filename},
    }

    if self.filename.endswith(".pt"):
        payload["lineage"]["parent"]["name"] = self.filename

    self.model.create_model(payload)

    if not self.model.id:
        raise HUBModelError(f"❌ Failed to create model '{self.filename}' on Ultralytics HUB. Please try again.")

    self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"

    # Start heartbeats for HUB to monitor agent
    self.model.start_heartbeat(self.rate_limits["heartbeat"])

    LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession.create_session` {#ultralytics.hub.session.HUBTrainingSession.create\_session}

```python
def create_session(cls, identifier: str, args: dict[str, Any] | None = None)
```

Create an authenticated HUBTrainingSession or return None.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `identifier` | `str` | Model identifier used to initialize the HUB training session. | *required* |
| `args` | `dict[str, Any], optional` | Arguments for creating a new model if identifier is not a HUB model URL. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `session (HUBTrainingSession | None)` | An authenticated session or None if creation fails. |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L93-L111"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@classmethod
def create_session(cls, identifier: str, args: dict[str, Any] | None = None):
    """Create an authenticated HUBTrainingSession or return None.

    Args:
        identifier (str): Model identifier used to initialize the HUB training session.
        args (dict[str, Any], optional): Arguments for creating a new model if identifier is not a HUB model URL.

    Returns:
        session (HUBTrainingSession | None): An authenticated session or None if creation fails.
    """
    try:
        session = cls(identifier)
        if args and not identifier.startswith(f"{HUB_WEB_ROOT}/models/"):  # not a HUB model URL
            session.create_model(args)
            assert session.model.id, "HUB model not loaded correctly"
        return session
    # PermissionError and ModuleNotFoundError indicate hub-sdk not installed
    except (PermissionError, ModuleNotFoundError, AssertionError):
        return None
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession.load_model` {#ultralytics.hub.session.HUBTrainingSession.load\_model}

```python
def load_model(self, model_id: str)
```

Load an existing model from Ultralytics HUB using the provided model identifier.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model_id` | `str` | The identifier of the model to load. | *required* |

**Raises**

| Type | Description |
| --- | --- |
| `ValueError` | If the specified HUB model does not exist. |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L113-L136"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def load_model(self, model_id: str):
    """Load an existing model from Ultralytics HUB using the provided model identifier.

    Args:
        model_id (str): The identifier of the model to load.

    Raises:
        ValueError: If the specified HUB model does not exist.
    """
    self.model = self.client.model(model_id)
    if not self.model.data:  # then model does not exist
        raise HUBModelError(f"❌ Model not found: '{model_id}'. Verify the model ID is correct.")

    self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"
    if self.model.is_trained():
        LOGGER.info(f"Loading trained HUB model {self.model_url} 🚀")
        url = self.model.get_weights_url("best")  # download URL with auth
        self.model_file = checks.check_file(url, download_dir=Path(SETTINGS["weights_dir"]) / "hub" / self.model.id)
        return

    # Set training args and start heartbeats for HUB to monitor agent
    self._set_train_args()
    self.model.start_heartbeat(self.rate_limits["heartbeat"])
    LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession.request_queue` {#ultralytics.hub.session.HUBTrainingSession.request\_queue}

```python
def request_queue(
    self,
    request_func,
    retry: int = 3,
    timeout: int = 30,
    thread: bool = True,
    verbose: bool = True,
    progress_total: int | None = None,
    stream_response: bool | None = None,
    *args,
    **kwargs,
)
```

Execute request_func with retries, timeout handling, optional threading, and progress tracking.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `request_func` | `callable` | The function to execute. | *required* |
| `retry` | `int` | Number of retry attempts. | `3` |
| `timeout` | `int` | Maximum time to wait for the request to complete. | `30` |
| `thread` | `bool` | Whether to run the request in a separate thread. | `True` |
| `verbose` | `bool` | Whether to log detailed messages. | `True` |
| `progress_total` | `int, optional` | Total size for progress tracking. | `None` |
| `stream_response` | `bool, optional` | Whether to stream the response. | `None` |
| `*args` | `Any` | Additional positional arguments for request_func. | *required* |
| `**kwargs` | `Any` | Additional keyword arguments for request_func. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `requests.Response | None` | The response object if thread=False, otherwise None. |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L243-L322"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def request_queue(
    self,
    request_func,
    retry: int = 3,
    timeout: int = 30,
    thread: bool = True,
    verbose: bool = True,
    progress_total: int | None = None,
    stream_response: bool | None = None,
    *args,
    **kwargs,
):
    """Execute request_func with retries, timeout handling, optional threading, and progress tracking.

    Args:
        request_func (callable): The function to execute.
        retry (int): Number of retry attempts.
        timeout (int): Maximum time to wait for the request to complete.
        thread (bool): Whether to run the request in a separate thread.
        verbose (bool): Whether to log detailed messages.
        progress_total (int, optional): Total size for progress tracking.
        stream_response (bool, optional): Whether to stream the response.
        *args (Any): Additional positional arguments for request_func.
        **kwargs (Any): Additional keyword arguments for request_func.

    Returns:
        (requests.Response | None): The response object if thread=False, otherwise None.
    """

    def retry_request():
        """Attempt to call request_func with retries, timeout, and optional threading."""
        t0 = time.time()  # Record the start time for the timeout
        response = None
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                LOGGER.warning(f"{PREFIX}Timeout for request reached. {HELP_MSG}")
                break  # Timeout reached, exit loop

            response = request_func(*args, **kwargs)
            if response is None:
                LOGGER.warning(f"{PREFIX}Received no response from the request. {HELP_MSG}")
                time.sleep(2**i)  # Exponential backoff before retrying
                continue  # Skip further processing and retry

            if progress_total:
                self._show_upload_progress(progress_total, response)
            elif stream_response:
                self._iterate_content(response)

            if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
                # if request related to metrics upload
                if kwargs.get("metrics"):
                    self.metrics_upload_failed_queue = {}
                return response  # Success, no need to retry

            if i == 0:
                # Initial attempt, check status code and provide messages
                message = self._get_failure_message(response, retry, timeout)

                if verbose:
                    LOGGER.warning(f"{PREFIX}{message} {HELP_MSG} ({response.status_code})")

            if not self._should_retry(response.status_code):
                LOGGER.warning(f"{PREFIX}Request failed. {HELP_MSG} ({response.status_code}")
                break  # Not an error that should be retried, exit loop

            time.sleep(2**i)  # Exponential backoff for retries

        # if request related to metrics upload and exceed retries
        if response is None and kwargs.get("metrics"):
            self.metrics_upload_failed_queue.update(kwargs.get("metrics"))

        return response

    if thread:
        # Start a new thread to run the retry_request function
        threading.Thread(target=retry_request, daemon=True).start()
    else:
        # If running in the main thread, call retry_request directly
        return retry_request()
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession.upload_metrics` {#ultralytics.hub.session.HUBTrainingSession.upload\_metrics}

```python
def upload_metrics(self)
```

Upload model metrics to Ultralytics HUB.

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L359-L361"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def upload_metrics(self):
    """Upload model metrics to Ultralytics HUB."""
    return self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)
```
</details>

<br>

### Method `ultralytics.hub.session.HUBTrainingSession.upload_model` {#ultralytics.hub.session.HUBTrainingSession.upload\_model}

```python
def upload_model(self, epoch: int, weights: str, is_best: bool = False, map: float = 0.0, final: bool = False) -> None
```

Upload a model checkpoint to Ultralytics HUB.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `epoch` | `int` | The current training epoch. | *required* |
| `weights` | `str` | Path to the model weights file. | *required* |
| `is_best` | `bool` | Indicates if the current model is the best one so far. | `False` |
| `map` | `float` | Mean average precision of the model. | `0.0` |
| `final` | `bool` | Indicates if the model is the final model after training. | `False` |

<details>
<summary>Source code in <code>ultralytics/hub/session.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/session.py#L363-L407"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def upload_model(
    self,
    epoch: int,
    weights: str,
    is_best: bool = False,
    map: float = 0.0,
    final: bool = False,
) -> None:
    """Upload a model checkpoint to Ultralytics HUB.

    Args:
        epoch (int): The current training epoch.
        weights (str): Path to the model weights file.
        is_best (bool): Indicates if the current model is the best one so far.
        map (float): Mean average precision of the model.
        final (bool): Indicates if the model is the final model after training.
    """
    weights = Path(weights)
    if not weights.is_file():
        last = weights.with_name(f"last{weights.suffix}")
        if final and last.is_file():
            LOGGER.warning(
                f"{PREFIX} Model 'best.pt' not found, copying 'last.pt' to 'best.pt' and uploading. "
                "This often happens when resuming training in transient environments like Google Colab. "
                "For more reliable training, consider using Ultralytics HUB Cloud. "
                "Learn more at https://docs.ultralytics.com/hub/cloud-training."
            )
            shutil.copy(last, weights)  # copy last.pt to best.pt
        else:
            LOGGER.warning(f"{PREFIX} Model upload issue. Missing model {weights}.")
            return

    self.request_queue(
        self.model.upload_model,
        epoch=epoch,
        weights=str(weights),
        is_best=is_best,
        map=map,
        final=final,
        retry=10,
        timeout=3600,
        thread=not final,
        progress_total=weights.stat().st_size if final else None,  # only show progress if final
        stream_response=True,
    )
```
</details>

<br><br>
