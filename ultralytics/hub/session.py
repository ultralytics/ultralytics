# Ultralytics YOLO 🚀, AGPL-3.0 license

import threading
import time
from http import HTTPStatus
from pathlib import Path

import requests

from ultralytics.hub.utils import HELP_MSG, HUB_WEB_ROOT, PREFIX, TQDM
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, __version__, checks, emojis
from ultralytics.utils.errors import HUBModelError

AGENT_NAME = f"python-{__version__}-colab" if IS_COLAB else f"python-{__version__}-local"


class HUBTrainingSession:
    """
    HUB training session for Ultralytics HUB YOLO models. Handles model initialization, heartbeats, and checkpointing.

    Attributes:
        model_id (str): Identifier for the YOLO model being trained.
        model_url (str): URL for the model in Ultralytics HUB.
        rate_limits (dict): Rate limits for different API calls (in seconds).
        timers (dict): Timers for rate limiting.
        metrics_queue (dict): Queue for the model's metrics.
        model (dict): Model data fetched from Ultralytics HUB.
    """

    def __init__(self, identifier):
        """
        Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session.
                It can be a URL string or a model key with specific format.

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

        # Parse input
        api_key, model_id, self.filename = self._parse_identifier(identifier)

        # Get credentials
        active_key = api_key or SETTINGS.get("api_key")
        credentials = {"api_key": active_key} if active_key else None  # set credentials

        # Initialize client
        self.client = HUBClient(credentials)

        # Load models if authenticated
        if self.client.authenticated:
            if model_id:
                self.load_model(model_id)  # load existing model
            else:
                self.model = self.client.model()  # load empty model

    @classmethod
    def create_session(cls, identifier, args=None):
        """Class method to create an authenticated HUBTrainingSession or return None."""
        try:
            session = cls(identifier)
            if not session.client.authenticated:
                if identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
                    LOGGER.warning(f"{PREFIX}WARNING ⚠️ Login to Ultralytics HUB with 'yolo hub login API_KEY'.")
                    exit()
                return None
            if args and not identifier.startswith(f"{HUB_WEB_ROOT}/models/"):  # not a HUB model URL
                session.create_model(args)
                assert session.model.id, "HUB model not loaded correctly"
            return session
        # PermissionError and ModuleNotFoundError indicate hub-sdk not installed
        except (PermissionError, ModuleNotFoundError, AssertionError):
            return None

    def load_model(self, model_id):
        """Loads an existing model from Ultralytics HUB using the provided model identifier."""
        self.model = self.client.model(model_id)
        if not self.model.data:  # then model does not exist
            raise ValueError(emojis("❌ The specified HUB model does not exist"))  # TODO: improve error handling

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"

        self._set_train_args()

        # Start heartbeats for HUB to monitor agent
        self.model.start_heartbeat(self.rate_limits["heartbeat"])
        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")

    def create_model(self, model_args):
        """Initializes a HUB training session with the specified model identifier."""
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

        # Model could not be created
        # TODO: improve error handling
        if not self.model.id:
            return None

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"

        # Start heartbeats for HUB to monitor agent
        self.model.start_heartbeat(self.rate_limits["heartbeat"])

        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")

    @staticmethod
    def _parse_identifier(identifier):
        """
        Parses the given identifier to determine the type of identifier and extract relevant components.

        The method supports different identifier formats:
            - A HUB URL, which starts with HUB_WEB_ROOT followed by '/models/'
            - An identifier containing an API key and a model ID separated by an underscore
            - An identifier that is solely a model ID of a fixed length
            - A local filename that ends with '.pt' or '.yaml'

        Args:
            identifier (str): The identifier string to be parsed.

        Returns:
            (tuple): A tuple containing the API key, model ID, and filename as applicable.

        Raises:
            HUBModelError: If the identifier format is not recognized.
        """

        # Initialize variables
        api_key, model_id, filename = None, None, None

        # Check if identifier is a HUB URL
        if identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
            # Extract the model_id after the HUB_WEB_ROOT URL
            model_id = identifier.split(f"{HUB_WEB_ROOT}/models/")[-1]
        else:
            # Split the identifier based on underscores only if it's not a HUB URL
            parts = identifier.split("_")

            # Check if identifier is in the format of API key and model ID
            if len(parts) == 2 and len(parts[0]) == 42 and len(parts[1]) == 20:
                api_key, model_id = parts
            # Check if identifier is a single model ID
            elif len(parts) == 1 and len(parts[0]) == 20:
                model_id = parts[0]
            # Check if identifier is a local filename
            elif identifier.endswith(".pt") or identifier.endswith(".yaml"):
                filename = identifier
            else:
                raise HUBModelError(
                    f"model='{identifier}' could not be parsed. Check format is correct. "
                    f"Supported formats are Ultralytics HUB URL, apiKey_modelId, modelId, local pt or yaml file."
                )

        return api_key, model_id, filename

    def _set_train_args(self):
        """
        Initializes training arguments and creates a model entry on the Ultralytics HUB.

        This method sets up training arguments based on the model's state and updates them with any additional
        arguments provided. It handles different states of the model, such as whether it's resumable, pretrained,
        or requires specific file setup.

        Raises:
            ValueError: If the model is already trained, if required dataset information is missing, or if there are
                issues with the provided training arguments.
        """
        if self.model.is_trained():
            raise ValueError(emojis(f"Model is already trained and uploaded to {self.model_url} 🚀"))

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

    def request_queue(
        self,
        request_func,
        retry=3,
        timeout=30,
        thread=True,
        verbose=True,
        progress_total=None,
        stream_response=None,
        *args,
        **kwargs,
    ):
        """Attempts to execute `request_func` with retries, timeout handling, optional threading, and progress."""

        def retry_request():
            """Attempts to call `request_func` with retries, timeout, and optional threading."""
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
                self.metrics_upload_failed_queue.update(kwargs.get("metrics", None))

            return response

        if thread:
            # Start a new thread to run the retry_request function
            threading.Thread(target=retry_request, daemon=True).start()
        else:
            # If running in the main thread, call retry_request directly
            return retry_request()

    @staticmethod
    def _should_retry(status_code):
        """Determines if a request should be retried based on the HTTP status code."""
        retry_codes = {
            HTTPStatus.REQUEST_TIMEOUT,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.GATEWAY_TIMEOUT,
        }
        return status_code in retry_codes

    def _get_failure_message(self, response: requests.Response, retry: int, timeout: int):
        """
        Generate a retry message based on the response status code.

        Args:
            response: The HTTP response object.
            retry: The number of retry attempts allowed.
            timeout: The maximum timeout duration.

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

    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB."""
        return self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)

    def upload_model(
        self,
        epoch: int,
        weights: str,
        is_best: bool = False,
        map: float = 0.0,
        final: bool = False,
    ) -> None:
        """
        Upload a model checkpoint to Ultralytics HUB.

        Args:
            epoch (int): The current training epoch.
            weights (str): Path to the model weights file.
            is_best (bool): Indicates if the current model is the best one so far.
            map (float): Mean average precision of the model.
            final (bool): Indicates if the model is the final model after training.
        """
        if Path(weights).is_file():
            progress_total = Path(weights).stat().st_size if final else None  # Only show progress if final
            self.request_queue(
                self.model.upload_model,
                epoch=epoch,
                weights=weights,
                is_best=is_best,
                map=map,
                final=final,
                retry=10,
                timeout=3600,
                thread=not final,
                progress_total=progress_total,
                stream_response=True,
            )
        else:
            LOGGER.warning(f"{PREFIX}WARNING ⚠️ Model upload issue. Missing model {weights}.")

    @staticmethod
    def _show_upload_progress(content_length: int, response: requests.Response) -> None:
        """
        Display a progress bar to track the upload progress of a file download.

        Args:
            content_length (int): The total size of the content to be downloaded in bytes.
            response (requests.Response): The response object from the file download request.

        Returns:
            None
        """
        with TQDM(total=content_length, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))

    @staticmethod
    def _iterate_content(response: requests.Response) -> None:
        """
        Process the streamed HTTP response data.

        Args:
            response (requests.Response): The response object from the file download request.

        Returns:
            None
        """
        for _ in response.iter_content(chunk_size=1024):
            pass  # Do nothing with data chunks
