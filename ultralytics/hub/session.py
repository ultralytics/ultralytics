# Ultralytics YOLO 🚀, AGPL-3.0 license

import signal
import sys
from pathlib import Path
import time
import threading

import requests
from ultralytics_hub_sdk import HUBClient, HUB_API_ROOT, HUB_WEB_ROOT
from ultralytics.hub.utils import PREFIX, TryExcept
from ultralytics.utils import SETTINGS, LOGGER, __version__, checks, emojis, is_colab, threaded
from ultralytics.utils.errors import HUBModelError

AGENT_NAME = f'python-{__version__}-colab' if is_colab() else f'python-{__version__}-local'


class HUBTrainingSession:
    """
    HUB training session for Ultralytics HUB YOLO models. Handles model initialization, heartbeats, and checkpointing.

    Args:
        url (str): Model identifier used to initialize the HUB training session.

    Attributes:
        agent_id (str): Identifier for the instance communicating with the server.
        model_id (str): Identifier for the YOLOv5 model being trained.
        model_url (str): URL for the model in Ultralytics HUB.
        api_url (str): API URL for the model in Ultralytics HUB.
        auth_header (dict): Authentication header for the Ultralytics HUB API requests.
        rate_limits (dict): Rate limits for different API calls (in seconds).
        timers (dict): Timers for rate limiting.
        metrics_queue (dict): Queue for the model's metrics.
        model (dict): Model data fetched from Ultralytics HUB.
        alive (bool): Indicates if the heartbeat loop is active.
    """

    def __init__(self, url):
        """
        Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            url (str): Model identifier used to initialize the HUB training session.
                         It can be a URL string or a model key with specific format.

        Raises:
            ValueError: If the provided model identifier is invalid.
            ConnectionError: If connecting with global API key is not supported.
        """

        # Parse input
        api_key, model_id = self.parse_model_url(url)

        # Get credentials
        active_key = api_key or SETTINGS.get("api_key")
        credentials = {"api_key": active_key} if active_key else None  # Set credentials

        # Initialize client
        client = HUBClient(credentials)

        # Initialize model
        self.model = client.model(model_id)
        self.agent_id = None  # identifies which instance is communicating with server
        self.model_url = f'{HUB_WEB_ROOT}/models/{self.model.id}'
        self.api_url = f'{HUB_API_ROOT}/v1/models/{self.model.id}'

        self.set_train_args()

        # Start heartbeats for HUB to monitor agent
        self.model.start_heartbeat()

        self.rate_limits = {'metrics': 3.0, 'ckpt': 900.0, 'heartbeat': 300.0}  # rate limits (seconds)
        self.timers = {}  # rate limit timers (seconds)
        self.metrics_queue = {}  # metrics queue

        LOGGER.info(f'{PREFIX}View model at {self.model_url} 🚀')

    def parse_model_url(self, url):
        # Split the URL based on the last occurrence of '/models/'
        parts = url.split(f'{HUB_WEB_ROOT}/models/')[-1].split('_')

        # Check if the parts have the expected lengths
        if len(parts) == 2 and len(parts[0]) == 42 and len(parts[1]) == 20:
            # Is old format 'key*42_id*20'
            api_key, model_id = parts
        elif len(parts) == 1 and len(parts[0]) == 20:
            # Is new format 'id*20'
            api_key, model_id = '', parts[0]
        else:
            raise HUBModelError(f"model='{url}' not found. Check format is correct, i.e. "
                                f"model='{HUB_WEB_ROOT}/models/MODEL_ID' and try again.")

        return api_key, model_id

    def set_train_args(self, **kwargs):
        if self.model.is_trained():
            # Model is already trained
            raise ValueError(emojis(f'Model is already trained and uploaded to {self.model_url} 🚀'))

        if self.model.is_resumable():
            # Model has saved weights
            self.train_args = {'data': self.model.get_dataset_url(), 'resume': True}
            self.model_file = self.model.get_weights_url('last')
        else:
            # Model has no saved weights
            def get_train_args(config):
                return {
                    'batch': config['batchSize'],
                    'epochs': config['epochs'],
                    'imgsz': config['imageSize'],
                    'patience': config['patience'],
                    'device': config['device'],
                    'cache': config['cache'],
                    'data': self.model.get_dataset_url()
                }
            self.train_args = get_train_args(self.model.data.get("config"))
            # Set the model file as either a *.pt or *.yaml file
            self.model_file = self.model.get_weights_url('parent') if self.model.is_pretrained() else self.model.get_architecture()

        if not self.train_args.get('data'):
            raise ValueError('Dataset may still be processing. Please wait a minute and try again.')  # RF fix

        self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False) # YOLOv5->YOLOv5u

    def request_queue(self, request_func, retry=3, timeout=30, thread=True, verbose=True, progress=False, *args, **kwargs,):
        retry_codes = (408, 500)  # retry only these codes

        @TryExcept(verbose=verbose)
        def func(func_method, **func_kwargs):
            r = None  # response
            t0 = time.time()  # initial time for timer
            for i in range(retry + 1):
                if (time.time() - t0) > timeout:
                    break
                r = request_func(*args, **kwargs)
                if r.status_code < 300:  # return codes in the 2xx range are generally considered "good" or "successful"
                    break
                try:
                    m = r.json().get('message', 'No JSON message.')
                except AttributeError:
                    m = 'Unable to read JSON.'
                if i == 0:
                    if r.status_code in retry_codes:
                        m += f' Retrying {retry}x for {timeout}s.' if retry else ''
                    elif r.status_code == 429:  # rate limit
                        h = r.headers  # response headers
                        m = f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). " \
                            f"Please retry after {h['Retry-After']}s."
                    if verbose:
                        LOGGER.warning(f'{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})')
                    if r.status_code not in retry_codes:
                        return r
                time.sleep(2 ** i)  # exponential standoff
            return r

        if thread:
            threading.Thread(target=func, args=[request_func], kwargs=kwargs, daemon=True).start()
        else:
            return func(request_func, **kwargs)


    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB."""
        self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)

    def upload_model(self, epoch, weights, is_best=False, mAP=0.0, final=False):
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
            self.model.upload_model(epoch=epoch, weights=weights, is_best=is_best, mAP=mAP, final=final, retry=10, timeout=3600, thread=not final, progress=True)
        else:
            LOGGER.warning(f'{PREFIX}WARNING ⚠️ Model upload issue. Missing model {weights}.')

