# Ultralytics YOLO 🚀, AGPL-3.0 license
import signal
import sys
from pathlib import Path
from time import sleep

import requests

from ultralytics.hub.utils import HUB_API_ROOT, PREFIX, smart_request
from ultralytics.yolo.utils import LOGGER, __version__, checks, emojis, is_colab, threaded
from ultralytics.yolo.utils.errors import HUBModelError

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
        auth_header (Dict): Authentication header for the Ultralytics HUB API requests.
        rate_limits (Dict): Rate limits for different API calls (in seconds).
        timers (Dict): Timers for rate limiting.
        metrics_queue (Dict): Queue for the model's metrics.
        model (Dict): Model data fetched from Ultralytics HUB.
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

        from ultralytics.hub.auth import Auth

        # Parse input
        if url.startswith('https://hub.ultralytics.com/models/'):
            url = url.split('https://hub.ultralytics.com/models/')[-1]
        if [len(x) for x in url.split('_')] == [42, 20]:
            key, model_id = url.split('_')
        elif len(url) == 20:
            key, model_id = '', url
        else:
            raise HUBModelError(f"model='{url}' not found. Check format is correct, i.e. "
                                f"model='https://hub.ultralytics.com/models/MODEL_ID' and try again.")

        # Authorize
        auth = Auth(key)
        self.agent_id = None  # identifies which instance is communicating with server
        self.model_id = model_id
        self.model_url = f'https://hub.ultralytics.com/models/{model_id}'
        self.api_url = f'{HUB_API_ROOT}/v1/models/{model_id}'
        self.auth_header = auth.get_auth_header()
        self.rate_limits = {'metrics': 3.0, 'ckpt': 900.0, 'heartbeat': 300.0}  # rate limits (seconds)
        self.timers = {}  # rate limit timers (seconds)
        self.metrics_queue = {}  # metrics queue
        self.model = self._get_model()
        self.alive = True
        self._start_heartbeat()  # start heartbeats
        self._register_signal_handlers()
        LOGGER.info(f'{PREFIX}View model at {self.model_url} 🚀')

    def _register_signal_handlers(self):
        """Register signal handlers for SIGTERM and SIGINT signals to gracefully handle termination."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """
        Handle kill signals and prevent heartbeats from being sent on Colab after termination.
        This method does not use frame, it is included as it is passed by signal.
        """
        if self.alive is True:
            LOGGER.info(f'{PREFIX}Kill signal received! ❌')
            self._stop_heartbeat()
            sys.exit(signum)

    def _stop_heartbeat(self):
        """Terminate the heartbeat loop."""
        self.alive = False

    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB."""
        payload = {'metrics': self.metrics_queue.copy(), 'type': 'metrics'}
        smart_request('post', self.api_url, json=payload, headers=self.auth_header, code=2)

    def _get_model(self):
        """Fetch and return model data from Ultralytics HUB."""
        api_url = f'{HUB_API_ROOT}/v1/models/{self.model_id}'

        try:
            response = smart_request('get', api_url, headers=self.auth_header, thread=False, code=0)
            data = response.json().get('data', None)

            if data.get('status', None) == 'trained':
                raise ValueError(emojis(f'Model is already trained and uploaded to {self.model_url} 🚀'))

            if not data.get('data', None):
                raise ValueError('Dataset may still be processing. Please wait a minute and try again.')  # RF fix
            self.model_id = data['id']

            if data['status'] == 'new':  # new model to start training
                self.train_args = {
                    # TODO: deprecate 'batch_size' key for 'batch' in 3Q23
                    'batch': data['batch' if ('batch' in data) else 'batch_size'],
                    'epochs': data['epochs'],
                    'imgsz': data['imgsz'],
                    'patience': data['patience'],
                    'device': data['device'],
                    'cache': data['cache'],
                    'data': data['data']}
                self.model_file = data.get('cfg') or data.get('weights')  # cfg for pretrained=False
                self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)  # YOLOv5->YOLOv5u
            elif data['status'] == 'training':  # existing model to resume training
                self.train_args = {'data': data['data'], 'resume': True}
                self.model_file = data['resume']

            return data
        except requests.exceptions.ConnectionError as e:
            raise ConnectionRefusedError('ERROR: The HUB server is not online. Please try again later.') from e
        except Exception:
            raise

    def upload_model(self, epoch, weights, is_best=False, map=0.0, final=False):
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
            with open(weights, 'rb') as f:
                file = f.read()
        else:
            LOGGER.warning(f'{PREFIX}WARNING ⚠️ Model upload issue. Missing model {weights}.')
            file = None
        url = f'{self.api_url}/upload'
        # url = 'http://httpbin.org/post'  # for debug
        data = {'epoch': epoch}
        if final:
            data.update({'type': 'final', 'map': map})
            smart_request('post',
                          url,
                          data=data,
                          files={'best.pt': file},
                          headers=self.auth_header,
                          retry=10,
                          timeout=3600,
                          thread=False,
                          progress=True,
                          code=4)
        else:
            data.update({'type': 'epoch', 'isBest': bool(is_best)})
            smart_request('post', url, data=data, files={'last.pt': file}, headers=self.auth_header, code=3)

    @threaded
    def _start_heartbeat(self):
        """Begin a threaded heartbeat loop to report the agent's status to Ultralytics HUB."""
        while self.alive:
            r = smart_request('post',
                              f'{HUB_API_ROOT}/v1/agent/heartbeat/models/{self.model_id}',
                              json={
                                  'agent': AGENT_NAME,
                                  'agentId': self.agent_id},
                              headers=self.auth_header,
                              retry=0,
                              code=5,
                              thread=False)  # already in a thread
            self.agent_id = r.json().get('data', {}).get('agentId', None)
            sleep(self.rate_limits['heartbeat'])
