# Ultralytics YOLO 🚀, GPL-3.0 license
import signal
import sys
from pathlib import Path
from time import sleep

import requests

from ultralytics.hub.utils import HUB_API_ROOT, check_dataset_disk_space, smart_request
from ultralytics.yolo.utils import LOGGER, PREFIX, __version__, checks, emojis, is_colab, threaded

AGENT_NAME = f'python-{__version__}-colab' if is_colab() else f'python-{__version__}-local'


class HUBTrainingSession:

    def __init__(self, model_id, auth):
        self.agent_id = None  # identifies which instance is communicating with server
        self.model_id = model_id
        self.api_url = f'{HUB_API_ROOT}/v1/models/{model_id}'
        self.auth_header = auth.get_auth_header()
        self.rate_limits = {'metrics': 3.0, 'ckpt': 900.0, 'heartbeat': 300.0}  # rate limits (seconds)
        self.timers = {}  # rate limit timers (seconds)
        self.metrics_queue = {}  # metrics queue
        self.model = self._get_model()
        self.alive = True
        self._start_heartbeat()  # start heartbeats
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """
        Prevent heartbeats from being sent on Colab after kill.
        This method does not use frame, it is included as it is
        passed by signal.
        """
        if self.alive is True:
            LOGGER.info(f'{PREFIX}Kill signal received! ❌')
            self._stop_heartbeat()
            sys.exit(signum)

    def _stop_heartbeat(self):
        """End the heartbeat loop"""
        self.alive = False

    def upload_metrics(self):
        payload = {'metrics': self.metrics_queue.copy(), 'type': 'metrics'}
        smart_request('post', self.api_url, json=payload, headers=self.auth_header, code=2)

    def _get_model(self):
        # Returns model from database by id
        api_url = f'{HUB_API_ROOT}/v1/models/{self.model_id}'

        try:
            response = smart_request('get', api_url, headers=self.auth_header, thread=False, code=0)
            data = response.json().get('data', None)

            if data.get('status', None) == 'trained':
                raise ValueError(
                    emojis(f'Model is already trained and uploaded to '
                           f'https://hub.ultralytics.com/models/{self.model_id} 🚀'))

            if not data.get('data', None):
                raise ValueError('Dataset may still be processing. Please wait a minute and try again.')  # RF fix
            self.model_id = data['id']

            # TODO: restore when server keys when dataset URL and GPU train is working

            self.train_args = {
                'batch': data['batch_size'],
                'epochs': data['epochs'],
                'imgsz': data['imgsz'],
                'patience': data['patience'],
                'device': data['device'],
                'cache': data['cache'],
                'data': data['data']}

            self.model_file = data.get('cfg', data['weights'])
            self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)  # YOLOv5->YOLOv5u

            return data
        except requests.exceptions.ConnectionError as e:
            raise ConnectionRefusedError('ERROR: The HUB server is not online. Please try again later.') from e
        except Exception:
            raise

    def check_disk_space(self):
        if not check_dataset_disk_space(self.model['data']):
            raise MemoryError('Not enough disk space')

    def upload_model(self, epoch, weights, is_best=False, map=0.0, final=False):
        # Upload a model to HUB
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
