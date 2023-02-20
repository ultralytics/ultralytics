# Ultralytics YOLO 🚀, GPL-3.0 license
import json
import signal
import sys
from pathlib import Path
from time import sleep, time

import requests

from ultralytics.hub.utils import HUB_API_ROOT, check_dataset_disk_space, smart_request
from ultralytics.yolo.utils import LOGGER, PREFIX, __version__, emojis, is_colab, threaded
from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

AGENT_NAME = f'python-{__version__}-colab' if is_colab() else f'python-{__version__}-local'
session = None


class HubTrainingSession:

    def __init__(self, model_id, auth):
        self.agent_id = None  # identifies which instance is communicating with server
        self.model_id = model_id
        self.api_url = f'{HUB_API_ROOT}/v1/models/{model_id}'
        self.auth_header = auth.get_auth_header()
        self._rate_limits = {'metrics': 3.0, 'ckpt': 900.0, 'heartbeat': 300.0}  # rate limits (seconds)
        self._timers = {}  # rate limit timers (seconds)
        self._metrics_queue = {}  # metrics queue
        self.model = self._get_model()
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
        payload = {'metrics': self._metrics_queue.copy(), 'type': 'metrics'}
        smart_request(f'{self.api_url}', json=payload, headers=self.auth_header, code=2)

    def upload_model(self, epoch, weights, is_best=False, map=0.0, final=False):
        # Upload a model to HUB
        file = None
        if Path(weights).is_file():
            with open(weights, 'rb') as f:
                file = f.read()
        if final:
            smart_request(
                f'{self.api_url}/upload',
                data={
                    'epoch': epoch,
                    'type': 'final',
                    'map': map},
                files={'best.pt': file},
                headers=self.auth_header,
                retry=10,
                timeout=3600,
                code=4,
            )
        else:
            smart_request(
                f'{self.api_url}/upload',
                data={
                    'epoch': epoch,
                    'type': 'epoch',
                    'isBest': bool(is_best)},
                headers=self.auth_header,
                files={'last.pt': file},
                code=3,
            )

    def _get_model(self):
        # Returns model from database by id
        api_url = f'{HUB_API_ROOT}/v1/models/{self.model_id}'
        headers = self.auth_header

        try:
            response = smart_request(api_url, method='get', headers=headers, thread=False, code=0)
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

            self.input_file = data.get('cfg', data['weights'])

            # hack for yolov5 cfg adds u
            if 'cfg' in data and 'yolov5' in data['cfg']:
                self.input_file = data['cfg'].replace('.yaml', 'u.yaml')

            return data
        except requests.exceptions.ConnectionError as e:
            raise ConnectionRefusedError('ERROR: The HUB server is not online. Please try again later.') from e
        except Exception:
            raise

    def check_disk_space(self):
        if not check_dataset_disk_space(self.model['data']):
            raise MemoryError('Not enough disk space')

    def register_callbacks(self, trainer):
        trainer.add_callback('on_pretrain_routine_end', self.on_pretrain_routine_end)
        trainer.add_callback('on_fit_epoch_end', self.on_fit_epoch_end)
        trainer.add_callback('on_model_save', self.on_model_save)
        trainer.add_callback('on_train_end', self.on_train_end)

    def on_pretrain_routine_end(self, trainer):
        """
        Start timer for upload rate limit.
        This method does not use trainer. It is passed to all callbacks by default.
        """
        # Start timer for upload rate limit
        LOGGER.info(f'{PREFIX}View model at https://hub.ultralytics.com/models/{self.model_id} 🚀')
        self._timers = {'metrics': time(), 'ckpt': time()}  # start timer on self.rate_limit

    def on_fit_epoch_end(self, trainer):
        # Upload metrics after val end
        all_plots = {**trainer.label_loss_items(trainer.tloss, prefix='train'), **trainer.metrics}

        if trainer.epoch == 0:
            model_info = {
                'model/parameters': get_num_params(trainer.model),
                'model/GFLOPs': round(get_flops(trainer.model), 3),
                'model/speed(ms)': round(trainer.validator.speed[1], 3)}
            all_plots = {**all_plots, **model_info}
        self._metrics_queue[trainer.epoch] = json.dumps(all_plots)
        if time() - self._timers['metrics'] > self._rate_limits['metrics']:
            self.upload_metrics()
            self._timers['metrics'] = time()  # reset timer
            self._metrics_queue = {}  # reset queue

    def on_model_save(self, trainer):
        # Upload checkpoints with rate limiting
        is_best = trainer.best_fitness == trainer.fitness
        if time() - self._timers['ckpt'] > self._rate_limits['ckpt']:
            LOGGER.info(f'{PREFIX}Uploading checkpoint {self.model_id}')
            self._upload_model(trainer.epoch, trainer.last, is_best)
            self._timers['ckpt'] = time()  # reset timer

    def on_train_end(self, trainer):
        # Upload final model and metrics with exponential standoff
        LOGGER.info(f'{PREFIX}Training completed successfully ✅')
        LOGGER.info(f'{PREFIX}Uploading final {self.model_id}')

        # hack for fetching mAP
        mAP = trainer.metrics.get('metrics/mAP50-95(B)', 0)
        self._upload_model(trainer.epoch, trainer.best, map=mAP, final=True)  # results[3] is mAP0.5:0.95
        self.alive = False  # stop heartbeats
        LOGGER.info(f'{PREFIX}View model at https://hub.ultralytics.com/models/{self.model_id} 🚀')

    def _upload_model(self, epoch, weights, is_best=False, map=0.0, final=False):
        # Upload a model to HUB
        file = None
        if Path(weights).is_file():
            with open(weights, 'rb') as f:
                file = f.read()
        file_param = {'best.pt' if final else 'last.pt': file}
        endpoint = f'{self.api_url}/upload'
        data = {'epoch': epoch}
        if final:
            data.update({'type': 'final', 'map': map})
        else:
            data.update({'type': 'epoch', 'isBest': bool(is_best)})

        smart_request(
            endpoint,
            data=data,
            files=file_param,
            headers=self.auth_header,
            retry=10 if final else None,
            timeout=3600 if final else None,
            code=4 if final else 3,
        )

    @threaded
    def _start_heartbeat(self):
        self.alive = True
        while self.alive:
            r = smart_request(
                f'{HUB_API_ROOT}/v1/agent/heartbeat/models/{self.model_id}',
                json={
                    'agent': AGENT_NAME,
                    'agentId': self.agent_id},
                headers=self.auth_header,
                retry=0,
                code=5,
                thread=False,
            )
            self.agent_id = r.json().get('data', {}).get('agentId', None)
            sleep(self._rate_limits['heartbeat'])
