# Ultralytics YOLO 🚀, GPL-3.0 license
import signal
from pathlib import Path
from time import sleep

import requests

from ultralytics import __version__
from ultralytics.hub.utils import HUB_API_ROOT, check_dataset_disk_space, smart_request
from ultralytics.yolo.utils import is_colab, threaded

AGENT_NAME = f'python-{__version__}-colab' if is_colab() else f'python-{__version__}-local'

session = None


class HubTrainingSession:

    def __init__(self, model_id, auth):
        self.agent_id = None  # identifies which instance is communicating with server
        self.model_id = model_id
        self.api_url = f'{HUB_API_ROOT}/v1/models/{model_id}'
        self.auth_header = auth.get_auth_header()
        self.rate_limits = {'metrics': 3.0, 'ckpt': 900.0, 'heartbeat': 300.0}  # rate limits (seconds)
        self.t = {}  # rate limit timers (seconds)
        self.metrics_queue = {}  # metrics queue
        self.alive = True  # for heartbeats
        self.model = self._get_model()
        self._heartbeats()  # start heartbeats
        signal.signal(signal.SIGTERM, self.shutdown)  # register the shutdown function to be called on exit
        signal.signal(signal.SIGINT, self.shutdown)

    def shutdown(self, *args):  # noqa
        self.alive = False  # stop heartbeats

    def upload_metrics(self):
        payload = {"metrics": self.metrics_queue.copy(), "type": "metrics"}
        smart_request(f'{self.api_url}', json=payload, headers=self.auth_header, code=2)

    def upload_model(self, epoch, weights, is_best=False, map=0.0, final=False):
        # Upload a model to HUB
        file = None
        if Path(weights).is_file():
            with open(weights, "rb") as f:
                file = f.read()
        if final:
            smart_request(f'{self.api_url}/upload',
                          data={
                              "epoch": epoch,
                              "type": "final",
                              "map": map},
                          files={"best.pt": file},
                          headers=self.auth_header,
                          retry=10,
                          timeout=3600,
                          code=4)
        else:
            smart_request(f'{self.api_url}/upload',
                          data={
                              "epoch": epoch,
                              "type": "epoch",
                              "isBest": bool(is_best)},
                          headers=self.auth_header,
                          files={"last.pt": file},
                          code=3)

    def _get_model(self):
        # Returns model from database by id
        api_url = f"{HUB_API_ROOT}/v1/models/{self.model_id}"
        headers = self.auth_header

        try:
            r = smart_request(api_url, method="get", headers=headers, thread=False, code=0)
            data = r.json().get("data", None)
            if not data:
                return
            assert data['data'], 'ERROR: Dataset may still be processing. Please wait a minute and try again.'  # RF fix
            self.model_id = data["id"]

            return data
        except requests.exceptions.ConnectionError as e:
            raise ConnectionRefusedError('ERROR: The HUB server is not online. Please try again later.') from e

    def check_disk_space(self):
        if not check_dataset_disk_space(self.model['data']):
            raise MemoryError("Not enough disk space")

    @threaded
    def _heartbeats(self):
        while self.alive:
            r = smart_request(f'{HUB_API_ROOT}/v1/agent/heartbeat/models/{self.model_id}',
                              json={
                                  "agent": AGENT_NAME,
                                  "agentId": self.agent_id},
                              headers=self.auth_header,
                              retry=0,
                              code=5,
                              thread=False)
            self.agent_id = r.json().get('data', {}).get('agentId', None)
            sleep(self.rate_limits['heartbeat'])
