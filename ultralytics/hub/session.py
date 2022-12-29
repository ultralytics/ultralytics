import json
import signal
import sys
from pathlib import Path
from time import sleep, time

import requests

from ultralytics.hub.config import HUB_API_ROOT
from ultralytics.hub.utils import PREFIX
from ultralytics.yolo.utils import LOGGER, emojis


class HubTrainingSession:

    def __init__(self, model_id, auth):
        self.agent_id = None  # identifies which instance is communicating with server
        self.model_id = model_id
        self.api_url = f'{HUB_API_ROOT}/v1/models/{model_id}'
        self.auth_header = auth.get_auth_header()
        self.rate_limits = {'metrics': 3.0, 'ckpt': 900.0, 'heartbeat': 300.0}  # rate limits (seconds)
        self.t = {}  # rate limit timers (seconds)
        self.metrics_queue = {}  # metrics queue
        self.keys = [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',  # metrics
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',  # val loss
            'x/lr0',
            'x/lr1',
            'x/lr2']  # metrics keys
        self.alive = True  # for heartbeats
        self._heartbeats()  # start heartbeats

    def __del__(self):
        # Class destructor
        self.alive = False

    # Internal functions ---
    def _upload_metrics(self):
        payload = {"metrics": self.metrics_queue.copy(), "type": "metrics"}
        smart_request(f'{self.api_url}', json=payload, headers=self.auth_header, code=2)

    def _upload_model(self, epoch, weights, is_best=False, map=0.0, final=False):
        # Upload a model to HUB
        file = None
        if Path(weights).is_file():
            with open(weights, "rb") as f:
                file = f.read()
        if final:
            smart_request(f'{self.api_url}/upload',
                            data={"epoch": epoch, "type": "final", "map": map},
                            files={"best.pt": file},
                            headers=self.auth_header,
                            retry=10,
                            timeout=3600,
                            code=4)
        else:
            smart_request(f'{self.api_url}/upload',
                            data={"epoch": epoch, "type": "epoch", "isBest": bool(is_best)},
                            headers=self.auth_header,
                            files={"last.pt": file},
                            code=3)

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
