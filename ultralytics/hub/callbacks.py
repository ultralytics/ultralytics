import json
import signal
import sys
from pathlib import Path
from time import sleep, time

import requests

from ultralytics.hub.config import HUB_API_ROOT
from ultralytics.hub.session import HubTrainingSession
from ultralytics.hub.utils import PREFIX
from ultralytics.yolo.utils import LOGGER, emojis


def signal_handler(signum, frame):
    """ Confirm exit """
    global hub_logger
    print(f'Signal received. {signum} {frame}')
    if isinstance(hub_logger, HUBLogger):
        hub_logger.alive = False
        del hub_logger
    sys.exit(signum)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class Trainer:
    # HUB Trainer class

    def __init__(self, model_id, auth):
        self.auth = auth
        self.model = self._get_model(model_id)
        if self.model is not None:
            self._connect_callbacks()

    def _get_model_by_id(self):
        # return a specific model
        return

    def _get_next_model(self):
        # return next model in queue
        return

    def _get_model(self, model_id):
        # Returns model from database by id
        api_url = f"{HUB_API_ROOT}/v1/models/{model_id}"
        headers = self.auth.get_auth_header()

        try:
            r = smart_request(api_url, method="get", headers=headers, thread=False, code=0)
            data = r.json().get("data", None)
            if not data: return
            assert data['data'], 'ERROR: Dataset may still be processing. Please wait a minute and try again.'  # RF fix
            self.model_id = data["id"]
            return data
        except requests.exceptions.ConnectionError as e:
            raise Exception('ERROR: The HUB server is not online. Please try again later.') from e

    def _connect_callbacks(self):
        global hub_logger
        callback_handler = YOLOv5.new_callback_handler()
        hub_logger = HUBLogger(self.model_id, self.auth)
        callback_handler.register_action("on_pretrain_routine_start", "HUB", hub_logger.on_pretrain_routine_start)
        callback_handler.register_action("on_pretrain_routine_end", "HUB", hub_logger.on_pretrain_routine_end)
        callback_handler.register_action("on_fit_epoch_end", "HUB", hub_logger.on_fit_epoch_end)
        callback_handler.register_action("on_model_save", "HUB", hub_logger.on_model_save)
        callback_handler.register_action("on_train_end", "HUB", hub_logger.on_train_end)
        self.callbacks = callback_handler

    def start(self):
        # Checks
        if not check_dataset_disk_space(self.model['data']):
            return

        # Train
        YOLOv5.train(self.callbacks, **self.model)


def on_pretrain_routine_end(trainer):
    # Start timer for upload rate limit
    session = HubTrainingSession()
    LOGGER.info(emojis(f"{PREFIX}View model at https://hub.ultralytics.com/models/{trainer.hub_session.model_id} 🚀"))
    trainer.hub_session.t = {'metrics': time(), 'ckpt': time()}  # start timer on self.rate_limit


def on_fit_epoch_end(trainer):
    # Upload metrics after val end
    session = trainer.hub_session
    session.metrics_queue[trainer.epoch] = json.dumps(trainer.metrics)  # json string
    if time() - session.t['metrics'] > session.rate_limits['metrics']:
        session._upload_metrics()
        session.t['metrics'] = time()  # reset timer
        session.metrics_queue = {}  # reset queue


def on_model_save(trainer):
    # Upload checkpoints with rate limiting
    session = trainer.hub_session
    is_best = trainer.best_fitness == trainer.fitness
    if time() - session.t['ckpt'] > session.rate_limits['ckpt']:
        LOGGER.info(f"{PREFIX}Uploading checkpoint {session.model_id}")
        session._upload_model(trainer.epoch, trainer.last, is_best)
        session.t['ckpt'] = time()  # reset timer


def on_train_end(trainer):
    # Upload final model and metrics with exponential standoff
    session = trainer.hub_session
    LOGGER.info(emojis(f"{PREFIX}Training completed successfully ✅"))
    LOGGER.info(f"{PREFIX}Uploading final {session.model_id}")
    session._upload_model(trainer.epoch, trainer.best, map=trainer.metrics['metrics/mAP50(B)'],
                          final=True)  # results[3] is mAP0.5:0.95
    session.alive = False  # stop heartbeats
    LOGGER.info(emojis(f"{PREFIX}View model at https://hub.ultralytics.com/models/{session.model_id} 🚀"))


callbacks = {
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_model_save": on_model_save,
    "on_train_end": on_train_end}
