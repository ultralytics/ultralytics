# Ultralytics YOLO 🚀, GPL-3.0 license

import json
from time import time

from ultralytics.hub.utils import PREFIX, traces
from ultralytics.yolo.utils import LOGGER


def on_pretrain_routine_end(trainer):
    session = getattr(trainer, 'hub_session', None)
    if session:
        # Start timer for upload rate limit
        LOGGER.info(f"{PREFIX}View model at https://hub.ultralytics.com/models/{session.model_id} 🚀")
        session.t = {'metrics': time(), 'ckpt': time()}  # start timer on self.rate_limit


def on_fit_epoch_end(trainer):
    session = getattr(trainer, 'hub_session', None)
    if session:
        session.metrics_queue[trainer.epoch] = json.dumps(trainer.metrics)  # json string
        if time() - session.t['metrics'] > session.rate_limits['metrics']:
            session.upload_metrics()
            session.t['metrics'] = time()  # reset timer
            session.metrics_queue = {}  # reset queue


def on_model_save(trainer):
    session = getattr(trainer, 'hub_session', None)
    if session:
        # Upload checkpoints with rate limiting
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.t['ckpt'] > session.rate_limits['ckpt']:
            LOGGER.info(f"{PREFIX}Uploading checkpoint {session.model_id}")
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.t['ckpt'] = time()  # reset timer


def on_train_end(trainer):
    session = getattr(trainer, 'hub_session', None)
    if session:
        # Upload final model and metrics with exponential standoff
        LOGGER.info(f"{PREFIX}Training completed successfully ✅\n"
                    f"{PREFIX}Uploading final {session.model_id}")
        session.upload_model(trainer.epoch, trainer.best, map=trainer.metrics['metrics/mAP50-95(B)'], final=True)
        session.shutdown()  # stop heartbeats
        LOGGER.info(f"{PREFIX}View model at https://hub.ultralytics.com/models/{session.model_id} 🚀")


def on_train_start(trainer):
    traces(trainer.args, traces_sample_rate=1.0)


def on_val_start(validator):
    traces(validator.args, traces_sample_rate=1.0)


def on_predict_start(predictor):
    traces(predictor.args, traces_sample_rate=1.0)


def on_export_start(exporter):
    traces(exporter.args, traces_sample_rate=1.0)


callbacks = {
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_model_save": on_model_save,
    "on_train_end": on_train_end,
    "on_train_start": on_train_start,
    "on_val_start": on_val_start,
    "on_predict_start": on_predict_start,
    "on_export_start": on_export_start}
