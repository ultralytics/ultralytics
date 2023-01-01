import json
from time import time

import torch

from ultralytics.hub.utils import PREFIX
from ultralytics.yolo.utils import LOGGER, emojis


def on_pretrain_routine_end(trainer):
    # Start timer for upload rate limit
    LOGGER.info(emojis(f"{PREFIX}View model at https://hub.ultralytics.com/models/{trainer.hub_session.model_id} 🚀"))
    trainer.hub_session.t = {'metrics': time(), 'ckpt': time()}  # start timer on self.rate_limit


def on_fit_epoch_end(trainer):
    # Upload metrics after val end
    session = trainer.hub_session

    # Temp. Figure out the source of this problm
    metrics = trainer.metrics
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            metrics[k] = v.item()

    session.metrics_queue[trainer.epoch] = json.dumps(metrics)  # json string
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
