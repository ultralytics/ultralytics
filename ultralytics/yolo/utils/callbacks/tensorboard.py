# Ultralytics YOLO 🚀, GPL-3.0 license
from ultralytics.yolo.utils import LOGGER, TESTS_RUNNING

try:
    from torch.utils.tensorboard import SummaryWriter

    assert not TESTS_RUNNING  # do not log pytest
except (ImportError, AssertionError):
    SummaryWriter = None

writer = None  # TensorBoard SummaryWriter instance


def _log_scalars(scalars, step=0):
    if writer:
        for k, v in scalars.items():
            writer.add_scalar(k, v, step)


def on_pretrain_routine_start(trainer):
    global writer
    try:
        writer = SummaryWriter(str(trainer.save_dir))
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}')


def on_batch_end(trainer):
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix='train'), trainer.epoch + 1)


def on_fit_epoch_end(trainer):
    _log_scalars(trainer.metrics, trainer.epoch + 1)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_batch_end': on_batch_end}
