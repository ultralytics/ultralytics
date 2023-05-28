# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.yolo.utils import TESTS_RUNNING
from ultralytics.yolo.utils.torch_utils import model_info_for_loggers

try:
    import wandb as wb

    assert hasattr(wb, '__version__')
    assert not TESTS_RUNNING  # do not log pytest
except (ImportError, AssertionError):
    wb = None


def on_pretrain_routine_start(trainer):
    """Initiate and start project if module is present."""
    wb.run or wb.init(project=trainer.args.project or 'YOLOv8', name=trainer.args.name, config=vars(trainer.args))


def on_fit_epoch_end(trainer):
    """Logs training metrics and model information at the end of an epoch."""
    wb.run.log(trainer.metrics, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        wb.run.log({f.stem: wb.Image(str(f))
                    for f in trainer.save_dir.glob('train_batch*.jpg')},
                   step=trainer.epoch + 1)


def on_train_end(trainer):
    """Save the best model as an artifact at end of training."""
    art = wb.Artifact(type='model', name=f'run_{wb.run.id}_model')
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if wb else {}
