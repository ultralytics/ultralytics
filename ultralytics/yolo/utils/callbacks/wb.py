# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import wandb as wb

    assert hasattr(wb, '__version__')
except (ImportError, AssertionError):
    wb = None


def on_pretrain_routine_start(trainer):
    wb.init(project=trainer.args.project or 'YOLOv8', name=trainer.args.name, config=vars(
        trainer.args)) if not wb.run else wb.run


def on_fit_epoch_end(trainer):
    wb.run.log(trainer.metrics, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        model_info = {
            'model/parameters': get_num_params(trainer.model),
            'model/GFLOPs': round(get_flops(trainer.model), 3),
            'model/speed(ms)': round(trainer.validator.speed['inference'], 3)}
        wb.run.log(model_info, step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        wb.run.log({f.stem: wb.Image(str(f))
                    for f in trainer.save_dir.glob('train_batch*.jpg')},
                   step=trainer.epoch + 1)


def on_train_end(trainer):
    art = wb.Artifact(type='model', name=f'run_{wb.run.id}_model')
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if wb else {}
