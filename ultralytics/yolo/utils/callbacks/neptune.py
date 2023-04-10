# Ultralytics YOLO 🚀, GPL-3.0 license
from ultralytics.yolo.utils import LOGGER, TESTS_RUNNING
from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import neptune
    from neptune.types import File
    
    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(neptune, '__version__')
except (ImportError, AssertionError):
    neptune = None

run = None  # NeptuneAI experiment logger instance

def _log_scalars(scalars):
    if run:
        for k, v in scalars.items():
            run[k].append(v)


def _log_images(imgs_dict, group=""):
    if run:
        for k, v in imgs_dict.items():
            run[f"{group}/{k}"].append(File(v))


def on_pretrain_routine_start(trainer):
    try:
        global run
        project = 'YOLOv8' if trainer.args.project is None else trainer.args.project
        run = neptune.init_run(project=project,
                               name=trainer.args.name,
                               tags=['YOLOv8'])
        run["Configuration/Hyperparameters"] = {k: "" if v is None else v for k, v in vars(trainer.args).items()}
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ NeptuneAI installed but not initialized correctly, not logging this run. {e}')


def on_train_epoch_end(trainer):
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix='train'))
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob('train_batch*.jpg')}, 'Mosaic')


def on_fit_epoch_end(trainer):
    if run and trainer.epoch == 0:
        model_info = {
            'parameters': get_num_params(trainer.model),
            'GFLOPs': round(get_flops(trainer.model), 3),
            'speed(ms)': round(trainer.validator.speed['inference'], 3)}
        run['Configuration/Model'] = model_info
    _log_scalars(trainer.metrics)


def on_train_end(trainer):
    if run:
        run[f"weights/{trainer.args.name or trainer.args.task}/{str(trainer.best.name)}"].upload(File(str(trainer.best)))
        run.stop()


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_train_epoch_end": on_train_epoch_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_train_end": on_train_end} if neptune else {}
