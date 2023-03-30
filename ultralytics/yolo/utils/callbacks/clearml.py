# Ultralytics YOLO 🚀, GPL-3.0 license
from ultralytics.yolo.utils import LOGGER, TESTS_RUNNING
from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import clearml
    from clearml import Task

    assert hasattr(clearml, '__version__')  # verify package is not directory
    assert not TESTS_RUNNING  # do not log pytest
except (ImportError, AssertionError):
    clearml = None


def _log_images(imgs_dict, group='', step=0):
    task = Task.current_task()
    if task:
        for k, v in imgs_dict.items():
            task.get_logger().report_image(group, k, step, v)


def on_pretrain_routine_start(trainer):
    try:
        task = Task.init(project_name=trainer.args.project or 'YOLOv8',
                         task_name=trainer.args.name,
                         tags=['YOLOv8'],
                         output_uri=True,
                         reuse_last_task_id=False,
                         auto_connect_frameworks={'pytorch': False})
        task.connect(vars(trainer.args), name='General')
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ ClearML installed but not initialized correctly, not logging this run. {e}')


def on_train_epoch_end(trainer):
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob('train_batch*.jpg')}, 'Mosaic', trainer.epoch)


def on_fit_epoch_end(trainer):
    task = Task.current_task()
    if task and trainer.epoch == 0:
        model_info = {
            'model/parameters': get_num_params(trainer.model),
            'model/GFLOPs': round(get_flops(trainer.model), 3),
            'model/speed(ms)': round(trainer.validator.speed['inference'], 3)}
        task.connect(model_info, name='Model')


def on_train_end(trainer):
    task = Task.current_task()
    if task:
        task.update_output_model(model_path=str(trainer.best), model_name=trainer.args.name, auto_delete_file=False)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if clearml else {}
