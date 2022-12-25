import os
from pathlib import Path

from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import clearml
    from clearml import Task

    assert hasattr(clearml, '__version__')
except (ImportError, AssertionError):
    clearml = None


def _log_images(imgs_dict, group="", step=0):
    task = Task.current_task()
    if task:
        for k, v in imgs_dict.items():
            task.get_logger().report_image(group, k, step, v)


def on_train_start(trainer):
    # TODO: reuse existing task
    task = Task.init(project_name=trainer.args.project if trainer.args.project != 'runs/train' else 'YOLOv8',
                     task_name=trainer.args.name,
                     tags=['YOLOv8'],
                     output_uri=True,
                     reuse_last_task_id=False,
                     auto_connect_frameworks={'pytorch': False})
    task.connect(dict(trainer.args), name='General')


def on_epoch_start(trainer):
    if trainer.epoch == 1:
        plots = [filename for filename in os.listdir(trainer.save_dir) if filename.startswith("train_batch")]
        imgs_dict = {f"train_batch_{i}": Path(trainer.save_dir) / img for i, img in enumerate(plots)}
        if imgs_dict:
            _log_images(imgs_dict, "Mosaic", trainer.epoch)


def on_val_end(trainer):
    if trainer.epoch == 0:
        model_info = {
            "Parameters": get_num_params(trainer.model),
            "GFLOPs": round(get_flops(trainer.model), 1),
            "Inference speed (ms/img)": round(trainer.validator.speed[1], 1)}
        Task.current_task().connect(model_info, name='Model')


def on_train_end(trainer):
    Task.current_task().update_output_model(model_path=str(trainer.best),
                                            model_name=trainer.args.name,
                                            auto_delete_file=False)


callbacks = {
    "on_train_start": on_train_start,
    "on_epoch_start": on_epoch_start,
    "on_val_end": on_val_end,
    "on_train_end": on_train_end} if clearml else {}
