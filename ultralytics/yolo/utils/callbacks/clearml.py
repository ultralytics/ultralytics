from pathlib import Path
import os

from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import clearml
    from clearml import Task

    assert hasattr(clearml, '__version__')
except (ImportError, AssertionError):
    clearml = None


def _log_scalers(metric_dict, group="", step=0):
    task = Task.current_task()
    if task:
        for k, v in metric_dict.items():
            task.get_logger().report_scalar(group, k, v, step)

def _log_images(imgs_dict, group="", step=0):
    task = Task.current_task()
    if task:
        for k, v in imgs_dict.items():    
            task.get_logger().report_image(group, k, step, v)

def before_train(trainer):
    # TODO: reuse existing task
    task = Task.init(project_name=trainer.args.project if trainer.args.project != 'runs/train' else 'YOLOv5',
                     task_name=trainer.args.name if trainer.args.name != 'exp' else 'Training',
                     tags=['YOLOv5'],
                     output_uri=True,
                     reuse_last_task_id=False,
                     auto_connect_frameworks={'pytorch': False})
    task.connect(dict(trainer.args), name='General')


def on_epoch_start(trainer):
    if trainer.epoch == 1:
        plots = [filename for filename in os.listdir(trainer.save_dir) if filename.startswith("train_batch")]
        imgs_dict = {f"train_batch_{i}": Path(trainer.save_dir)/img for i,img in enumerate(plots)}
        if imgs_dict:
            _log_images(imgs_dict, "Mosaic", trainer.epoch)


def on_batch_end(trainer):
    _log_scalers(trainer.label_loss_items(trainer.tloss, prefix="train"), "train", trainer.epoch)


def on_val_end(trainer):
    _log_scalers(trainer.label_loss_items(trainer.validator.loss, prefix="val"), "val", trainer.epoch)
    _log_scalers({k: v for k, v in trainer.metrics.items() if k.startswith("metrics")}, "metrics", trainer.epoch)
    if trainer.epoch == 0:
        model_info = {
            "inference_speed": trainer.validator.speed[1],
            "flops@640": get_flops(trainer.model),
            "params": get_num_params(trainer.model)}
        Task.current_task().connect(model_info, 'Model')


def on_train_end(trainer):
    task = Task.current_task()
    if task:
        task.update_output_model(model_path=str(trainer.best), model_name='Best Model', auto_delete_file=False)


callbacks = {
    "before_train": before_train,
    "on_epoch_start": on_epoch_start,
    "on_val_end": on_val_end,
    "on_batch_end": on_batch_end,
    "on_train_end": on_train_end}
