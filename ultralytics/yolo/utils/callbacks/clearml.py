from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import clearml
    from clearml import Task

    assert hasattr(clearml, '__version__')
except (ImportError, AssertionError):
    clearml = None


def on_train_start(trainer):
    # TODO: reuse existing task
    task = Task.init(project_name=trainer.args.project if trainer.args.project != 'runs/train' else 'YOLOv8',
                     task_name=trainer.args.name,
                     tags=['YOLOv8'],
                     output_uri=True,
                     reuse_last_task_id=False,
                     auto_connect_frameworks={'pytorch': False})
    task.connect(dict(trainer.args), name='General')


def on_val_end(trainer):
    if trainer.epoch == 0:
        model_info = {
            "Inference speed (ms/img)": round(trainer.validator.speed[1], 1),
            "GFLOPs": round(get_flops(trainer.model), 1),
            "Parameters": get_num_params(trainer.model)}
        Task.current_task().connect(model_info, name='Model')


def on_train_end(trainer):
    Task.current_task().update_output_model(model_path=str(trainer.best),
                                            model_name=trainer.args.name,
                                            auto_delete_file=False)


callbacks = {
    "on_train_start": on_train_start,
    "on_val_end": on_val_end,
    "on_train_end": on_train_end} if clearml else {}
