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


def before_train(trainer):
    # TODO: reuse existing task
    task = Task.init(project_name=trainer.args.project if trainer.args.project != 'runs/train' else 'YOLOv5',
                     task_name=trainer.args.name if trainer.args.name != 'exp' else 'Training',
                     tags=['YOLOv5'],
                     output_uri=True,
                     reuse_last_task_id=False,
                     auto_connect_frameworks={'pytorch': False})

    task.connect(trainer.args, name='parameters')


def on_batch_end(trainer):
    train_loss = trainer.tloss
    _log_scalers(trainer.label_loss_items(train_loss), "train", trainer.epoch)


def on_val_end(trainer):
    metrics = trainer.metrics
    val_losses = trainer.validator.loss
    val_loss_dict = trainer.label_loss_items(val_losses)
    _log_scalers(val_loss_dict, "val", trainer.epoch)
    _log_scalers(metrics, "metrics", trainer.epoch)


callbacks = {
    "before_train": before_train,
    "on_val_end": on_val_end,
    "on_batch_end": on_batch_end,}
