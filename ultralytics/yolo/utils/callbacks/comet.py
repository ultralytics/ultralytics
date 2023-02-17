# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import comet_ml

except ImportError:
    comet_ml = None


def on_pretrain_routine_start(trainer):
    experiment = comet_ml.Experiment(project_name=trainer.args.project or 'YOLOv8')
    experiment.log_parameters(vars(trainer.args))


def on_train_epoch_end(trainer):
    experiment = comet_ml.get_global_experiment()
    experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1)
    if trainer.epoch == 1:
        for f in trainer.save_dir.glob('train_batch*.jpg'):
            experiment.log_image(f, name=f.stem, step=trainer.epoch + 1)


def on_fit_epoch_end(trainer):
    experiment = comet_ml.get_global_experiment()
    experiment.log_metrics(trainer.metrics, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        model_info = {
            'model/parameters': get_num_params(trainer.model),
            'model/GFLOPs': round(get_flops(trainer.model), 3),
            'model/speed(ms)': round(trainer.validator.speed[1], 3)}
        experiment.log_metrics(model_info, step=trainer.epoch + 1)


def on_train_end(trainer):
    experiment = comet_ml.get_global_experiment()
    experiment.log_model('YOLOv8', file_or_folder=str(trainer.best), file_name='best.pt', overwrite=True)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if comet_ml else {}
