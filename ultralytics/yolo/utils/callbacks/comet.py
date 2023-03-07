# Ultralytics YOLO 🚀, GPL-3.0 license
import os

from ultralytics.yolo.utils import LOGGER, RANK, TESTS_RUNNING
from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

COMET_MODE = os.getenv('COMET_MODE', 'online')
MODEL_NAME = os.getenv('COMET_MODEL_NAME', 'YOLOv8')

try:
    import comet_ml

    assert not TESTS_RUNNING  # do not log pytest
    assert comet_ml.__version__  # verify package is not directory
except (ImportError, AssertionError):
    comet_ml = None


def _fetch_run_metadata(trainer):
    curr_epoch = trainer.epoch + 1

    train_num_samples = len(trainer.train_loader.dataset)
    train_batch_size = trainer.batch_size
    train_num_steps_per_epoch = train_num_samples // train_batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    save_assets = ((trainer.args.save) and (trainer.save_period > 0) and ((curr_epoch % trainer.args.save_period) == 0))

    output = dict(curr_epoch=curr_epoch, curr_step=curr_step, save_assets=save_assets)
    return output


def _get_experiment(mode, project_name):
    if mode == 'offline':
        return comet_ml.OfflineExperiment(project_name=project_name)

    return comet_ml.Experiment(project_name=project_name)


def _log_confusion_matrix(trainer, experiment, curr_epoch):
    conf_mat = trainer.validator.confusion_matrix.matrix
    # Ensure dictionary with label names is sorted based on keys
    label_names = [v for k, v in sorted(trainer.data['names'].items(), key=lambda x: x[0])]
    label_names.append('background')
    experiment.log_confusion_matrix(
        matrix=conf_mat,
        labels=label_names,
        max_categories=len(label_names),
        epoch=curr_epoch,
    )


def on_pretrain_routine_start(trainer):
    # Ensures that the experiment object is created in a single process
    # during distributed training.
    if RANK in {-1, 0}:
        try:
            experiment = _get_experiment(COMET_MODE, trainer.args.project)
            experiment.log_parameters(vars(trainer.args))
            experiment.log_other('Created from', 'yolov8')
        except Exception as e:
            LOGGER.warning(f'WARNING ⚠️ Comet installed but not initialized correctly, not logging this run. {e}')


def on_train_epoch_end(trainer):
    experiment = comet_ml.get_global_experiment()

    metadata = _fetch_run_metadata(trainer)
    curr_epoch = metadata['curr_epoch']
    curr_step = metadata['curr_step']

    if experiment:
        experiment.log_metrics(
            trainer.label_loss_items(trainer.tloss, prefix='train'),
            epoch=curr_epoch,
        )

        for f in trainer.save_dir.glob('train_batch*.jpg'):
            experiment.log_image(f, name=f.stem, step=curr_step)


def on_fit_epoch_end(trainer):
    experiment = comet_ml.get_global_experiment()

    metadata = _fetch_run_metadata(trainer)
    curr_epoch = metadata['curr_epoch']
    curr_step = metadata['curr_step']
    save_assets = metadata['save_assets']

    if experiment:
        experiment.log_metrics(trainer.metrics, epoch=curr_epoch)
        if trainer.epoch == 0:
            model_info = {
                'model/parameters': get_num_params(trainer.model),
                'model/GFLOPs': round(get_flops(trainer.model), 3),
                'model/speed(ms)': round(trainer.validator.speed['inference'], 3),}
            experiment.log_metrics(model_info, epoch=curr_epoch)

        if save_assets:
            for f in trainer.save_dir.glob('val_batch*.jpg'):
                experiment.log_image(f, name=f.stem, step=curr_step)

            experiment.log_model(
                'YOLOv8',
                file_or_folder=str(trainer.best),
                file_name='best.pt',
                overwrite=True,
            )
            _log_confusion_matrix(trainer, experiment, curr_epoch)


def on_train_end(trainer):
    experiment = comet_ml.get_global_experiment()

    metadata = _fetch_run_metadata(trainer)
    curr_epoch = metadata['curr_epoch']

    if experiment:
        experiment.log_model(
            'YOLOv8',
            file_or_folder=str(trainer.best),
            file_name='best.pt',
            overwrite=True,
        )
        _log_confusion_matrix(trainer, experiment, curr_epoch)

        try:
            for plots in [
                    'F1_curve',
                    'P_curve',
                    'R_curve',
                    'PR_curve',
                    'confusion_matrix',]:
                filepath = trainer.save_dir / f'{plots}.png'
                experiment.log_image(filepath, name=filepath.stem)

            for labels in ['labels', 'labels_correlogram']:
                filepath = trainer.save_dir / f'{labels}.jpg'
                experiment.log_image(filepath, name=filepath.stem)

        except Exception as e:
            LOGGER.warning(f'COMET WARNING: {e}')


def teardown(trainer):
    experiment = comet_ml.get_global_experiment()
    if experiment:
        experiment.end()


callbacks = ({
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end,
    'teardown': teardown,} if comet_ml else {})
