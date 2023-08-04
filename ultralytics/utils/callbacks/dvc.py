# Ultralytics YOLO 🚀, GPL-3.0 license
import os

import pkg_resources as pkg

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    from importlib.metadata import version

    import dvclive

    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS['dvc'] is True  # verify integration is enabled

    ver = version('dvclive')
    if pkg.parse_version(ver) < pkg.parse_version('2.11.0'):
        LOGGER.debug(f'DVCLive is detected but version {ver} is incompatible (>=2.11 required).')
        dvclive = None  # noqa: F811
except (ImportError, AssertionError, TypeError):
    dvclive = None

# DVCLive logger instance
live = None
_processed_plots = {}

# `on_fit_epoch_end` is called on final validation (probably need to be fixed)
# for now this is the way we distinguish final evaluation of the best model vs
# last epoch validation
_training_epoch = False


def _logger_disabled():
    return os.getenv('ULTRALYTICS_DVC_DISABLED', 'false').lower() == 'true'


def _log_images(image_path, prefix=''):
    if live:
        live.log_image(os.path.join(prefix, image_path.name), image_path)


def _log_plots(plots, prefix=''):
    for name, params in plots.items():
        timestamp = params['timestamp']
        if _processed_plots.get(name) != timestamp:
            _log_images(name, prefix)
            _processed_plots[name] = timestamp


def _log_confusion_matrix(validator):
    targets = []
    preds = []
    matrix = validator.confusion_matrix.matrix
    names = list(validator.names.values())
    if validator.confusion_matrix.task == 'detect':
        names += ['background']

    for ti, pred in enumerate(matrix.T.astype(int)):
        for pi, num in enumerate(pred):
            targets.extend([names[ti]] * num)
            preds.extend([names[pi]] * num)

    live.log_sklearn_plot('confusion_matrix', targets, preds, name='cf.json', normalized=True)


def on_pretrain_routine_start(trainer):
    try:
        global live
        if not _logger_disabled():
            live = dvclive.Live(save_dvc_exp=True, cache_images=True)
            LOGGER.info(
                'DVCLive is detected and auto logging is enabled (can be disabled with `ULTRALYTICS_DVC_DISABLED=true`).'
            )
        else:
            LOGGER.debug('DVCLive is detected and auto logging is disabled via `ULTRALYTICS_DVC_DISABLED`.')
            live = None
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ DVCLive installed but not initialized correctly, not logging this run. {e}')


def on_pretrain_routine_end(trainer):
    _log_plots(trainer.plots, 'train')


def on_train_start(trainer):
    if live:
        live.log_params(trainer.args)


def on_train_epoch_start(trainer):
    global _training_epoch
    _training_epoch = True


def on_fit_epoch_end(trainer):
    global _training_epoch
    if live and _training_epoch:
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix='train'), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value)

        if trainer.epoch == 0:
            for metric, value in model_info_for_loggers(trainer).items():
                live.log_metric(metric, value, plot=False)

        _log_plots(trainer.plots, 'train')
        _log_plots(trainer.validator.plots, 'val')

        live.next_step()
        _training_epoch = False


def on_train_end(trainer):
    if live:
        # At the end log the best metrics. It runs validator on the best model internally.
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix='train'), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value, plot=False)

        _log_plots(trainer.plots, 'val')
        _log_plots(trainer.validator.plots, 'val')
        _log_confusion_matrix(trainer.validator)

        if trainer.best.exists():
            live.log_artifact(trainer.best, copy=True, type='model')

        live.end()


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_train_start': on_train_start,
    'on_train_epoch_start': on_train_epoch_start,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if dvclive else {}
