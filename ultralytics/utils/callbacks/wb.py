# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS['wandb'] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, '__version__')  # verify package is not directory

    import numpy as np
    import pandas as pd

    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def create_custom_wandb_metric(x,
                               y,
                               classes,
                               title='Precision Recall Curve',
                               x_axis_title='Recall',
                               y_axis_title='Precision'):
    """
    Creates a custom wandb metric similar to default wandb.plot.pr_curve.

    Args:
        x (List): list of N values to plot on the x-axis
        y (List): list of N values to plot on the y-axis
        classes (List): class labels for each point (list of N values)
        title (str, optional): plot title. Defaults to 'Precision Recall Curve'.
        x_axis_title (str, optional): title for x-axis. Defaults to 'Recall'.
        y_axis_title (str, optional): title for y-axis. Defaults to 'Precision'.

    Returns:
        wandb object to log
    """
    df = pd.DataFrame({'class': classes, 'y': y, 'x': x}).round(3)
    fields = {'x': 'x', 'y': 'y', 'class': 'class'}
    string_fields = {'title': title, 'x-axis-title': x_axis_title, 'y-axis-title': y_axis_title}
    return wb.plot_table('wandb/area-under-curve/v0',
                         wb.Table(dataframe=df),
                         fields=fields,
                         string_fields=string_fields)


def plot_curve_wandb(x,
                     y,
                     names=None,
                     id='precision-recall',
                     title='Precision Recall Curve',
                     x_axis_title='Recall',
                     y_axis_title='Precision',
                     num_x=100,
                     only_mean=True):
    """
    Adds a metric curve to wandb.

    Args:
        x (np.ndarray): X-axis of N values.
        y (np.ndarray): Y-axis of C by N values where C is the number of classes.
        names (list, optional): list of class names (length C). Defaults to [].
        id (str, optional): log id in wandb. Defaults to 'precision-recall'.
        title (str, optional): plot title in wandb. Defaults to 'Precision Recall Curve'.
        x_axis_title (str, optional): title for x-axis. Defaults to 'Recall'.
        y_axis_title (str, optional): title for y-axis. Defaults to 'Precision'.
        num_x (int, optional): number of points to interpolate to. Defaults to 100.
        only_mean (bool, optional): if True, only the mean curve is plotted. Defaults to True.
    """
    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).tolist()
    classes = ['mean'] * len(x_log)

    if not only_mean:
        for i, yi in enumerate(y):
            # Add new x
            x_log.extend(x_new)
            # Interpolate y to new x
            y_log.extend(np.interp(x_new, x, yi))
            # Add class names
            classes.extend([names[i]] * len(x_new))

    wb.log(
        {id: create_custom_wandb_metric(
            x_log,
            y_log,
            classes,
            title,
            x_axis_title,
            y_axis_title,
        )},
        commit=False,
    )


def _log_plots(plots, step):
    """Logs plots from the input dictionary if they haven't been logged already at the specified step."""
    for name, params in plots.items():
        timestamp = params['timestamp']
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """Initiate and start project if module is present."""
    wb.run or wb.init(project=trainer.args.project or 'YOLOv8', name=trainer.args.name, config=vars(trainer.args))


def on_fit_epoch_end(trainer):
    """Logs training metrics and model information at the end of an epoch."""
    wb.run.log(trainer.metrics, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """Save the best model as an artifact at end of training."""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type='model', name=f'run_{wb.run.id}_model')
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=['best'])
    for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
        x, y, x_axis_title, y_axis_title = curve_values
        plot_curve_wandb(
            x,
            y,
            id=f'curves/{curve_name}',
            title=f'{curve_name}',
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )
    wb.run.finish()  # required or run continues on dashboard


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if wb else {}
