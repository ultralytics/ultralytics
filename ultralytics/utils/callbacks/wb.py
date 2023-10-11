# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import pandas as pd

from ultralytics.utils import SETTINGS, TESTS_RUNNING, TryExcept
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS['wandb'] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, '__version__')

    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


@TryExcept("create_custom_wandb_metric")
def create_custom_wandb_metric(
    xs: list,
    ys: list,
    classes: list,
    title: str = "Precision Recall Curve",
    x_axis_title: str = "Recall",
    y_axis_title: str = "Precision",
):
    """Creates a custom wandb metric similar to default wandb.plot.pr_curve

    Args:
        xs: list of N values to plot on the x-axis
        ys: list of N values to plot on the y-axis
        classes: class labels for each point (list of N values)
        title: plot title

    Returns:
        wandb object to log
    """
    df = pd.DataFrame({
        "class": classes,
        "y": ys,
        "x": xs, }).round(3)

    return wb.plot_table(
        "wandb/area-under-curve/v0",
        wb.Table(dataframe=df),
        {
            "x": "x",
            "y": "y",
            "class": "class"},
        {
            "title": title,
            "x-axis-title": x_axis_title,
            "y-axis-title": y_axis_title, },
    )


@TryExcept("plot_curve_wandb")
def plot_curve_wandb(
    xs: np.ndarray,
    ys: np.ndarray,
    names: list = [],
    id: str = "precision-recall",
    title: str = "Precision Recall Curve",
    x_axis_title: str = "Recall",
    y_axis_title: str = "Precision",
    num_xs: int = 100,
    only_mean: bool = True,
):
    """adds a metric curve to wandb

    Args:
        xs: np.array of N values
        ys: np.array of C by N values where C is the number of classes
        names: list of class names (length C)
        id: log id in wandb
        title: plot title in wandb
        num_xs: number of points to interpolate to
        only_mean: if True, only the mean curve is plotted
    """
    # create new xs
    xs_new = np.linspace(xs[0], xs[-1], num_xs)

    # create arrays for logging
    xs_log = xs_new.tolist()
    ys_log = np.interp(xs_new, xs, np.mean(ys, axis=0)).tolist()
    classes = ["mean"] * len(xs_log)

    if not only_mean:
        for i, y in enumerate(ys):
            # add new xs
            xs_log.extend(xs_new)
            # interpolate y to new xs
            ys_log.extend(np.interp(xs_new, xs, y))
            # add class names
            classes.extend([names[i]] * len(xs_new))

    wb.log(
        {id: create_custom_wandb_metric(
            xs_log,
            ys_log,
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
        xs, ys, x_axis_title, y_axis_title = curve_values
        plot_curve_wandb(
            xs,
            ys,
            id=f"curves/{curve_name}",
            title=f"{curve_name}",
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
        )
    wb.run.finish()  # required or run continues on dashboard


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if wb else {}
