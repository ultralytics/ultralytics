# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert SETTINGS["wandb"] is True  # verify integration is enabled
    import wandb

    DISABLED = False
except (ImportError, AssertionError):
    DISABLED = True


def on_pretrain_routine_start(trainer):
    """Initiate and start project if module is present."""
    if wandb.run:  # run should be created in user code
        trainer.wandb_run = wandb.run
        trainer.wandb_run.config.update({"train": (wandb.config.get("train") or {}) | (vars(trainer.args))})


def on_fit_epoch_end(trainer):
    """Logs training metrics and model information at the end of an epoch."""
    wandb_run = getattr(trainer, "wandb_run", None)
    if wandb_run:
        wandb_run.log(trainer.metrics, step=trainer.epoch + 1)
        _log_plots(wandb_run, trainer.plots, step=trainer.epoch + 1)
        _log_plots(wandb_run, trainer.validator.plots, step=trainer.epoch + 1)
        if trainer.epoch == 0:
            wandb_run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wandb_run = getattr(trainer, "wandb_run", None)
    if wandb_run:
        wandb_run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
        wandb_run.log(trainer.lr, step=trainer.epoch + 1)
        if trainer.epoch == 1:
            _log_plots(wandb_run, trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """Save the best model as an artifact at end of training."""
    wandb_run = getattr(trainer, "wandb_run", None)
    if wandb_run:
        _log_plots(wandb_run, trainer.validator.plots, step=trainer.epoch + 1)
        _log_plots(wandb_run, trainer.plots, step=trainer.epoch + 1)
        art = wandb.Artifact(type="model", name=f"run_{wandb_run.id}_model")
        if trainer.best.exists():
            art.add_file(trainer.best)
            wandb_run.log_artifact(art, aliases=["best"])
        if hasattr(trainer.validator.metrics, "curves") and hasattr(trainer.validator.metrics, "curves_results"):
            for curve, curve_result in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
                x, y, x_title, y_title = curve_result
                _plot_curve(
                    wandb_run,
                    x,
                    y,
                    names=list(trainer.validator.metrics.names.values()),
                    id=f"curves/{curve}",
                    title=curve,
                    x_title=x_title,
                    y_title=y_title,
                )


def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """
    Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of the default wandb precision-recall
    curve while allowing for enhanced customization. The visual metric is useful for monitoring model performance across
    different classes.

    Args:
        x (List): Values for the x-axis; expected to have length N.
        y (List): Corresponding values for the y-axis; also expected to have length N.
        classes (List): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot; defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis; defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis; defaults to 'Precision'.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    import pandas  # scope for faster 'import ultralytics'

    df = pandas.DataFrame({"class": classes, "y": y, "x": x}).round(3)
    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wandb.plot_table(
        "wandb/area-under-curve/v0", wandb.Table(dataframe=df), fields=fields, string_fields=string_fields
    )


def _plot_curve(
    wandb_run,
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """
    Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb.
    The curve can represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        wandb_run (wandb.Run): Current wandb run object.
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape CxN, where C is the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C. Defaults to [].
        id (str, optional): Unique identifier for the logged data in wandb. Defaults to 'precision-recall'.
        title (str, optional): Title for the visualization plot. Defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis. Defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis. Defaults to 'Precision'.
        num_x (int, optional): Number of interpolated data points for visualization. Defaults to 100.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted. Defaults to True.

    Note:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    import numpy as np

    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wandb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wandb_run.log({title: wandb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # add new x
            y_log.extend(np.interp(x_new, x, yi))  # interpolate y to new x
            classes.extend([names[i]] * len(x_new))  # add class names
        wandb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


_processed_plots = {}


def _log_plots(wandb_run, plots, step):
    """Logs plots from the input dictionary if they haven't been logged already at the specified step."""
    for name, params in plots.copy().items():  # shallow copy to prevent plots dict changing during iteration
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wandb_run.log({name.stem: wandb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


callbacks = (
    {}
    if DISABLED
    else {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
)
