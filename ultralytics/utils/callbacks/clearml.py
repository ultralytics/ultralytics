# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["clearml"] is True  # verify integration is enabled
    import clearml
    from clearml import Task

    assert hasattr(clearml, "__version__")  # verify package is not directory

except (ImportError, AssertionError):
    clearml = None


def _log_debug_samples(files, title: str = "Debug Samples") -> None:
    """Log files (images) as debug samples in the ClearML task.

    Args:
        files (list[Path]): A list of file paths in PosixPath format.
        title (str): A title that groups together images with the same values.
    """
    import re

    if task := Task.current_task():
        for f in files:
            if f.exists():
                it = re.search(r"_batch(\d+)", f.name)
                iteration = int(it.groups()[0]) if it else 0
                task.get_logger().report_image(
                    title=title, series=f.name.replace(it.group(), ""), local_path=str(f), iteration=iteration
                )


def _log_plot(title: str, plot_path: str) -> None:
    """Log an image as a plot in the plot section of ClearML.

    Args:
        title (str): The title of the plot.
        plot_path (str): The path to the saved image file.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # no ticks
    ax.imshow(img)

    Task.current_task().get_logger().report_matplotlib_figure(
        title=title, series="", figure=fig, report_interactive=False
    )


def on_pretrain_routine_start(trainer) -> None:
    """Initialize and connect ClearML task at the start of pretraining routine."""
    try:
        if task := Task.current_task():
            # WARNING: make sure the automatic pytorch and matplotlib bindings are disabled!
            # We are logging these plots and model files manually in the integration
            from clearml.binding.frameworks.pytorch_bind import PatchPyTorchModelIO
            from clearml.binding.matplotlib_bind import PatchedMatplotlib

            PatchPyTorchModelIO.update_current_task(None)
            PatchedMatplotlib.update_current_task(None)
        else:
            task = Task.init(
                project_name=trainer.args.project or "Ultralytics",
                task_name=trainer.args.name,
                tags=["Ultralytics"],
                output_uri=True,
                reuse_last_task_id=False,
                auto_connect_frameworks={"pytorch": False, "matplotlib": False},
            )
            LOGGER.warning(
                "ClearML Initialized a new task. If you want to run remotely, "
                "please add clearml-init and connect your arguments before initializing YOLO."
            )
        task.connect(vars(trainer.args), name="General")
    except Exception as e:
        LOGGER.warning(f"ClearML installed but not initialized correctly, not logging this run. {e}")


def on_train_epoch_end(trainer) -> None:
    """Log debug samples for the first epoch and report current training progress."""
    if task := Task.current_task():
        # Log debug samples for first epoch only
        if trainer.epoch == 1:
            _log_debug_samples(sorted(trainer.save_dir.glob("train_batch*.jpg")), "Mosaic")
        # Report the current training progress
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            task.get_logger().report_scalar("train", k, v, iteration=trainer.epoch)
        for k, v in trainer.lr.items():
            task.get_logger().report_scalar("lr", k, v, iteration=trainer.epoch)


def on_fit_epoch_end(trainer) -> None:
    """Report model information and metrics to logger at the end of an epoch."""
    if task := Task.current_task():
        # Report epoch time and validation metrics
        task.get_logger().report_scalar(
            title="Epoch Time", series="Epoch Time", value=trainer.epoch_time, iteration=trainer.epoch
        )
        for k, v in trainer.metrics.items():
            title = k.split("/")[0]
            task.get_logger().report_scalar(title, k, v, iteration=trainer.epoch)
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for k, v in model_info_for_loggers(trainer).items():
                task.get_logger().report_single_value(k, v)


def on_val_end(validator) -> None:
    """Log validation results including labels and predictions."""
    if Task.current_task():
        # Log validation labels and predictions
        _log_debug_samples(sorted(validator.save_dir.glob("val*.jpg")), "Validation")


def on_train_end(trainer) -> None:
    """Log final model and training results on training completion."""
    if task := Task.current_task():
        # Log final results, confusion matrix and PR plots
        for f in [*trainer.plots.keys(), *trainer.validator.plots.keys()]:
            if "batch" not in f.name:
                _log_plot(title=f.stem, plot_path=f)
        # Report final metrics
        for k, v in trainer.validator.metrics.results_dict.items():
            task.get_logger().report_single_value(k, v)
        # Log the final model
        task.update_output_model(model_path=str(trainer.best), model_name=trainer.args.name, auto_delete_file=False)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if clearml
    else {}
)
