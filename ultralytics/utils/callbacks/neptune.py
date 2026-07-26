# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["neptune"] is True  # verify integration is enabled

    import neptune
    from neptune.types import File

    assert hasattr(neptune, "__version__")

    run = None  # NeptuneAI experiment logger instance

except (ImportError, AssertionError):
    neptune = None


def _log_scalars(scalars: dict, step: int = 0) -> None:
    """Log scalars to the NeptuneAI experiment logger.

    Args:
        scalars (dict): Dictionary of scalar values to log to NeptuneAI.
        step (int, optional): The current step or iteration number for logging.

    Examples:
        >>> metrics = {"mAP": 0.85, "loss": 0.32}
        >>> _log_scalars(metrics, step=100)
    """
    if run:
        for k, v in scalars.items():
            run[k].append(value=v, step=step)


def _log_images(imgs_dict: dict, group: str = "") -> None:
    """Log images to the NeptuneAI experiment logger.

    This function logs image data to Neptune.ai when a valid Neptune run is active. Images are organized under the
    specified group name.

    Args:
        imgs_dict (dict): Dictionary of images to log, with keys as image names and values as image data.
        group (str, optional): Group name to organize images under in the Neptune UI.

    Examples:
        >>> # Log validation images
        >>> _log_images({"val_batch": img_tensor}, group="validation")
    """
    if run:
        for k, v in imgs_dict.items():
            run[f"{group}/{k}"].upload(File(v))


def _log_plot(title: str, plot_path: str) -> None:
    """Log plots to the NeptuneAI experiment logger."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # no ticks
    ax.imshow(img)
    run[f"Plots/{title}"].upload(fig)


def on_pretrain_routine_start(trainer) -> None:
    """Initialize NeptuneAI run and log hyperparameters before training starts."""
    try:
        global run
        run = neptune.init_run(
            project=trainer.args.project or "Ultralytics",
            name=trainer.args.name,
            tags=["Ultralytics"],
        )
        run["Configuration/Hyperparameters"] = {k: "" if v is None else v for k, v in vars(trainer.args).items()}
    except Exception as e:
        LOGGER.warning(f"NeptuneAI installed but not initialized correctly, not logging this run. {e}")


def on_train_epoch_end(trainer) -> None:
    """Log training metrics and learning rate at the end of each training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob("train_batch*.jpg")}, "Mosaic")


def on_fit_epoch_end(trainer) -> None:
    """Log model info and validation metrics at the end of each fit epoch."""
    if run and trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        run["Configuration/Model"] = model_info_for_loggers(trainer)
    _log_scalars(trainer.metrics, trainer.epoch + 1)


def on_val_end(validator) -> None:
    """Log validation images at the end of validation."""
    if run:
        # Log val_labels and val_pred
        _log_images({f.stem: str(f) for f in validator.save_dir.glob("val*.jpg")}, "Validation")


def on_train_end(trainer) -> None:
    """Log final results, plots, and model weights at the end of training."""
    if run:
        # Log final results, CM matrix + PR plots
        for f in [*trainer.plots.keys(), *trainer.validator.plots.keys()]:
            if "batch" not in f.name:
                _log_plot(title=f.stem, plot_path=f)
        # Log the final model
        run[f"weights/{trainer.args.name or trainer.args.task}/{trainer.best.name}"].upload(File(str(trainer.best)))


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if neptune
    else {}
)
