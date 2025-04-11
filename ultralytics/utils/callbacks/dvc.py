# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, checks

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["dvc"] is True  # verify integration is enabled
    import dvclive

    assert checks.check_version("dvclive", "2.11.0", verbose=True)

    import os
    import re

    # DVCLive logger instance
    live = None
    _processed_plots = {}

    # `on_fit_epoch_end` is called on final validation (probably need to be fixed) for now this is the way we
    # distinguish final evaluation of the best model vs last epoch validation
    _training_epoch = False

except (ImportError, AssertionError, TypeError):
    dvclive = None


def _log_images(path: Path, prefix: str = "") -> None:
    """
    Log images at specified path with an optional prefix using DVCLive.

    This function logs images found at the given path to DVCLive, organizing them by batch to enable slider
    functionality in the UI. It processes image filenames to extract batch information and restructures the path
    accordingly.

    Args:
        path (Path): Path to the image file to be logged.
        prefix (str): Optional prefix to add to the image name when logging.

    Examples:
        >>> from pathlib import Path
        >>> _log_images(Path("runs/train/exp/val_batch0_pred.jpg"), prefix="validation")
    """
    if live:
        name = path.name

        # Group images by batch to enable sliders in UI
        if m := re.search(r"_batch(\d+)", name):
            ni = m[1]
            new_stem = re.sub(r"_batch(\d+)", "_batch", path.stem)
            name = (Path(new_stem) / ni).with_suffix(path.suffix)

        live.log_image(os.path.join(prefix, name), path)


def _log_plots(plots: dict, prefix: str = "") -> None:
    """
    Log plot images for training progress if they have not been previously processed.

    Args:
        plots (dict): Dictionary containing plot information with timestamps.
        prefix (str, optional): Optional prefix to add to the logged image paths.
    """
    for name, params in plots.items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            _log_images(name, prefix)
            _processed_plots[name] = timestamp


def _log_confusion_matrix(validator) -> None:
    """
    Log confusion matrix for a validator using DVCLive.

    This function processes the confusion matrix from a validator object and logs it to DVCLive by converting
    the matrix into lists of target and prediction labels.

    Args:
        validator (BaseValidator): The validator object containing the confusion matrix and class names.
            Must have attributes: confusion_matrix.matrix, confusion_matrix.task, and names.

    Returns:
        None
    """
    targets = []
    preds = []
    matrix = validator.confusion_matrix.matrix
    names = list(validator.names.values())
    if validator.confusion_matrix.task == "detect":
        names += ["background"]

    for ti, pred in enumerate(matrix.T.astype(int)):
        for pi, num in enumerate(pred):
            targets.extend([names[ti]] * num)
            preds.extend([names[pi]] * num)

    live.log_sklearn_plot("confusion_matrix", targets, preds, name="cf.json", normalized=True)


def on_pretrain_routine_start(trainer) -> None:
    """Initializes DVCLive logger for training metadata during pre-training routine."""
    try:
        global live
        live = dvclive.Live(save_dvc_exp=True, cache_images=True)
        LOGGER.info("DVCLive is detected and auto logging is enabled (run 'yolo settings dvc=False' to disable).")
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ DVCLive installed but not initialized correctly, not logging this run. {e}")


def on_pretrain_routine_end(trainer) -> None:
    """Logs plots related to the training process at the end of the pretraining routine."""
    _log_plots(trainer.plots, "train")


def on_train_start(trainer) -> None:
    """Logs the training parameters if DVCLive logging is active."""
    if live:
        live.log_params(trainer.args)


def on_train_epoch_start(trainer) -> None:
    """Sets the global variable _training_epoch value to True at the start of training each epoch."""
    global _training_epoch
    _training_epoch = True


def on_fit_epoch_end(trainer) -> None:
    """
    Log training metrics, model info, and advance to next step at the end of each fit epoch.

    This function is called at the end of each fit epoch during training. It logs various metrics including
    training loss items, validation metrics, and learning rates. On the first epoch, it also logs model
    information. Additionally, it logs training and validation plots and advances the DVCLive step counter.

    Args:
        trainer (BaseTrainer): The trainer object containing training state, metrics, and plots.

    Notes:
        This function only performs logging operations when DVCLive logging is active and during a training epoch.
        The global variable _training_epoch is used to track whether the current epoch is a training epoch.
    """
    global _training_epoch
    if live and _training_epoch:
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value)

        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for metric, value in model_info_for_loggers(trainer).items():
                live.log_metric(metric, value, plot=False)

        _log_plots(trainer.plots, "train")
        _log_plots(trainer.validator.plots, "val")

        live.next_step()
        _training_epoch = False


def on_train_end(trainer) -> None:
    """
    Log best metrics, plots, and confusion matrix at the end of training.

    This function is called at the conclusion of the training process to log final metrics, visualizations, and
    model artifacts if DVCLive logging is active. It captures the best model performance metrics, training plots,
    validation plots, and confusion matrix for later analysis.

    Args:
        trainer (BaseTrainer): The trainer object containing training state, metrics, and validation results.

    Examples:
        >>> # Inside a custom training loop
        >>> from ultralytics.utils.callbacks.dvc import on_train_end
        >>> on_train_end(trainer)  # Log final metrics and artifacts
    """
    if live:
        # At the end log the best metrics. It runs validator on the best model internally.
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value, plot=False)

        _log_plots(trainer.plots, "val")
        _log_plots(trainer.validator.plots, "val")
        _log_confusion_matrix(trainer.validator)

        if trainer.best.exists():
            live.log_artifact(trainer.best, copy=True, type="model")

        live.end()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if dvclive
    else {}
)
