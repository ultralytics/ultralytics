---
description: Learn to integrate DVCLive with Ultralytics for enhanced logging during training. Step-by-step methods for setting up and optimizing DVC callbacks.
keywords: Ultralytics, DVC, DVCLive, machine learning, logging, training, callbacks, integration
---

# Reference for `ultralytics/utils/callbacks/dvc.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`_log_images`](#ultralytics.utils.callbacks.dvc._log_images)
        - [`_log_plots`](#ultralytics.utils.callbacks.dvc._log_plots)
        - [`_log_confusion_matrix`](#ultralytics.utils.callbacks.dvc._log_confusion_matrix)
        - [`on_pretrain_routine_start`](#ultralytics.utils.callbacks.dvc.on_pretrain_routine_start)
        - [`on_pretrain_routine_end`](#ultralytics.utils.callbacks.dvc.on_pretrain_routine_end)
        - [`on_train_start`](#ultralytics.utils.callbacks.dvc.on_train_start)
        - [`on_train_epoch_start`](#ultralytics.utils.callbacks.dvc.on_train_epoch_start)
        - [`on_fit_epoch_end`](#ultralytics.utils.callbacks.dvc.on_fit_epoch_end)
        - [`on_train_end`](#ultralytics.utils.callbacks.dvc.on_train_end)


## Function `ultralytics.utils.callbacks.dvc._log_images` {#ultralytics.utils.callbacks.dvc.\_log\_images}

```python
def _log_images(path: Path, prefix: str = "") -> None
```

Log images at specified path with an optional prefix using DVCLive.

This function logs images found at the given path to DVCLive, organizing them by batch to enable slider functionality in the UI. It processes image filenames to extract batch information and restructures the path accordingly.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `Path` | Path to the image file to be logged. | *required* |
| `prefix` | `str, optional` | Optional prefix to add to the image name when logging. | `""` |

**Examples**

```python
>>> from pathlib import Path
>>> _log_images(Path("runs/train/exp/val_batch0_pred.jpg"), prefix="validation")
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L29-L53"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_images(path: Path, prefix: str = "") -> None:
    """Log images at specified path with an optional prefix using DVCLive.

    This function logs images found at the given path to DVCLive, organizing them by batch to enable slider
    functionality in the UI. It processes image filenames to extract batch information and restructures the path
    accordingly.

    Args:
        path (Path): Path to the image file to be logged.
        prefix (str, optional): Optional prefix to add to the image name when logging.

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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.dvc._log_plots` {#ultralytics.utils.callbacks.dvc.\_log\_plots}

```python
def _log_plots(plots: dict, prefix: str = "") -> None
```

Log plot images for training progress if they have not been previously processed.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `plots` | `dict` | Dictionary containing plot information with timestamps. | *required* |
| `prefix` | `str, optional` | Optional prefix to add to the logged image paths. | `""` |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L56-L67"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_plots(plots: dict, prefix: str = "") -> None:
    """Log plot images for training progress if they have not been previously processed.

    Args:
        plots (dict): Dictionary containing plot information with timestamps.
        prefix (str, optional): Optional prefix to add to the logged image paths.
    """
    for name, params in plots.items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            _log_images(name, prefix)
            _processed_plots[name] = timestamp
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.dvc._log_confusion_matrix` {#ultralytics.utils.callbacks.dvc.\_log\_confusion\_matrix}

```python
def _log_confusion_matrix(validator) -> None
```

Log confusion matrix for a validator using DVCLive.

This function processes the confusion matrix from a validator object and logs it to DVCLive by converting the matrix into lists of target and prediction labels.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `validator` | `BaseValidator` | The validator object containing the confusion matrix and class names. Must have<br>    attributes confusion_matrix.matrix, confusion_matrix.task, and names. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L70-L92"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_confusion_matrix(validator) -> None:
    """Log confusion matrix for a validator using DVCLive.

    This function processes the confusion matrix from a validator object and logs it to DVCLive by converting the matrix
    into lists of target and prediction labels.

    Args:
        validator (BaseValidator): The validator object containing the confusion matrix and class names. Must have
            attributes confusion_matrix.matrix, confusion_matrix.task, and names.
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.dvc.on_pretrain_routine_start` {#ultralytics.utils.callbacks.dvc.on\_pretrain\_routine\_start}

```python
def on_pretrain_routine_start(trainer) -> None
```

Initialize DVCLive logger for training metadata during pre-training routine.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L95-L102"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_pretrain_routine_start(trainer) -> None:
    """Initialize DVCLive logger for training metadata during pre-training routine."""
    try:
        global live
        live = dvclive.Live(save_dvc_exp=True, cache_images=True)
        LOGGER.info("DVCLive is detected and auto logging is enabled (run 'yolo settings dvc=False' to disable).")
    except Exception as e:
        LOGGER.warning(f"DVCLive installed but not initialized correctly, not logging this run. {e}")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.dvc.on_pretrain_routine_end` {#ultralytics.utils.callbacks.dvc.on\_pretrain\_routine\_end}

```python
def on_pretrain_routine_end(trainer) -> None
```

Log plots related to the training process at the end of the pretraining routine.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L105-L107"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_pretrain_routine_end(trainer) -> None:
    """Log plots related to the training process at the end of the pretraining routine."""
    _log_plots(trainer.plots, "train")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.dvc.on_train_start` {#ultralytics.utils.callbacks.dvc.on\_train\_start}

```python
def on_train_start(trainer) -> None
```

Log the training parameters if DVCLive logging is active.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L110-L113"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_start(trainer) -> None:
    """Log the training parameters if DVCLive logging is active."""
    if live:
        live.log_params(trainer.args)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.dvc.on_train_epoch_start` {#ultralytics.utils.callbacks.dvc.on\_train\_epoch\_start}

```python
def on_train_epoch_start(trainer) -> None
```

Set the global variable _training_epoch value to True at the start of each training epoch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L116-L119"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_epoch_start(trainer) -> None:
    """Set the global variable _training_epoch value to True at the start of each training epoch."""
    global _training_epoch
    _training_epoch = True
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.dvc.on_fit_epoch_end` {#ultralytics.utils.callbacks.dvc.on\_fit\_epoch\_end}

```python
def on_fit_epoch_end(trainer) -> None
```

Log training metrics, model info, and advance to next step at the end of each fit epoch.

This function is called at the end of each fit epoch during training. It logs various metrics including training loss items, validation metrics, and learning rates. On the first epoch, it also logs model information. Additionally, it logs training and validation plots and advances the DVCLive step counter.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `BaseTrainer` | The trainer object containing training state, metrics, and plots. | *required* |

!!! note "Notes"

    This function only performs logging operations when DVCLive logging is active and during a training epoch.
    The global variable _training_epoch is used to track whether the current epoch is a training epoch.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L122-L152"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_fit_epoch_end(trainer) -> None:
    """Log training metrics, model info, and advance to next step at the end of each fit epoch.

    This function is called at the end of each fit epoch during training. It logs various metrics including training
    loss items, validation metrics, and learning rates. On the first epoch, it also logs model
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.dvc.on_train_end` {#ultralytics.utils.callbacks.dvc.on\_train\_end}

```python
def on_train_end(trainer) -> None
```

Log best metrics, plots, and confusion matrix at the end of training.

This function is called at the conclusion of the training process to log final metrics, visualizations, and model artifacts if DVCLive logging is active. It captures the best model performance metrics, training plots, validation plots, and confusion matrix for later analysis.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `BaseTrainer` | The trainer object containing training state, metrics, and validation results. | *required* |

**Examples**

```python
>>> # Inside a custom training loop
>>> from ultralytics.utils.callbacks.dvc import on_train_end
>>> on_train_end(trainer)  # Log final metrics and artifacts
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/dvc.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/dvc.py#L155-L183"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_end(trainer) -> None:
    """Log best metrics, plots, and confusion matrix at the end of training.

    This function is called at the conclusion of the training process to log final metrics, visualizations, and model
    artifacts if DVCLive logging is active. It captures the best model performance metrics, training plots, validation
    plots, and confusion matrix for later analysis.

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
```
</details>

<br><br>
