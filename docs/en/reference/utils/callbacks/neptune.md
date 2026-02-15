---
description: Learn how to use NeptuneAI with Ultralytics for advanced logging and tracking of experiments. Detailed setup and callback functions included.
keywords: Ultralytics, NeptuneAI, YOLO, experiment logging, machine learning, AI, callbacks, training, validation
---

# Reference for `ultralytics/utils/callbacks/neptune.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`_log_scalars`](#ultralytics.utils.callbacks.neptune._log_scalars)
        - [`_log_images`](#ultralytics.utils.callbacks.neptune._log_images)
        - [`_log_plot`](#ultralytics.utils.callbacks.neptune._log_plot)
        - [`on_pretrain_routine_start`](#ultralytics.utils.callbacks.neptune.on_pretrain_routine_start)
        - [`on_train_epoch_end`](#ultralytics.utils.callbacks.neptune.on_train_epoch_end)
        - [`on_fit_epoch_end`](#ultralytics.utils.callbacks.neptune.on_fit_epoch_end)
        - [`on_val_end`](#ultralytics.utils.callbacks.neptune.on_val_end)
        - [`on_train_end`](#ultralytics.utils.callbacks.neptune.on_train_end)


## Function `ultralytics.utils.callbacks.neptune._log_scalars` {#ultralytics.utils.callbacks.neptune.\_log\_scalars}

```python
def _log_scalars(scalars: dict, step: int = 0) -> None
```

Log scalars to the NeptuneAI experiment logger.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `scalars` | `dict` | Dictionary of scalar values to log to NeptuneAI. | *required* |
| `step` | `int, optional` | The current step or iteration number for logging. | `0` |

**Examples**

```python
>>> metrics = {"mAP": 0.85, "loss": 0.32}
>>> _log_scalars(metrics, step=100)
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/neptune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py#L20-L33"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.neptune._log_images` {#ultralytics.utils.callbacks.neptune.\_log\_images}

```python
def _log_images(imgs_dict: dict, group: str = "") -> None
```

Log images to the NeptuneAI experiment logger.

This function logs image data to Neptune.ai when a valid Neptune run is active. Images are organized under the specified group name.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `imgs_dict` | `dict` | Dictionary of images to log, with keys as image names and values as image data. | *required* |
| `group` | `str, optional` | Group name to organize images under in the Neptune UI. | `""` |

**Examples**

```python
>>> # Log validation images
>>> _log_images({"val_batch": img_tensor}, group="validation")
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/neptune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py#L36-L52"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.neptune._log_plot` {#ultralytics.utils.callbacks.neptune.\_log\_plot}

```python
def _log_plot(title: str, plot_path: str) -> None
```

Log plots to the NeptuneAI experiment logger.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `title` | `str` |  | *required* |
| `plot_path` | `str` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/neptune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py#L55-L64"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_plot(title: str, plot_path: str) -> None:
    """Log plots to the NeptuneAI experiment logger."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # no ticks
    ax.imshow(img)
    run[f"Plots/{title}"].upload(fig)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.neptune.on_pretrain_routine_start` {#ultralytics.utils.callbacks.neptune.on\_pretrain\_routine\_start}

```python
def on_pretrain_routine_start(trainer) -> None
```

Initialize NeptuneAI run and log hyperparameters before training starts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/neptune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py#L67-L78"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.neptune.on_train_epoch_end` {#ultralytics.utils.callbacks.neptune.on\_train\_epoch\_end}

```python
def on_train_epoch_end(trainer) -> None
```

Log training metrics and learning rate at the end of each training epoch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/neptune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py#L81-L86"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_epoch_end(trainer) -> None:
    """Log training metrics and learning rate at the end of each training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob("train_batch*.jpg")}, "Mosaic")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.neptune.on_fit_epoch_end` {#ultralytics.utils.callbacks.neptune.on\_fit\_epoch\_end}

```python
def on_fit_epoch_end(trainer) -> None
```

Log model info and validation metrics at the end of each fit epoch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/neptune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py#L89-L95"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_fit_epoch_end(trainer) -> None:
    """Log model info and validation metrics at the end of each fit epoch."""
    if run and trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        run["Configuration/Model"] = model_info_for_loggers(trainer)
    _log_scalars(trainer.metrics, trainer.epoch + 1)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.neptune.on_val_end` {#ultralytics.utils.callbacks.neptune.on\_val\_end}

```python
def on_val_end(validator) -> None
```

Log validation images at the end of validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `validator` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/neptune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py#L98-L102"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_val_end(validator) -> None:
    """Log validation images at the end of validation."""
    if run:
        # Log val_labels and val_pred
        _log_images({f.stem: str(f) for f in validator.save_dir.glob("val*.jpg")}, "Validation")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.neptune.on_train_end` {#ultralytics.utils.callbacks.neptune.on\_train\_end}

```python
def on_train_end(trainer) -> None
```

Log final results, plots, and model weights at the end of training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/neptune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/neptune.py#L105-L113"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_end(trainer) -> None:
    """Log final results, plots, and model weights at the end of training."""
    if run:
        # Log final results, CM matrix + PR plots
        for f in [*trainer.plots.keys(), *trainer.validator.plots.keys()]:
            if "batch" not in f.name:
                _log_plot(title=f.stem, plot_path=f)
        # Log the final model
        run[f"weights/{trainer.args.name or trainer.args.task}/{trainer.best.name}"].upload(File(str(trainer.best)))
```
</details>

<br><br>
