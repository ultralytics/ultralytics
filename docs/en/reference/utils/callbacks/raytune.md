---
description: Learn how to integrate Ray Tune with Ultralytics YOLO for efficient hyperparameter tuning and performance tracking.
keywords: Ultralytics, Ray Tune, hyperparameter tuning, YOLO, machine learning, deep learning, callbacks, integration, training metrics
---

# Reference for `ultralytics/utils/callbacks/raytune.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/raytune.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/raytune.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`on_fit_epoch_end`](#ultralytics.utils.callbacks.raytune.on_fit_epoch_end)


## Function `ultralytics.utils.callbacks.raytune.on_fit_epoch_end` {#ultralytics.utils.callbacks.raytune.on\_fit\_epoch\_end}

```python
def on_fit_epoch_end(trainer)
```

Report training metrics to Ray Tune at epoch end when a Ray session is active.

Captures metrics from the trainer object and sends them to Ray Tune with the current epoch number, enabling hyperparameter tuning optimization. Only executes when within an active Ray Tune session.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `ultralytics.engine.trainer.BaseTrainer` | The Ultralytics trainer object containing metrics and epochs. | *required* |

**Examples**

```python
>>> # Called automatically by the Ultralytics training loop
    >>> on_fit_epoch_end(trainer)

References:
    Ray Tune docs: https://docs.ray.io/en/latest/tune/index.html
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/raytune.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/raytune.py#L15-L33"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_fit_epoch_end(trainer):
    """Report training metrics to Ray Tune at epoch end when a Ray session is active.

    Captures metrics from the trainer object and sends them to Ray Tune with the current epoch number, enabling
    hyperparameter tuning optimization. Only executes when within an active Ray Tune session.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The Ultralytics trainer object containing metrics and epochs.

    Examples:
        >>> # Called automatically by the Ultralytics training loop
        >>> on_fit_epoch_end(trainer)

    References:
        Ray Tune docs: https://docs.ray.io/en/latest/tune/index.html
    """
    if ray.train._internal.session.get_session():  # check if Ray Tune session is active
        metrics = trainer.metrics
        session.report({**metrics, **{"epoch": trainer.epoch + 1}})
```
</details>

<br><br>
