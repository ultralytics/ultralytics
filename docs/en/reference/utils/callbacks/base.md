---
title: utils.callbacks.base API Reference
description: Discover the essential base callbacks in Ultralytics for training, validation, prediction, and exporting models efficiently.
keywords: Ultralytics, base callbacks, training, validation, prediction, model export, ML, machine learning, deep learning
---

# Reference for `ultralytics/utils/callbacks/base.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`on_pretrain_routine_start`](#ultralytics.utils.callbacks.base.on_pretrain_routine_start)
        - [`on_pretrain_routine_end`](#ultralytics.utils.callbacks.base.on_pretrain_routine_end)
        - [`on_train_start`](#ultralytics.utils.callbacks.base.on_train_start)
        - [`on_train_epoch_start`](#ultralytics.utils.callbacks.base.on_train_epoch_start)
        - [`on_train_batch_start`](#ultralytics.utils.callbacks.base.on_train_batch_start)
        - [`optimizer_step`](#ultralytics.utils.callbacks.base.optimizer_step)
        - [`on_before_zero_grad`](#ultralytics.utils.callbacks.base.on_before_zero_grad)
        - [`on_train_batch_end`](#ultralytics.utils.callbacks.base.on_train_batch_end)
        - [`on_train_epoch_end`](#ultralytics.utils.callbacks.base.on_train_epoch_end)
        - [`on_fit_epoch_end`](#ultralytics.utils.callbacks.base.on_fit_epoch_end)
        - [`on_model_save`](#ultralytics.utils.callbacks.base.on_model_save)
        - [`on_train_end`](#ultralytics.utils.callbacks.base.on_train_end)
        - [`on_params_update`](#ultralytics.utils.callbacks.base.on_params_update)
        - [`teardown`](#ultralytics.utils.callbacks.base.teardown)
        - [`on_val_start`](#ultralytics.utils.callbacks.base.on_val_start)
        - [`on_val_batch_start`](#ultralytics.utils.callbacks.base.on_val_batch_start)
        - [`on_val_batch_end`](#ultralytics.utils.callbacks.base.on_val_batch_end)
        - [`on_val_end`](#ultralytics.utils.callbacks.base.on_val_end)
        - [`on_predict_start`](#ultralytics.utils.callbacks.base.on_predict_start)
        - [`on_predict_batch_start`](#ultralytics.utils.callbacks.base.on_predict_batch_start)
        - [`on_predict_batch_end`](#ultralytics.utils.callbacks.base.on_predict_batch_end)
        - [`on_predict_postprocess_end`](#ultralytics.utils.callbacks.base.on_predict_postprocess_end)
        - [`on_predict_end`](#ultralytics.utils.callbacks.base.on_predict_end)
        - [`on_export_start`](#ultralytics.utils.callbacks.base.on_export_start)
        - [`on_export_end`](#ultralytics.utils.callbacks.base.on_export_end)
        - [`get_default_callbacks`](#ultralytics.utils.callbacks.base.get_default_callbacks)
        - [`add_integration_callbacks`](#ultralytics.utils.callbacks.base.add_integration_callbacks)


## Function `ultralytics.utils.callbacks.base.on_pretrain_routine_start` {#ultralytics.utils.callbacks.base.on\_pretrain\_routine\_start}

```python
def on_pretrain_routine_start(trainer)
```

Called at the beginning of the pre-training routine, before data loading and model setup.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L10-L11">View on GitHub</a>
```python
def on_pretrain_routine_start(trainer):
    """Called at the beginning of the pre-training routine, before data loading and model setup."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_pretrain_routine_end` {#ultralytics.utils.callbacks.base.on\_pretrain\_routine\_end}

```python
def on_pretrain_routine_end(trainer)
```

Called at the end of the pre-training routine, after data loading and model setup are complete.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L14-L15">View on GitHub</a>
```python
def on_pretrain_routine_end(trainer):
    """Called at the end of the pre-training routine, after data loading and model setup are complete."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_start` {#ultralytics.utils.callbacks.base.on\_train\_start}

```python
def on_train_start(trainer)
```

Called when the training starts, before the first epoch begins.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L18-L19">View on GitHub</a>
```python
def on_train_start(trainer):
    """Called when the training starts, before the first epoch begins."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_epoch_start` {#ultralytics.utils.callbacks.base.on\_train\_epoch\_start}

```python
def on_train_epoch_start(trainer)
```

Called at the start of each training epoch, before batch iteration begins.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L22-L23">View on GitHub</a>
```python
def on_train_epoch_start(trainer):
    """Called at the start of each training epoch, before batch iteration begins."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_batch_start` {#ultralytics.utils.callbacks.base.on\_train\_batch\_start}

```python
def on_train_batch_start(trainer)
```

Called at the start of each training batch, before the forward pass.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L26-L27">View on GitHub</a>
```python
def on_train_batch_start(trainer):
    """Called at the start of each training batch, before the forward pass."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.optimizer_step` {#ultralytics.utils.callbacks.base.optimizer\_step}

```python
def optimizer_step(trainer)
```

Called during the optimizer step. Reserved for custom integrations; not called by default.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L30-L31">View on GitHub</a>
```python
def optimizer_step(trainer):
    """Called during the optimizer step. Reserved for custom integrations; not called by default."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_before_zero_grad` {#ultralytics.utils.callbacks.base.on\_before\_zero\_grad}

```python
def on_before_zero_grad(trainer)
```

Called before the gradients are set to zero. Reserved for custom integrations; not called by default.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L34-L35">View on GitHub</a>
```python
def on_before_zero_grad(trainer):
    """Called before the gradients are set to zero. Reserved for custom integrations; not called by default."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_batch_end` {#ultralytics.utils.callbacks.base.on\_train\_batch\_end}

```python
def on_train_batch_end(trainer)
```

Called at the end of each training batch, after the backward pass. Optimizer step may be deferred by

accumulation.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L38-L41">View on GitHub</a>
```python
def on_train_batch_end(trainer):
    """Called at the end of each training batch, after the backward pass. Optimizer step may be deferred by
    accumulation.
    """
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_epoch_end` {#ultralytics.utils.callbacks.base.on\_train\_epoch\_end}

```python
def on_train_epoch_end(trainer)
```

Called at the end of each training epoch, after all batches but before validation.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L44-L45">View on GitHub</a>
```python
def on_train_epoch_end(trainer):
    """Called at the end of each training epoch, after all batches but before validation."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_fit_epoch_end` {#ultralytics.utils.callbacks.base.on\_fit\_epoch\_end}

```python
def on_fit_epoch_end(trainer)
```

Called at the end of each fit epoch (train + val), after validation and any checkpoint save.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L48-L49">View on GitHub</a>
```python
def on_fit_epoch_end(trainer):
    """Called at the end of each fit epoch (train + val), after validation and any checkpoint save."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_model_save` {#ultralytics.utils.callbacks.base.on\_model\_save}

```python
def on_model_save(trainer)
```

Called when the model checkpoint is saved, after validation.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L52-L53">View on GitHub</a>
```python
def on_model_save(trainer):
    """Called when the model checkpoint is saved, after validation."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_end` {#ultralytics.utils.callbacks.base.on\_train\_end}

```python
def on_train_end(trainer)
```

Called when the training ends, after final evaluation of the best model.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L56-L57">View on GitHub</a>
```python
def on_train_end(trainer):
    """Called when the training ends, after final evaluation of the best model."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_params_update` {#ultralytics.utils.callbacks.base.on\_params\_update}

```python
def on_params_update(trainer)
```

Called when the model parameters are updated. Reserved for custom integrations; not called by default.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L60-L61">View on GitHub</a>
```python
def on_params_update(trainer):
    """Called when the model parameters are updated. Reserved for custom integrations; not called by default."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.teardown` {#ultralytics.utils.callbacks.base.teardown}

```python
def teardown(trainer)
```

Called during the teardown of the training process.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L64-L65">View on GitHub</a>
```python
def teardown(trainer):
    """Called during the teardown of the training process."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_val_start` {#ultralytics.utils.callbacks.base.on\_val\_start}

```python
def on_val_start(validator)
```

Called when the validation starts.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L71-L72">View on GitHub</a>
```python
def on_val_start(validator):
    """Called when the validation starts."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_val_batch_start` {#ultralytics.utils.callbacks.base.on\_val\_batch\_start}

```python
def on_val_batch_start(validator)
```

Called at the start of each validation batch.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L75-L76">View on GitHub</a>
```python
def on_val_batch_start(validator):
    """Called at the start of each validation batch."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_val_batch_end` {#ultralytics.utils.callbacks.base.on\_val\_batch\_end}

```python
def on_val_batch_end(validator)
```

Called at the end of each validation batch.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L79-L80">View on GitHub</a>
```python
def on_val_batch_end(validator):
    """Called at the end of each validation batch."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_val_end` {#ultralytics.utils.callbacks.base.on\_val\_end}

```python
def on_val_end(validator)
```

Called when the validation ends.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L83-L84">View on GitHub</a>
```python
def on_val_end(validator):
    """Called when the validation ends."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_start` {#ultralytics.utils.callbacks.base.on\_predict\_start}

```python
def on_predict_start(predictor)
```

Called when the prediction starts.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L90-L91">View on GitHub</a>
```python
def on_predict_start(predictor):
    """Called when the prediction starts."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_batch_start` {#ultralytics.utils.callbacks.base.on\_predict\_batch\_start}

```python
def on_predict_batch_start(predictor)
```

Called at the start of each prediction batch.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L94-L95">View on GitHub</a>
```python
def on_predict_batch_start(predictor):
    """Called at the start of each prediction batch."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_batch_end` {#ultralytics.utils.callbacks.base.on\_predict\_batch\_end}

```python
def on_predict_batch_end(predictor)
```

Called at the end of each prediction batch.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L98-L99">View on GitHub</a>
```python
def on_predict_batch_end(predictor):
    """Called at the end of each prediction batch."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_postprocess_end` {#ultralytics.utils.callbacks.base.on\_predict\_postprocess\_end}

```python
def on_predict_postprocess_end(predictor)
```

Called after the post-processing of the prediction ends.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L102-L103">View on GitHub</a>
```python
def on_predict_postprocess_end(predictor):
    """Called after the post-processing of the prediction ends."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_end` {#ultralytics.utils.callbacks.base.on\_predict\_end}

```python
def on_predict_end(predictor)
```

Called when the prediction ends.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L106-L107">View on GitHub</a>
```python
def on_predict_end(predictor):
    """Called when the prediction ends."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_export_start` {#ultralytics.utils.callbacks.base.on\_export\_start}

```python
def on_export_start(exporter)
```

Called when the model export starts.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L113-L114">View on GitHub</a>
```python
def on_export_start(exporter):
    """Called when the model export starts."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_export_end` {#ultralytics.utils.callbacks.base.on\_export\_end}

```python
def on_export_end(exporter)
```

Called when the model export ends.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L117-L118">View on GitHub</a>
```python
def on_export_end(exporter):
    """Called when the model export ends."""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.get_default_callbacks` {#ultralytics.utils.callbacks.base.get\_default\_callbacks}

```python
def get_default_callbacks()
```

Get the default callbacks for Ultralytics training, validation, prediction, and export processes.

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Dictionary of default callbacks for various training events. Each key represents an event during the training process, and the corresponding value is a list of callback functions executed when that event occurs. |

**Examples**

```python
>>> callbacks = get_default_callbacks()
>>> print(list(callbacks.keys()))  # show all available callback events
['on_pretrain_routine_start', 'on_pretrain_routine_end', ...]
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L154-L167">View on GitHub</a>
```python
def get_default_callbacks():
    """Get the default callbacks for Ultralytics training, validation, prediction, and export processes.

    Returns:
        (dict): Dictionary of default callbacks for various training events. Each key represents an event during the
            training process, and the corresponding value is a list of callback functions executed when that
            event occurs.

    Examples:
        >>> callbacks = get_default_callbacks()
        >>> print(list(callbacks.keys()))  # show all available callback events
        ['on_pretrain_routine_start', 'on_pretrain_routine_end', ...]
    """
    return defaultdict(list, deepcopy(default_callbacks))
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.add_integration_callbacks` {#ultralytics.utils.callbacks.base.add\_integration\_callbacks}

```python
def add_integration_callbacks(instance)
```

Add integration callbacks to the instance's callbacks dictionary.

This function loads and adds analytics callbacks to every instance. Trainer instances also receive Platform and experiment logger callbacks for ClearML, Comet, DVC, MLflow, Neptune, Ray Tune, TensorBoard, and Weights & Biases.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `instance` | `Trainer \| Predictor \| Validator \| Exporter` | The object instance to which callbacks will be added. The type of instance determines which callbacks are loaded. | *required* |

**Examples**

```python
>>> from ultralytics.engine.trainer import BaseTrainer
>>> trainer = BaseTrainer()
>>> add_integration_callbacks(trainer)
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L170-L207">View on GitHub</a>
```python
def add_integration_callbacks(instance):
    """Add integration callbacks to the instance's callbacks dictionary.

    This function loads and adds analytics callbacks to every instance. Trainer instances also receive Platform and
    experiment logger callbacks for ClearML, Comet, DVC, MLflow, Neptune, Ray Tune, TensorBoard, and Weights & Biases.

    Args:
        instance (Trainer | Predictor | Validator | Exporter): The object instance to which callbacks will be added. The
            type of instance determines which callbacks are loaded.

    Examples:
        >>> from ultralytics.engine.trainer import BaseTrainer
        >>> trainer = BaseTrainer()
        >>> add_integration_callbacks(trainer)
    """
    from ultralytics.utils.events import callbacks as events_cb

    callbacks_list = [events_cb]

    # Load training callbacks
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb
        from .comet import callbacks as comet_cb
        from .dvc import callbacks as dvc_cb
        from .mlflow import callbacks as mlflow_cb
        from .neptune import callbacks as neptune_cb
        from .platform import callbacks as platform_cb
        from .raytune import callbacks as tune_cb
        from .tensorboard import callbacks as tb_cb
        from .wb import callbacks as wb_cb

        callbacks_list.extend([platform_cb, clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])

    # Add the callbacks to the callbacks dictionary
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
```
</details>

<br><br>
