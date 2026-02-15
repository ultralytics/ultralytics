---
description: Discover the essential base callbacks in Ultralytics for training, validation, prediction, and exporting models efficiently.
keywords: Ultralytics, base callbacks, training, validation, prediction, model export, ML, machine learning, deep learning
---

# Reference for `ultralytics/utils/callbacks/base.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

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

Called before the pretraining routine starts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L10-L12"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_pretrain_routine_start(trainer):
    """Called before the pretraining routine starts."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_pretrain_routine_end` {#ultralytics.utils.callbacks.base.on\_pretrain\_routine\_end}

```python
def on_pretrain_routine_end(trainer)
```

Called after the pretraining routine ends.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L15-L17"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_pretrain_routine_end(trainer):
    """Called after the pretraining routine ends."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_start` {#ultralytics.utils.callbacks.base.on\_train\_start}

```python
def on_train_start(trainer)
```

Called when the training starts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L20-L22"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_start(trainer):
    """Called when the training starts."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_epoch_start` {#ultralytics.utils.callbacks.base.on\_train\_epoch\_start}

```python
def on_train_epoch_start(trainer)
```

Called at the start of each training epoch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L25-L27"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_epoch_start(trainer):
    """Called at the start of each training epoch."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_batch_start` {#ultralytics.utils.callbacks.base.on\_train\_batch\_start}

```python
def on_train_batch_start(trainer)
```

Called at the start of each training batch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L30-L32"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_batch_start(trainer):
    """Called at the start of each training batch."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.optimizer_step` {#ultralytics.utils.callbacks.base.optimizer\_step}

```python
def optimizer_step(trainer)
```

Called when the optimizer takes a step.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L35-L37"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def optimizer_step(trainer):
    """Called when the optimizer takes a step."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_before_zero_grad` {#ultralytics.utils.callbacks.base.on\_before\_zero\_grad}

```python
def on_before_zero_grad(trainer)
```

Called before the gradients are set to zero.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L40-L42"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_before_zero_grad(trainer):
    """Called before the gradients are set to zero."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_batch_end` {#ultralytics.utils.callbacks.base.on\_train\_batch\_end}

```python
def on_train_batch_end(trainer)
```

Called at the end of each training batch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L45-L47"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_batch_end(trainer):
    """Called at the end of each training batch."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_epoch_end` {#ultralytics.utils.callbacks.base.on\_train\_epoch\_end}

```python
def on_train_epoch_end(trainer)
```

Called at the end of each training epoch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L50-L52"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_epoch_end(trainer):
    """Called at the end of each training epoch."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_fit_epoch_end` {#ultralytics.utils.callbacks.base.on\_fit\_epoch\_end}

```python
def on_fit_epoch_end(trainer)
```

Called at the end of each fit epoch (train + val).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L55-L57"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_fit_epoch_end(trainer):
    """Called at the end of each fit epoch (train + val)."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_model_save` {#ultralytics.utils.callbacks.base.on\_model\_save}

```python
def on_model_save(trainer)
```

Called when the model is saved.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L60-L62"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_model_save(trainer):
    """Called when the model is saved."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_train_end` {#ultralytics.utils.callbacks.base.on\_train\_end}

```python
def on_train_end(trainer)
```

Called when the training ends.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L65-L67"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_end(trainer):
    """Called when the training ends."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_params_update` {#ultralytics.utils.callbacks.base.on\_params\_update}

```python
def on_params_update(trainer)
```

Called when the model parameters are updated.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L70-L72"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_params_update(trainer):
    """Called when the model parameters are updated."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.teardown` {#ultralytics.utils.callbacks.base.teardown}

```python
def teardown(trainer)
```

Called during the teardown of the training process.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L75-L77"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def teardown(trainer):
    """Called during the teardown of the training process."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_val_start` {#ultralytics.utils.callbacks.base.on\_val\_start}

```python
def on_val_start(validator)
```

Called when the validation starts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `validator` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L83-L85"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_val_start(validator):
    """Called when the validation starts."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_val_batch_start` {#ultralytics.utils.callbacks.base.on\_val\_batch\_start}

```python
def on_val_batch_start(validator)
```

Called at the start of each validation batch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `validator` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L88-L90"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_val_batch_start(validator):
    """Called at the start of each validation batch."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_val_batch_end` {#ultralytics.utils.callbacks.base.on\_val\_batch\_end}

```python
def on_val_batch_end(validator)
```

Called at the end of each validation batch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `validator` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L93-L95"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_val_batch_end(validator):
    """Called at the end of each validation batch."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_val_end` {#ultralytics.utils.callbacks.base.on\_val\_end}

```python
def on_val_end(validator)
```

Called when the validation ends.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `validator` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L98-L100"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_val_end(validator):
    """Called when the validation ends."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_start` {#ultralytics.utils.callbacks.base.on\_predict\_start}

```python
def on_predict_start(predictor)
```

Called when the prediction starts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predictor` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L106-L108"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_predict_start(predictor):
    """Called when the prediction starts."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_batch_start` {#ultralytics.utils.callbacks.base.on\_predict\_batch\_start}

```python
def on_predict_batch_start(predictor)
```

Called at the start of each prediction batch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predictor` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L111-L113"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_predict_batch_start(predictor):
    """Called at the start of each prediction batch."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_batch_end` {#ultralytics.utils.callbacks.base.on\_predict\_batch\_end}

```python
def on_predict_batch_end(predictor)
```

Called at the end of each prediction batch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predictor` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L116-L118"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_predict_batch_end(predictor):
    """Called at the end of each prediction batch."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_postprocess_end` {#ultralytics.utils.callbacks.base.on\_predict\_postprocess\_end}

```python
def on_predict_postprocess_end(predictor)
```

Called after the post-processing of the prediction ends.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predictor` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L121-L123"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_predict_postprocess_end(predictor):
    """Called after the post-processing of the prediction ends."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_predict_end` {#ultralytics.utils.callbacks.base.on\_predict\_end}

```python
def on_predict_end(predictor)
```

Called when the prediction ends.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predictor` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L126-L128"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_predict_end(predictor):
    """Called when the prediction ends."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_export_start` {#ultralytics.utils.callbacks.base.on\_export\_start}

```python
def on_export_start(exporter)
```

Called when the model export starts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `exporter` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L134-L136"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_export_start(exporter):
    """Called when the model export starts."""
    pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.base.on_export_end` {#ultralytics.utils.callbacks.base.on\_export\_end}

```python
def on_export_end(exporter)
```

Called when the model export ends.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `exporter` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L139-L141"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_export_end(exporter):
    """Called when the model export ends."""
    pass
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
| `dict` | Dictionary of default callbacks for various training events. Each key represents an event during the |

**Examples**

```python
>>> callbacks = get_default_callbacks()
>>> print(list(callbacks.keys()))  # show all available callback events
['on_pretrain_routine_start', 'on_pretrain_routine_end', ...]
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L177-L190"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

This function loads and adds various integration callbacks to the provided instance. The specific callbacks added depend on the type of instance provided. All instances receive HUB callbacks, while Trainer instances also receive additional callbacks for various integrations like ClearML, Comet, DVC, MLflow, Neptune, Ray Tune, TensorBoard, and Weights & Biases.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `instance` | `Trainer | Predictor | Validator | Exporter` | The object instance to which callbacks will be added. The<br>    type of instance determines which callbacks are loaded. | *required* |

**Examples**

```python
>>> from ultralytics.engine.trainer import BaseTrainer
>>> trainer = BaseTrainer()
>>> add_integration_callbacks(trainer)
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py#L193-L233"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def add_integration_callbacks(instance):
    """Add integration callbacks to the instance's callbacks dictionary.

    This function loads and adds various integration callbacks to the provided instance. The specific callbacks added
    depend on the type of instance provided. All instances receive HUB callbacks, while Trainer instances also receive
    additional callbacks for various integrations like ClearML, Comet, DVC, MLflow, Neptune, Ray Tune, TensorBoard, and
    Weights & Biases.

    Args:
        instance (Trainer | Predictor | Validator | Exporter): The object instance to which callbacks will be added. The
            type of instance determines which callbacks are loaded.

    Examples:
        >>> from ultralytics.engine.trainer import BaseTrainer
        >>> trainer = BaseTrainer()
        >>> add_integration_callbacks(trainer)
    """
    from .hub import callbacks as hub_cb
    from .platform import callbacks as platform_cb

    # Load Ultralytics callbacks
    callbacks_list = [hub_cb, platform_cb]

    # Load training callbacks
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb
        from .comet import callbacks as comet_cb
        from .dvc import callbacks as dvc_cb
        from .mlflow import callbacks as mlflow_cb
        from .neptune import callbacks as neptune_cb
        from .raytune import callbacks as tune_cb
        from .tensorboard import callbacks as tb_cb
        from .wb import callbacks as wb_cb

        callbacks_list.extend([clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])

    # Add the callbacks to the callbacks dictionary
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
```
</details>

<br><br>
