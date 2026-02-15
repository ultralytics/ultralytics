---
description: Explore detailed guides on Ultralytics callbacks, including pretrain, model save, train start/end, and more. Enhance your ML training workflows with ease.
keywords: Ultralytics, callbacks, pretrain, model save, train start, train end, validation, predict, export, training, machine learning
---

# Reference for `ultralytics/utils/callbacks/hub.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`on_pretrain_routine_start`](#ultralytics.utils.callbacks.hub.on_pretrain_routine_start)
        - [`on_pretrain_routine_end`](#ultralytics.utils.callbacks.hub.on_pretrain_routine_end)
        - [`on_fit_epoch_end`](#ultralytics.utils.callbacks.hub.on_fit_epoch_end)
        - [`on_model_save`](#ultralytics.utils.callbacks.hub.on_model_save)
        - [`on_train_end`](#ultralytics.utils.callbacks.hub.on_train_end)
        - [`on_train_start`](#ultralytics.utils.callbacks.hub.on_train_start)
        - [`on_val_start`](#ultralytics.utils.callbacks.hub.on_val_start)
        - [`on_predict_start`](#ultralytics.utils.callbacks.hub.on_predict_start)
        - [`on_export_start`](#ultralytics.utils.callbacks.hub.on_export_start)


## Function `ultralytics.utils.callbacks.hub.on_pretrain_routine_start` {#ultralytics.utils.callbacks.hub.on\_pretrain\_routine\_start}

```python
def on_pretrain_routine_start(trainer)
```

Create a remote Ultralytics HUB session to log local model training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L11-L14"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_pretrain_routine_start(trainer):
    """Create a remote Ultralytics HUB session to log local model training."""
    if RANK in {-1, 0} and SETTINGS["hub"] is True and SETTINGS["api_key"] and trainer.hub_session is None:
        trainer.hub_session = HUBTrainingSession.create_session(trainer.args.model, trainer.args)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.hub.on_pretrain_routine_end` {#ultralytics.utils.callbacks.hub.on\_pretrain\_routine\_end}

```python
def on_pretrain_routine_end(trainer)
```

Initialize timers for upload rate limiting before training begins.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L17-L21"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_pretrain_routine_end(trainer):
    """Initialize timers for upload rate limiting before training begins."""
    if session := getattr(trainer, "hub_session", None):
        # Start timer for upload rate limit
        session.timers = {"metrics": time(), "ckpt": time()}  # start timer for session rate limiting
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.hub.on_fit_epoch_end` {#ultralytics.utils.callbacks.hub.on\_fit\_epoch\_end}

```python
def on_fit_epoch_end(trainer)
```

Upload training progress metrics to Ultralytics HUB at the end of each epoch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L24-L46"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_fit_epoch_end(trainer):
    """Upload training progress metrics to Ultralytics HUB at the end of each epoch."""
    if session := getattr(trainer, "hub_session", None):
        # Upload metrics after validation ends
        all_plots = {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
        }
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            all_plots = {**all_plots, **model_info_for_loggers(trainer)}

        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)

        # If any metrics failed to upload previously, add them to the queue to attempt uploading again
        if session.metrics_upload_failed_queue:
            session.metrics_queue.update(session.metrics_upload_failed_queue)

        if time() - session.timers["metrics"] > session.rate_limits["metrics"]:
            session.upload_metrics()
            session.timers["metrics"] = time()  # reset timer
            session.metrics_queue = {}  # reset queue
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.hub.on_model_save` {#ultralytics.utils.callbacks.hub.on\_model\_save}

```python
def on_model_save(trainer)
```

Upload model checkpoints to Ultralytics HUB with rate limiting.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L49-L57"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_model_save(trainer):
    """Upload model checkpoints to Ultralytics HUB with rate limiting."""
    if session := getattr(trainer, "hub_session", None):
        # Upload checkpoints with rate limiting
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.timers["ckpt"] > session.rate_limits["ckpt"]:
            LOGGER.info(f"{PREFIX}Uploading checkpoint {HUB_WEB_ROOT}/models/{session.model.id}")
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.timers["ckpt"] = time()  # reset timer
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.hub.on_train_end` {#ultralytics.utils.callbacks.hub.on\_train\_end}

```python
def on_train_end(trainer)
```

Upload final model and metrics to Ultralytics HUB at the end of training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L60-L72"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_end(trainer):
    """Upload final model and metrics to Ultralytics HUB at the end of training."""
    if session := getattr(trainer, "hub_session", None):
        # Upload final model and metrics with exponential standoff
        LOGGER.info(f"{PREFIX}Syncing final model...")
        session.upload_model(
            trainer.epoch,
            trainer.best,
            map=trainer.metrics.get("metrics/mAP50-95(B)", 0),
            final=True,
        )
        session.alive = False  # stop heartbeats
        LOGGER.info(f"{PREFIX}Done ✅\n{PREFIX}View model at {session.model_url} 🚀")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.hub.on_train_start` {#ultralytics.utils.callbacks.hub.on\_train\_start}

```python
def on_train_start(trainer)
```

Run events on train start.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L75-L77"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_start(trainer):
    """Run events on train start."""
    events(trainer.args, trainer.device)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.hub.on_val_start` {#ultralytics.utils.callbacks.hub.on\_val\_start}

```python
def on_val_start(validator)
```

Run events on validation start.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `validator` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L80-L83"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_val_start(validator):
    """Run events on validation start."""
    if not validator.training:
        events(validator.args, validator.device)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.hub.on_predict_start` {#ultralytics.utils.callbacks.hub.on\_predict\_start}

```python
def on_predict_start(predictor)
```

Run events on predict start.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predictor` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L86-L88"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_predict_start(predictor):
    """Run events on predict start."""
    events(predictor.args, predictor.device)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.hub.on_export_start` {#ultralytics.utils.callbacks.hub.on\_export\_start}

```python
def on_export_start(exporter)
```

Run events on export start.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `exporter` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/hub.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/hub.py#L91-L93"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_export_start(exporter):
    """Run events on export start."""
    events(exporter.args, exporter.device)
```
</details>

<br><br>
