---
description: Learn how to set up and customize MLflow logging for Ultralytics YOLO. Log metrics, parameters, and model artifacts easily.
keywords: MLflow, Ultralytics YOLO, logging, metrics, parameters, model artifacts, setup, tracking, customization
---

# Reference for `ultralytics/utils/callbacks/mlflow.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/mlflow.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/mlflow.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`sanitize_dict`](#ultralytics.utils.callbacks.mlflow.sanitize_dict)
        - [`on_pretrain_routine_end`](#ultralytics.utils.callbacks.mlflow.on_pretrain_routine_end)
        - [`on_train_epoch_end`](#ultralytics.utils.callbacks.mlflow.on_train_epoch_end)
        - [`on_fit_epoch_end`](#ultralytics.utils.callbacks.mlflow.on_fit_epoch_end)
        - [`on_train_end`](#ultralytics.utils.callbacks.mlflow.on_train_end)


## Function `ultralytics.utils.callbacks.mlflow.sanitize_dict` {#ultralytics.utils.callbacks.mlflow.sanitize\_dict}

```python
def sanitize_dict(x: dict) -> dict
```

Sanitize dictionary keys by removing parentheses and converting values to floats.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `dict` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/mlflow.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/mlflow.py#L42-L44"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def sanitize_dict(x: dict) -> dict:
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.mlflow.on_pretrain_routine_end` {#ultralytics.utils.callbacks.mlflow.on\_pretrain\_routine\_end}

```python
def on_pretrain_routine_end(trainer)
```

Log training parameters to MLflow at the end of the pretraining routine.

This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI, experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters from the trainer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `ultralytics.engine.trainer.BaseTrainer` | The training object with arguments and parameters to log. | *required* |

!!! note "Notes"

    MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
    MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
    MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
    MLFLOW_KEEP_RUN_ACTIVE: Boolean indicating whether to keep the MLflow run active after training ends.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/mlflow.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/mlflow.py#L47-L84"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_pretrain_routine_end(trainer):
    """Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters from
    the trainer.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.

    Notes:
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
        MLFLOW_KEEP_RUN_ACTIVE: Boolean indicating whether to keep the MLflow run active after training ends.
    """
    global mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/Ultralytics"
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f"{PREFIX}Failed to initialize: {e}")
        LOGGER.warning(f"{PREFIX}Not tracking this run")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.mlflow.on_train_epoch_end` {#ultralytics.utils.callbacks.mlflow.on\_train\_epoch\_end}

```python
def on_train_epoch_end(trainer)
```

Log training metrics at the end of each train epoch to MLflow.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/mlflow.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/mlflow.py#L87-L96"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(
            metrics={
                **sanitize_dict(trainer.lr),
                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
            step=trainer.epoch,
        )
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.mlflow.on_fit_epoch_end` {#ultralytics.utils.callbacks.mlflow.on\_fit\_epoch\_end}

```python
def on_fit_epoch_end(trainer)
```

Log training metrics at the end of each fit epoch to MLflow.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/mlflow.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/mlflow.py#L99-L102"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.mlflow.on_train_end` {#ultralytics.utils.callbacks.mlflow.on\_train\_end}

```python
def on_train_end(trainer)
```

Log model artifacts at the end of training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/mlflow.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/mlflow.py#L105-L122"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_end(trainer):
    """Log model artifacts at the end of training."""
    if not mlflow:
        return
    mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt
    for f in trainer.save_dir.glob("*"):  # log all other files in save_dir
        if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
            mlflow.log_artifact(str(f))
    keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    if keep_run_active:
        LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")
    else:
        mlflow.end_run()
        LOGGER.debug(f"{PREFIX}mlflow run ended")

    LOGGER.info(
        f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n{PREFIX}disable with 'yolo settings mlflow=False'"
    )
```
</details>

<br><br>
