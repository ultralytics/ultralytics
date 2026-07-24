# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MLflow Logging for Ultralytics YOLO.

This module enables MLflow logging for Ultralytics YOLO. It logs metrics, parameters, and model artifacts.
For setting up, a tracking URI should be specified. The logging can be customized using environment variables.

Commands:
    1. To set a project name:
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument

    2. To set a run name:
        `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument

    3. To start a local MLflow server:
        mlflow server --backend-store-uri runs/mlflow
       It will by default start a local server at http://127.0.0.1:5000.
       To specify a different URI, set the MLFLOW_TRACKING_URI environment variable.

    4. To kill all running MLflow server instances:
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

import os
from pathlib import Path

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr, env_bool

PREFIX = colorstr("MLflow: ")

try:
    import mlflow

    assert hasattr(mlflow, "__version__")  # verify package is not a local directory
except (ImportError, AssertionError):
    mlflow = None


def sanitize_dict(x: dict) -> dict:
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


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
        MLFLOW_KEEP_RUN_ACTIVE: Whether to keep the MLflow run active after training ends. Truthy values are
            "1", "true", "yes", "on", "y", "t" (case-insensitive); anything else is False.
    """
    # Resolve enablement at call time (not import time) so test/training order can never permanently disable MLflow:
    # `add_integration_callbacks` imports this module on the first training, which may run with mlflow off.
    if not mlflow or SETTINGS["mlflow"] is not True:
        return
    if TESTS_RUNNING and "test_mlflow" not in os.environ.get("PYTEST_CURRENT_TEST", ""):
        return  # do not log during unrelated pytest tests

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")

    # Set experiment and run names
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/Ultralytics"
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name

    trainer._mlflow_active = False
    trainer._mlflow_started_run = False
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        mlflow.autolog()
        active_run = mlflow.active_run()
        if active_run is None:
            active_run = mlflow.start_run(run_name=run_name)
            trainer._mlflow_started_run = True
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
        trainer._mlflow_active = True
    except Exception as e:
        LOGGER.warning(f"{PREFIX}Failed to initialize: {e}")
        LOGGER.warning(f"{PREFIX}Not tracking this run")
        if trainer._mlflow_started_run:
            try:
                mlflow.end_run()
            except Exception:
                pass


def _log_metrics(trainer, metrics):
    """Log metrics to MLflow, disabling tracking for this run on failure so it never crashes training."""
    try:
        mlflow.log_metrics(metrics=metrics, step=trainer.epoch)
    except Exception as e:
        LOGGER.warning(f"{PREFIX}metric logging failed, disabling tracking for this run: {e}")
        trainer._mlflow_active = False


def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow and getattr(trainer, "_mlflow_active", False):
        _log_metrics(
            trainer,
            {
                **sanitize_dict(trainer.lr),
                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
        )


def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow and getattr(trainer, "_mlflow_active", False):
        _log_metrics(trainer, sanitize_dict(trainer.metrics))


def on_train_end(trainer):
    """Log model artifacts at the end of training and close any run this callback opened."""
    if not mlflow:
        return
    if getattr(trainer, "_mlflow_active", False):
        try:
            mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt
            for f in trainer.save_dir.glob("*"):  # log all other files in save_dir
                if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
                    mlflow.log_artifact(str(f))
            LOGGER.info(
                f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n{PREFIX}disable with 'yolo settings mlflow=False'"
            )
        except Exception as e:
            LOGGER.warning(f"{PREFIX}failed to log artifacts: {e}")
    if getattr(trainer, "_mlflow_started_run", False):  # only close a run we created
        if env_bool("MLFLOW_KEEP_RUN_ACTIVE"):
            LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")
        else:
            try:
                mlflow.end_run()
                LOGGER.debug(f"{PREFIX}mlflow run ended")
            except Exception:
                pass


callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if mlflow
    else {}
)
