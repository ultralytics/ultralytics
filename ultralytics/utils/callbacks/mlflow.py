# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
MLflow Logging for Ultralytics YOLO
===================================

This module enables MLflow logging for Ultralytics YOLO. It logs metrics, parameters, and model artifacts.
For setting up, a tracking URI should be specified. The logging can be customized using environment variables.

Commands:
---------
1. To set a project name:
    `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument

2. To set a run name:
    `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument

3. To start a local MLflow server:
    mlflow ui
   It will by default start a local server at http://127.0.0.1:5000.
   To specify a different URI, set the MLFLOW_TRACKING_URI environment variable.

4. To kill all running MLflow server instances:
    ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

from pathlib import Path

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS['mlflow'] is True  # verify integration is enabled
    import mlflow

    assert hasattr(mlflow, '__version__')  # verify package is not directory
    PREFIX = colorstr('MLflow: ')
    import os

except (ImportError, AssertionError):
    mlflow = None


def on_pretrain_routine_end(trainer):
    """Log training parameters to MLflow."""
    global mlflow

    uri = os.environ.get('MLFLOW_TRACKING_URI') or str(RUNS_DIR / 'mlflow')
    LOGGER.debug(f'{PREFIX} tracking uri: {uri}')
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or trainer.args.project or '/Shared/YOLOv8'
    run_name = os.environ.get('MLFLOW_RUN') or trainer.args.name
    experiment = mlflow.set_experiment(experiment_name)  # change since mlflow does this now by default

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
        LOGGER.info(f'{PREFIX}logging run_id({active_run.info.run_id}) to {uri}')
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n'
                       f'{PREFIX}WARNING ⚠️ Not tracking this run')


def on_fit_epoch_end(trainer):
    """Log training metrics to MLflow."""
    if mlflow:
        sanitized_metrics = {k.replace('(', '').replace(')', ''): float(v) for k, v in trainer.metrics.items()}
        mlflow.log_metrics(metrics=sanitized_metrics, step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if mlflow:
        mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt
        for f in trainer.save_dir.glob('*'):  # log all other files in save_dir
            if f.suffix in {'.png', '.jpg', '.csv', '.pt', '.yaml'}:
                mlflow.log_artifact(str(f))

        mlflow.end_run()
        LOGGER.info(f'{PREFIX}results logged to {mlflow.get_tracking_uri()}\n'
                    f"{PREFIX}disable with 'yolo settings mlflow=False'")


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if mlflow else {}
