# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import re
from pathlib import Path

from ultralytics.yolo.utils import LOGGER, TESTS_RUNNING, colorstr

try:
    import mlflow

    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(mlflow, '__version__')  # verify package is not directory
except (ImportError, AssertionError):
    mlflow = None


def on_pretrain_routine_end(trainer):
    """Logs training parameters to MLflow."""
    global mlflow, run, run_id, experiment_name

    if os.environ.get('MLFLOW_TRACKING_URI') is None:
        mlflow = None

    if mlflow:
        mlflow_location = os.environ['MLFLOW_TRACKING_URI']  # "http://192.168.xxx.xxx:5000"
        mlflow.set_tracking_uri(mlflow_location)

        experiment_name = trainer.args.project or '/Shared/YOLOv8'
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        prefix = colorstr('MLFlow: ')
        try:
            run, active_run = mlflow, mlflow.active_run()
            if not active_run:
                active_run = mlflow.start_run(experiment_id=experiment.experiment_id)
            run_id = active_run.info.run_id
            LOGGER.info(f'{prefix}Using run_id({run_id}) at {mlflow_location}')
            run.log_params(vars(trainer.model.args))
        except Exception as err:
            LOGGER.error(f'{prefix}Failing init - {repr(err)}')
            LOGGER.warning(f'{prefix}Continuing without Mlflow')


def on_fit_epoch_end(trainer):
    """Logs training metrics to Mlflow."""
    if mlflow:
        metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
        run.log_metrics(metrics=metrics_dict, step=trainer.epoch)


def on_model_save(trainer):
    """Logs model and metrics to mlflow on save."""
    if mlflow:
        run.log_artifact(trainer.last)


def on_train_end(trainer):
    """Called at end of train loop to log model artifact info."""
    if mlflow:
        root_dir = Path(__file__).resolve().parents[3]
        run.log_artifact(trainer.best)
        model_uri = f'runs:/{run_id}/'
        run.register_model(model_uri, experiment_name)
        run.pyfunc.log_model(artifact_path=experiment_name,
                             code_path=[str(root_dir)],
                             artifacts={'model_path': str(trainer.save_dir)},
                             python_model=run.pyfunc.PythonModel())


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_model_save': on_model_save,
    'on_train_end': on_train_end} if mlflow else {}
