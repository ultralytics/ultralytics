# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import re
from pathlib import Path

from ultralytics.yolo.utils import LOGGER, colorstr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]

try:
    import mlflow
    assert hasattr(mlflow, '__version__')
except (ImportError, AssertionError):
    mlflow = None


def on_pretrain_routine_end(trainer):

    global mlflow, _mlflow, _run_id, _expr_name

    if os.environ.get('MLFLOW_TRACKING_URI') is None:
        mlflow = None

    if mlflow:

        mlflow_location = os.environ['MLFLOW_TRACKING_URI']  # "http://192.168.xxx.xxx:5000"
        mlflow.set_tracking_uri(mlflow_location)

        _expr_name = 'QYOLOv8'
        experiment = mlflow.get_experiment_by_name(_expr_name)
        if experiment is None:
            mlflow.create_experiment(_expr_name)
        mlflow.set_experiment(_expr_name)

        prefix = colorstr('MLFlow: ')
        try:
            _mlflow, mlflow_active_run = mlflow, None if not mlflow else mlflow.start_run()
            if mlflow_active_run is not None:
                _run_id = mlflow_active_run.info.run_id
                LOGGER.info(f'{prefix}Using run_id({_run_id}) at {mlflow_location}')
        except Exception as err:
            LOGGER.error(f'{prefix}Failing init - {repr(err)}')
            LOGGER.warning(f'{prefix}Continuining without Mlflow')
            _mlflow = None
            mlflow_active_run = None

        _mlflow.log_params(vars(trainer.model.args))


def on_fit_epoch_end(trainer):
    if mlflow:
        metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
        _mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)


def on_model_save(trainer):

    if mlflow:
        # Save last model
        _mlflow.log_artifact(trainer.last)


def on_train_end(trainer):

    if mlflow:
        # Save best model
        _mlflow.log_artifact(trainer.best)
        model_uri = f'runs:/{_run_id}/'

        # Save model
        _mlflow.register_model(model_uri, _expr_name)

        _mlflow.pyfunc.log_model(artifact_path=_expr_name,
                                 code_path=[str(ROOT.resolve())],
                                 artifacts={'model_path': str(trainer.save_dir)},
                                 python_model=_mlflow.pyfunc.PythonModel())


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_model_save': on_model_save,
    'on_train_end': on_train_end}
