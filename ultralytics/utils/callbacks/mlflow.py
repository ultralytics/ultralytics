# Ultralytics YOLO 🚀, AGPL-3.0 license
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

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr

try:
    import os

    assert not TESTS_RUNNING or 'test_mlflow' in os.environ.get('PYTEST_CURRENT_TEST', '')  # do not log pytest
    assert SETTINGS['mlflow'] is True  # verify integration is enabled
    import mlflow

    assert hasattr(mlflow, '__version__')  # verify package is not directory
    from pathlib import Path
    PREFIX = colorstr('MLflow: ')

except (ImportError, AssertionError):
    mlflow = None


def on_pretrain_routine_end(trainer):
    """
    Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters
    from the trainer.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.

    Global:
        mlflow: The imported mlflow module to use for logging.

    Environment Variables:
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
    """
    global mlflow

    # The tracking uri need file: in front of it if logging to a local directory
    uri = os.environ.get('MLFLOW_TRACKING_URI') or 'file:' + str(RUNS_DIR / 'mlflow')
    LOGGER.debug(f'{PREFIX} tracking uri: {uri}')
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or trainer.args.project or '/Shared/YOLOv8'
    run_name = os.environ.get('MLFLOW_RUN') or trainer.args.name
    # Keep experiments in the Shared workspace folder on Databricks
    if uri == "databricks":
        workspace_path = os.sep + "Workspace"
        exp_dir = os.sep + os.sep.join(["Shared", "mlflow"])
        os.makedirs("".join([workspace_path, exp_dir]), exist_ok=True)
        experiment_name = os.sep.join([exp_dir, experiment_name])
    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f'{PREFIX}logging run_id({active_run.info.run_id}) to {uri}')
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n'
                       f'{PREFIX}WARNING ⚠️ Not tracking this run')


def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        sanitized_metrics = {k.replace('(', '').replace(')', ''): float(v) for k, v in trainer.metrics.items()}
        mlflow.log_metrics(metrics=sanitized_metrics, step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if mlflow:

        # The wrapper is required for pyfunc.log_model() to log data properly
        # even if we would not use the wrapper for predictions
        class YOLOWrapper(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                from ultralytics import YOLO
                self.model = YOLO(os.path.join(context.artifacts["model_path"], "weights", "best.pt"))

            def predict(self, context, model_input):
                """
                Args:
                    context:
                    model_input: The input to the model. The last element has to be a dictionary containing YOLO predict arguments

                Returns:

                """
                return self.model.predict(model_input[:-1], **model_input[-1])

        # log_model is preferred over log_artifacts as it also saves
        # additional information required for making predictions using a
        # logged model in the future. This information includes:
        # - automatically generated conda.yaml and requirements.txt files.
        # - the Databricks runtime version used.
        # - a copy of the source code used when the model was serialized.
        root_dir = Path(__file__).resolve().parents[2]
        mlflow.pyfunc.log_model(artifact_path="model",
                                code_path=[str(root_dir)],
                                artifacts={'model_path': str(trainer.save_dir) + os.path.sep},
                                python_model=YOLOWrapper()
                                )

        mlflow.end_run()
        LOGGER.info(f'{PREFIX}results logged to {mlflow.get_tracking_uri()}\n'
                    f"{PREFIX}disable with 'yolo settings mlflow=False'")


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if mlflow else {}
