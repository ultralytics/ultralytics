# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr

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
    global mlflow, run

    uri = os.environ.get('MLFLOW_TRACKING_URI')
    if uri is None:
        mlflow = None
        LOGGER.info(f'{PREFIX}installed but no tracking URI set, '
                    "i.e. run 'mlflow ui && MLFLOW_TRACKING_URI='http://127.0.0.1:5000'")

    else:
        LOGGER.info(f'{PREFIX} tracking uri: {uri}')
        mlflow.set_tracking_uri(uri)

        # Set experiment and run names
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or trainer.args.project or '/Shared/YOLOv8'
        run_name = os.environ.get('MLFLOW_RUN') or trainer.args.name
        experiment = mlflow.set_experiment(experiment_name)  # change since mlflow does this now by default

        mlflow.autolog()
        try:
            run, active_run = mlflow, mlflow.active_run()
            if not active_run:
                active_run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
            LOGGER.info(f'{PREFIX}Using run_id({active_run.info.run_id}) at {uri}')
            run.log_params(dict(trainer.args))
        except Exception as e:
            LOGGER.warning(f'{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n'
                           f'{PREFIX}Continuing without MLflow')


def on_fit_epoch_end(trainer):
    """Log training metrics to MLflow."""
    if mlflow:
        sanitized_metrics = {k.replace('(', '').replace(')', ''): float(v) for k, v in trainer.metrics.items()}
        run.log_metrics(metrics=sanitized_metrics, step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if mlflow:
        run.log_artifact(trainer.last)
        run.log_artifact(trainer.best)
        run.log_artifact(trainer.save_dir)
        mlflow.end_run()
        LOGGER.info(f'{PREFIX} ending run')


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if mlflow else {}
