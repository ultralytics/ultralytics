# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """
    Sends training metrics to Ray Tune at end of each epoch.

    This function checks if a Ray Tune session is active and reports the current training metrics along with the
    epoch number to Ray Tune's session.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The Ultralytics trainer object containing metrics and epochs.

    Examples:
        >>> # Called automatically by the Ultralytics training loop
        >>> on_fit_epoch_end(trainer)
    """
    if ray.train._internal.session.get_session():  # check if Ray Tune session is active
        metrics = trainer.metrics
        session.report({**metrics, **{"epoch": trainer.epoch + 1}})


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
