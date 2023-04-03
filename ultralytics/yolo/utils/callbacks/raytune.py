
try:
    from ray import tune
    from ray.air import session
except (ImportError, AssertionError):
    tune = None

def on_fit_epoch_end(trainer):
    metrics = trainer.metrics
    metrics["epoch"] = trainer.epoch
    session.report(metrics)

callbacks = {
    'on_fit_epoch_end': on_fit_epoch_end,
    } if tune else {}
