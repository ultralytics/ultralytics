# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers


try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["swanlab"] is True  # verify integration is enabled
    import swanlab

    assert hasattr(swanlab, "__version__")
    _processed_plots = {}

except (ImportError, AssertionError):
    swanlab = None


def _log_plots(plots: dict, step: int, tag: str):
    """Record metric plotting and inference images"""
    image_list = []
    for (
        name,
        params,
    ) in plots.copy().items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            image_list.append(swanlab.Image(str(name), caption=name.stem))
            _processed_plots[name] = timestamp

    if image_list:
        swanlab.log({tag: image_list}, step=step)

def on_pretrain_routine_start(trainer):
    """Initialize the experiment logger"""
    swanlab.config["FRAMEWORK"] = "ultralytics"
    if swanlab.get_run() is None:
        swanlab.init(
            project=str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
            name=str(trainer.args.name).replace("/", "-"),
            config=vars(trainer.args),
        )


def on_fit_epoch_end(trainer):
    """Record metrics and plot (including training and validation) at the end of each epoch."""
    swanlab.log(trainer.metrics, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        swanlab.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)
     
        
def on_train_epoch_end(trainer):
    """Record metrics at the end of each training epoch (training only)"""
    swanlab.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    swanlab.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1, tag="Plots")


def on_train_end(self, trainer):
    """Finish training"""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1, tag="TrainEnd/ValPlots")
    _log_plots(trainer.plots, step=trainer.epoch + 1, tag="TrainEnd/Plots")
    swanlab.finish()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if swanlab
    else {}
)
