def on_pretrain_routine_start(trainer):
    trainer.force_stop_flag = False#添加一个强行停止的标志
    trainer.state = {}#添加一个状态字典
    trainer.state.update({"state":"Builds dataloaders and optimizer on correct rank process"})

def on_pretrain_routine_end(trainer):
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"Builds dataloaders and optimizer end"})

def on_train_start(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"train start"})

def on_train_epoch_start(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"train epoch start"})

def on_train_batch_start(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"train batch start"})

def optimizer_step(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"optimizer step"})

def on_before_zero_grad(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"before zero grad"})

def on_train_batch_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    trainer.state.update({"state":"train batch end"})

def on_train_epoch_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"train epoch end"})

def on_fit_epoch_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"fit epoch end"})
    if hasattr(trainer,'force_stop_flag') and trainer.force_stop_flag:
        trainer.stop = trainer.force_stop_flag

def on_model_save(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"model save"})

def on_train_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"train end"})

def on_params_update(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"params update"})

def teardown(trainer):
    """Logs epoch metrics at end of training epoch."""
    if hasattr(trainer,'state'):
        trainer.state.update({"state":"teardown"})


callbacks = (
    {
        "on_pretrain_routine_start": [on_pretrain_routine_start],
        "on_pretrain_routine_end": [on_pretrain_routine_end],
        "on_train_start": [on_train_start],
        "on_train_epoch_start": [on_train_epoch_start],
        "on_train_batch_start": [on_train_batch_start],
        "optimizer_step": [optimizer_step],
        "on_before_zero_grad": [on_before_zero_grad],
        "on_train_batch_end": [on_train_batch_end],
        "on_train_epoch_end": [on_train_epoch_end],
        "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
        "on_model_save": [on_model_save],
        "on_train_end": [on_train_end],
        "on_params_update": [on_params_update],
        "teardown": [teardown],
    }
)