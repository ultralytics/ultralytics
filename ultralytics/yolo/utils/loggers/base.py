def before_train(trainer):
    # Initialize tensorboard logger
    pass


def on_epoch_start(trainer):
    pass


def on_batch_start(trainer):
    pass


def on_val_start(trainer):
    pass


def on_val_end(trainer):
    pass


def on_model_save(trainer):
    pass


default_callbacks = {
    "before_train": before_train,
    "on_epoch_start": on_epoch_start,
    "on_batch_start": on_batch_start,
    "on_val_start": on_val_start,
    "on_val_end": on_val_end,
    "on_model_save": on_model_save}
