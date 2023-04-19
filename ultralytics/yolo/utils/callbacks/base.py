# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Base callbacks
"""

from collections import defaultdict
from copy import deepcopy

# Trainer callbacks ----------------------------------------------------------------------------------------------------


def on_pretrain_routine_start(trainer):
    """Called before the pretraining routine starts."""
    pass


def on_pretrain_routine_end(trainer):
    """Called after the pretraining routine ends."""
    pass


def on_train_start(trainer):
    """Called when the training starts."""
    pass


def on_train_epoch_start(trainer):
    """Called at the start of each training epoch."""
    pass


def on_train_batch_start(trainer):
    """Called at the start of each training batch."""
    pass


def optimizer_step(trainer):
    """Called when the optimizer takes a step."""
    pass


def on_before_zero_grad(trainer):
    """Called before the gradients are set to zero."""
    pass


def on_train_batch_end(trainer):
    """Called at the end of each training batch."""
    pass


def on_train_epoch_end(trainer):
    """Called at the end of each training epoch."""
    pass


def on_fit_epoch_end(trainer):
    """Called at the end of each fit epoch (train + val)."""
    pass


def on_model_save(trainer):
    """Called when the model is saved."""
    pass


def on_train_end(trainer):
    """Called when the training ends."""
    pass


def on_params_update(trainer):
    """Called when the model parameters are updated."""
    pass


def teardown(trainer):
    """Called during the teardown of the training process."""
    pass


# Validator callbacks --------------------------------------------------------------------------------------------------


def on_val_start(validator):
    """Called when the validation starts."""
    pass


def on_val_batch_start(validator):
    """Called at the start of each validation batch."""
    pass


def on_val_batch_end(validator):
    """Called at the end of each validation batch."""
    pass


def on_val_end(validator):
    """Called when the validation ends."""
    pass


# Predictor callbacks --------------------------------------------------------------------------------------------------


def on_predict_start(predictor):
    """Called when the prediction starts."""
    pass


def on_predict_batch_start(predictor):
    """Called at the start of each prediction batch."""
    pass


def on_predict_batch_end(predictor):
    """Called at the end of each prediction batch."""
    pass


def on_predict_postprocess_end(predictor):
    """Called after the post-processing of the prediction ends."""
    pass


def on_predict_end(predictor):
    """Called when the prediction ends."""
    pass


# Exporter callbacks ---------------------------------------------------------------------------------------------------


def on_export_start(exporter):
    """Called when the model export starts."""
    pass


def on_export_end(exporter):
    """Called when the model export ends."""
    pass


default_callbacks = {
    # Run in trainer
    'on_pretrain_routine_start': [on_pretrain_routine_start],
    'on_pretrain_routine_end': [on_pretrain_routine_end],
    'on_train_start': [on_train_start],
    'on_train_epoch_start': [on_train_epoch_start],
    'on_train_batch_start': [on_train_batch_start],
    'optimizer_step': [optimizer_step],
    'on_before_zero_grad': [on_before_zero_grad],
    'on_train_batch_end': [on_train_batch_end],
    'on_train_epoch_end': [on_train_epoch_end],
    'on_fit_epoch_end': [on_fit_epoch_end],  # fit = train + val
    'on_model_save': [on_model_save],
    'on_train_end': [on_train_end],
    'on_params_update': [on_params_update],
    'teardown': [teardown],

    # Run in validator
    'on_val_start': [on_val_start],
    'on_val_batch_start': [on_val_batch_start],
    'on_val_batch_end': [on_val_batch_end],
    'on_val_end': [on_val_end],

    # Run in predictor
    'on_predict_start': [on_predict_start],
    'on_predict_batch_start': [on_predict_batch_start],
    'on_predict_postprocess_end': [on_predict_postprocess_end],
    'on_predict_batch_end': [on_predict_batch_end],
    'on_predict_end': [on_predict_end],

    # Run in exporter
    'on_export_start': [on_export_start],
    'on_export_end': [on_export_end]}


def get_default_callbacks():
    """
    Return a copy of the default_callbacks dictionary with lists as default values.

    Returns:
        (defaultdict): A defaultdict with keys from default_callbacks and empty lists as default values.
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """
    Add integration callbacks from various sources to the instance's callbacks.

    Args:
        instance (Trainer, Predictor, Validator, Exporter): An object with a 'callbacks' attribute that is a dictionary
            of callback lists.
    """
    from .clearml import callbacks as clearml_callbacks
    from .comet import callbacks as comet_callbacks
    from .hub import callbacks as hub_callbacks
    from .mlflow import callbacks as mf_callbacks
    from .raytune import callbacks as tune_callbacks
    from .tensorboard import callbacks as tb_callbacks
    from .wb import callbacks as wb_callbacks

    for x in clearml_callbacks, comet_callbacks, hub_callbacks, tb_callbacks, mf_callbacks, tune_callbacks, wb_callbacks:
        for k, v in x.items():
            if v not in instance.callbacks[k]:  # prevent duplicate callbacks addition
                instance.callbacks[k].append(v)  # callback[name].append(func)
