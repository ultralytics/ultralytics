# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Base callbacks
"""


# Trainer callbacks ----------------------------------------------------------------------------------------------------
def on_pretrain_routine_start(trainer):
    pass


def on_pretrain_routine_end(trainer):
    pass


def on_train_start(trainer):
    pass


def on_train_epoch_start(trainer):
    pass


def on_train_batch_start(trainer):
    pass


def optimizer_step(trainer):
    pass


def on_before_zero_grad(trainer):
    pass


def on_train_batch_end(trainer):
    pass


def on_train_epoch_end(trainer):
    pass


def on_fit_epoch_end(trainer):
    pass


def on_model_save(trainer):
    pass


def on_train_end(trainer):
    pass


def on_params_update(trainer):
    pass


def teardown(trainer):
    pass


# Validator callbacks --------------------------------------------------------------------------------------------------
def on_val_start(validator):
    pass


def on_val_batch_start(validator):
    pass


def on_val_batch_end(validator):
    pass


def on_val_end(validator):
    pass


# Predictor callbacks --------------------------------------------------------------------------------------------------
def on_predict_start(predictor):
    pass


def on_predict_batch_start(predictor):
    pass


def on_predict_batch_end(predictor):
    pass


def on_predict_postprocess_end(predictor):
    pass


def on_predict_end(predictor):
    pass


# Exporter callbacks ---------------------------------------------------------------------------------------------------
def on_export_start(exporter):
    pass


def on_export_end(exporter):
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


def add_integration_callbacks(instance):
    from .clearml import callbacks as clearml_callbacks
    from .comet import callbacks as comet_callbacks
    from .hub import callbacks as hub_callbacks
    from .tensorboard import callbacks as tb_callbacks

    for x in clearml_callbacks, comet_callbacks, hub_callbacks, tb_callbacks:
        for k, v in x.items():
            if v not in instance.callbacks[k]:  # prevent duplicate callbacks addition
                instance.callbacks[k].append(v)  # callback[name].append(func)
