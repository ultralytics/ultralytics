# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import deepcopy
from typing import Union

import torch

from ultralytics.models.sam.predict import Predictor as SAMPredictor
from ultralytics.models.yolo.classify import ClassificationPredictor, ClassificationTrainer, ClassificationValidator
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.pose import PosePredictor, PoseTrainer, PoseValidator
from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationTrainer, SegmentationValidator
from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

PREDICTOR_DTYPE = Union[DetectionPredictor, ClassificationPredictor, PosePredictor, SegmentationPredictor, SAMPredictor]
TRAINER_DTYPE = Union[DetectionTrainer, SegmentationTrainer, PoseTrainer, ClassificationTrainer]
VALIDATOR_DTYPE = Union[DetectionValidator, SegmentationValidator, PoseValidator, ClassificationValidator]

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS['wandb'] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, '__version__')  # verify package is not directory

    import numpy as np
    import pandas as pd

    from ultralytics.utils.callbacks.wb_utils.bbox import plot_bbox_predictions, plot_detection_validation_results
    from ultralytics.utils.callbacks.wb_utils.classification import (plot_classification_predictions,
                                                                     plot_classification_validation_results)
    from ultralytics.utils.callbacks.wb_utils.pose import plot_pose_predictions, plot_pose_validation_results
    from ultralytics.utils.callbacks.wb_utils.segment import (plot_mask_predictions, plot_sam_predictions,
                                                              plot_segmentation_validation_results)

    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def _custom_table(x, y, classes, title='Precision Recall Curve', x_title='Recall', y_title='Precision'):
    """
    Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of wandb's default precision-recall curve
    while allowing for enhanced customization. The visual metric is useful for monitoring model performance across different classes.

    Args:
        x (List): Values for the x-axis; expected to have length N.
        y (List): Corresponding values for the y-axis; also expected to have length N.
        classes (List): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot; defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis; defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis; defaults to 'Precision'.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    df = pd.DataFrame({'class': classes, 'y': y, 'x': x}).round(3)
    fields = {'x': 'x', 'y': 'y', 'class': 'class'}
    string_fields = {'title': title, 'x-axis-title': x_title, 'y-axis-title': y_title}
    return wb.plot_table('wandb/area-under-curve/v0',
                         wb.Table(dataframe=df),
                         fields=fields,
                         string_fields=string_fields)


def _plot_curve(x,
                y,
                names=None,
                id='precision-recall',
                title='Precision Recall Curve',
                x_title='Recall',
                y_title='Precision',
                num_x=100,
                only_mean=False):
    """
    Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb.
    The curve can represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape CxN, where C represents the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C. Defaults to an empty list.
        id (str, optional): Unique identifier for the logged data in wandb. Defaults to 'precision-recall'.
        title (str, optional): Title for the visualization plot. Defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis. Defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis. Defaults to 'Precision'.
        num_x (int, optional): Number of interpolated data points for visualization. Defaults to 100.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted. Defaults to True.

    Note:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ['mean'] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # add new x
            y_log.extend(np.interp(x_new, x, yi))  # interpolate y to new x
            classes.extend([names[i]] * len(x_new))  # add class names
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    """Logs plots from the input dictionary if they haven't been logged already at the specified step."""
    for name, params in plots.items():
        timestamp = params['timestamp']
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


class WandBCallbackState:

    def __init__(self):
        self.wandb_train_val_table = None
        self.wandb_prediction_table = None
        self.wandb_validation_table = None
        self.predictor = None
        self.mode = None
        self.prompts = None
        self.predictor_dict = {
            'detect': DetectionPredictor,
            'segment': SegmentationPredictor,
            'pose': PosePredictor,
            'classify': ClassificationPredictor, }

    def on_pretrain_routine_start(self, trainer: TRAINER_DTYPE):
        """Initiate and start project if module is present."""
        self.mode = trainer.args.mode
        wb.run or wb.init(project=trainer.args.project or 'YOLOv8',
                          name=trainer.args.name,
                          config=vars(trainer.args),
                          job_type='train_' + trainer.args.task)
        if trainer.args.task in ['detect', 'segment']:
            self.wandb_train_val_table = wb.Table(columns=[
                'Model-Name',
                'Epoch',
                'Data-Index',
                'Batch-Index',
                'Image',
                'Mean-Confidence',
                'Speed', ])
        elif trainer.args.task == 'pose':
            self.wandb_train_val_table = wb.Table(columns=[
                'Model-Name',
                'Epoch',
                'Data-Index',
                'Batch-Index',
                'Image-Ground-Truth',
                'Image-Prediction',
                'Num-Instances',
                'Mean-Confidence',
                'Speed', ])
        elif trainer.args.task == 'classify':
            self.wandb_train_val_table = wb.Table(columns=[
                'Model-Name',
                'Epoch',
                'Data-Index',
                'Batch-Index',
                'Image',
                'Ground-Truth-Category',
                'Predicted-Category',
                'Prediction-Confidence',
                'Top-5-Prediction-Categories',
                'Top-5-Prediction-Confindence',
                'Probabilities',
                'Speed', ])

    @torch.no_grad()
    def on_fit_epoch_end(self, trainer: TRAINER_DTYPE):
        wb.run.log(trainer.metrics, step=trainer.epoch + 1)
        _log_plots(trainer.plots, step=trainer.epoch + 1)
        _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
        if trainer.epoch == 0:
            wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)
        if isinstance(trainer, DetectionTrainer):
            dataloader = trainer.validator.dataloader
            class_label_map = trainer.validator.names
            overrides = trainer.args
            overrides.conf = 0.1
            if self.predictor is None:
                self.predictor = self.predictor_dict[trainer.args.task](overrides=overrides)
                self.predictor.callbacks = {}
                self.predictor.args.save = False
                self.predictor.args.save_txt = False
                self.predictor.args.save_crop = False
                self.predictor.args.verbose = None
            if trainer.args.task == 'detect':
                self.wandb_train_val_table = plot_detection_validation_results(
                    dataloader=dataloader,
                    class_label_map=class_label_map,
                    model_name=trainer.args.model,
                    predictor=self.predictor,
                    table=self.wandb_train_val_table,
                    max_validation_batches=1,
                    epoch=trainer.epoch,
                )
            elif trainer.args.task == 'segment':
                self.wandb_train_val_table = plot_segmentation_validation_results(
                    dataloader=dataloader,
                    class_label_map=class_label_map,
                    model_name=trainer.args.model,
                    predictor=self.predictor,
                    table=self.wandb_train_val_table,
                    max_validation_batches=1,
                    epoch=trainer.epoch,
                )
            elif trainer.args.task == 'pose':
                self.wandb_train_val_table = plot_pose_validation_results(
                    dataloader=dataloader,
                    class_label_map=class_label_map,
                    model_name=trainer.args.model,
                    predictor=self.predictor,
                    visualize_skeleton=True,
                    table=self.wandb_train_val_table,
                    max_validation_batches=1,
                    epoch=trainer.epoch,
                )
            elif trainer.args.task == 'classify':
                self.wandb_train_val_table = plot_classification_validation_results(
                    dataloader=dataloader,
                    model_name=trainer.args.model,
                    predictor=self.predictor,
                    table=self.wandb_train_val_table,
                    max_validation_batches=1,
                    epoch=trainer.epoch,
                )

    def on_train_end(self, trainer):
        """Save the best model as an artifact at end of training."""
        _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
        _log_plots(trainer.plots, step=trainer.epoch + 1)
        art = wb.Artifact(type='model', name=f'run_{wb.run.id}_model')
        if trainer.best.exists():
            art.add_file(trainer.best)
            wb.run.log_artifact(art, aliases=['best'])
        for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f'curves/{curve_name}',
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
        if self.wandb_train_val_table is not None:
            if len(self.wandb_train_val_table.data) > 0:
                wb.log({'Validation-Table': self.wandb_train_val_table})
        wb.run.finish()  # required or run continues on dashboard

    def on_predict_start(self, predictor: PREDICTOR_DTYPE):
        wb.run or wb.init(project='YOLOv8', job_type='predict_' + predictor.args.task, config=vars(predictor.args))
        if predictor.args.task == 'classify':
            self.wandb_prediction_table = wb.Table(columns=[
                'Model-Name',
                'Image',
                'Predicted-Category',
                'Prediction-Confidence',
                'Top-5-Prediction-Categories',
                'Top-5-Prediction-Confindence',
                'Probabilities',
                'Speed', ])
        elif predictor.args.task == 'detect':
            self.wandb_prediction_table = wb.Table(columns=[
                'Model-Name',
                'Image',
                'Number-of-Predictions',
                'Mean-Confidence',
                'Speed', ])
        elif predictor.args.task == 'pose':
            self.wandb_prediction_table = wb.Table(columns=[
                'Model-Name',
                'Image-Prediction',
                'Num-Instances',
                'Mean-Confidence',
                'Speed', ])
        elif predictor.args.task == 'segment':
            if isinstance(predictor, SegmentationPredictor):
                self.wandb_prediction_table = wb.Table(columns=[
                    'Model-Name',
                    'Image',
                    'Number-of-Predictions',
                    'Mean-Confidence',
                    'Speed', ])
            elif isinstance(predictor, SAMPredictor):
                self.prompts = deepcopy(predictor.prompts)
                self.wandb_prediction_table = wb.Table(columns=['Image'])

    def on_predict_end(self, predictor: PREDICTOR_DTYPE):
        if wb.run:
            for result in predictor.results:
                if predictor.args.task == 'classify':
                    self.wandb_prediction_table = plot_classification_predictions(result, predictor.args.model,
                                                                                  self.wandb_prediction_table)
                elif predictor.args.task == 'detect':
                    self.wandb_prediction_table = plot_bbox_predictions(result, predictor.args.model,
                                                                        self.wandb_prediction_table)
                elif predictor.args.task == 'pose':
                    self.wandb_prediction_table = plot_pose_predictions(result,
                                                                        predictor.args.model,
                                                                        table=self.wandb_prediction_table,
                                                                        visualize_skeleton=True)
                elif predictor.args.task == 'segment':
                    if isinstance(predictor, SegmentationPredictor):
                        self.wandb_prediction_table = plot_mask_predictions(result, predictor.args.model,
                                                                            self.wandb_prediction_table)
                    elif isinstance(predictor, SAMPredictor):
                        self.wandb_prediction_table = plot_sam_predictions(result, self.prompts,
                                                                           self.wandb_prediction_table)
            if self.wandb_prediction_table is not None:
                if len(self.wandb_prediction_table.data) > 0:
                    wb.log({'Prediction-Table': self.wandb_prediction_table})
            wb.run.finish()


wandb_callback_state = WandBCallbackState()
callbacks = {
    'on_pretrain_routine_start': wandb_callback_state.on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': wandb_callback_state.on_fit_epoch_end,
    'on_train_end': wandb_callback_state.on_train_end,
    'on_predict_start': wandb_callback_state.on_predict_start,
    'on_predict_end': wandb_callback_state.on_predict_end} if wb else {}
