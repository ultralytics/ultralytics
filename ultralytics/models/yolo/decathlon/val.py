# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics
from ultralytics.utils.plotting import plot_images


class DecathlonValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a decathlon model.

    This class is not used, decathlon validation relies on the validation of each sub task.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.metrics = ClassifyMetrics()  # not used, any metric

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        return 'Unused'

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""
        pass

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        pass

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        pass

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        return {}

    def build_dataset(self, img_path):
        return None

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        return None

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""
        pass
