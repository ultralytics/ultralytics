# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data import RegressionDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import RegressMetrics
from ultralytics.utils.plotting import plot_images


class RegressionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a regression model.

    Example:
        ```python
        from ultralytics.models.yolo.regress import RegressionValidator

        args = dict(model='yolov8n-regress.pt', data='imdb-age.yaml')
        validator = RegressionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes RegressionValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "regress"
        self.metrics = RegressMetrics()

    def get_desc(self):
        """Returns a formatted string summarizing regression metrics."""
        return ("%11s" * 2) % ("mae", "mse")

    def init_metrics(self, model):
        """Initialize storages for metrics - mean absolute error and mean squared error."""
        self.names = model.names
        self.pred = []
        self.targets = []
        self.max = model.metadata["max_value"] if (isinstance(model, AutoBackend) and not model.pt) else 6
        self.min = model.metadata["min_value"] if (isinstance(model, AutoBackend) and not model.pt) else 0

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["value"] = batch["value"].to(self.device)
        return batch
    
    def postprocess(self, preds, img_shape):
        for i in range(len(preds)):
            preds[i] = preds[i] * (self.max - self.min) / 6 + self.min
        return preds

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        self.pred.append(preds.view(preds.size()[0]))
        self.targets.append(batch["value"])

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as speed."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        """Creates and returns a RegressionDataset instance using given image path and preprocessing parameters."""
        return RegressionDataset(args=self.args, img_path=img_path, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Prints evaluation metrics for YOLO regression model."""
        pf = "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % (self.metrics.mae, self.metrics.mse))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["value"].view(-1),  # warning: use .view(), not .squeeze() for Regress models
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            on_plot=self.on_plot)  # pred
