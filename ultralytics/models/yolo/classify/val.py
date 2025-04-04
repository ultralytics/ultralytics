# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a classification model.

    This validator handles the validation process for classification models, including metrics calculation,
    confusion matrix generation, and visualization of results.

    Attributes:
        targets (List[torch.Tensor]): Ground truth class labels.
        pred (List[torch.Tensor]): Model predictions.
        metrics (ClassifyMetrics): Object to calculate and store classification metrics.
        names (dict): Mapping of class indices to class names.
        nc (int): Number of classes.
        confusion_matrix (ConfusionMatrix): Matrix to evaluate model performance across classes.

    Methods:
        get_desc: Return a formatted string summarizing classification metrics.
        init_metrics: Initialize confusion matrix, class names, and tracking containers.
        preprocess: Preprocess input batch by moving data to device.
        update_metrics: Update running metrics with model predictions and batch targets.
        finalize_metrics: Finalize metrics including confusion matrix and processing speed.
        postprocess: Extract the primary prediction from model output.
        get_stats: Calculate and return a dictionary of metrics.
        build_dataset: Create a ClassificationDataset instance for validation.
        get_dataloader: Build and return a data loader for classification validation.
        print_results: Print evaluation metrics for the classification model.
        plot_val_samples: Plot validation image samples with their ground truth labels.
        plot_predictions: Plot images with their predicted class labels.

    Examples:
        >>> from ultralytics.models.yolo.classify import ClassificationValidator
        >>> args = dict(model="yolo11n-cls.pt", data="imagenet10")
        >>> validator = ClassificationValidator(args=args)
        >>> validator()

    Notes:
        Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize ClassificationValidator with dataloader, save directory, and other parameters.

        This validator handles the validation process for classification models, including metrics calculation,
        confusion matrix generation, and visualization of results.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (str | Path, optional): Directory to save results.
            pbar (bool, optional): Display a progress bar.
            args (dict, optional): Arguments containing model and validation configuration.
            _callbacks (list, optional): List of callback functions to be called during validation.

        Examples:
            >>> from ultralytics.models.yolo.classify import ClassificationValidator
            >>> args = dict(model="yolo11n-cls.pt", data="imagenet10")
            >>> validator = ClassificationValidator(args=args)
            >>> validator()
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        """Return a formatted string summarizing classification metrics."""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and tracking containers for predictions and targets."""
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf, task="classify")
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocess input batch by moving data to device and converting to appropriate dtype."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """
        Update running metrics with model predictions and batch targets.

        Args:
            preds (torch.Tensor): Model predictions, typically logits or probabilities for each class.
            batch (dict): Batch data containing images and class labels.

        This method appends the top-N predictions (sorted by confidence in descending order) to the
        prediction list for later evaluation. N is limited to the minimum of 5 and the number of classes.
        """
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self, *args, **kwargs):
        """
        Finalize metrics including confusion matrix and processing speed.

        This method processes the accumulated predictions and targets to generate the confusion matrix,
        optionally plots it, and updates the metrics object with speed information.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Examples:
            >>> validator = ClassificationValidator()
            >>> validator.pred = [torch.tensor([[0, 1, 2]])]  # Top-3 predictions for one sample
            >>> validator.targets = [torch.tensor([0])]  # Ground truth class
            >>> validator.finalize_metrics()
            >>> print(validator.metrics.confusion_matrix)  # Access the confusion matrix
        """
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def postprocess(self, preds):
        """Extract the primary prediction from model output if it's in a list or tuple format."""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self):
        """Calculate and return a dictionary of metrics by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        """Create a ClassificationDataset instance for validation."""
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        """
        Build and return a data loader for classification validation.

        Args:
            dataset_path (str | Path): Path to the dataset directory.
            batch_size (int): Number of samples per batch.

        Returns:
            (torch.utils.data.DataLoader): DataLoader object for the classification validation dataset.
        """
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Print evaluation metrics for the classification model."""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        """
        Plot validation image samples with their ground truth labels.

        Args:
            batch (dict): Dictionary containing batch data with 'img' (images) and 'cls' (class labels).
            ni (int): Batch index used for naming the output file.

        Examples:
            >>> validator = ClassificationValidator()
            >>> batch = {"img": torch.rand(16, 3, 224, 224), "cls": torch.randint(0, 10, (16,))}
            >>> validator.plot_val_samples(batch, 0)
        """
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        Plot images with their predicted class labels and save the visualization.

        Args:
            batch (dict): Batch data containing images and other information.
            preds (torch.Tensor): Model predictions with shape (batch_size, num_classes).
            ni (int): Batch index used for naming the output file.

        Examples:
            >>> validator = ClassificationValidator()
            >>> batch = {"img": torch.rand(16, 3, 224, 224)}
            >>> preds = torch.rand(16, 10)  # 16 images, 10 classes
            >>> validator.plot_predictions(batch, preds, 0)
        """
        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=torch.argmax(preds, dim=1),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
