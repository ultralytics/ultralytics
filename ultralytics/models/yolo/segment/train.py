# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    This trainer specializes in handling segmentation tasks, extending the detection trainer with segmentation-specific
    functionality including model initialization, validation, and visualization.

    Attributes:
        loss_names (Tuple[str]): Names of the loss components used during training.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize a SegmentationTrainer object.

        This initializes a trainer for segmentation tasks, extending the detection trainer with segmentation-specific
        functionality. It sets the task to 'segment' and prepares the trainer for training segmentation models.

        Args:
            cfg (dict): Configuration dictionary with default training settings. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Examples:
            >>> from ultralytics.models.yolo.segment import SegmentationTrainer
            >>> args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml", epochs=3)
            >>> trainer = SegmentationTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return a SegmentationModel with specified configuration and weights.

        Args:
            cfg (dict | str | None): Model configuration. Can be a dictionary, a path to a YAML file, or None.
            weights (str | Path | None): Path to pretrained weights file.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (SegmentationModel): Initialized segmentation model with loaded weights if specified.

        Examples:
            >>> trainer = SegmentationTrainer()
            >>> model = trainer.get_model(cfg="yolov8n-seg.yaml")
            >>> model = trainer.get_model(weights="yolov8n-seg.pt", verbose=False)
        """
        model = SegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """
        Plot training sample images with labels, bounding boxes, and masks.

        This method creates a visualization of training batch images with their corresponding labels, bounding boxes,
        and segmentation masks, saving the result to a file for inspection and debugging.

        Args:
            batch (dict): Dictionary containing batch data with the following keys:
                'img': Images tensor
                'batch_idx': Batch indices for each box
                'cls': Class labels tensor (squeezed to remove last dimension)
                'bboxes': Bounding box coordinates tensor
                'masks': Segmentation masks tensor
                'im_file': List of image file paths
            ni (int): Current training iteration number, used for naming the output file.

        Examples:
            >>> trainer = SegmentationTrainer()
            >>> batch = {
            ...     "img": torch.rand(16, 3, 640, 640),
            ...     "batch_idx": torch.zeros(16),
            ...     "cls": torch.randint(0, 80, (16, 1)),
            ...     "bboxes": torch.rand(16, 4),
            ...     "masks": torch.rand(16, 640, 640),
            ...     "im_file": ["image1.jpg", "image2.jpg"],
            ... }
            >>> trainer.plot_training_samples(batch, ni=5)
        """
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
