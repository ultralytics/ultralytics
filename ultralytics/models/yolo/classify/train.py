# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first


class ClassificationTrainer(BaseTrainer):
    """
    A trainer class extending BaseTrainer for training image classification models.

    This trainer handles the training process for image classification tasks, supporting both YOLO classification models
    and torchvision models with comprehensive dataset handling and validation.

    Attributes:
        model (ClassificationModel): The classification model to be trained.
        data (dict[str, Any]): Dictionary containing dataset information including class names and number of classes.
        loss_names (list[str]): Names of the loss functions used during training.
        validator (ClassificationValidator): Validator instance for model evaluation.

    Methods:
        set_model_attributes: Set the model's class names from the loaded dataset.
        get_model: Return a modified PyTorch model configured for training.
        setup_model: Load, create or download model for classification.
        build_dataset: Create a ClassificationDataset instance.
        get_dataloader: Return PyTorch DataLoader with transforms for image preprocessing.
        preprocess_batch: Preprocess a batch of images and classes.
        progress_string: Return a formatted string showing training progress.
        get_validator: Return an instance of ClassificationValidator.
        label_loss_items: Return a loss dict with labelled training loss items.
        final_eval: Evaluate trained model and save validation results.
        plot_training_samples: Plot training samples with their annotations.

    Examples:
        Initialize and train a classification model
        >>> from ultralytics.models.yolo.classify import ClassificationTrainer
        >>> args = dict(model="yolo11n-cls.pt", data="imagenet10", epochs=3)
        >>> trainer = ClassificationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize a ClassificationTrainer object.

        Args:
            cfg (dict[str, Any], optional): Default configuration dictionary containing training parameters.
            overrides (dict[str, Any], optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list[Any], optional): List of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """
        Return a modified PyTorch model configured for training YOLO classification.

        Args:
            cfg (Any, optional): Model configuration.
            weights (Any, optional): Pre-trained model weights.
            verbose (bool, optional): Whether to display model information.

        Returns:
            (ClassificationModel): Configured PyTorch model for classification.
        """
        model = ClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def setup_model(self):
        """
        Load, create or download model for classification tasks.

        Returns:
            (Any): Model checkpoint if applicable, otherwise None.
        """
        import torchvision  # scope for faster 'import ultralytics'

        if str(self.model) in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None
        else:
            ckpt = super().setup_model()
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """
        Create a ClassificationDataset instance given an image path and mode.

        Args:
            img_path (str): Path to the dataset images.
            mode (str, optional): Dataset mode ('train', 'val', or 'test').
            batch (Any, optional): Batch information (unused in this implementation).

        Returns:
            (ClassificationDataset): Dataset for the specified mode.
        """
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Return PyTorch DataLoader with transforms to preprocess images.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int, optional): Number of images per batch.
            rank (int, optional): Process rank for distributed training.
            mode (str, optional): 'train', 'val', or 'test' mode.

        Returns:
            (torch.utils.data.DataLoader): DataLoader for the specified dataset and mode.
        """
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank, drop_last=self.args.compile)
        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preprocess a batch of images and classes."""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def progress_string(self) -> str:
        """Return a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """Return an instance of ClassificationValidator for validation."""
        self.loss_names = ["loss"]
        return yolo.classify.ClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: torch.Tensor | None = None, prefix: str = "train"):
        """
        Return a loss dict with labelled training loss items tensor.

        Args:
            loss_items (torch.Tensor, optional): Loss tensor items.
            prefix (str, optional): Prefix to prepend to loss names.

        Returns:
            keys (list[str]): List of loss keys if loss_items is None.
            loss_dict (dict[str, float]): Dictionary of loss items if loss_items is provided.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def plot_training_samples(self, batch: dict[str, torch.Tensor], ni: int):
        """
        Plot training samples with their annotations.

        Args:
            batch (dict[str, torch.Tensor]): Batch containing images and class labels.
            ni (int): Number of iterations.
        """
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # add batch index for plotting
        plot_images(
            labels=batch,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
