# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.data import build_reid_dataloader
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import ReidModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first
from ..classify.train import ClassificationTrainer


class ReidTrainer(ClassificationTrainer):
    """Trainer for person re-identification models.

    Extends BaseTrainer with ReID-specific dataset handling (Market-1501), PK batch sampling,
    and multi-loss training (cross-entropy + triplet).

    Attributes:
        model (ReidModel): The ReID model to be trained.
        data (dict): Dataset information including identity names and count.
        loss_names (list[str]): Names of loss components: ['ce_loss', 'tri_loss'].

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidTrainer
        >>> args = dict(model="yolo26n-reid.yaml", data="Market-1501.yaml", epochs=60)
        >>> trainer = ReidTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize ReidTrainer.

        Args:
            cfg (dict): Default configuration dictionary.
            overrides (dict, optional): Parameter overrides.
            _callbacks (list, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "reid"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 256
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """Set the model's identity names and configure loss from trainer args."""
        super().set_model_attributes()
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a ReidModel configured for training.

        Args:
            cfg: Model configuration.
            weights: Pre-trained weights (path or dict).
            verbose (bool): Whether to display model info.

        Returns:
            (ReidModel): Configured model.
        """
        # Forward ReID loss hparams so ReidModel.init_criterion works without the trainer side-effect.
        reid_kwargs = {
            k: getattr(self.args, k)
            for k in (
                "triplet_margin",
                "label_smoothing",
                "triplet_weight",
                "ce_weight",
                "center_weight",
                "center_momentum",
                "focal_gamma",
                "supcon_temp",
            )
            if hasattr(self.args, k)
        }
        model = ReidModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and RANK == -1,
            **reid_kwargs,
        )
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        return model

    def setup_model(self):
        """Load or create model for ReID tasks, then resize the classifier to the dataset identity count."""
        ckpt = super().setup_model()
        ReidModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Create a ReidDataset instance via the centralised build_yolo_dataset() entry point.

        Args:
            img_path (str): Path to dataset split.
            mode (str): 'train', 'val', 'test', or 'gallery'.
            batch (int, optional): Batch size (unused for ReID — kept for parent signature parity).

        Returns:
            (ReidDataset): Dataset for the specified split.
        """
        return build_yolo_dataset(self.args, img_path, batch or 0, self.data, mode=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return dataloader with PK sampling for training.

        Args:
            dataset_path (str): Path to dataset.
            batch_size (int): Batch size.
            rank (int): Process rank for DDP.
            mode (str): 'train' or 'val'.

        Returns:
            (DataLoader): Configured dataloader.
        """
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        if mode == "train":
            # PK sampling: P identities x K images
            p = getattr(self.args, "reid_p", 16)
            k = getattr(self.args, "reid_k", 4)
            loader = build_reid_dataloader(dataset, batch_size, self.args.workers, p=p, k=k, shuffle=True, rank=rank)
        else:
            from ultralytics.data import build_dataloader

            loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)

        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def get_validator(self):
        """Return a ReidValidator instance."""
        self.loss_names = ["ce_loss", "tri_loss"]
        return yolo.reid.ReidValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def plot_training_samples(self, batch, ni):
        """Plotting training samples is a no-op for ReID — pid integers (often hundreds in
        Market-1501) are not human-meaningful class names, so a mosaic of pid-labelled crops
        adds visual noise without conveying anything useful. ``ReidValidator.plot_predictions``
        is similarly a no-op."""
        pass

    def label_loss_items(self, loss_items=None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items.

        ReID validation uses query-gallery mAP, not loss computation, so val-prefixed
        loss items are omitted to keep results.csv and plots clean.

        Args:
            loss_items: Loss tensor items.
            prefix (str): Prefix for loss names.

        Returns:
            Loss keys or dict of loss items.
        """
        if prefix == "val":
            return [] if loss_items is None else {}
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))
