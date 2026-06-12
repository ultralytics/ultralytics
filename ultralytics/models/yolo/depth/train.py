# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation trainer for YOLO models."""

from __future__ import annotations

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import DepthModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


class DepthTrainer(yolo.detect.DetectionTrainer):
    """Trainer for YOLO depth estimation models.

    Multi-source training (list of img_paths) is handled transparently by the base DetectionTrainer/BaseDataset.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthTrainer
        >>> args = dict(model="yolo26s-depth.yaml", data="depth-mixed.yaml", epochs=100)
        >>> trainer = DepthTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DepthTrainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "depth"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a DepthModel initialized with the given config and weights.

        If the dataset YAML declares ``max_depth`` (meters), it overrides the head's output
        range (``sigmoid × max_depth``). Depth range is a property of the data: training on
        GT beyond the head's range is otherwise unrepresentable. The value persists in saved
        checkpoints, so fine-tuned models predict in the new range at inference.
        """
        model = DepthModel(cfg, ch=self.data.get("channels", 3), nc=self.data["nc"], verbose=verbose and RANK in {-1, 0})
        if weights:
            model.load(weights)
        max_depth = self.data.get("max_depth")
        if max_depth is not None:
            head = model.model[-1]
            if hasattr(head, "max_depth"):
                if RANK in {-1, 0}:
                    LOGGER.info(f"Depth head max_depth: {head.max_depth} → {float(max_depth)} m (from dataset YAML)")
                head.max_depth = float(max_depth)
        return model

    def preprocess_batch(self, batch):
        """Preprocess batch: normalize images and keep depth as float32."""
        batch = super().preprocess_batch(batch)
        if "depth" in batch:
            batch["depth"] = batch["depth"].float()
        return batch

    def get_validator(self):
        """Return a DepthValidator for model validation."""
        self.loss_names = "silog", "grad"
        return yolo.depth.DepthValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
