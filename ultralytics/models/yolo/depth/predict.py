# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation predictor for YOLO models."""

from __future__ import annotations

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import ops


class DepthPredictor(BasePredictor):
    """Predictor for YOLO depth estimation models.

    Produces per-pixel depth maps from RGB images.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthPredictor
        >>> predictor = DepthPredictor(overrides=dict(model="yolo26n-depth.pt"))
        >>> results = predictor("image.jpg")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DepthPredictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "depth"

    def postprocess(self, preds, img, orig_imgs):
        """Post-process depth predictions to Results objects."""
        # preds is the depth map tensor from model
        if isinstance(preds, (tuple, list)):
            depth_maps = preds[0]  # first element is depth
        else:
            depth_maps = preds

        if not isinstance(orig_imgs, list):
            orig_imgs = [orig_imgs]

        if depth_maps.ndim == 3:
            depth_maps = depth_maps.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)

        results = []
        for i, orig_img in enumerate(orig_imgs):
            depth = depth_maps[i] if depth_maps.ndim == 4 else depth_maps
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)  # (H, W) → (1, H, W)

            # Remove letterbox padding and rescale to the original image size. The model output is
            # padded to a (square) inference shape, so the padding must be cropped before resizing
            # or the depth map ends up stretched relative to the RGB image (e.g. on ONNX/TorchScript
            # exports that run at a fixed square imgsz instead of rectangular inference).
            depth = ops.scale_masks(depth.unsqueeze(0).float(), orig_img.shape[:2])
            depth = depth.squeeze().cpu().numpy()

            results.append(
                Results(
                    orig_img=orig_img,
                    path=self.batch[0][i] if self.batch else "",
                    names=self.model.names,
                    depth=depth,
                )
            )

        return results
