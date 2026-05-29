# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation predictor for YOLO models."""

from __future__ import annotations

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG


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

        results = []
        for i, orig_img in enumerate(orig_imgs):
            depth = depth_maps[i] if depth_maps.ndim == 4 else depth_maps
            if depth.ndim == 3:
                depth = depth.squeeze(0)  # Remove channel dim → (H, W)

            # Resize to original image size
            import torch.nn.functional as F
            oh, ow = orig_img.shape[:2]
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0).float(),
                size=(oh, ow),
                mode="bilinear",
                align_corners=True,
            ).squeeze().cpu().numpy()

            results.append(
                Results(
                    orig_img=orig_img,
                    path=self.batch[0][i] if self.batch else "",
                    names=self.model.names,
                    depth=depth,
                )
            )

        return results
