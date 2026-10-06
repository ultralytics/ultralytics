# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation predictor for YOLO models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class DepthPredictor(BasePredictor):
    """Predictor for YOLO depth estimation models.

    Produces per-pixel depth maps from RGB images.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthPredictor
        >>> predictor = DepthPredictor(overrides=dict(model="yolo26n-depth.pt"))
        >>> results = predictor("image.jpg")
    """

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None
    ) -> None:
        """Initialize DepthPredictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "depth"

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Stretch images to the model input size without padding, matching depth validation and calibration."""
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]

    def postprocess(
        self, preds: torch.Tensor | tuple | list, img: torch.Tensor, orig_imgs: list[np.ndarray] | torch.Tensor
    ) -> list[Results]:
        """Post-process depth predictions to Results objects."""
        depth_maps = preds[0] if isinstance(preds, (tuple, list)) else preds  # (B, 1, H, W)
        if depth_maps.ndim == 3:
            depth_maps = depth_maps.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
        # Restore model-input resolution so all backends upsample the same way before scaling to the original image.
        # align_corners=True matches the depth loss, validator and exported head upsample.
        depth_maps = F.interpolate(depth_maps.float(), size=img.shape[2:], mode="bilinear", align_corners=True)

        if not isinstance(orig_imgs, list):  # torch.Tensor source (B, 3, H, W)
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for i, orig_img in enumerate(orig_imgs):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            depth = F.interpolate(depth_maps[i : i + 1], orig_img.shape[:2], mode="bilinear", align_corners=True)
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, depth=depth.squeeze()))

        return results
