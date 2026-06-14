# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class SemanticSegmentationPredictor(BasePredictor):
    """Predictor for semantic segmentation models.

    This predictor processes model outputs to produce per-pixel class label maps.

    Examples:
        >>> from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor
        >>> args = dict(model="yolo26n-sem.pt", source="path/to/image.jpg")
        >>> predictor = SemanticSegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize SemanticSegmentationPredictor.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "semantic"

    @staticmethod
    def _class_map_dtype(num_classes: int) -> torch.dtype:
        """Return the smallest practical integer dtype for semantic class IDs."""
        return torch.uint8 if num_classes <= 256 else torch.int16 if num_classes <= 32768 else torch.int32

    def postprocess(self, preds, img, orig_imgs):
        """Convert model logits to semantic segmentation results.

        Args:
            preds (torch.Tensor | tuple): Model output logits.
            img (torch.Tensor): Preprocessed input image tensor.
            orig_imgs (list | torch.Tensor): Original images.

        Returns:
            (list[Results]): List of Results objects with semantic masks.
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        classes = (
            torch.as_tensor(self.args.classes, device=preds.device).flatten()
            if self.args.classes is not None and len(self.model.names) > 1
            else None
        )

        results = []
        for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            class_map_input = pred.ndim == 2  # exports with in-graph ArgMax emit a [H, W] class map directly
            pred = (pred.unsqueeze(0).unsqueeze(0) if class_map_input else pred.unsqueeze(0)).float()
            # Upsample pred to the model input resolution first so LetterBox padding is an integer in this space
            if pred.shape[2:] != img.shape[2:]:
                pred = F.interpolate(pred, img.shape[2:], mode="nearest" if class_map_input else "bilinear")
            # scale_masks then crops it cleanly without the sub-pixel asymmetry; class maps must never blend IDs
            pred = ops.scale_masks(pred, orig_img.shape[:2], mode="nearest" if class_map_input else "bilinear")[0]
            if class_map_input:
                class_map = pred.squeeze(0).to(self._class_map_dtype(int(pred.max().item()) + 1))
            else:
                dtype = self._class_map_dtype(max(pred.shape[0], 2))
                class_map = pred.argmax(0).to(dtype) if pred.shape[0] > 1 else pred.gt(0).squeeze(0).to(dtype)
            if classes is not None:  # keep only selected classes, mark the rest as ignore
                class_map[~(class_map.unsqueeze(-1) == classes).any(-1)] = 255
            results.append(Results(orig_img, path=img_path, names=self.model.names, semantic_mask=class_map))
        return results
