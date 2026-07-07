# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops


class AnomalyPredictor(DetectionPredictor):
    """Predictor for YOLOA anomaly-detection models.

    Extends ``DetectionPredictor`` by extracting the anomaly heatmap that
    ``AnomalyDetect`` emits alongside the detection tensor and attaching it to
    each ``Results`` object as ``result.heatmap``.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-process YOLO predictions and return output detections with proto.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed detection predictions with masks.
        """
        heatmap = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, heatmap=heatmap)

    def construct_results(self, preds, img, orig_imgs, heatmap=None):
        """Build Results objects, forwarding the optional batch heatmap."""
        return [
            self.construct_result(pred, img, orig_img, img_path, idx=i, heatmap=heatmap)
            for i, (pred, orig_img, img_path) in enumerate(zip(preds, orig_imgs, self.batch[0]))
        ]

    def construct_result(self, pred, img, orig_img, img_path, idx=0, heatmap=None):
        """Build one Result object and attach the scaled heatmap."""
        result = super().construct_result(pred, img, orig_img, img_path)
        if heatmap is not None:
            result.heatmap = ops.scale_masks(heatmap[idx : idx + 1], orig_img.shape[:2]).squeeze(0).squeeze(0)
        return result
