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

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Extract the batch heatmap, run NMS, and build ``Results`` objects."""
        # AnomalyDetect returns (detections, heatmap, raw_preds) in training-free
        # inference and (detections, heatmap) when exported.
        heatmap = None
        if isinstance(preds, (list, tuple)) and len(preds) >= 2 and isinstance(preds[1], torch.Tensor):
            h = preds[1]
            if h.ndim == 4 and h.shape[1] == 1:
                heatmap = h

        # Let the detection predictor handle NMS and per-image Result construction;
        # pass the heatmap through to construct_result below.
        return super().postprocess(preds, img, orig_imgs, heatmap=heatmap, **kwargs)

    def construct_results(self, preds, img, orig_imgs, heatmap=None, **kwargs):
        """Build Results objects, forwarding the optional batch heatmap."""
        return [
            self.construct_result(pred, img, orig_img, img_path, idx=i, heatmap=heatmap, **kwargs)
            for i, (pred, orig_img, img_path) in enumerate(zip(preds, orig_imgs, self.batch[0]))
        ]

    def construct_result(self, pred, img, orig_img, img_path, idx=0, heatmap=None, **kwargs):
        """Build one Result object and attach the scaled heatmap."""
        result = super().construct_result(pred, img, orig_img, img_path, **kwargs)
        if heatmap is not None:
            result.heatmap = ops.scale_masks(heatmap[idx : idx + 1], orig_img.shape[:2]).squeeze(0).squeeze(0)
        return result
