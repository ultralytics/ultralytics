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
            preds (torch.Tensor | tuple): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed detection predictions with masks.
        """
        # Baseline AnomalyDetect returns ((detections, heatmap), raw_preds).
        # The neck-fusion variant uses a standard Detect head and returns
        # (detections, raw_preds) with no heatmap tensor.
        if isinstance(preds, (list, tuple)) and isinstance(preds[0], tuple):
            heatmap = preds[0][1]
            detections = preds[0]
        else:
            heatmap = None
            detections = preds[0] if isinstance(preds, (list, tuple)) else preds
        return super().postprocess(detections, img, orig_imgs, heatmap=heatmap)

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


class AnomalyPredictorHM(AnomalyPredictor):
    """Anomaly predictor that derives bounding boxes from the heatmap via connected components.

    Instead of using the detection head's NMS-decoded boxes, ``postprocess``
    thresholds the heatmap emitted by ``AnomalyDetect``, fits connected-component
    bounding boxes via ``_heatmap_to_boxes``, and forwards them as the detection
    result.  The heatmap is still attached to each ``Results`` object as
    ``result.heatmap``.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Derive detections from the AnomalyDetect heatmap via connected components.

        Args:
            preds (tuple): Raw model output — ``((detections, heatmap), raw_preds)``
                from ``AnomalyDetect``.
            img (torch.Tensor): Preprocessed batch tensor ``(B, C, H, W)``.
            orig_imgs (list[np.ndarray]): Original images before preprocessing.

        Returns:
            list[Results]: One Results per image; boxes come from heatmap connected
                components scaled to original image coordinates.  ``result.heatmap``
                holds the 2-D heatmap in original-image space.
        """
        if not (isinstance(preds, (list, tuple)) and isinstance(preds[0], tuple)):
            # No heatmap available — fall back to parent behaviour.
            return super().postprocess(preds, img, orig_imgs)

        heatmap = preds[0][1]  # (B, 1, mH, mW)

        # Scale factors: heatmap pixel space → inference image space.
        ih, iw = img.shape[2], img.shape[3]
        mh, mw = heatmap.shape[2], heatmap.shape[3]
        sx, sy = iw / mw, ih / mh

        hm_boxes = []
        for i in range(heatmap.shape[0]):
            boxes = self._heatmap_to_boxes(heatmap[i, 0])  # (N, 6) in heatmap pixel coords
            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] *= sx  # x1, x2
                boxes[:, [1, 3]] *= sy  # y1, y2
            hm_boxes.append(boxes)

        # construct_results (from AnomalyPredictor) scales boxes to orig_img space
        # and attaches the per-image heatmap as result.heatmap.
        return self.construct_results(hm_boxes, img, orig_imgs, heatmap=heatmap)

    @staticmethod
    def _heatmap_to_boxes(
        heatmap: "torch.Tensor",
        thresh: float = 0.5,
        max_det: int = 9,
        min_area: int = 5,
    ) -> "torch.Tensor":
        """Threshold a spatial heatmap and fit bounding boxes via connected components.

        Args:
            heatmap (Tensor): (H, W) float tensor, values in [0, 1].
            thresh (float): Score threshold for foreground pixels.
            max_det (int): Maximum number of boxes returned.
            min_area (int): Minimum connected-component area (pixels) to keep.

        Returns:
            Tensor: Shape (N, 6) — ``[x1, y1, x2, y2, score, class_id=0]``,
                sorted by score descending.  Returns empty (0, 6) when no
                component passes the threshold or min_area filter.
        """
        import cv2
        import numpy as np

        h_np = heatmap.detach().cpu().float().numpy()
        mask = (h_np >= thresh).astype(np.uint8)
        if mask.sum() == 0:
            return torch.zeros((0, 6), dtype=torch.float32)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        H, W = h_np.shape
        boxes = []
        for lbl in range(1, num):  # skip background label 0
            x, y, w, h, area = stats[lbl]
            if area < min_area:
                continue
            # Skip components that touch the image border (resize / padding artifacts).
            if x == 0 or y == 0 or (x + w) >= W or (y + h) >= H:
                continue
            score = float(h_np[labels == lbl].mean())
            boxes.append([float(x), float(y), float(x + w), float(y + h), score, 0.0])
        if not boxes:
            return torch.zeros((0, 6), dtype=torch.float32)
        t = torch.tensor(boxes, dtype=torch.float32)
        order = t[:, 4].argsort(descending=True)[:max_det]
        return t[order]
