# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Extract per-image properties from object detection datasets."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

COCO_AREA_SMALL = 32**2  # COCO small-object area threshold (px^2), Lin et al. 2014


class ImagePropertyExtractor:
    """Augment a ``YOLODataset``'s labels in place with six per-image properties.

    Compute object count, small-object ratio, object-scale variation, class count, center spread, and maximum pairwise
    IoU from image headers and annotations.

    Attributes:
        labels (list[dict]): The same list as ``dataset.labels``, with an ``im_properties`` dict added per image.
    """

    def __init__(self, dataset):
        """Extract properties into dataset labels."""
        self.labels = dataset.labels
        for label in self.labels:
            self._augment_label(label)

    @staticmethod
    def _augment_label(lbl: dict) -> None:
        """Compute the six properties for one label into its ``im_properties`` dict."""
        cls_arr = lbl["cls"].reshape(-1)
        bboxes_n = lbl["bboxes"].reshape(-1, 4)
        with Image.open(lbl["im_file"]) as image:
            w, h = image.size
        n = len(bboxes_n)
        areas_n = bboxes_n[:, 2] * bboxes_n[:, 3]
        lbl["im_properties"] = {
            "num_objects": n,
            "small_object_ratio": float(np.mean(areas_n * w * h < COCO_AREA_SMALL)) if n else np.nan,
            "object_scale_variance": (float(np.std(areas_n) / max(np.mean(areas_n), 1e-9)) if n else np.nan),
            "num_classes_present": int(np.unique(cls_arr).size),
            "center_spread": (float(np.sqrt(np.var(bboxes_n[:, 0]) + np.var(bboxes_n[:, 1]))) if n else np.nan),
            "max_pairwise_iou": (ImagePropertyExtractor._max_pairwise_iou(xywh2xyxy(bboxes_n)) if n >= 2 else np.nan),
        }

    @staticmethod
    def _max_pairwise_iou(xyxy: np.ndarray) -> float:
        """Calculate the maximum pairwise IoU among boxes in xyxy format."""
        boxes, maximum = torch.as_tensor(xyxy, dtype=torch.float32), 0.0
        for i in range(0, len(boxes), 1024):
            for j in range(i, len(boxes), 1024):
                iou = box_iou(boxes[i : i + 1024], boxes[j : j + 1024])
                if i == j:
                    iou.triu_(diagonal=1)
                maximum = max(maximum, float(iou.max()))
        return maximum
