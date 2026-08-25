# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-image property extraction for object detection datasets.

``ImagePropertyExtractor(yolo_dataset)`` augments each ``dataset.labels`` entry in place with six scalar image
properties for downstream analysis. It uses image headers and annotations only, with no model, metrics, pixel decoding,
or disk output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

COCO_AREA_SMALL = 32**2  # COCO small-object area threshold (px^2), Lin et al. 2014


class ImagePropertyExtractor:
    """Augment a ``YOLODataset``'s labels in place with six per-image properties.

    Computes object count, small-object ratio, object-scale variation, class count, center spread, and maximum pairwise
    IoU from image headers and annotations. Each label in ``dataset.labels`` gains an ``im_properties`` dict, and the
    same label list is exposed as ``self.labels`` for chaining.

    Attributes:
        labels (list[dict]): The same list as ``dataset.labels``, with an ``im_properties`` dict added per image.

    Examples:
        >>> from ultralytics.cfg import get_cfg
        >>> from ultralytics.data.build import build_yolo_dataset
        >>> from ultralytics.data.utils import check_det_dataset
        >>> from ultralytics.utils.analysis import ImagePropertyExtractor
        >>> data = check_det_dataset("coco128.yaml")
        >>> cfg = get_cfg(overrides={"task": "detect", "imgsz": 320})
        >>> ds = build_yolo_dataset(cfg, data["val"], 1, data, mode="val", rect=False, stride=32)
        >>> labels = ImagePropertyExtractor(ds).labels
        >>> num_objects = labels[0]["im_properties"]["num_objects"]
    """

    def __init__(self, dataset: Any):
        """Extract per-image properties and mutate ``dataset.labels`` in place.

        Args:
            dataset (Any): A ``YOLODataset`` instance with non-empty ``labels`` and ``im_file`` per entry.
        """
        labels = getattr(dataset, "labels", None)
        if not labels:
            raise ValueError("ImagePropertyExtractor requires a YOLODataset with non-empty labels.")
        for label in labels:
            self._augment_label(label)
        self.labels = labels

    @staticmethod
    def _augment_label(lbl: dict) -> dict:
        """Compute the six properties for one label into its ``im_properties`` dict."""
        cls_arr = np.asarray(lbl.get("cls", np.zeros((0, 1)))).reshape(-1).astype(int)
        bboxes_n = np.asarray(lbl.get("bboxes", np.zeros((0, 4)))).reshape(-1, 4)
        with Image.open(lbl["im_file"]) as image:
            w, h = image.size
        n = int(bboxes_n.shape[0])
        areas_n = bboxes_n[:, 2] * bboxes_n[:, 3]
        lbl["im_properties"] = {
            "num_objects": n,
            "small_object_ratio": float(np.mean(areas_n * w * h < COCO_AREA_SMALL)) if n else np.nan,
            "object_scale_variance": (float(np.std(areas_n) / max(np.mean(areas_n), 1e-9)) if n else np.nan),
            "num_classes_present": int(np.unique(cls_arr).size),
            "center_spread": (float(np.sqrt(np.var(bboxes_n[:, 0]) + np.var(bboxes_n[:, 1]))) if n else np.nan),
            "max_pairwise_iou": (ImagePropertyExtractor._max_pairwise_iou(xywh2xyxy(bboxes_n)) if n >= 2 else np.nan),
        }
        return lbl

    @staticmethod
    def _max_pairwise_iou(xyxy: np.ndarray) -> float:
        """Calculate the maximum pairwise IoU among boxes in xyxy format."""
        t = torch.as_tensor(xyxy, dtype=torch.float32)
        iou = box_iou(t, t).triu_(diagonal=1)
        return float(iou.max())
