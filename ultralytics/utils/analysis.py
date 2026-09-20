# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Correlate dataset properties with per-image accuracy and score possible label issues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import DataExportMixin
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

COCO_AREA_SMALL = 32**2  # COCO small-object area threshold (px^2), Lin et al. 2014
_PROPERTIES = (
    "num_objects",
    "small_object_ratio",
    "object_scale_variance",
    "num_classes_present",
    "center_spread",
    "max_pairwise_iou",
)
_LABEL_ISSUES = ("possible_fp", "possible_fn", "possible_label_confusion")


@dataclass
class AnalysisReport(DataExportMixin):
    """Store per-image metrics and property correlations.

    Attributes:
        per_image (dict[str, dict]): Per-image metrics and properties keyed by absolute image path.
        correlations (dict[str, dict]): Per-property Spearman correlation and sample count against F1.
    """

    per_image: dict[str, dict]
    correlations: dict[str, dict]

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict]:
        """Return one numeric row per image property."""
        return [
            {
                "property": prop,
                "spearman_r": None if row["spearman_r"] is None else round(row["spearman_r"], decimals),
                "n": row["n"],
            }
            for prop, row in self.correlations.items()
        ]


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


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, assigning tied values their mean rank."""
    sorter = np.argsort(values, kind="stable")
    inverse = np.empty(values.size, dtype=int)
    inverse[sorter] = np.arange(values.size)
    sorted_values = values[sorter]
    observed = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense = observed.cumsum()[inverse]
    count = np.r_[np.nonzero(observed)[0], values.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def analyze_correlations(dataset, metrics) -> AnalysisReport:
    """Correlate per-image dataset properties with per-image F1.

    Derive object count, small-object ratio, object-scale variation, class count, center spread, and maximum pairwise
    IoU from each label's cached shape and annotations, then rank-correlate them against the per-image F1 recorded
    during validation.

    Args:
        dataset (YOLODataset): Dataset whose ``labels`` supply annotations and cached image shapes.
        metrics (DetMetrics): Validation metrics carrying ``box.image_metrics`` keyed by absolute image path.

    Returns:
        (AnalysisReport): Per-image rows and per-property Spearman correlations.
    """
    per_image = {}
    for label in dataset.labels:
        h, w = label["shape"]
        bboxes = label["bboxes"]
        n = len(bboxes)
        areas = bboxes[:, 2] * bboxes[:, 3]
        im_file = str(Path(label["im_file"]).absolute())
        per_image[im_file] = {
            **metrics.box.image_metrics.get(im_file, {}),
            "num_objects": n,
            "small_object_ratio": float(np.mean(areas * w * h < COCO_AREA_SMALL)) if n else np.nan,
            "object_scale_variance": float(np.std(areas) / max(np.mean(areas), 1e-9)) if n else np.nan,
            "num_classes_present": int(np.unique(label["cls"]).size),
            "center_spread": float(np.sqrt(np.var(bboxes[:, 0]) + np.var(bboxes[:, 1]))) if n else np.nan,
            "max_pairwise_iou": _max_pairwise_iou(xywh2xyxy(bboxes)) if n >= 2 else np.nan,
        }

    f1 = np.array([row.get("f1", np.nan) for row in per_image.values()], dtype=float)
    correlations = {}
    for prop in _PROPERTIES:
        values = np.array([row[prop] for row in per_image.values()], dtype=float)
        mask = np.isfinite(values) & np.isfinite(f1)
        r = None
        if mask.sum() > 1 and np.ptp(values[mask]) and np.ptp(f1[mask]):
            r = float(np.corrcoef(_rankdata(values[mask]), _rankdata(f1[mask]))[0, 1])
        correlations[prop] = {"spearman_r": r, "n": int(mask.sum())}
    return AnalysisReport(per_image, correlations)


def _label_issue_scores(
    iou: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_cls: np.ndarray,
) -> dict[str, float]:
    """Return image-level possible FP, FN, and label-confusion scores."""
    pred_cls, gt_cls = pred_cls.astype(int), gt_cls.astype(int)
    same_class = gt_cls[:, None] == pred_cls[None]
    weighted_iou = iou * pred_conf[None]
    scores = (
        np.max(pred_conf * (1 - np.max(iou, axis=0, initial=0)), initial=0),
        np.mean(1 - np.max(np.where(same_class, weighted_iou, 0), axis=1, initial=0)) if len(gt_cls) else 0.0,
        np.max(np.where(~same_class, weighted_iou, 0), initial=0),
    )
    return dict(zip(_LABEL_ISSUES, map(float, scores)))
