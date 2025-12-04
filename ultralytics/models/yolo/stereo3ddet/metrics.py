# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""3D Detection Metrics for Stereo 3D Object Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.utils import DataExportMixin, SimpleClass


class Stereo3DDetMetrics(SimpleClass, DataExportMixin):
    """3D Detection Metrics Calculator.

    Similar to DetMetrics but for 3D detection with AP3D calculation.
    Computes AP3D (3D Average Precision) at IoU thresholds 0.5 and 0.7
    for each class and difficulty level (easy, moderate, hard).

    Attributes:
        names (dict[int, str]): Class name mapping {0: "Car", 1: "Pedestrian", 2: "Cyclist"}.
        ap3d_50 (dict[str, np.ndarray]): AP3D at IoU 0.5 per class and difficulty.
        ap3d_70 (dict[str, np.ndarray]): AP3D at IoU 0.7 per class and difficulty.
        precision (dict[str, np.ndarray]): Precision per class and difficulty.
        recall (dict[str, np.ndarray]): Recall per class and difficulty.
        f1 (dict[str, np.ndarray]): F1 score per class and difficulty.
        stats (dict): Raw statistics (tp, fp, conf, pred_cls, target_cls).
        speed (dict[str, float]): Processing speed metrics.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """Initialize metrics calculator.

        Args:
            names: Class name mapping {0: "Car", 1: "Pedestrian", 2: "Cyclist"}.
        """
        self.names = names
        self.nc = len(names)
        self.ap3d_50 = {}
        self.ap3d_70 = {}
        self.precision = {}
        self.recall = {}
        self.f1 = {}
        self.stats = []
        self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

    def update_stats(self, stat: dict[str, Any]) -> None:
        """Update statistics with batch results.

        Args:
            stat: Dictionary containing:
                - tp: True positives [N]
                - fp: False positives [N]
                - conf: Confidence scores [N]
                - pred_cls: Predicted classes [N]
                - target_cls: Target classes [M]
                - boxes3d_pred: Predicted 3D boxes [N]
                - boxes3d_target: Target 3D boxes [M]
        """
        self.stats.append(stat)

    def process(
        self,
        save_dir: Path = Path("."),
        plot: bool = False,
        on_plot: callable | None = None,
    ) -> dict[str, np.ndarray]:
        """Process statistics and compute AP3D metrics.

        Args:
            save_dir: Directory to save plots.
            plot: Whether to generate plots.
            on_plot: Plot callback.

        Returns:
            Dictionary of computed metrics.
        """
        if not self.stats:
            return {}

        # Concatenate all statistics
        stats = {
            "tp": np.concatenate([s["tp"] for s in self.stats], axis=0) if self.stats[0]["tp"].size > 0 else np.zeros((0, 2), dtype=bool),
            "fp": np.concatenate([s["fp"] for s in self.stats], axis=0) if self.stats[0]["fp"].size > 0 else np.zeros((0, 2), dtype=bool),
            "conf": np.concatenate([s["conf"] for s in self.stats], axis=0) if len(self.stats[0]["conf"]) > 0 else np.array([]),
            "pred_cls": np.concatenate([s["pred_cls"] for s in self.stats], axis=0) if len(self.stats[0]["pred_cls"]) > 0 else np.array([], dtype=int),
            "target_cls": np.concatenate([s["target_cls"] for s in self.stats], axis=0) if len(self.stats[0]["target_cls"]) > 0 else np.array([], dtype=int),
        }

        if len(stats["conf"]) == 0:
            return {}

        # Import compute_ap from utils.metrics
        from ultralytics.utils.metrics import compute_ap

        # Sort by confidence (descending)
        i = np.argsort(-stats["conf"])
        tp = stats["tp"][i]
        fp = stats["fp"][i]
        conf = stats["conf"][i]
        pred_cls = stats["pred_cls"][i]

        # Find unique classes
        unique_classes = np.unique(stats["target_cls"]) if len(stats["target_cls"]) > 0 else np.unique(pred_cls) if len(pred_cls) > 0 else np.array([])
        if len(unique_classes) == 0:
            return {}

        nc = len(unique_classes)
        nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc) if len(stats["target_cls"]) > 0 else np.zeros(self.nc, dtype=int)

        # Compute AP3D for IoU 0.5 and 0.7
        iou_thresholds = [0.5, 0.7]
        ap3d_results = {0.5: {}, 0.7: {}}
        precision_results = {}
        recall_results = {}
        f1_results = {}

        for iou_idx, iou_thresh in enumerate(iou_thresholds):
            ap3d_results[iou_thresh] = {}
            precision_results[iou_thresh] = {}
            recall_results[iou_thresh] = {}
            f1_results[iou_thresh] = {}

            for ci, c in enumerate(unique_classes):
                # Filter predictions for this class
                class_mask = pred_cls == c
                if not class_mask.any():
                    ap3d_results[iou_thresh][c] = 0.0
                    precision_results[iou_thresh][c] = 0.0
                    recall_results[iou_thresh][c] = 0.0
                    f1_results[iou_thresh][c] = 0.0
                    continue

                # Get TP/FP for this class and IoU threshold
                tp_class = tp[class_mask, iou_idx]
                fp_class = fp[class_mask, iou_idx]
                conf_class = conf[class_mask]

                # Number of ground truth for this class
                n_gt = int(nt_per_class[c]) if c < len(nt_per_class) else 0

                if n_gt == 0:
                    ap3d_results[iou_thresh][c] = 0.0
                    precision_results[iou_thresh][c] = 0.0
                    recall_results[iou_thresh][c] = 0.0
                    f1_results[iou_thresh][c] = 0.0
                    continue

                # Sort by confidence (already sorted, but ensure)
                sort_idx = np.argsort(-conf_class)
                tp_class = tp_class[sort_idx]
                fp_class = fp_class[sort_idx]
                conf_class = conf_class[sort_idx]

                # Accumulate TP and FP
                tp_cumsum = tp_class.cumsum()
                fp_cumsum = fp_class.cumsum()

                # Compute precision and recall
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                recall = tp_cumsum / (n_gt + 1e-16)

                # Compute AP using 11-point interpolation (KITTI standard)
                ap, _, _ = compute_ap(recall.tolist(), precision.tolist())

                # Store results
                ap3d_results[iou_thresh][c] = float(ap)
                precision_results[iou_thresh][c] = float(precision[-1]) if len(precision) > 0 else 0.0
                recall_results[iou_thresh][c] = float(recall[-1]) if len(recall) > 0 else 0.0
                f1_results[iou_thresh][c] = (
                    2 * precision_results[iou_thresh][c] * recall_results[iou_thresh][c] / (precision_results[iou_thresh][c] + recall_results[iou_thresh][c] + 1e-16)
                )

        # Store results
        self.ap3d_50 = {self.names.get(c, f"class_{c}"): ap3d_results[0.5].get(c, 0.0) for c in unique_classes}
        self.ap3d_70 = {self.names.get(c, f"class_{c}"): ap3d_results[0.7].get(c, 0.0) for c in unique_classes}
        self.precision = precision_results
        self.recall = recall_results
        self.f1 = f1_results

        return {
            "ap3d_50": self.ap3d_50,
            "ap3d_70": self.ap3d_70,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    @property
    def results_dict(self) -> dict[str, Any]:
        """Return results as dictionary."""
        return {
            "ap3d_50": self.ap3d_50,
            "ap3d_70": self.ap3d_70,
            "maps3d_50": self.maps3d_50,
            "maps3d_70": self.maps3d_70,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    @property
    def keys(self) -> list[str]:
        """Return list of metric keys."""
        return ["ap3d_50", "ap3d_70", "maps3d_50", "maps3d_70"]

    @property
    def maps3d_50(self) -> float:
        """Mean AP3D@0.5 across all classes."""
        if not self.ap3d_50:
            return 0.0
        values = [v for v in self.ap3d_50.values() if v > 0]
        return float(np.mean(values)) if values else 0.0

    @property
    def maps3d_70(self) -> float:
        """Mean AP3D@0.7 across all classes."""
        if not self.ap3d_70:
            return 0.0
        values = [v for v in self.ap3d_70.values() if v > 0]
        return float(np.mean(values)) if values else 0.0

    def clear_stats(self) -> None:
        """Clear stored statistics."""
        self.stats = []

