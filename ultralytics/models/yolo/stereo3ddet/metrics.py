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
        self.nc = len(names) if names else 0
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

        # Concatenate all statistics (filter to non-empty arrays to avoid first-element dependency)
        tp_arrays = [s["tp"] for s in self.stats if s["tp"].size > 0]
        fp_arrays = [s["fp"] for s in self.stats if s["fp"].size > 0]
        conf_arrays = [s["conf"] for s in self.stats if len(s["conf"]) > 0]
        pred_cls_arrays = [s["pred_cls"] for s in self.stats if len(s["pred_cls"]) > 0]
        target_cls_arrays = [s["target_cls"] for s in self.stats if len(s["target_cls"]) > 0]
        stats = {
            "tp": np.concatenate(tp_arrays, axis=0) if tp_arrays else np.zeros((0, 2), dtype=bool),
            "fp": np.concatenate(fp_arrays, axis=0) if fp_arrays else np.zeros((0, 2), dtype=bool),
            "conf": np.concatenate(conf_arrays, axis=0) if conf_arrays else np.array([]),
            "pred_cls": np.concatenate(pred_cls_arrays, axis=0) if pred_cls_arrays else np.array([], dtype=int),
            "target_cls": np.concatenate(target_cls_arrays, axis=0) if target_cls_arrays else np.array([], dtype=int),
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
        unique_classes = np.unique(np.concatenate([stats["target_cls"], pred_cls])) if len(stats["target_cls"]) > 0 or len(pred_cls) > 0 else np.array([])

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
                # Number of ground truth for this class
                n_gt = int(nt_per_class[c]) if c < len(nt_per_class) else 0
                
                # Filter predictions for this class
                class_mask = pred_cls == c if len(pred_cls) > 0 else np.array([], dtype=bool)
                
                if not class_mask.any():
                    # No predictions for this class - AP is 0
                    ap3d_results[iou_thresh][c] = 0.0
                    precision_results[iou_thresh][c] = 0.0
                    recall_results[iou_thresh][c] = 0.0
                    f1_results[iou_thresh][c] = 0.0
                    continue

                # Get TP/FP for this class and IoU threshold
                tp_class = tp[class_mask, iou_idx] if tp.size > 0 else np.array([], dtype=bool)
                fp_class = fp[class_mask, iou_idx] if fp.size > 0 else np.array([], dtype=bool)
                conf_class = conf[class_mask] if len(conf) > 0 else np.array([])

                if n_gt == 0:
                    # No ground truth for this class - AP is 0
                    ap3d_results[iou_thresh][c] = 0.0
                    precision_results[iou_thresh][c] = 0.0
                    recall_results[iou_thresh][c] = 0.0
                    f1_results[iou_thresh][c] = 0.0
                    continue

                # Sort by confidence (already sorted, but ensure)
                if len(conf_class) > 0:
                    sort_idx = np.argsort(-conf_class)
                    tp_class = tp_class[sort_idx]
                    fp_class = fp_class[sort_idx]
                    conf_class = conf_class[sort_idx]

                # Accumulate TP and FP
                tp_cumsum = tp_class.cumsum() if len(tp_class) > 0 else np.array([0], dtype=int)
                fp_cumsum = fp_class.cumsum() if len(fp_class) > 0 else np.array([0], dtype=int)

                # Compute precision and recall
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                recall = tp_cumsum / (n_gt + 1e-16)

                # Compute AP using 11-point interpolation (KITTI standard)
                # If no predictions, AP is 0
                if len(precision) == 0 or len(recall) == 0:
                    ap = 0.0
                else:
                    ap, _, _ = compute_ap(recall.tolist(), precision.tolist())
                    
                    # DIAGNOSTIC: Log AP computation details
                    if ci < 3 and iou_idx < 2:  # Log for first few classes and both IoU thresholds
                        class_name = self.names.get(c, f'class_{c}')
                        


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
        """Return results as dictionary with flat scalar values (T148, T149).
        
        Flattens nested precision/recall/f1 dictionaries and ap3d dictionaries
        into scalar mean values to be compatible with BaseValidator's rounding logic.
        """
        # Flatten nested metric dict to scalar mean value
        def flatten_metric(metric_dict: dict) -> float:
            """Flatten nested metric dict to scalar mean value."""
            if not metric_dict:
                return 0.0
            # Collect all values from nested dict
            all_values = []
            for key, value in metric_dict.items():
                if isinstance(value, dict):
                    # Nested dict: {0.5: {class_id: value}, 0.7: {class_id: value}}
                    all_values.extend([v for v in value.values() if isinstance(v, (int, float))])
                elif isinstance(value, (int, float)):
                    # Flat dict: {class_name: value}
                    all_values.append(value)
            return float(np.mean(all_values)) if all_values else 0.0
        
        # Flatten precision, recall, f1 from nested dict {0.5: {class_id: value}, 0.7: {class_id: value}}
        # to scalar mean values across all IoU thresholds and classes
        precision_scalar = flatten_metric(self.precision) if isinstance(self.precision, dict) else (float(self.precision) if isinstance(self.precision, (int, float)) else 0.0)
        recall_scalar = flatten_metric(self.recall) if isinstance(self.recall, dict) else (float(self.recall) if isinstance(self.recall, (int, float)) else 0.0)
        f1_scalar = flatten_metric(self.f1) if isinstance(self.f1, dict) else (float(self.f1) if isinstance(self.f1, (int, float)) else 0.0)
        
        # Flatten ap3d_50 and ap3d_70 from dict {class_name: value} to scalar mean values
        ap3d_50_scalar = flatten_metric(self.ap3d_50) if isinstance(self.ap3d_50, dict) else (float(self.ap3d_50) if isinstance(self.ap3d_50, (int, float)) else 0.0)
        ap3d_70_scalar = flatten_metric(self.ap3d_70) if isinstance(self.ap3d_70, dict) else (float(self.ap3d_70) if isinstance(self.ap3d_70, (int, float)) else 0.0)
        
        maps3d_50_val = self.maps3d_50
        maps3d_70_val = self.maps3d_70
        
        return {
            "ap3d_50": ap3d_50_scalar,
            "ap3d_70": ap3d_70_scalar,
            "maps3d_50": maps3d_50_val,
            "maps3d_70": maps3d_70_val,
            "precision": precision_scalar,
            "recall": recall_scalar,
            "f1": f1_scalar,
            "fitness": self.fitness,
        }

    @property
    def keys(self) -> list[str]:
        """Return list of metric keys."""
        return ["ap3d_50", "ap3d_70", "maps3d_50", "maps3d_70", "precision", "recall", "f1"]

    @property
    def fitness(self) -> float:
        """Return model fitness as mean AP3D@0.5 for early stopping and best model selection."""
        return self.maps3d_50

    @property
    def maps3d_50(self) -> float:
        """Mean AP3D@0.5 across all classes."""
        if not self.ap3d_50:
            return 0.0
        values = list(self.ap3d_50.values())
        result = float(np.mean(values)) if values else 0.0
        
        # DIAGNOSTIC: Log maps3d_50 computation
        
        return result

    @property
    def maps3d_70(self) -> float:
        """Mean AP3D@0.7 across all classes."""
        if not self.ap3d_70:
            return 0.0
        values = list(self.ap3d_70.values())
        result = float(np.mean(values)) if values else 0.0
        
        # DIAGNOSTIC: Log maps3d_70 computation
        
        return result

    def clear_stats(self) -> None:
        """Clear stored statistics."""
        self.stats = []

