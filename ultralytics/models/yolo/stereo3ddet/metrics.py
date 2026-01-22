# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""3D Detection Metrics for Stereo 3D Object Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.utils import DataExportMixin, LOGGER, SimpleClass


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

    def _diagnostic_log_statistics_accumulation(self, stats_list: list[dict], batch_idx: int) -> None:
        """Log statistics accumulation across batches for diagnostic purposes.

        Args:
            stats_list: List of batch statistics dictionaries.
            batch_idx: Current batch index.
        """
        try:
            LOGGER.info(f"[DIAG] Batch {batch_idx}: Statistics Accumulation")
            LOGGER.info(f"  Total batches: {len(stats_list)}")
            if len(stats_list) > 0:
                first_batch = stats_list[0]
                LOGGER.info(f"  Batch 0: tp shape={first_batch.get('tp', np.array([])).shape}, fp shape={first_batch.get('fp', np.array([])).shape}, conf len={len(first_batch.get('conf', []))}, pred_cls len={len(first_batch.get('pred_cls', []))}, target_cls len={len(first_batch.get('target_cls', []))}")
                if len(stats_list) > 1:
                    # Show concatenated shape if we have multiple batches
                    try:
                        tp_concat = np.concatenate([s["tp"] for s in stats_list], axis=0) if stats_list[0]["tp"].size > 0 else np.zeros((0, 2), dtype=bool)
                        fp_concat = np.concatenate([s["fp"] for s in stats_list], axis=0) if stats_list[0]["fp"].size > 0 else np.zeros((0, 2), dtype=bool)
                        conf_concat = np.concatenate([s["conf"] for s in stats_list], axis=0) if len(stats_list[0]["conf"]) > 0 else np.array([])
                        LOGGER.info(f"  After concatenation: tp shape={tp_concat.shape}, fp shape={fp_concat.shape}, conf len={len(conf_concat)}")
                    except Exception as e:
                        LOGGER.warning(f"  Could not compute concatenated shapes: {e}")
        except Exception as e:
            LOGGER.warning(f"Diagnostic logging failed (_diagnostic_log_statistics_accumulation): {e}")

    def _diagnostic_log_ground_truth_counting(
        self, target_cls: np.ndarray, nt_per_class: np.ndarray, nc: int
    ) -> None:
        """Log ground truth counting results for diagnostic purposes.

        Args:
            target_cls: Target class IDs array of shape [Total_M].
            nt_per_class: Count per class array of shape [nc].
            nc: Number of classes.
        """
        try:
            LOGGER.info("[DIAG] Ground Truth Counting")
            LOGGER.info(f"  nc (num classes): {nc}")
            unique_target_cls = np.unique(target_cls).tolist() if len(target_cls) > 0 else []
            target_min = int(np.min(target_cls)) if len(target_cls) > 0 else 0
            target_max = int(np.max(target_cls)) if len(target_cls) > 0 else 0
            LOGGER.info(f"  target_cls: shape={target_cls.shape}, unique={unique_target_cls}, min={target_min}, max={target_max}")
            LOGGER.info(f"  nt_per_class: {nt_per_class.tolist()}")
            class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
            for i in range(min(nc, len(nt_per_class))):
                class_name = class_names.get(i, f"class_{i}")
                LOGGER.info(f"    Class {i} ({class_name}): {int(nt_per_class[i])}")
        except Exception as e:
            LOGGER.warning(f"Diagnostic logging failed (_diagnostic_log_ground_truth_counting): {e}")

    def _diagnostic_log_ap3d_computation(
        self, tp_class: np.ndarray, fp_class: np.ndarray, n_gt: int, class_id: int, iou_thresh: float
    ) -> None:
        """Log AP3D computation inputs and results for diagnostic purposes.

        Args:
            tp_class: TP array for this class and threshold.
            fp_class: FP array for this class and threshold.
            n_gt: Number of ground truth objects for this class.
            class_id: Class ID being processed.
            iou_thresh: IoU threshold (0.5 or 0.7).
        """
        try:
            from ultralytics.utils.metrics import compute_ap

            tp_count = int(np.sum(tp_class))
            fp_count = int(np.sum(fp_class))
            tp_sum = int(np.sum(tp_class))
            fp_sum = int(np.sum(fp_class))

            # Compute precision and recall for diagnostic output
            tp_cumsum = tp_class.cumsum()
            fp_cumsum = fp_class.cumsum()
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            recall = tp_cumsum / (n_gt + 1e-16)

            ap, _, _ = compute_ap(recall.tolist(), precision.tolist())

            LOGGER.info(f"[DIAG] AP3D Computation: Class {class_id} @ IoU {iou_thresh}")
            LOGGER.info(f"  n_gt: {n_gt}")
            LOGGER.info(f"  TP: shape={tp_class.shape}, count={tp_count}, sum={tp_sum}")
            LOGGER.info(f"  FP: shape={fp_class.shape}, count={fp_count}, sum={fp_sum}")
            if len(precision) > 0:
                LOGGER.info(f"  Precision array: len={len(precision)}, range=[{float(np.min(precision)):.4f}, {float(np.max(precision)):.4f}]")
            if len(recall) > 0:
                LOGGER.info(f"  Recall array: len={len(recall)}, range=[{float(np.min(recall)):.4f}, {float(np.max(recall)):.4f}]")
            LOGGER.info(f"  AP result: {ap:.4f}")
        except Exception as e:
            LOGGER.warning(f"Diagnostic logging failed (_diagnostic_log_ap3d_computation): {e}")

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

        # Handle case where there are no predictions but there are ground truth boxes
        # In this case, we still want to compute metrics (they'll be 0, but structure should be there)
        if len(stats["conf"]) == 0:
            # No predictions, but may have ground truth - use target_cls to determine classes
            unique_classes = np.unique(stats["target_cls"]) if len(stats["target_cls"]) > 0 else np.array([])
            if len(unique_classes) == 0:
                return {}
            # Create empty arrays for predictions
            tp = np.zeros((0, 2), dtype=bool)
            fp = np.zeros((0, 2), dtype=bool)
            conf = np.array([])
            pred_cls = np.array([], dtype=int)
        else:
            # Sort by confidence (descending)
            i = np.argsort(-stats["conf"])
            tp = stats["tp"][i]
            fp = stats["fp"][i]
            conf = stats["conf"][i]
            pred_cls = stats["pred_cls"][i]
            # Find unique classes from both predictions and ground truth
            unique_classes = np.unique(np.concatenate([stats["target_cls"], pred_cls])) if len(stats["target_cls"]) > 0 or len(pred_cls) > 0 else np.array([])
        
        if len(unique_classes) == 0:
            return {}

        nc = len(unique_classes)
        nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc) if len(stats["target_cls"]) > 0 else np.zeros(self.nc, dtype=int)

        # DIAGNOSTIC START
        # self._diagnostic_log_ground_truth_counting(stats["target_cls"], nt_per_class, self.nc)
        # DIAGNOSTIC END

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
                        

                # DIAGNOSTIC START
                # self._diagnostic_log_ap3d_computation(tp_class, fp_class, n_gt, c, iou_thresh)
                # DIAGNOSTIC END

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
        result = float(np.mean(values)) if values else 0.0
        
        # DIAGNOSTIC: Log maps3d_50 computation
        
        return result

    @property
    def maps3d_70(self) -> float:
        """Mean AP3D@0.7 across all classes."""
        if not self.ap3d_70:
            return 0.0
        values = [v for v in self.ap3d_70.values() if v > 0]
        result = float(np.mean(values)) if values else 0.0
        
        # DIAGNOSTIC: Log maps3d_70 computation
        
        return result

    def clear_stats(self) -> None:
        """Clear stored statistics."""
        self.stats = []

