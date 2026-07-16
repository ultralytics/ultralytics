# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""3D Detection Metrics for Stereo 3D Object Detection.

Implements KITTI-standard R40 evaluation with difficulty splits (Easy/Moderate/Hard).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.utils import DataExportMixin, SimpleClass

# KITTI difficulty levels
DIFFICULTY_EASY = 0
DIFFICULTY_MODERATE = 1
DIFFICULTY_HARD = 2
DIFFICULTY_NAMES = ["Easy", "Mod", "Hard"]

# Minimum 2D bbox height in pixels for valid predictions (KITTI standard)
MIN_HEIGHT_2D = 25


def classify_difficulty(truncated: float, occluded: int, bbox_height_2d: float) -> int:
    """Classify GT difficulty per KITTI criteria.

    Args:
        truncated: Truncation level [0.0, 1.0] (0=fully visible, 1=fully truncated).
        occluded: Occlusion level (0=visible, 1=partial, 2=heavy, 3=unknown).
        bbox_height_2d: 2D bounding box height in pixels.

    Returns:
        Difficulty level: 0=Easy, 1=Moderate, 2=Hard, -1=DontCare.
    """
    if bbox_height_2d >= 40 and occluded == 0 and truncated <= 0.15:
        return DIFFICULTY_EASY
    if bbox_height_2d >= 25 and occluded <= 1 and truncated <= 0.30:
        return DIFFICULTY_MODERATE
    if bbox_height_2d >= 25 and occluded <= 2 and truncated <= 0.50:
        return DIFFICULTY_HARD
    return -1  # DontCare


def compute_ap_r40(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using 40-point interpolation (KITTI R40 standard).

    Uses max-precision-at-recall-threshold method (not COCO-style sentinel interpolation). For each of 40 recall
    thresholds, finds the maximum precision at recall >= threshold.

    Args:
        recall: Cumulative recall values (ascending).
        precision: Corresponding precision values.

    Returns:
        AP value (float).
    """
    # Precision envelope: monotonically decreasing from right
    envelope = precision.copy()
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])

    # Sample at 40 recall points [1/40, 2/40, ..., 40/40]
    ap = 0.0
    for i in range(40):
        threshold = (i + 1) / 40.0
        mask = recall >= threshold
        ap += float(envelope[mask].max()) if mask.any() else 0.0
    return ap / 40.0


class Stereo3DDetMetrics(SimpleClass, DataExportMixin):
    """3D Detection Metrics Calculator with KITTI-standard R40 evaluation.

    Computes AP3D at IoU thresholds 0.5 and 0.7 for each class and difficulty level (Easy, Moderate, Hard) using
    40-point recall interpolation.

    Stats format (per image):
        pred_boxes: list[Box3D] with confidence and class_id
        gt_boxes: list[Box3D] with truncated and occluded
        iou_matrix: np.ndarray (N_pred, N_gt) 3D IoU matrix
        gt_difficulties: np.ndarray (N_gt,) difficulty per GT (0/1/2/-1)
        pred_heights_2d: np.ndarray (N_pred,) 2D bbox heights in pixels
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        self.names = names
        self.nc = len(names) if names else 0
        self.stats: list[dict[str, Any]] = []
        self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        # metric[iou_thresh][difficulty][class_id] = float
        self.ap3d: dict[float, dict[int, dict[int, float]]] = {}
        self.apbev: dict[float, dict[int, dict[int, float]]] = {}
        self.aos: dict[float, dict[int, dict[int, float]]] = {}

    def update_stats(self, stat: dict[str, Any]) -> None:
        """Store per-image raw data for later processing."""
        self.stats.append(stat)

    def process(
        self,
        save_dir: Path = Path("."),
        plot: bool = False,
        on_plot: callable | None = None,
    ) -> dict[float, dict[int, dict[int, float]]]:
        """Process statistics and compute KITTI-standard AP3D metrics with R40."""
        if not self.stats:
            return {}

        iou_thresholds = [0.5, 0.7]
        difficulties = [DIFFICULTY_EASY, DIFFICULTY_MODERATE, DIFFICULTY_HARD]

        self.ap3d = {iou_t: {diff: {} for diff in difficulties} for iou_t in iou_thresholds}
        self.apbev = {iou_t: {diff: {} for diff in difficulties} for iou_t in iou_thresholds}
        self.aos = {iou_t: {diff: {} for diff in difficulties} for iou_t in iou_thresholds}

        # Collect unique class IDs
        all_classes = set()
        for stat in self.stats:
            for box in stat.get("gt_boxes", []):
                all_classes.add(box.class_id)
            for box in stat.get("pred_boxes", []):
                all_classes.add(box.class_id)
        all_classes = sorted(all_classes)

        if not all_classes:
            return {}

        for diff in difficulties:
            for iou_t in iou_thresholds:
                for cls_id in all_classes:
                    # AP3D + AOS share the 3D matching (AOS = orientation similarity of TPs).
                    ap3d, aos = self._eval_class_difficulty(cls_id, diff, iou_t, "iou_matrix", compute_aos=True)
                    self.ap3d[iou_t][diff][cls_id] = ap3d
                    self.aos[iou_t][diff][cls_id] = aos
                    # AP_BEV uses the ground-plane footprint IoU matrix.
                    apbev, _ = self._eval_class_difficulty(cls_id, diff, iou_t, "bev_iou_matrix", compute_aos=False)
                    self.apbev[iou_t][diff][cls_id] = apbev

        return self.ap3d

    def _eval_class_difficulty(
        self, cls_id: int, max_difficulty: int, iou_thresh: float, iou_key: str, compute_aos: bool = False
    ) -> tuple[float, float]:
        """Compute AP (and optionally AOS) for one class/difficulty/IoU threshold.

        Uses KITTI convention: difficulty X includes all GTs at difficulty <= X.
        Predictions matching same-class GTs outside the difficulty range are ignored
        (neither TP nor FP), following the KITTI "don't care" matching protocol.
        Matching is per-image, but predictions are sorted globally by confidence.

        Args:
            cls_id: Class id to evaluate.
            max_difficulty: Highest difficulty level to include (Easy/Mod/Hard).
            iou_thresh: IoU threshold for a true positive.
            iou_key: Stat key selecting the IoU matrix ("iou_matrix" for 3D, "bev_iou_matrix" for BEV).
            compute_aos: If True, also return Average Orientation Similarity over true positives.

        Returns:
            (ap, aos): R40 AP, and R40 AOS (0.0 when ``compute_aos`` is False).
        """
        all_preds = []  # (confidence, image_idx, local_pred_idx)
        n_gt_total = 0
        per_image_data = []

        for img_idx, stat in enumerate(self.stats):
            gt_boxes = stat.get("gt_boxes", [])
            pred_boxes = stat.get("pred_boxes", [])
            iou_matrix = stat.get(iou_key, np.zeros((0, 0)))
            gt_difficulties = stat.get("gt_difficulties", np.array([], dtype=int))
            pred_heights_2d = stat.get("pred_heights_2d", np.array([]))

            # Evaluate GTs: class match AND valid difficulty within range
            gt_eval_indices = [
                i
                for i, box in enumerate(gt_boxes)
                if box.class_id == cls_id and i < len(gt_difficulties) and 0 <= gt_difficulties[i] <= max_difficulty
            ]

            # Ignored GTs: same class but difficulty outside range (or DontCare=-1)
            # Predictions matching these are neither TP nor FP (KITTI protocol)
            gt_ignored_indices = [
                i
                for i, box in enumerate(gt_boxes)
                if box.class_id == cls_id
                and i < len(gt_difficulties)
                and not (0 <= gt_difficulties[i] <= max_difficulty)
            ]

            # Filter pred: class match AND min 25px height
            pred_indices = [
                i
                for i, box in enumerate(pred_boxes)
                if box.class_id == cls_id and (i >= len(pred_heights_2d) or pred_heights_2d[i] >= MIN_HEIGHT_2D)
            ]

            n_gt_total += len(gt_eval_indices)

            # Sub-IoU matrix for preds x eval GTs
            if iou_matrix.size > 0 and pred_indices and gt_eval_indices:
                sub_iou_eval = iou_matrix[np.ix_(pred_indices, gt_eval_indices)]
            else:
                sub_iou_eval = np.zeros((len(pred_indices), len(gt_eval_indices)))

            # Sub-IoU matrix for preds x ignored GTs (for "don't care" matching)
            if iou_matrix.size > 0 and pred_indices and gt_ignored_indices:
                sub_iou_ignored = iou_matrix[np.ix_(pred_indices, gt_ignored_indices)]
            else:
                sub_iou_ignored = np.zeros((len(pred_indices), len(gt_ignored_indices)))

            # Orientations for AOS (aligned with pred_indices / gt_eval_indices order)
            pred_oris = np.array([pred_boxes[p].orientation for p in pred_indices]) if compute_aos else None
            gt_eval_oris = np.array([gt_boxes[g].orientation for g in gt_eval_indices]) if compute_aos else None

            per_image_data.append(
                {
                    "sub_iou_eval": sub_iou_eval,
                    "sub_iou_ignored": sub_iou_ignored,
                    "pred_oris": pred_oris,
                    "gt_eval_oris": gt_eval_oris,
                    "matched_gt": set(),
                }
            )

            # Collect predictions with global image index
            for local_idx, pred_idx in enumerate(pred_indices):
                all_preds.append((pred_boxes[pred_idx].confidence, img_idx, local_idx))

        if n_gt_total == 0 or not all_preds:
            return 0.0, 0.0

        # Sort globally by confidence (descending)
        all_preds.sort(key=lambda x: x[0], reverse=True)

        # Match predictions to GT in confidence order (per-image greedy matching)
        # Result per prediction: +1 = TP, 0 = FP, -1 = ignored
        match_result = np.zeros(len(all_preds), dtype=int)
        orient_sim = np.zeros(len(all_preds))  # (1 + cos d_theta) / 2 for TPs, else 0
        for i, (_, img_idx, local_idx) in enumerate(all_preds):
            img_data = per_image_data[img_idx]
            sub_iou_eval = img_data["sub_iou_eval"]

            # Try to match with eval GTs first
            best_gt = -1
            best_iou = iou_thresh
            if sub_iou_eval.shape[1] > 0:
                ious = sub_iou_eval[local_idx]
                for gi in range(len(ious)):
                    if gi in img_data["matched_gt"]:
                        continue
                    if ious[gi] >= best_iou:
                        best_iou = ious[gi]
                        best_gt = gi

            if best_gt >= 0:
                match_result[i] = 1  # TP
                img_data["matched_gt"].add(best_gt)
                if compute_aos:
                    dtheta = float(img_data["pred_oris"][local_idx] - img_data["gt_eval_oris"][best_gt])
                    orient_sim[i] = (1.0 + np.cos(dtheta)) / 2.0
            else:
                # Check if prediction overlaps an ignored GT — if so, ignore it
                sub_iou_ignored = img_data["sub_iou_ignored"]
                if sub_iou_ignored.shape[1] > 0:
                    max_ignored_iou = sub_iou_ignored[local_idx].max()
                    if max_ignored_iou >= iou_thresh:
                        match_result[i] = -1  # Ignored (don't count as FP)

        # Filter out ignored predictions, compute cumulative TP/FP
        valid_mask = match_result >= 0
        valid_tp = (match_result == 1)[valid_mask]

        if len(valid_tp) == 0:
            return 0.0, 0.0

        tp_cum = np.cumsum(valid_tp)
        fp_cum = np.cumsum(~valid_tp)
        n_valid = tp_cum + fp_cum
        precision = tp_cum / n_valid
        recall = tp_cum / n_gt_total
        ap = compute_ap_r40(recall, precision)

        aos = 0.0
        if compute_aos:
            # KITTI AOS: orientation-similarity-weighted precision, R40 interpolated.
            sim_cum = np.cumsum(orient_sim[valid_mask])
            aos_precision = sim_cum / n_valid
            aos = compute_ap_r40(recall, aos_precision)

        return ap, aos

    def _mean_metric(
        self, metric: dict[float, dict[int, dict[int, float]]], iou_thresh: float, difficulty: int
    ) -> float:
        """Mean of a metric across real classes (excluding Aux_) for an IoU/difficulty."""
        if not metric or iou_thresh not in metric:
            return 0.0
        diff_dict = metric[iou_thresh].get(difficulty, {})
        if not diff_dict:
            return 0.0
        # Average over model's real classes only (not Aux_ pseudo-classes)
        real_ids = [cid for cid, name in self.names.items() if not name.startswith("Aux_")]
        if not real_ids:
            # Fallback: no names set, average over all classes in diff_dict
            return float(np.mean(list(diff_dict.values())))
        return float(np.mean([diff_dict.get(cid, 0.0) for cid in real_ids]))

    def _mean_ap(self, iou_thresh: float, difficulty: int) -> float:
        """Compute mean AP3D across real classes (excluding Aux_) for given IoU and difficulty."""
        return self._mean_metric(self.ap3d, iou_thresh, difficulty)

    @property
    def results_dict(self) -> dict[str, Any]:
        """Return results as flat dictionary for CSV logging."""
        result = {}

        # Per-class per-difficulty per-IoU for AP3D, AP_BEV, and AOS (skip Aux_ pseudo-classes)
        for prefix, metric in (("AP3D", self.ap3d), ("APBEV", self.apbev), ("AOS", self.aos)):
            for iou_t, diff_dict in metric.items():
                iou_str = str(int(iou_t * 100))
                for diff, cls_dict in diff_dict.items():
                    diff_str = DIFFICULTY_NAMES[diff]
                    for cls_id, val in cls_dict.items():
                        cls_name = self.names.get(cls_id, f"class_{cls_id}")
                        if cls_name.startswith("Aux_"):
                            continue
                        result[f"{prefix}_{cls_name}_{diff_str}_{iou_str}"] = val

        # Summary means (Moderate) across classes
        result["ap3d_50"] = self.maps3d_50
        result["ap3d_70"] = self.maps3d_70
        result["apbev_50"] = self._mean_metric(self.apbev, 0.5, DIFFICULTY_MODERATE)
        result["apbev_70"] = self._mean_metric(self.apbev, 0.7, DIFFICULTY_MODERATE)
        result["aos_50"] = self._mean_metric(self.aos, 0.5, DIFFICULTY_MODERATE)
        result["aos_70"] = self._mean_metric(self.aos, 0.7, DIFFICULTY_MODERATE)
        result["fitness"] = self.fitness

        return result

    @property
    def keys(self) -> list[str]:
        """Return list of metric keys."""
        keys = []
        for prefix in ["AP3D", "APBEV", "AOS"]:
            for iou_str in ["50", "70"]:
                for diff_str in DIFFICULTY_NAMES:
                    for _, cls_name in sorted(self.names.items()):
                        if cls_name.startswith("Aux_"):
                            continue
                        keys.append(f"{prefix}_{cls_name}_{diff_str}_{iou_str}")
        keys.extend(["ap3d_50", "ap3d_70", "apbev_50", "apbev_70", "aos_50", "aos_70"])
        return keys

    @property
    def fitness(self) -> float:
        """Model fitness = mean AP3D@0.5 Moderate across classes."""
        return self.maps3d_50

    @property
    def maps3d_50(self) -> float:
        """Mean AP3D@0.5 Moderate across all classes."""
        return self._mean_ap(0.5, DIFFICULTY_MODERATE)

    @property
    def maps3d_70(self) -> float:
        """Mean AP3D@0.7 Moderate across all classes."""
        return self._mean_ap(0.7, DIFFICULTY_MODERATE)

    def clear_stats(self) -> None:
        """Clear stored statistics."""
        self.stats = []
        self.ap3d = {}
        self.apbev = {}
        self.aos = {}
