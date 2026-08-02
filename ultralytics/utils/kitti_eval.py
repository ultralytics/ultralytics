# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""KITTI object label I/O and official-style R40 evaluation for monocular 3D detection.

The R40 matching and threshold semantics are an adapted NumPy/OpenCV implementation of the public
``traveller59/kitti-object-eval-python`` protocol. See ``THIRD_PARTY_NOTICES.md`` for attribution and the MIT terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

KITTI_CLASSES = ("Car", "Pedestrian", "Cyclist")
KITTI_DIFFICULTIES = ("easy", "moderate", "hard")
KITTI_IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
_MIN_HEIGHT = (40, 25, 25)
_MAX_OCCLUSION = (0, 1, 2)
_MAX_TRUNCATION = (0.15, 0.30, 0.50)
_NEIGHBOR_CLASSES = {"car": "van", "pedestrian": "person_sitting"}


@dataclass
class KittiAnnotation:
    """Annotations for one KITTI image using camera coordinates and KITTI `(h, w, l)` dimensions."""

    name: np.ndarray
    truncated: np.ndarray
    occluded: np.ndarray
    alpha: np.ndarray
    bbox: np.ndarray
    dimensions: np.ndarray
    location: np.ndarray
    rotation_y: np.ndarray
    score: np.ndarray

    def __len__(self) -> int:
        """Return the number of objects in this image."""
        return len(self.name)


def empty_kitti_annotation() -> KittiAnnotation:
    """Return an empty, correctly shaped KITTI annotation."""
    return KittiAnnotation(
        name=np.empty(0, dtype=str),
        truncated=np.empty(0, dtype=np.float64),
        occluded=np.empty(0, dtype=np.int64),
        alpha=np.empty(0, dtype=np.float64),
        bbox=np.empty((0, 4), dtype=np.float64),
        dimensions=np.empty((0, 3), dtype=np.float64),
        location=np.empty((0, 3), dtype=np.float64),
        rotation_y=np.empty(0, dtype=np.float64),
        score=np.empty(0, dtype=np.float64),
    )


def parse_kitti_label(path: str | Path) -> KittiAnnotation:
    """Parse an official KITTI `label_2` file, including optional prediction scores."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) not in {15, 16}:
                raise ValueError(f"{path}:{line_number}: expected 15 or 16 KITTI fields, found {len(fields)}")
            try:
                rows.append((fields[0], *map(float, fields[1:])))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric KITTI field") from error
    if not rows:
        return empty_kitti_annotation()

    return KittiAnnotation(
        name=np.asarray([row[0] for row in rows]),
        truncated=np.asarray([row[1] for row in rows], dtype=np.float64),
        occluded=np.asarray([row[2] for row in rows], dtype=np.int64),
        alpha=np.asarray([row[3] for row in rows], dtype=np.float64),
        bbox=np.asarray([row[4:8] for row in rows], dtype=np.float64),
        dimensions=np.asarray([row[8:11] for row in rows], dtype=np.float64),
        location=np.asarray([row[11:14] for row in rows], dtype=np.float64),
        rotation_y=np.asarray([row[14] for row in rows], dtype=np.float64),
        score=np.asarray([row[15] if len(row) == 16 else 0.0 for row in rows], dtype=np.float64),
    )


def build_kitti_predictions(
    names: list[str],
    bbox: np.ndarray,
    score: np.ndarray,
    alpha: np.ndarray,
    dimensions: np.ndarray,
    location: np.ndarray,
    rotation_y: np.ndarray,
) -> KittiAnnotation:
    """Build one prediction annotation after validating aligned array lengths and shapes."""
    n = len(names)
    arrays = {
        "bbox": np.asarray(bbox, dtype=np.float64).reshape(-1, 4),
        "score": np.asarray(score, dtype=np.float64).reshape(-1),
        "alpha": np.asarray(alpha, dtype=np.float64).reshape(-1),
        "dimensions": np.asarray(dimensions, dtype=np.float64).reshape(-1, 3),
        "location": np.asarray(location, dtype=np.float64).reshape(-1, 3),
        "rotation_y": np.asarray(rotation_y, dtype=np.float64).reshape(-1),
    }
    if any(len(value) != n for value in arrays.values()):
        lengths = {key: len(value) for key, value in arrays.items()}
        raise ValueError(f"KITTI prediction arrays must contain {n} rows, got {lengths}")
    return KittiAnnotation(
        name=np.asarray(names),
        truncated=np.full(n, -1.0, dtype=np.float64),
        occluded=np.full(n, -1, dtype=np.int64),
        alpha=arrays["alpha"],
        bbox=arrays["bbox"],
        dimensions=arrays["dimensions"],
        location=arrays["location"],
        rotation_y=arrays["rotation_y"],
        score=arrays["score"],
    )


def write_kitti_predictions(annotation: KittiAnnotation, path: str | Path) -> None:
    """Write predictions in the official 16-column KITTI result format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(len(annotation)):
        values = (
            annotation.name[i],
            annotation.truncated[i],
            int(annotation.occluded[i]),
            annotation.alpha[i],
            *annotation.bbox[i],
            *annotation.dimensions[i],
            *annotation.location[i],
            annotation.rotation_y[i],
            annotation.score[i],
        )
        lines.append(
            f"{values[0]} {values[1]:.2f} {values[2]:d} {values[3]:.6f} "
            + " ".join(f"{value:.6f}" for value in values[4:])
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def image_box_overlap(boxes: np.ndarray, query_boxes: np.ndarray, criterion: int = -1) -> np.ndarray:
    """Return pairwise 2D overlap using IoU, box-area, query-area, or raw-intersection normalization."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    query_boxes = np.asarray(query_boxes, dtype=np.float64).reshape(-1, 4)
    if not len(boxes) or not len(query_boxes):
        return np.zeros((len(boxes), len(query_boxes)), dtype=np.float64)
    left_top = np.maximum(boxes[:, None, :2], query_boxes[None, :, :2])
    right_bottom = np.minimum(boxes[:, None, 2:], query_boxes[None, :, 2:])
    intersection = np.prod(np.clip(right_bottom - left_top, 0.0, None), axis=2)
    box_area = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None), axis=1)[:, None]
    query_area = np.prod(np.clip(query_boxes[:, 2:] - query_boxes[:, :2], 0.0, None), axis=1)[None, :]
    if criterion == -1:
        denominator = box_area + query_area - intersection
    elif criterion == 0:
        denominator = np.broadcast_to(box_area, intersection.shape)
    elif criterion == 1:
        denominator = np.broadcast_to(query_area, intersection.shape)
    else:
        denominator = np.ones_like(intersection)
    return np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection),
        where=denominator > 0,
    )


def _bev_corners(annotation: KittiAnnotation) -> tuple[np.ndarray, np.ndarray]:
    """Return oriented footprint corners and valid-dimension flags in camera `(x, z)` coordinates."""
    n = len(annotation)
    corners = np.zeros((n, 4, 2), dtype=np.float64)
    valid = np.isfinite(annotation.dimensions).all(1) & (annotation.dimensions > 0).all(1)
    for i in np.flatnonzero(valid):
        _, width, length = annotation.dimensions[i]
        local = np.asarray(
            [
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2],
            ],
            dtype=np.float64,
        )
        c, s = np.cos(annotation.rotation_y[i]), np.sin(annotation.rotation_y[i])
        rotation = np.asarray([[c, s], [-s, c]], dtype=np.float64)
        corners[i] = local @ rotation.T + annotation.location[i, [0, 2]]
    return corners, valid


def _pairwise_bev_intersection(boxes: KittiAnnotation, query_boxes: KittiAnnotation) -> np.ndarray:
    """Return pairwise exact convex-polygon intersection areas for oriented ground-plane boxes."""
    box_corners, box_valid = _bev_corners(boxes)
    query_corners, query_valid = _bev_corners(query_boxes)
    intersections = np.zeros((len(boxes), len(query_boxes)), dtype=np.float64)
    if not len(boxes) or not len(query_boxes):
        return intersections
    box_min, box_max = box_corners.min(1), box_corners.max(1)
    query_min, query_max = query_corners.min(1), query_corners.max(1)
    candidates = (
        box_valid[:, None]
        & query_valid[None, :]
        & (box_max[:, None, 0] > query_min[None, :, 0])
        & (query_max[None, :, 0] > box_min[:, None, 0])
        & (box_max[:, None, 1] > query_min[None, :, 1])
        & (query_max[None, :, 1] > box_min[:, None, 1])
    )
    for box_index, query_index in np.argwhere(candidates):
        area, _ = cv2.intersectConvexConvex(
            box_corners[box_index].astype(np.float32),
            query_corners[query_index].astype(np.float32),
        )
        intersections[box_index, query_index] = max(float(area), 0.0)
    return intersections


def bev_box_overlap(boxes: KittiAnnotation, query_boxes: KittiAnnotation) -> np.ndarray:
    """Return pairwise rotated BEV IoU for KITTI camera-coordinate boxes."""
    intersection = _pairwise_bev_intersection(boxes, query_boxes)
    box_area = (boxes.dimensions[:, 1] * boxes.dimensions[:, 2])[:, None]
    query_area = (query_boxes.dimensions[:, 1] * query_boxes.dimensions[:, 2])[None, :]
    union = box_area + query_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def d3_box_overlap(boxes: KittiAnnotation, query_boxes: KittiAnnotation) -> np.ndarray:
    """Return pairwise rotated 3D IoU for bottom-centered KITTI camera boxes."""
    bev_intersection = _pairwise_bev_intersection(boxes, query_boxes)
    box_bottom = boxes.location[:, 1][:, None]
    query_bottom = query_boxes.location[:, 1][None, :]
    box_top = box_bottom - boxes.dimensions[:, 0][:, None]
    query_top = query_bottom - query_boxes.dimensions[:, 0][None, :]
    height_intersection = np.clip(np.minimum(box_bottom, query_bottom) - np.maximum(box_top, query_top), 0.0, None)
    intersection = bev_intersection * height_intersection
    box_volume = np.prod(boxes.dimensions, axis=1)[:, None]
    query_volume = np.prod(query_boxes.dimensions, axis=1)[None, :]
    union = box_volume + query_volume - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def paired_d3_box_overlap(boxes: KittiAnnotation, query_boxes: KittiAnnotation) -> np.ndarray:
    """Return exact rotated 3D IoU for aligned pairs of bottom-centered KITTI camera boxes.

    Unlike :func:`d3_box_overlap`, this helper computes only ``boxes[i]`` against ``query_boxes[i]``. It is intended
    for matched-pair diagnostics, where constructing a full ``N x N`` overlap matrix would waste memory and polygon
    intersections. Invalid or non-positive boxes receive IoU zero.
    """
    if len(boxes) != len(query_boxes):
        raise ValueError(f"Aligned 3D IoU requires equal lengths, got {len(boxes)} and {len(query_boxes)}")
    n = len(boxes)
    if not n:
        return np.zeros(0, dtype=np.float64)

    box_corners, box_valid = _bev_corners(boxes)
    query_corners, query_valid = _bev_corners(query_boxes)
    box_valid &= np.isfinite(boxes.location).all(1) & np.isfinite(boxes.rotation_y)
    query_valid &= np.isfinite(query_boxes.location).all(1) & np.isfinite(query_boxes.rotation_y)
    iou = np.zeros(n, dtype=np.float64)
    for index in np.flatnonzero(box_valid & query_valid):
        bev_intersection, _ = cv2.intersectConvexConvex(
            box_corners[index].astype(np.float32),
            query_corners[index].astype(np.float32),
        )
        bev_intersection = max(float(bev_intersection), 0.0)
        box_bottom, query_bottom = (
            boxes.location[index, 1],
            query_boxes.location[index, 1],
        )
        box_top = box_bottom - boxes.dimensions[index, 0]
        query_top = query_bottom - query_boxes.dimensions[index, 0]
        height_intersection = max(min(box_bottom, query_bottom) - max(box_top, query_top), 0.0)
        intersection = bev_intersection * height_intersection
        box_volume = float(np.prod(boxes.dimensions[index]))
        query_volume = float(np.prod(query_boxes.dimensions[index]))
        union = box_volume + query_volume - intersection
        if np.isfinite(union) and union > 0.0:
            iou[index] = np.clip(intersection / union, 0.0, 1.0)
    return iou


def _bev_and_d3_overlap(boxes: KittiAnnotation, query_boxes: KittiAnnotation) -> tuple[np.ndarray, np.ndarray]:
    """Compute BEV and 3D IoU together so full KITTI evaluation intersects each polygon pair only once."""
    bev_intersection = _pairwise_bev_intersection(boxes, query_boxes)
    box_bev_area = (boxes.dimensions[:, 1] * boxes.dimensions[:, 2])[:, None]
    query_bev_area = (query_boxes.dimensions[:, 1] * query_boxes.dimensions[:, 2])[None, :]
    bev_union = box_bev_area + query_bev_area - bev_intersection
    bev_iou = np.divide(
        bev_intersection,
        bev_union,
        out=np.zeros_like(bev_intersection),
        where=bev_union > 0,
    )

    box_bottom = boxes.location[:, 1][:, None]
    query_bottom = query_boxes.location[:, 1][None, :]
    box_top = box_bottom - boxes.dimensions[:, 0][:, None]
    query_top = query_bottom - query_boxes.dimensions[:, 0][None, :]
    height_intersection = np.clip(np.minimum(box_bottom, query_bottom) - np.maximum(box_top, query_top), 0.0, None)
    d3_intersection = bev_intersection * height_intersection
    box_volume = np.prod(boxes.dimensions, axis=1)[:, None]
    query_volume = np.prod(query_boxes.dimensions, axis=1)[None, :]
    d3_union = box_volume + query_volume - d3_intersection
    d3_iou = np.divide(
        d3_intersection,
        d3_union,
        out=np.zeros_like(d3_intersection),
        where=d3_union > 0,
    )
    return bev_iou, d3_iou


def _clean_data(
    ground_truth: KittiAnnotation,
    detections: KittiAnnotation,
    class_name: str,
    difficulty: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Apply KITTI class aliases, difficulty filters and DontCare extraction for one image."""
    target = class_name.lower()
    neighbor = _NEIGHBOR_CLASSES.get(target)
    ignored_gt = np.full(len(ground_truth), -1, dtype=np.int64)
    valid_gt = 0
    dontcare = []
    for i, name in enumerate(ground_truth.name):
        name_lower = name.lower()
        if name_lower == "dontcare":
            dontcare.append(ground_truth.bbox[i])
            continue
        is_target = name_lower == target
        is_neighbor = neighbor is not None and name_lower == neighbor
        height = ground_truth.bbox[i, 3] - ground_truth.bbox[i, 1]
        outside_difficulty = (
            ground_truth.occluded[i] > _MAX_OCCLUSION[difficulty]
            or ground_truth.truncated[i] > _MAX_TRUNCATION[difficulty]
            or height <= _MIN_HEIGHT[difficulty]
        )
        if is_target and not outside_difficulty:
            ignored_gt[i] = 0
            valid_gt += 1
        elif is_neighbor or (is_target and outside_difficulty):
            ignored_gt[i] = 1

    ignored_dt = np.full(len(detections), -1, dtype=np.int64)
    for i, name in enumerate(detections.name):
        height = abs(detections.bbox[i, 3] - detections.bbox[i, 1])
        if height < _MIN_HEIGHT[difficulty]:
            ignored_dt[i] = 1
        elif name.lower() == target:
            ignored_dt[i] = 0
    dontcare_array = np.asarray(dontcare, dtype=np.float64).reshape(-1, 4)
    return valid_gt, ignored_gt, ignored_dt, dontcare_array


def _compute_statistics(
    overlaps: np.ndarray,
    ground_truth: KittiAnnotation,
    detections: KittiAnnotation,
    ignored_gt: np.ndarray,
    ignored_dt: np.ndarray,
    dontcare: np.ndarray,
    metric: str,
    min_overlap: float,
    threshold: float = 0.0,
    compute_fp: bool = False,
    compute_aos: bool = False,
) -> tuple[int, int, int, float, list[float]]:
    """Match detections using the official KITTI score-selection and ignored-object semantics."""
    assigned = np.zeros(len(detections), dtype=bool)
    below_threshold = detections.score < threshold if compute_fp else np.zeros(len(detections), dtype=bool)
    eligible = ignored_dt != -1
    tp = fp = fn = 0
    similarity = 0.0
    matched_scores: list[float] = []
    orientation_deltas: list[float] = []

    for gt_index in range(len(ground_truth)):
        if ignored_gt[gt_index] == -1:
            continue
        candidates = np.flatnonzero(eligible & ~assigned & ~below_threshold & (overlaps[:, gt_index] > min_overlap))
        detection_index = -1
        if len(candidates) and not compute_fp:
            detection_index = int(candidates[np.argmax(detections.score[candidates])])
        elif len(candidates):
            valid_candidates = candidates[ignored_dt[candidates] == 0]
            if len(valid_candidates):
                detection_index = int(valid_candidates[np.argmax(overlaps[valid_candidates, gt_index])])
            else:
                detection_index = int(candidates[0])

        if detection_index == -1 and ignored_gt[gt_index] == 0:
            fn += 1
        elif detection_index >= 0 and (ignored_gt[gt_index] == 1 or ignored_dt[detection_index] == 1):
            assigned[detection_index] = True
        elif detection_index >= 0:
            tp += 1
            assigned[detection_index] = True
            matched_scores.append(float(detections.score[detection_index]))
            if compute_aos:
                orientation_deltas.append(float(ground_truth.alpha[gt_index] - detections.alpha[detection_index]))

    if compute_fp:
        valid_unassigned = ~assigned & (ignored_dt == 0) & ~below_threshold
        fp = int(valid_unassigned.sum())
        if metric == "bbox" and len(dontcare) and fp:
            dc_overlap = image_box_overlap(detections.bbox, dontcare, criterion=0)
            suppressed = valid_unassigned & (dc_overlap > min_overlap).any(axis=1)
            fp -= int(suppressed.sum())
        if compute_aos and (tp or fp):
            similarity = float(sum((1.0 + np.cos(delta)) * 0.5 for delta in orientation_deltas))
        elif compute_aos:
            similarity = -1.0
    return tp, fp, fn, similarity, matched_scores


def _get_thresholds(scores: list[float], num_gt: int, num_sample_points: int = 41) -> list[float]:
    """Select detection score thresholds at the KITTI recall sampling positions."""
    if num_gt <= 0 or not scores:
        return []
    sorted_scores = sorted(scores, reverse=True)
    current_recall = 0.0
    thresholds = []
    for i, score in enumerate(sorted_scores):
        left_recall = (i + 1) / num_gt
        right_recall = (i + 2) / num_gt if i < len(sorted_scores) - 1 else left_recall
        if i < len(sorted_scores) - 1 and right_recall - current_recall < current_recall - left_recall:
            continue
        thresholds.append(score)
        current_recall += 1.0 / (num_sample_points - 1)
    return thresholds


def _metric_overlaps(
    ground_truth: list[KittiAnnotation], detections: list[KittiAnnotation], metric: str
) -> list[np.ndarray]:
    """Precompute per-image detection-to-ground-truth overlaps for one metric."""
    overlaps = []
    for gt, dt in zip(ground_truth, detections):
        if metric == "bbox":
            overlap = image_box_overlap(dt.bbox, gt.bbox)
        elif metric == "bev":
            overlap = bev_box_overlap(dt, gt)
        elif metric == "3d":
            overlap = d3_box_overlap(dt, gt)
        else:
            raise ValueError(f"Unknown KITTI metric '{metric}'")
        overlaps.append(overlap)
    return overlaps


def _evaluate_class_metric(
    ground_truth: list[KittiAnnotation],
    detections: list[KittiAnnotation],
    overlaps: list[np.ndarray],
    class_name: str,
    difficulty: int,
    metric: str,
    min_overlap: float,
    compute_aos: bool = False,
) -> tuple[float, int, np.ndarray]:
    """Evaluate one class/difficulty/metric, returning AP R40 percent, valid GT count and its 41-point curve."""
    cleaned = [_clean_data(gt, dt, class_name, difficulty) for gt, dt in zip(ground_truth, detections)]
    num_gt = sum(item[0] for item in cleaned)
    threshold_scores = []
    for gt, dt, overlap, (_, ignored_gt, ignored_dt, dontcare) in zip(ground_truth, detections, overlaps, cleaned):
        threshold_scores.extend(
            _compute_statistics(
                overlap,
                gt,
                dt,
                ignored_gt,
                ignored_dt,
                dontcare,
                metric,
                min_overlap,
                compute_fp=False,
            )[4]
        )
    thresholds = _get_thresholds(threshold_scores, num_gt)
    precision = np.zeros(41, dtype=np.float64)
    for threshold_index, threshold in enumerate(thresholds[:41]):
        tp = fp = fn = 0
        similarity = 0.0
        for gt, dt, overlap, (_, ignored_gt, ignored_dt, dontcare) in zip(ground_truth, detections, overlaps, cleaned):
            values = _compute_statistics(
                overlap,
                gt,
                dt,
                ignored_gt,
                ignored_dt,
                dontcare,
                metric,
                min_overlap,
                threshold=threshold,
                compute_fp=True,
                compute_aos=compute_aos,
            )
            tp += values[0]
            fp += values[1]
            fn += values[2]
            if values[3] >= 0:
                similarity += values[3]
        denominator = tp + fp
        if denominator:
            precision[threshold_index] = similarity / denominator if compute_aos else tp / denominator
    if thresholds:
        precision[: len(thresholds)] = np.maximum.accumulate(precision[: len(thresholds)][::-1])[::-1]
    return float(precision[1:].mean() * 100.0), num_gt, precision


def evaluate_kitti_metric(
    ground_truth: list[KittiAnnotation],
    detections: list[KittiAnnotation],
    class_name: str = "Car",
    difficulty: str = "moderate",
    metric: str = "3d",
    return_curve: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Compute one exact KITTI R40 class/difficulty/metric slice.

    This is the lightweight counterpart to :func:`evaluate_kitti_r40` for training-time checkpoint selection. It uses
    the same class cleaning, rotated overlap, score thresholds and 41-point interpolation as the full evaluator, but
    avoids calculating unused classes, difficulties and metrics. For example, the defaults return the exact same value
    as ``evaluate_kitti_r40(..., classes=("Car",))["kitti/Car_AP3D_R40_moderate"]``.

    Args:
        ground_truth: Per-image KITTI ground-truth annotations.
        detections: Per-image KITTI prediction annotations aligned with ``ground_truth``.
        class_name: Official KITTI class to evaluate (Car, Pedestrian or Cyclist).
        difficulty: KITTI difficulty level (easy, moderate or hard).
        metric: Metric slice to evaluate (3d, bev or aos).
        return_curve: Return the exact 41-point interpolated precision/similarity curve with the AP value.

    Returns:
        AP R40 in percent, or ``(AP R40, curve)`` when ``return_curve=True``.
    """
    if len(ground_truth) != len(detections):
        raise ValueError(f"KITTI ground truth/prediction image counts differ: {len(ground_truth)} vs {len(detections)}")
    if class_name not in KITTI_CLASSES:
        raise ValueError(f"Unsupported KITTI class '{class_name}'; expected one of {KITTI_CLASSES}")
    if difficulty not in KITTI_DIFFICULTIES:
        raise ValueError(f"Unsupported KITTI difficulty '{difficulty}'; expected one of {KITTI_DIFFICULTIES}")
    metric = metric.lower()
    if metric not in {"3d", "bev", "aos"}:
        raise ValueError("Unsupported KITTI metric; expected one of ('3d', 'bev', 'aos')")

    if metric == "aos":
        has_detections = any(len(annotation) for annotation in detections)
        valid_alpha = has_detections and all(
            not len(annotation) or (np.isfinite(annotation.alpha).all() and np.not_equal(annotation.alpha, -10.0).all())
            for annotation in detections
        )
        if not valid_alpha:
            raise ValueError("KITTI AOS is unavailable because predictions do not provide valid alpha values")
        overlaps = _metric_overlaps(ground_truth, detections, "bbox")
    else:
        overlap_index = 1 if metric == "3d" else 0
        # Use the same joint BEV/3D overlap helper as the full evaluator to guarantee identical floating-point results.
        overlaps = [_bev_and_d3_overlap(dt, gt)[overlap_index] for gt, dt in zip(ground_truth, detections)]

    ap, _, curve = _evaluate_class_metric(
        ground_truth,
        detections,
        overlaps,
        class_name,
        KITTI_DIFFICULTIES.index(difficulty),
        "bbox" if metric == "aos" else metric,
        KITTI_IOU_THRESHOLDS[class_name],
        compute_aos=metric == "aos",
    )
    return (ap, curve) if return_curve else ap


def evaluate_kitti_r40(
    ground_truth: list[KittiAnnotation],
    detections: list[KittiAnnotation],
    classes: tuple[str, ...] | list[str] = KITTI_CLASSES,
    return_curves: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute KITTI AP3D, APBEV and AOS R40, optionally returning the exact interpolated curves used for AP."""
    if len(ground_truth) != len(detections):
        raise ValueError(f"KITTI ground truth/prediction image counts differ: {len(ground_truth)} vs {len(detections)}")
    classes = tuple(name for name in classes if name in KITTI_CLASSES)
    if not classes:
        raise ValueError(f"No supported KITTI classes requested; expected some of {KITTI_CLASSES}")

    # KITTI uses alpha=-10 as the sentinel for predictions that do not provide orientation. Official-style evaluators
    # omit AOS in that case instead of treating -10 radians as a real observation angle.
    has_detections = any(len(annotation) for annotation in detections)
    compute_aos = has_detections and all(
        not len(annotation) or (np.isfinite(annotation.alpha).all() and np.not_equal(annotation.alpha, -10.0).all())
        for annotation in detections
    )

    overlap_cache = {
        "bbox": _metric_overlaps(ground_truth, detections, "bbox"),
        "bev": [],
        "3d": [],
    }
    for gt, dt in zip(ground_truth, detections):
        bev_overlap, d3_overlap = _bev_and_d3_overlap(dt, gt)
        overlap_cache["bev"].append(bev_overlap)
        overlap_cache["3d"].append(d3_overlap)
    results: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}
    moderate_ap3d = []
    for class_name in classes:
        min_overlap = KITTI_IOU_THRESHOLDS[class_name]
        for difficulty_index, difficulty_name in enumerate(KITTI_DIFFICULTIES):
            ap_bev, bev_gt, bev_curve = _evaluate_class_metric(
                ground_truth,
                detections,
                overlap_cache["bev"],
                class_name,
                difficulty_index,
                "bev",
                min_overlap,
            )
            ap_3d, d3_gt, d3_curve = _evaluate_class_metric(
                ground_truth,
                detections,
                overlap_cache["3d"],
                class_name,
                difficulty_index,
                "3d",
                min_overlap,
            )
            prefix = f"kitti/{class_name}"
            results[f"{prefix}_AP3D_R40_{difficulty_name}"] = ap_3d
            results[f"{prefix}_APBEV_R40_{difficulty_name}"] = ap_bev
            curves[f"{prefix}_AP3D_R40_{difficulty_name}"] = d3_curve
            curves[f"{prefix}_APBEV_R40_{difficulty_name}"] = bev_curve
            if compute_aos:
                aos, _, aos_curve = _evaluate_class_metric(
                    ground_truth,
                    detections,
                    overlap_cache["bbox"],
                    class_name,
                    difficulty_index,
                    "bbox",
                    min_overlap,
                    compute_aos=True,
                )
                results[f"{prefix}_AOS_R40_{difficulty_name}"] = aos
                curves[f"{prefix}_AOS_R40_{difficulty_name}"] = aos_curve
            if difficulty_name == "moderate" and d3_gt > 0 and bev_gt > 0:
                moderate_ap3d.append(ap_3d)
    results["kitti/AP3D_R40_moderate"] = float(np.mean(moderate_ap3d)) if moderate_ap3d else 0.0
    return (results, curves) if return_curves else results


def format_kitti_r40(results: dict[str, float], classes: tuple[str, ...] | list[str]) -> str:
    """Format the class-standard KITTI R40 metrics as a compact table."""
    lines = [
        "KITTI R40 (class-standard IoU)",
        f"{'Class':<12}{'Metric':<9}{'Easy':>10}{'Moderate':>12}{'Hard':>10}",
    ]
    for class_name in classes:
        if class_name not in KITTI_CLASSES:
            continue
        for metric in ("AP3D", "APBEV", "AOS"):
            keys = [f"kitti/{class_name}_{metric}_R40_{difficulty}" for difficulty in KITTI_DIFFICULTIES]
            if not all(key in results for key in keys):
                continue
            values = [results[key] for key in keys]
            lines.append(f"{class_name:<12}{metric:<9}{values[0]:>10.2f}{values[1]:>12.2f}{values[2]:>10.2f}")
    return "\n".join(lines)


def plot_kitti_r40(
    results: dict[str, float],
    curves: dict[str, np.ndarray],
    classes: tuple[str, ...] | list[str],
    save_dir: str | Path,
) -> list[Path]:
    """Save a scalar summary chart and one set of R40 curves per evaluated class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = tuple(name for name in classes if name in KITTI_CLASSES)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    difficulty_colors = {"easy": "#159A9C", "moderate": "#E09F3E", "hard": "#C44536"}
    metrics = ("AP3D", "APBEV")
    if all(
        f"kitti/{class_name}_AOS_R40_{difficulty}" in results
        for class_name in classes
        for difficulty in KITTI_DIFFICULTIES
    ):
        metrics += ("AOS",)

    labels = [f"{class_name}\n{metric}" for class_name in classes for metric in metrics]
    x_positions = np.arange(len(labels), dtype=np.float64)
    bar_width = 0.24
    figure_width = max(8.0, len(labels) * 1.7)
    fig, axis = plt.subplots(figsize=(figure_width, 5.2))
    for offset_index, difficulty in enumerate(KITTI_DIFFICULTIES):
        values = [
            results[f"kitti/{class_name}_{metric}_R40_{difficulty}"] for class_name in classes for metric in metrics
        ]
        offset = (offset_index - 1) * bar_width
        bars = axis.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=difficulty.capitalize(),
            color=difficulty_colors[difficulty],
        )
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=8)
    axis.set_title("KITTI R40 summary (class-standard IoU)")
    axis.set_ylabel("Score (%)")
    axis.set_xticks(x_positions, labels)
    axis.set_ylim(0.0, 108.0)
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    summary_path = save_dir / "KITTI_R40_summary.png"
    fig.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(summary_path)

    recall = np.linspace(0.0, 1.0, 41)
    for class_name in classes:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.5), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for axis, metric in zip(axes, metrics):
            for difficulty in KITTI_DIFFICULTIES:
                key = f"kitti/{class_name}_{metric}_R40_{difficulty}"
                axis.plot(
                    recall,
                    curves[key] * 100.0,
                    color=difficulty_colors[difficulty],
                    linewidth=2.0,
                    label=f"{difficulty.capitalize()}  AP={results[key]:.2f}",
                )
            axis.set_title(f"{metric} R40")
            axis.set_xlabel("Recall")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 102.0)
            axis.grid(color="#D9DEE3", linewidth=0.8, alpha=0.8)
            axis.legend(frameon=False, fontsize=8, loc="best")
        axes[0].set_ylabel("Interpolated precision / orientation similarity (%)")
        fig.suptitle(f"KITTI R40 curves - {class_name} (IoU {KITTI_IOU_THRESHOLDS[class_name]:.1f})")
        fig.tight_layout()
        curve_path = save_dir / f"KITTI_R40_{class_name}_curves.png"
        fig.savefig(curve_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(curve_path)
    return output_paths
