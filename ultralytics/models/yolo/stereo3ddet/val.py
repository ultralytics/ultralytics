# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations


import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.stereo.box3d import Box3D

from ultralytics.data.stereo.calib import CalibrationParameters
from ultralytics.models.yolo.stereo3ddet.dense_align_optimized import (
    create_dense_alignment_optimized,
    DenseAlignmentOptimized,
)
from ultralytics.models.yolo.stereo3ddet.occlusion import (
    classify_occlusion,
    should_skip_dense_alignment,
)

from ultralytics.models.yolo.stereo3ddet.geometric import (
    GeometricConstruction,
    solve_geometric_batch,
)

from ultralytics.models.yolo.stereo3ddet.metrics import Stereo3DDetMetrics
from ultralytics.utils import LOGGER, RANK, YAML
from ultralytics.utils.metrics import DetMetrics, box_iou, compute_3d_iou
from ultralytics.utils.plotting import plot_stereo3d_boxes
from ultralytics.utils.profiling import profile_function, profile_section
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.nms import non_max_suppression

# ---------------------------------------------------------------------------
# YOLO11-mapped (Detect-based) decode helpers
# ---------------------------------------------------------------------------


def _decode_orientation_multibin(alpha_bins: torch.Tensor) -> float:
    """Decode Multi-Bin orientation encoding to alpha (observation angle) in radians.

    alpha_bins: Tensor [8] with layout:
      [conf0, conf1, sin0, cos0, sin1, cos1, pad, pad]
    """
    conf0 = float(alpha_bins[0].item())
    conf1 = float(alpha_bins[1].item())
    if conf0 >= conf1:
        bin_center = -math.pi / 2
        sin_v = float(alpha_bins[2].item())
        cos_v = float(alpha_bins[3].item())
    else:
        bin_center = math.pi / 2
        sin_v = float(alpha_bins[4].item())
        cos_v = float(alpha_bins[5].item())
    residual = math.atan2(sin_v, cos_v)
    alpha = bin_center + residual
    return math.atan2(math.sin(alpha), math.cos(alpha))


def decode_stereo3d_outputs_yolo11_p3(
    outputs: dict[str, torch.Tensor],
    conf_threshold: float = 0.25,
    top_k: int = 100,
    calib: dict[str, float] | list[dict[str, float]] | None = None,
    imgsz: int | tuple[int, int] | list[int] | None = None,
    ori_shapes: list[tuple[int, int]] | None = None,
    iou_thres: float = 0.45,
    mean_dims: dict[int, tuple[float, float, float]] | None = None,
    std_dims: dict[int, tuple[float, float, float]] | None = None,
    class_names: dict[int, str] | None = None,
) -> list[Box3D] | list[list[Box3D]]:
    """Decode YOLO11-mapped stereo3ddet outputs (P3-only) to Box3D objects.

    This uses Detect inference output for candidate 2D boxes and class scores, then samples the auxiliary
    stereo/3D maps at the kept P3 indices to estimate depth/dimensions/orientation.

    Args:
        outputs: Model outputs dictionary.
        conf_threshold: Confidence threshold for filtering detections.
        top_k: Maximum number of detections to extract.
        calib: Calibration parameters (dict or list of dicts).
        imgsz: Input image size.
        ori_shapes: Original image shapes per batch item.
        iou_thres: IoU threshold for NMS.
        mean_dims: Mean dimensions per class (class ID -> (H, W, L) in meters).
            Should be provided by dataset configuration.
        std_dims: Standard deviation of dimensions per class (class ID -> (H, W, L) in meters).
            Should be provided by dataset configuration for normalized offset decoding.
            If None, defaults to reasonable estimates.
        class_names: Mapping from class ID to class name (e.g., {0: "Car",1: "Pedestrian", ...}).
            Should be provided by dataset configuration.
    """
    if "det" not in outputs:
        raise KeyError("decode_stereo3d_outputs_yolo11_p3 expected outputs['det']")

    det_out = outputs["det"]
    det_inf = (
        det_out[0] if isinstance(det_out, (tuple, list)) else det_out
    )  # [B, 4+nc, HW]
    bs = int(det_inf.shape[0])
    nc = int(det_inf.shape[1] - 4)

    # Determine letterbox input size
    if imgsz is None:
        imgsz = (384, 384)
    input_h, input_w = (
        (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
    )

    # Feature map size from any aux map (P3 grid)
    sample_aux = None
    for k in ("lr_distance", "dimensions", "orientation"):
        if k in outputs and isinstance(outputs[k], torch.Tensor):
            sample_aux = outputs[k]
            break
    if sample_aux is None:
        raise KeyError(
            "decode_stereo3d_outputs_yolo11_p3 expected at least one aux map in outputs"
        )
    _, _, fh, fw = sample_aux.shape

    # Infer stride in pixels
    stride_w = input_w / fw
    stride_h = input_h / fh
    stride = (stride_w + stride_h) / 2.0

    # NMS on Detect inference output (BCN format)
    dets, keepi = non_max_suppression(
        det_inf,
        conf_thres=conf_threshold,
        iou_thres=iou_thres,
        max_det=top_k,
        nc=nc,
        return_idxs=True,
    )

    # Use mean_dims, std_dims, and class_names from dataset configuration
    # mean_dims is required (should always be provided by dataset YAML)
    # std_dims is required for normalized offset decoding (defaults provided if None)
    # class_names is required (should always be provided by dataset YAML)
    if mean_dims is None:
        raise ValueError("mean_dims must be provided by dataset configuration")
    if std_dims is None:
        raise ValueError("std_dims must be provided by dataset configuration")
    if class_names is None:
        raise ValueError("class_names must be provided by dataset configuration")

    # Original shapes fallback
    if ori_shapes is None or len(ori_shapes) == 0:
        ori_shapes = [(375, 1242)] * bs

    results_per_batch: list[list[Box3D]] = []
    eps = 1e-6

    for b in range(bs):
        # Calibration per sample
        # Guard against empty calib lists (should not happen if dataset is correct, but don't crash validation)
        if len(calib) == 0:
            fx = fy = 721.5377
            cx, cy = 609.5593, 172.8540
            baseline = 0.54
        else:
            cdict = calib[b] if b < len(calib) else calib[0]
            fx = float(cdict.get("fx", 721.5377))
            fy = float(cdict.get("fy", 721.5377))
            cx = float(cdict.get("cx", 609.5593))
            cy = float(cdict.get("cy", 172.8540))
            baseline = float(cdict.get("baseline", 0.54))

        ori_h, ori_w = ori_shapes[b]
        letterbox_scale, pad_left, pad_top = _compute_letterbox_params(
            ori_h, ori_w, imgsz
        )

        # Convert calib and measurements back to original coordinate frame
        fx_orig = fx / letterbox_scale
        fy_orig = fy / letterbox_scale
        cx_orig = (cx - pad_left) / letterbox_scale
        cy_orig = (cy - pad_top) / letterbox_scale

        boxes3d: list[Box3D] = []
        det_b = dets[b]
        idx_b = keepi[b].view(-1).long() if keepi is not None else None
        if det_b is None or det_b.numel() == 0:
            results_per_batch.append(boxes3d)
            continue

        for j, det_row in enumerate(det_b):
            x1_l, y1_l, x2_l, y2_l, conf, cls_f = det_row[:6]
            c = int(cls_f.item())
            confidence = float(conf.item())

            # Map kept index -> feature map location for sampling aux maps
            if idx_b is None or j >= idx_b.numel():
                # Fallback: approximate by bbox center grid location
                u_letterbox = float(((x1_l + x2_l) / 2.0).item())
                v_letterbox = float(((y1_l + y2_l) / 2.0).item())
                gx = int(max(0, min(fw - 1, u_letterbox / stride)))
                gy = int(max(0, min(fh - 1, v_letterbox / stride)))
            else:
                flat = int(idx_b[j].item())
                gy = flat // fw
                gx = flat % fw
                gy = max(0, min(fh - 1, gy))
                gx = max(0, min(fw - 1, gx))

            # Sample aux predictions
            lr_feat = (
                float(outputs["lr_distance"][b, 0, gy, gx].item())
                if "lr_distance" in outputs
                else 0.0
            )
            dim_off = (
                outputs["dimensions"][b, :, gy, gx].float()
                if "dimensions" in outputs
                else torch.zeros(3)
            )
            ori_enc = (
                outputs["orientation"][b, :, gy, gx].float()
                if "orientation" in outputs
                else torch.zeros(8)
            )

            # Depth from disparity
            disparity_letterbox = lr_feat * stride  # pixels in letterboxed space
            disparity_orig = disparity_letterbox / letterbox_scale
            z_3d = (fx_orig * baseline) / max(disparity_orig, eps)

            # Use bbox center as (u,v)
            u_letterbox = float(((x1_l + x2_l) / 2.0).item())
            v_letterbox = float(((y1_l + y2_l) / 2.0).item())
            u_orig = (u_letterbox - pad_left) / letterbox_scale
            v_orig = (v_letterbox - pad_top) / letterbox_scale

            x_3d = (u_orig - cx_orig) * z_3d / fx_orig
            y_3d = (v_orig - cy_orig) * z_3d / fy_orig

            # Dimensions decode (H, W, L) - de-normalize from offset to actual dimensions
            # Prediction is normalized offset: offset = (dim - mean) / std
            # So actual dimension: dim = mean + offset * std
            mean_h, mean_w, mean_l = mean_dims.get(c, mean_dims[0])
            std_h, std_w, std_l = std_dims.get(c, std_dims[0])

            # De-normalize: dimension = mean + offset * std
            height = mean_h + float(dim_off[0].item()) * std_h
            width = mean_w + float(dim_off[1].item()) * std_w
            length = mean_l + float(dim_off[2].item()) * std_l

            # Clamp to reasonable bounds to prevent invalid dimensions
            height = np.clip(height, 0.5, 3.0)  # height in [0.5, 3.0] meters
            width = np.clip(width, 0.3, 2.5)  # width in [0.3, 2.5] meters
            length = np.clip(length, 0.5, 6.0)  # length in [0.5, 6.0] meters

            # Orientation decode: alpha -> theta
            alpha = _decode_orientation_multibin(ori_enc)
            ray_angle = math.atan2(x_3d, z_3d)
            theta = alpha + ray_angle
            theta = math.atan2(math.sin(theta), math.cos(theta))

            # Map bbox back to original image coords
            x1 = (float(x1_l.item()) - pad_left) / letterbox_scale
            y1 = (float(y1_l.item()) - pad_top) / letterbox_scale
            x2 = (float(x2_l.item()) - pad_left) / letterbox_scale
            y2 = (float(y2_l.item()) - pad_top) / letterbox_scale
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            box3d = Box3D(
                center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                dimensions=(float(length), float(width), float(height)),
                orientation=float(theta),
                class_label=class_names.get(c, str(c)),
                class_id=c,
                confidence=confidence,
                bbox_2d=(float(x1), float(y1), float(x2), float(y2)),
            )
            boxes3d.append(box3d)

        results_per_batch.append(boxes3d)

    # Match legacy return format
    if bs == 1:
        return results_per_batch[0]
    return results_per_batch




def get_geometric_config(config_path: str | Path | None = None) -> dict:
    """Load geometric construction configuration from stereo3ddet_full.yaml.

    T015: Parse geometric_construction config section for solver parameters.

    Args:
        config_path: Optional path to config YAML. If None, uses default path.

    Returns:
        Dict with geometric_construction settings:
        - enabled: bool (default True)
        - max_iterations: int (default 10)
        - tolerance: float (default 1e-6)
        - damping: float (default 1e-3)
        - fallback_on_failure: bool (default True)
    """
    global _geometric_config

    # Return cached config if available
    if _geometric_config is not None:
        return _geometric_config

    # Default configuration
    default_config = {
        "enabled": True,
        "max_iterations": 10,
        "tolerance": 1e-6,
        "damping": 1e-3,
        "fallback_on_failure": True,
    }

    # Try to load from config file
    if config_path is None:
        # Try default location
        config_path = (
            Path(__file__).parent.parent.parent
            / "cfg"
            / "models"
            / "stereo3ddet_full.yaml"
        )
    else:
        config_path = Path(config_path)

    if config_path.exists():
        try:
            full_config = YAML.load(str(config_path))
            geo_config = full_config.get("geometric_construction", {})
            # Merge with defaults
            _geometric_config = {**default_config, **geo_config}
        except Exception as e:
            LOGGER.debug("Failed to load geometric config from %s: %s", config_path, e)
            _geometric_config = default_config
    else:
        _geometric_config = default_config

    return _geometric_config


def get_geometric_solver() -> GeometricConstruction:
    """Get or create the global geometric solver instance.

    T014/T016: Provides a singleton solver for consistent convergence tracking.

    Returns:
        GeometricConstruction solver instance with config-based parameters.
    """
    global _geometric_solver

    if _geometric_solver is None:
        config = get_geometric_config()
        _geometric_solver = GeometricConstruction(
            max_iterations=config.get("max_iterations", 10),
            tolerance=config.get("tolerance", 1e-6),
            damping=config.get("damping", 1e-3),
        )

    return _geometric_solver


def reset_geometric_solver() -> None:
    """Reset the global geometric solver and its statistics.

    Call this at the start of validation to reset convergence tracking.
    """
    global _geometric_solver, _geometric_config
    _geometric_solver = None
    _geometric_config = None


def get_geometric_convergence_rate() -> float:
    """Get the convergence rate for SC-007 validation.

    T016: Returns the fraction of geometric solver calls that converged.

    Returns:
        Convergence rate between 0.0 and 1.0. Returns 1.0 if no solves yet.
    """
    if _geometric_solver is None:
        return 1.0
    return _geometric_solver.convergence_rate


def log_geometric_statistics() -> None:
    """Log geometric solver statistics for SC-007 validation.

    T016: Logs convergence rate and total solve count.
    """
    if _geometric_solver is None:
        LOGGER.info("Geometric solver: not initialized (disabled or no detections)")
        return

    rate = _geometric_solver.convergence_rate
    total = _geometric_solver._total_solves
    converged = _geometric_solver._converged_solves

    status = "✓ PASS" if rate >= 0.95 else "✗ FAIL"
    LOGGER.info(
        "Geometric solver: %d/%d converged (%.1f%%) [%s]",
        converged,
        total,
        rate * 100,
        status,
    )


# Global dense alignment instance (GAP-002)
_dense_aligner: DenseAlignmentOptimized | None = None


def get_dense_alignment_config(config_path: str | Path | None = None) -> dict:
    """Load dense alignment configuration from stereo3ddet_full.yaml.

    T023: Parse dense_alignment config section for depth refinement parameters.

    Args:
        config_path: Optional path to config YAML. If None, uses default path.

    Returns:
        Dict with dense_alignment settings:
        - enabled: bool (default True)
        - method: "ncc" or "sad" (default "ncc")
        - depth_search_range: float in meters (default 2.0)
        - depth_steps: int (default 32)
        - patch_size: int in pixels (default 7)
    """
    global _dense_alignment_config

    # Return cached config if available

    # Default configuration
    default_config = {
        "enabled": True,
        "method": "ncc",
        "depth_search_range": 2.0,
        "depth_steps": 32,
        "patch_size": 7,
    }

    # Try to load from config file
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent
            / "cfg"
            / "models"
            / "stereo3ddet_full.yaml"
        )
    else:
        config_path = Path(config_path)

    if config_path.exists():
        try:
            full_config = YAML.load(str(config_path))
            dense_config = full_config.get("dense_alignment", {})
            # Merge with defaults
            _dense_alignment_config = {**default_config, **dense_config}
        except Exception as e:
            LOGGER.debug(
                "Failed to load dense alignment config from %s: %s", config_path, e
            )
            _dense_alignment_config = default_config
    else:
        _dense_alignment_config = default_config

    return _dense_alignment_config


def get_dense_aligner() -> DenseAlignmentOptimized | None:
    """Get or create the global dense alignment instance.

    T023: Provides a singleton aligner for consistent depth refinement.

    Returns:
        DenseAlignment instance if enabled, None if disabled.
    """
    global _dense_aligner
    
    if _dense_aligner is None:
        config = get_dense_alignment_config()
        if not config["enabled"]:
            return None
        _dense_aligner = create_dense_alignment_optimized(config)
    return _dense_aligner


def reset_dense_aligner() -> None:
    """Reset the global dense aligner.

    Call this at the start of validation to reset dense alignment state.
    """
    global _dense_aligner, _dense_alignment_config
    _dense_aligner = None
    _dense_alignment_config = None


# Import occlusion classification for GAP-006 (T040)

# Global occlusion config cache
_occlusion_config: dict | None = None


def get_occlusion_config(config_path: str | Path | None = None) -> dict:
    """Load occlusion configuration from stereo3ddet_full.yaml.

    T040: Parse occlusion config section for dense alignment skipping.

    Args:
        config_path: Optional path to config YAML. If None, uses default path.

    Returns:
        Dict with occlusion settings:
            - enabled: Whether occlusion classification is enabled
            - skip_dense_for_occluded: Whether to skip dense alignment for occluded objects
    """
    global _occlusion_config

    # Return cached config if available
    if _occlusion_config is not None:
        return _occlusion_config

    # Default configuration
    default_config = {
        "enabled": True,
        "skip_dense_for_occluded": True,
    }

    # Try to load from YAML config
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent
            / "cfg"
            / "models"
            / "stereo3ddet_full.yaml"
        )

    try:
        if Path(config_path).exists():
            full_config = YAML.load(str(config_path))
            occ_config = full_config.get("occlusion", {})
            _occlusion_config = {**default_config, **occ_config}
        else:
            LOGGER.debug("Occlusion config file not found at %s", config_path)
            _occlusion_config = default_config
    except Exception as e:
        LOGGER.debug("Failed to load occlusion config from %s: %s", config_path, e)
        _occlusion_config = default_config

    return _occlusion_config


def reset_occlusion_config() -> None:
    """Reset the cached occlusion configuration.

    Useful for testing or when configuration changes.
    """
    global _occlusion_config
    _occlusion_config = None


@profile_function(name="compute_3d_iou_batch")
def compute_3d_iou_batch(
    pred_boxes: list[Box3D],
    gt_boxes: list[Box3D],
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute 3D IoU matrix between prediction and ground truth boxes using vectorized operations.

    This function optimizes IoU computation by:
    1. Batch-generating corners for all boxes (avoiding repeated computation)
    2. Using vectorized operations where possible
    3. Only computing IoU for boxes with matching class_id

    Args:
        pred_boxes: List of predicted Box3D objects.
        gt_boxes: List of ground truth Box3D objects.
        eps: Small value to avoid division by zero.

    Returns:
        IoU matrix of shape (len(pred_boxes), len(gt_boxes)) with IoU values.
        IoU is only computed for boxes with matching class_id; others are set to 0.0.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    # Initialize IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

    # Extract class IDs for filtering
    pred_class_ids = np.array([box.class_id for box in pred_boxes])  # [N]
    gt_class_ids = np.array([box.class_id for box in gt_boxes])  # [M]

    # Create class matching mask: [N, M]
    class_match = pred_class_ids[:, None] == gt_class_ids[None, :]

    # For each matching class, compute IoU in batch
    # This optimization batches corner generation but still computes IoU per pair
    # The main speedup comes from avoiding repeated corner generation

    # Pre-compute corners for all boxes (batch operation)
    def get_box_corners_world(boxes):
        """Get world coordinates of 8 corners for each box."""
        corners_world = []
        for box in boxes:
            x, y, z = box.center_3d
            l, w, h = box.dimensions
            rot = box.orientation

            # Generate 8 corners in object coordinates (KITTI convention)
            # KITTI convention: rotation_y=0 means object faces camera X direction
            # So object's length (forward direction) should be along X axis
            corners_obj = np.array(
                [
                    [
                        -l / 2,
                        l / 2,
                        l / 2,
                        -l / 2,
                        -l / 2,
                        l / 2,
                        l / 2,
                        -l / 2,
                    ],  # x (length)
                    [
                        -h / 2,
                        -h / 2,
                        -h / 2,
                        -h / 2,
                        h / 2,
                        h / 2,
                        h / 2,
                        h / 2,
                    ],  # y (height)
                    [
                        w / 2,
                        w / 2,
                        -w / 2,
                        -w / 2,
                        w / 2,
                        w / 2,
                        -w / 2,
                        -w / 2,
                    ],  # z (width)
                ]
            )  # [3, 8]

            # Rotation matrix around y-axis
            cos_rot, sin_rot = np.cos(rot), np.sin(rot)
            R = np.array([[cos_rot, 0, sin_rot], [0, 1, 0], [-sin_rot, 0, cos_rot]])

            # Rotate and translate to world coordinates
            corners_world_box = R @ corners_obj  # [3, 8]
            corners_world_box[0, :] += x
            corners_world_box[1, :] += y
            corners_world_box[2, :] += z
            corners_world.append(corners_world_box.T)  # [8, 3]

        return np.array(corners_world)  # [N, 8, 3]

    # Batch-generate corners for all boxes
    pred_corners = get_box_corners_world(pred_boxes)  # [N, 8, 3]
    gt_corners = get_box_corners_world(gt_boxes)  # [M, 8, 3]

    # Compute IoU for matching pairs
    # Use the same logic as compute_3d_iou but with pre-computed corners
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if not class_match[i, j]:
                continue

            try:
                # Use existing compute_3d_iou function for correctness
                # The optimization is in batch corner generation above
                iou = compute_3d_iou(pred_boxes[i], gt_boxes[j], eps=eps)
                iou_matrix[i, j] = iou
            except Exception:
                iou_matrix[i, j] = 0.0

    return iou_matrix


def _compute_letterbox_params(
    ori_h: int, ori_w: int, imgsz: int | tuple[int, int] | list[int]
) -> tuple[float, int, int]:
    """Compute letterbox scale and padding from original image size.

    Args:
        ori_h: Original image height.
        ori_w: Original image width.
        imgsz: Letterboxed input size. May be a square int (e.g., 384) or (H, W).

    Returns:
        (scale, pad_left, pad_top) tuple where:
        - scale: Letterbox scale factor (min(imgsz / ori_h, imgsz / ori_w))
        - pad_left: Left padding added by letterbox
        - pad_top: Top padding added by letterbox
    """
    if isinstance(imgsz, int):
        out_h, out_w = imgsz, imgsz
    else:
        out_h, out_w = int(imgsz[0]), int(imgsz[1])
    scale = min(out_h / ori_h, out_w / ori_w)
    new_unpad_w = int(round(ori_w * scale))
    new_unpad_h = int(round(ori_h * scale))
    dw = out_w - new_unpad_w
    dh = out_h - new_unpad_h
    pad_left = dw // 2
    pad_top = dh // 2
    return scale, pad_left, pad_top

def _labels_to_box3d_list(
    labels: list[dict[str, Any]],
    calib: dict[str, float] | None = None,
    names: dict[int, str] | None = None,
    letterbox_scale: float | None = None,
    pad_left: int | None = None,
    pad_top: int | None = None,
    in_h: int | None = None,
    in_w: int | None = None,
) -> list[Box3D]:
    """Convert label dictionaries to Box3D objects.

    Uses provided names mapping, or falls back to paper classes (Car, Pedestrian, Cyclist) if not available.

    Args:
        labels: List of label dictionaries from dataset.
        calib: Calibration parameters (dict or CalibrationParameters object).
        names: Optional class names mapping {class_id: class_name}.
        letterbox_scale: Letterbox scale factor (original -> letterboxed).
        pad_left: Left padding added by letterbox.
        pad_top: Top padding added by letterbox.
        in_h: Letterboxed input image height.
        in_w: Letterboxed input image width.

    Returns:
        List of Box3D objects with filtered and remapped class IDs.
    """
    boxes3d = []

    # Validate required parameters - names must be provided
    if names is None:
        raise ValueError("class_names mapping must be provided")
    class_names = names

    for label in labels:
        class_id = label["class_id"]
        height = label["dimensions"]["height"]
        width = label["dimensions"]["width"]
        length = label["dimensions"]["length"]

        # Prefer GT 3D location if present (more accurate for large/near objects).
        loc = label.get("location_3d", None)
        if isinstance(loc, dict) and all(k in loc for k in ("x", "y", "z")):
            x_3d = float(loc["x"])
            y_bottom = float(loc["y"])
            z_3d = float(loc["z"])
            y_3d = (
                y_bottom - float(height) / 2.0
            )  # bottom-center -> geometric center (Y points down)
        else:
            # Fallback: reconstruct 3D center from stereo disparity (matching prediction pipeline)
            left_box = label["left_box"]
            right_box = label["right_box"]

            # Handle both dict and CalibrationParameters objects
            if isinstance(calib, dict):
                calib_obj = CalibrationParameters.from_dict(calib)
            else:
                calib_obj = calib
            fx_val = calib_obj.fx
            fy_val = calib_obj.fy
            cx_val = calib_obj.cx
            cy_val = calib_obj.cy
            baseline_val = calib_obj.baseline

            img_width = calib_obj.image_width
            img_height = calib_obj.image_height
            left_u = left_box["center_x"] * img_width
            right_u = right_box["center_x"] * img_width
            disparity = left_u - right_u
            # Compute depth from disparity: Z = (f × baseline) / disparity
            depth = (fx_val * baseline_val) / disparity

            # Convert 2D center to 3D
            center_x_2d = left_u
            center_y_2d = left_box.get("center_y", 0.5) * img_height

            x_3d = (center_x_2d - cx_val) * depth / fx_val
            # y_3d is at geometric center (matching prediction decoder convention)
            y_3d = (center_y_2d - cy_val) * depth / fy_val
            z_3d = depth

        rotation_y = label["rotation_y"]
        truncated = label["truncated"]
        occluded = label["occluded"]

        bbox_2d_xywh = label["left_box"]

        # Convert normalized (letterboxed) -> pixels (letterboxed) -> pixels (original)
        if (
            letterbox_scale is not None
            and pad_left is not None
            and pad_top is not None
            and in_h is not None
            and in_w is not None
        ):
            cx_lb_px = bbox_2d_xywh["center_x"] * in_w
            cy_lb_px = bbox_2d_xywh["center_y"] * in_h
            bw_lb_px = bbox_2d_xywh["width"] * in_w
            bh_lb_px = bbox_2d_xywh["height"] * in_h

            x1_lb = cx_lb_px - bw_lb_px / 2
            y1_lb = cy_lb_px - bh_lb_px / 2
            x2_lb = cx_lb_px + bw_lb_px / 2
            y2_lb = cy_lb_px + bh_lb_px / 2

            x1 = (x1_lb - pad_left) / letterbox_scale
            y1 = (y1_lb - pad_top) / letterbox_scale
            x2 = (x2_lb - pad_left) / letterbox_scale
            y2 = (y2_lb - pad_top) / letterbox_scale
        else:
            # Fallback: assume labels are normalized to original image
            if calib is not None:
                if isinstance(calib, dict):
                    calib_obj = CalibrationParameters.from_dict(calib)
                else:
                    calib_obj = calib
                orig_w = calib_obj.image_width
                orig_h = calib_obj.image_height
            else:
                orig_w, orig_h = 1242, 375

            x1 = (bbox_2d_xywh["center_x"] - bbox_2d_xywh["width"] / 2) * orig_w
            y1 = (bbox_2d_xywh["center_y"] - bbox_2d_xywh["height"] / 2) * orig_h
            x2 = (bbox_2d_xywh["center_x"] + bbox_2d_xywh["width"] / 2) * orig_w
            y2 = (bbox_2d_xywh["center_y"] + bbox_2d_xywh["height"] / 2) * orig_h

        bbox_2d_x1y1x2y2 = (x1, y1, x2, y2)

        box3d = Box3D(
            center_3d=(float(x_3d), float(y_3d), float(z_3d)),
            dimensions=(float(length), float(width), float(height)),
            orientation=float(rotation_y),
            class_label=class_names.get(class_id, f"class_{class_id}"),
            class_id=class_id,
            confidence=1.0,  # Ground truth has confidence 1.0
            bbox_2d=bbox_2d_x1y1x2y2,
            truncated=truncated,
            occluded=occluded,
        )
        boxes3d.append(box3d)
    return boxes3d


class Stereo3DDetValidator(BaseValidator):
    """Stereo 3D Detection Validator.

    Extends BaseValidator to implement 3D detection validation with AP3D metrics.
    Computes 3D IoU, matches predictions to ground truth, and calculates AP3D at IoU 0.5 and 0.7.
    """

    def __init__(
        self, dataloader=None, save_dir=None, args=None, _callbacks=None
    ) -> None:
        """Initialize Stereo3DDetValidator.

        Args:
            dataloader: DataLoader for validation data.
            save_dir: Directory to save results.
            args: Configuration arguments.
            _callbacks: Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "stereo3ddet"
        self.iouv = torch.tensor([0.5, 0.7])  # IoU thresholds for AP3D
        self.niou = len(self.iouv)
        self.metrics = Stereo3DDetMetrics()

        # 2D bbox metrics (YOLO-style mAP50/mAP50-95) for debugging bbox head quality.
        self.det_iouv = torch.linspace(0.5, 0.95, 10)
        self.det_metrics = DetMetrics()

        # Mean and std dimensions from dataset config (for decoding)
        self.mean_dims = None
        self.std_dims = None

    def get_dataset(self) -> dict[str, Any]:
        """Parse stereo dataset YAML and return metadata for KITTIStereoDataset.

        This overrides the base implementation to avoid the default YOLO detection dataset checks
        and instead wire up paths/splits intended for the custom `KITTIStereoDataset` loader.

        Returns:
            dict: Dataset dictionary with fields used by the validator and model.
        """
        # Load YAML if a path is provided; accept dicts directly
        data_cfg = self.args.data
        if isinstance(data_cfg, (str, Path)):
            data_cfg = YAML.load(str(data_cfg))

        if not isinstance(data_cfg, dict):
            raise RuntimeError("stereo3ddet: data must be a YAML path or dict")

        # Validate channels for stereo (must be 6 = left RGB + right RGB)
        channels = data_cfg.get("channels", 6)
        if channels != 6:
            raise ValueError(
                f"Stereo3DDet requires 6 input channels (left + right RGB), "
                f"but dataset config has channels={channels}. "
                f"Please set 'channels: 6' in your dataset YAML."
            )

        # Root path and splits
        root_path = data_cfg.get("path") or "."
        root = Path(str(root_path)).resolve()
        # Accept either directory-style train/val or txt; KITTIStereoDataset uses split names
        train_split = data_cfg.get("train_split", "train")
        val_split = data_cfg.get("val_split", "val")

        # Names/nc - must be provided by dataset configuration
        names = data_cfg.get("names")
        if names is None:
            raise ValueError("Dataset configuration must include 'names' mapping")
        nc = data_cfg.get("nc", len(names))

        # Mean dimensions per class (for dimension decoding)
        mean_dims = data_cfg.get("mean_dims")

        # Standard deviation of dimensions per class (for normalized offset decoding)
        std_dims = data_cfg.get("std_dims")

        # Return a dict compatible with BaseValidator expectations, plus stereo descriptors
        return {
            "yaml_file": (
                str(self.args.data) if isinstance(self.args.data, (str, Path)) else None
            ),
            "path": str(root),
            # Channels for model input (6 = left+right stacked)
            "channels": 6,
            # Signal to our get_dataloader/build_dataset that this is a stereo dataset
            "train": {"type": "kitti_stereo", "root": str(root), "split": train_split},
            "val": {"type": "kitti_stereo", "root": str(root), "split": val_split},
            "names": names,
            "nc": nc,
            # carry over optional stereo metadata if present
            "stereo": data_cfg.get("stereo", True),
            "image_size": data_cfg.get("image_size", [375, 1242]),
            "baseline": data_cfg.get("baseline"),
            "focal_length": data_cfg.get("focal_length"),
            "mean_dims": mean_dims,
            "std_dims": std_dims,
        }

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize 6-channel images to float [0,1] and move targets to device.

        Targets are now generated in the dataset's collate_fn, so we just need to
        move them to the device if they're not already there.
        """
        imgs = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (imgs.half() if self.args.half else imgs.float()) / 255.0

        # Make the *current* batch available to postprocess(); BaseValidator calls postprocess() before update_metrics().
        self._current_batch = batch
        # Move targets to device if present (dataset-dependent)
        if "targets" in batch and isinstance(batch["targets"], dict):
            batch["targets"] = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch["targets"].items()
            }
        if "aux_targets" in batch and isinstance(batch["aux_targets"], dict):
            batch["aux_targets"] = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch["aux_targets"].items()
            }
        for k in ("batch_idx", "cls", "bboxes"):
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device, non_blocking=True)

        return batch

    def postprocess(self, preds: dict[str, torch.Tensor]) -> list[list[Box3D]]:
        """Postprocess model outputs to Box3D objects.

        Args:
            preds: Dictionary of 10-branch model outputs.

        Returns:
            List of Box3D lists (one per batch item).
        """
        det_out = preds["det"]
        det_inf = det_out[0] if isinstance(det_out, (tuple, list)) else det_out
        batch_size = int(det_inf.shape[0])
        mode = "det"

        # T212: Get calibration from batch if available
        calib = None
        calibs = []
        ori_shapes = None
        if hasattr(self, "_current_batch") and self._current_batch:
            # Try to get calibration from batch
            calibs = self._current_batch.get("calib", [])
            if calibs:
                # T211: Handle batch calibration - pass list if different per sample, single dict if shared
                if len(calibs) == batch_size and all(
                    isinstance(c, dict) for c in calibs
                ):
                    # Different calibration per sample
                    calib = calibs
                elif len(calibs) > 0 and isinstance(calibs[0], dict):
                    # Shared calibration (use first one)
                    calib = calibs[0]

            # Get original shapes from batch
            ori_shapes = self._current_batch.get("ori_shape", [])

        # Get imgsz from args
        imgsz = getattr(self.args, "imgsz", 384)

        results = decode_stereo3d_outputs_yolo11_p3(
            preds,
            conf_threshold=self.args.conf,
            top_k=100,
            # IMPORTANT: pass resolved `calib` (None, dict, or list[dict]) rather than raw `calibs`.
            # `calibs` may be an empty list if the batch has no calibration (shouldn't happen, but guard anyway).
            calib=calib,
            imgsz=imgsz,
            ori_shapes=ori_shapes,
            iou_thres=getattr(self.args, "iou", 0.45),
            mean_dims=self.mean_dims if hasattr(self, "mean_dims") else None,
            std_dims=self.std_dims if hasattr(self, "std_dims") else None,
            class_names=self.names if hasattr(self, "names") else None,
        )

        # T212: Ensure results is list of lists
        # decode_stereo3d_outputs returns list[list[Box3D]] for batch_size > 1
        # or list[Box3D] for batch_size == 1 (backward compatibility)
        if (
            batch_size == 1
            and isinstance(results, list)
            and len(results) > 0
            and isinstance(results[0], Box3D)
        ):
            # Single sample result - wrap in list for consistency
            results = [results]

        # Apply geometric construction to refine initial 3D estimates
        use_geometric = getattr(self.args, "use_geometric", None)
        geo_config = get_geometric_config()
        should_apply_geo = use_geometric is True or (
            use_geometric is None and geo_config.get("enabled", True)
        )

        if should_apply_geo and hasattr(self, "_current_batch") and self._current_batch:
            results = self._apply_geometric_construction(results, calibs, batch_size)

        # T023: Apply dense alignment for depth refinement (GAP-002)
        # Only apply if enabled in config and we have access to images
        use_dense_alignment = getattr(self.args, "use_dense_alignment", None)
        dense_config = get_dense_alignment_config()

        # Check if dense alignment should be applied
        should_apply_dense = use_dense_alignment is True or (
            use_dense_alignment is None and dense_config.get("enabled", True)
        )

        if (
            should_apply_dense
            and hasattr(self, "_current_batch")
            and self._current_batch
        ):
            results = self._apply_dense_alignment(results, calibs, batch_size)

        return results

    def _apply_dense_alignment(
        self,
        results: list[list[Box3D]],
        calibs: list[dict],
        batch_size: int,
    ) -> list[list[Box3D]]:
        """Apply dense photometric alignment to refine depth estimates.

        T023: Integrate dense alignment into decode pipeline after geometric construction.
        T025: Performance profiling to ensure ≥20 FPS (SC-004/SC-005).

        This method refines the depth estimates of detected 3D boxes using photometric
        matching between left and right stereo images.

        Performance Notes (T025):
            - Typical runtime: ~1-3ms per object (CPU)
            - With 20 objects at 30 FPS: budget ~1.5ms/object → OK
            - Skip occluded objects to reduce overhead
            - Consider reducing depth_steps if too slow

        Args:
            results: List of Box3D lists (one per batch item).
            calibs: List of calibration dictionaries.
            batch_size: Number of images in batch.

        Returns:
            Results with refined depth values.
        """
        # T025: Profile dense alignment for performance monitoring (SC-004/SC-005)
        with profile_section("dense_alignment"):
            # Get dense aligner (returns None if disabled)
            aligner = get_dense_aligner()
            if aligner is None:
                return results

            # Get images from batch
            imgs = self._current_batch.get("img", None)
            if imgs is None:
                return results

            # Images are [B, 6, H, W] tensor - split into left [B, 3, H, W] and right [B, 3, H, W]
            # Channels: [0:3] = left RGB, [3:6] = right RGB
            # Convert to numpy HWC format for dense alignment
            try:
                # Move to CPU and convert to numpy
                imgs_np = imgs.cpu().numpy()  # [B, 6, H, W]

                # Get occlusion config for skipping heavily occluded objects
                occlusion_config = get_dense_alignment_config()
                skip_occluded = occlusion_config.get("skip_dense_for_occluded", True)

                for b in range(min(batch_size, len(results))):
                    boxes = results[b]
                    if not boxes:
                        continue

                    # Extract left and right images for this batch item
                    # Note: images are normalized [0, 1] - convert back to [0, 255] for patch matching
                    left_img = (imgs_np[b, :3].transpose(1, 2, 0) * 255).astype(
                        np.uint8
                    )  # [H, W, 3] RGB
                    right_img = (imgs_np[b, 3:].transpose(1, 2, 0) * 255).astype(
                        np.uint8
                    )  # [H, W, 3] RGB

                    # Get calibration for this sample
                    if calibs and b < len(calibs):
                        sample_calib = (
                            calibs[b] if isinstance(calibs[b], dict) else calibs[0]
                        )
                    elif calibs and len(calibs) > 0:
                        sample_calib = calibs[0] if isinstance(calibs[0], dict) else {}
                    else:
                        sample_calib = {}

                    # Optionally classify occlusion to skip heavily occluded objects
                    occluded_indices = []
                    if skip_occluded and len(boxes) > 1:
                        try:
                            # Build detection list for occlusion classification
                            detections = []
                            for box in boxes:
                                det = {
                                    "bbox_2d": (
                                        box.bbox_2d if box.bbox_2d else (0, 0, 100, 100)
                                    ),
                                    "center_3d": box.center_3d,
                                }
                                detections.append(det)
                            occluded_indices, _ = classify_occlusion(detections)
                        except Exception as e:
                            LOGGER.debug(
                                "Occlusion classification failed for box %d: %s", i, e
                            )
                            occluded_indices = []

                    # Refine depth for each box
                    refined_boxes = []
                    for i, box in enumerate(boxes):
                        # Skip dense alignment for heavily occluded objects
                        if skip_occluded and should_skip_dense_alignment(
                            i, occluded_indices
                        ):
                            refined_boxes.append(box)
                            continue

                        # Create box dict for dense alignment
                        box3d_dict = {
                            "center_3d": box.center_3d,
                            "dimensions": box.dimensions,
                            "orientation": box.orientation,
                        }

                        try:
                            # Refine depth using photometric alignment
                            refined_depth = aligner.refine_depth(
                                left_img=left_img,
                                right_img=right_img,
                                box3d_init=box3d_dict,
                                calib=sample_calib,
                            )

                            # Create new Box3D with refined depth
                            # Update x_3d proportionally with depth change
                            x, y, z = box.center_3d
                            if z > 0 and refined_depth > 0:
                                # Scale x proportionally to depth change
                                depth_ratio = refined_depth / z
                                x_refined = x * depth_ratio
                                y_refined = y * depth_ratio
                            else:
                                x_refined, y_refined = x, y

                            refined_box = Box3D(
                                center_3d=(
                                    float(x_refined),
                                    float(y_refined),
                                    float(refined_depth),
                                ),
                                dimensions=box.dimensions,
                                orientation=box.orientation,
                                class_label=box.class_label,
                                class_id=box.class_id,
                                confidence=box.confidence,
                                bbox_2d=box.bbox_2d,
                                truncated=box.truncated,
                                occluded=box.occluded,
                            )
                            refined_boxes.append(refined_box)
                        except Exception as e:
                            LOGGER.debug("Dense alignment failed for box %d: %s", i, e)
                            refined_boxes.append(box)  # Keep original on failure

                    results[b] = refined_boxes

            except Exception as e:
                LOGGER.debug("Dense alignment batch processing failed: %s", e)
                # Return original results on failure

            return results

    def _apply_geometric_construction(
        self,
        results: list[list[Box3D]],
        calibs: list[dict],
        batch_size: int,
    ) -> list[list[Box3D]]:
        """Apply geometric construction to refine initial 3D estimates.

        Refines 3D box center (x, y, z) and orientation (θ) using Gauss-Newton
        optimization with 7 geometric constraint equations. This improves initial
        estimates before dense alignment.

        Args:
            results: List of Box3D lists (one per batch item).
            calibs: List of calibration dictionaries.
            batch_size: Number of images in batch.

        Returns:
            Results with geometrically refined 3D estimates.
        """
        with profile_section("geometric_construction"):
            config = get_geometric_config()
            if not config.get("enabled", True):
                return results

            solver = get_geometric_solver()
            if solver is None:
                return results

            for b in range(min(batch_size, len(results))):
                boxes = results[b]
                if not boxes:
                    continue

                # Get calibration for this sample
                sample_calib = {}
                if calibs and b < len(calibs) and isinstance(calibs[b], dict):
                    sample_calib = calibs[b]
                elif calibs and len(calibs) > 0 and isinstance(calibs[0], dict):
                    sample_calib = calibs[0]

                # Convert Box3D to detection dicts for geometric solver
                detections = []
                for box in boxes:
                    if box.bbox_2d is None:
                        continue
                    x1, y1, x2, y2 = box.bbox_2d
                    u_left = (x1 + x2) / 2.0
                    v_left = (y1 + y2) / 2.0

                    # Extract center_3d and compute disparity
                    _, _, z_3d = box.center_3d
                    if sample_calib and z_3d > 0:
                        fx = sample_calib.get("fx", 721.5)
                        baseline = sample_calib.get("baseline", 0.54)
                        # Disparity from depth: d = (f * baseline) / z
                        disparity = (fx * baseline) / z_3d
                    else:
                        disparity = 0.0

                    # Dimensions: Box3D stores (length, width, height)
                    l, w, h = box.dimensions

                    det = {
                        "center_2d": (u_left, v_left),
                        "lr_distance": disparity,
                        "dimensions": (l, w, h),
                        "orientation": box.orientation,
                    }
                    detections.append(det)

                if not detections:
                    continue

                # Solve geometric construction batch
                refined_dets, _ = solve_geometric_batch(
                    detections=detections,
                    calib=sample_calib,
                    max_iterations=config.get("max_iterations", 10),
                    tolerance=config.get("tolerance", 1e-6),
                    damping=config.get("damping", 1e-3),
                    fallback_on_failure=True,
                )

                # Convert refined detections back to Box3D
                refined_boxes = []
                for i, box in enumerate(boxes):
                    if i >= len(refined_dets):
                        refined_boxes.append(box)
                        continue

                    refined_det = refined_dets[i]
                    if "center_3d" in refined_det:
                        refined_box = Box3D(
                            center_3d=refined_det["center_3d"],
                            dimensions=box.dimensions,
                            orientation=refined_det.get("orientation", box.orientation),
                            class_label=box.class_label,
                            class_id=box.class_id,
                            confidence=box.confidence,
                            bbox_2d=box.bbox_2d,
                            truncated=box.truncated,
                            occluded=box.occluded,
                        )
                        refined_boxes.append(refined_box)
                    else:
                        refined_boxes.append(box)

                results[b] = refined_boxes

        return results

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics with model information.

        Args:
            model: Model being validated.
        """
        # T016: Reset geometric solver statistics at start of validation
        reset_geometric_solver()

        # Get class names from dataset, not from metrics results_dict (which contains metric keys, not class names)
        # Get class names from dataset configuration - names must be provided
        if hasattr(self, "data") and self.data and "names" in self.data:
            self.names = self.data["names"]
        elif hasattr(model, "names") and model.names:
            self.names = model.names
        else:
            raise ValueError("Dataset configuration must include 'names' mapping")

        self.nc = (
            len(self.names)
            if isinstance(self.names, dict)
            else len(self.names) if isinstance(self.names, (list, tuple)) else 0
        )
        self.seen = 0
        self.metrics.names = self.names
        self.metrics.nc = (
            self.nc
        )  # Also update metrics.nc to match the correct number of classes

        # Parse and convert mean_dims from YAML format (class ID -> [L, W, H]) to (class ID -> (H, W, L))
        mean_dims_raw = (
            self.data.get("mean_dims") if hasattr(self, "data") and self.data else None
        )
        if mean_dims_raw is not None:
            # Convert from {class_id: [L, W, H]} to {class_id: (H, W, L)}
            self.mean_dims = {}
            for class_id, dims in mean_dims_raw.items():
                if isinstance(dims, (list, tuple)) and len(dims) == 3:
                    l, w, h = dims  # YAML has [L, W, H]
                    self.mean_dims[class_id] = (h, w, l)  # Store as (H, W, L)
            LOGGER.info("Loaded mean_dims with %d classes", len(self.mean_dims))
        else:
            self.mean_dims = None
            LOGGER.info("No mean_dims in dataset config, will use defaults")

        # Parse and convert std_dims from YAML format (class ID -> [L, W, H]) to (class ID -> (H, W, L))
        std_dims_raw = (
            self.data.get("std_dims") if hasattr(self, "data") and self.data else None
        )
        if std_dims_raw is not None:
            # Convert from {class_id: [L, W, H]} to {class_id: (H, W, L)}
            self.std_dims = {}
            for class_id, dims in std_dims_raw.items():
                if isinstance(dims, (list, tuple)) and len(dims) == 3:
                    l, w, h = dims  # YAML has [L, W, H]
                    self.std_dims[class_id] = (h, w, l)  # Store as (H, W, L)
            LOGGER.info("Loaded std_dims with %d classes", len(self.std_dims))
        else:
            self.std_dims = None
            LOGGER.info("No std_dims in dataset config, will use defaults")

        # Init 2D detection metrics (bbox mAP)
        self.det_metrics.names = self.names
        self.det_metrics.clear_stats()

    def update_metrics(self, preds: list[list[Box3D]], batch: dict[str, Any]) -> None:
        """Update metrics with predictions and ground truth.

        Args:
            preds: List of predicted Box3D lists (one per image).
            batch: Batch containing ground truth labels.
        """
        with profile_section("update_metrics"):
            self._current_batch = batch  # Store for calibration access

            labels_list = batch.get("labels", [])
            calibs = batch.get("calib", [])
            ori_shapes = batch.get("ori_shape", [])

            for si, (pred_boxes, labels) in enumerate(zip(preds, labels_list)):
                self.seen += 1

                # Get calibration for this sample
                calib = (
                    calibs[si]
                    if si < len(calibs) and isinstance(calibs[si], dict)
                    else None
                )

                # Convert labels to Box3D - need to reverse letterbox transformation on calibration
                # Labels use original image normalized coordinates, but calib is letterboxed
                if calib is not None and si < len(ori_shapes):
                    ori_shape = ori_shapes[si]
                    if isinstance(ori_shape, (list, tuple)) and len(ori_shape) >= 2:
                        actual_h, actual_w = ori_shape[0], ori_shape[1]
                        imgsz = getattr(self.args, "imgsz", 384)

                        # Compute letterbox parameters
                        letterbox_scale, pad_left, pad_top = _compute_letterbox_params(
                            actual_h, actual_w, imgsz
                        )

                        # Reverse letterbox transformation on calibration
                        if isinstance(calib, dict):
                            calib_orig = calib.copy()
                            calib_orig["fx"] = calib["fx"] / letterbox_scale
                            calib_orig["fy"] = calib["fy"] / letterbox_scale
                            calib_orig["cx"] = (
                                calib["cx"] - pad_left
                            ) / letterbox_scale
                            calib_orig["cy"] = (calib["cy"] - pad_top) / letterbox_scale
                            calib_orig["image_width"] = actual_w
                            calib_orig["image_height"] = actual_h
                        else:
                            if isinstance(calib, CalibrationParameters):
                                calib_orig = {
                                    "fx": calib.fx / letterbox_scale,
                                    "fy": calib.fy / letterbox_scale,
                                    "cx": (calib.cx - pad_left) / letterbox_scale,
                                    "cy": (calib.cy - pad_top) / letterbox_scale,
                                    "baseline": calib.baseline,
                                    "image_width": actual_w,
                                    "image_height": actual_h,
                                }
                            else:
                                calib_orig = calib
                        calib = calib_orig
                        # Get letterboxed input size
                        if isinstance(imgsz, int):
                            in_h, in_w = imgsz, imgsz
                        else:
                            in_h, in_w = int(imgsz[0]), int(imgsz[1])

                # Convert labels to Box3D - passing letterbox parameters for bbox_2d conversion
                try:
                    gt_boxes = _labels_to_box3d_list(
                        labels,
                        calib,
                        names=self.names,
                        letterbox_scale=letterbox_scale,
                        pad_left=pad_left,
                        pad_top=pad_top,
                        in_h=in_h,
                        in_w=in_w,
                    )
                except Exception as e:
                    LOGGER.warning(
                        "Error converting labels to Box3D (sample %d): %s", si, e
                    )
                    gt_boxes = []

                # ------------------------------------------------------------
                # 2D bbox metrics (original-image xyxy)
                # ------------------------------------------------------------
                try:
                    # Pred boxes2d from decoded results (already in original coords)
                    pred_bboxes2d = []
                    pred_conf2d = []
                    pred_cls2d = []
                    for pb in pred_boxes:
                        if pb.bbox_2d is None:
                            continue
                        x1, y1, x2, y2 = pb.bbox_2d
                        pred_bboxes2d.append([x1, y1, x2, y2])
                        pred_conf2d.append(float(pb.confidence))
                        pred_cls2d.append(int(pb.class_id))

                    # GT boxes2d from labels (labels are normalized to *letterboxed* input space).
                    gt_bboxes2d = []
                    gt_cls2d = []

                    # Use ori_shapes for inverse-letterbox.
                    imgsz = getattr(self.args, "imgsz", 384)
                    if (
                        si < len(ori_shapes)
                        and isinstance(ori_shapes[si], (list, tuple))
                        and len(ori_shapes[si]) >= 2
                    ):
                        ori_h, ori_w = int(ori_shapes[si][0]), int(ori_shapes[si][1])
                    else:
                        ori_h, ori_w = 375, 1242
                    letterbox_scale, pad_left, pad_top = _compute_letterbox_params(
                        ori_h, ori_w, imgsz
                    )
                    if isinstance(imgsz, int):
                        in_h, in_w = imgsz, imgsz
                    else:
                        in_h, in_w = int(imgsz[0]), int(imgsz[1])

                    for lab in labels:
                        lb = lab.get("left_box", None)
                        if lb is None:
                            continue
                        cls_i = int(lab.get("class_id", 0))
                        cx = float(lb.get("center_x", 0.0)) * in_w
                        cy = float(lb.get("center_y", 0.0)) * in_h
                        bw = float(lb.get("width", 0.0)) * in_w
                        bh = float(lb.get("height", 0.0)) * in_h
                        x1_l = cx - bw / 2
                        y1_l = cy - bh / 2
                        x2_l = cx + bw / 2
                        y2_l = cy + bh / 2
                        # letterbox -> original
                        x1 = (x1_l - pad_left) / letterbox_scale
                        y1 = (y1_l - pad_top) / letterbox_scale
                        x2 = (x2_l - pad_left) / letterbox_scale
                        y2 = (y2_l - pad_top) / letterbox_scale
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        gt_bboxes2d.append([x1, y1, x2, y2])
                        gt_cls2d.append(cls_i)

                    # Compute tp matrix (N,10) for bbox metrics
                    n_pred = len(pred_bboxes2d)
                    if n_pred == 0:
                        tp2d = np.zeros((0, self.det_iouv.numel()), dtype=bool)
                        conf2d = np.zeros((0,), dtype=np.float32)
                        pred_cls_np = np.zeros((0,), dtype=np.int64)
                    else:
                        pred_boxes_t = torch.tensor(pred_bboxes2d, dtype=torch.float32)
                        pred_cls_t = torch.tensor(pred_cls2d, dtype=torch.int64)
                        gt_boxes_t = (
                            torch.tensor(gt_bboxes2d, dtype=torch.float32)
                            if gt_bboxes2d
                            else torch.zeros((0, 4), dtype=torch.float32)
                        )
                        gt_cls_t = (
                            torch.tensor(gt_cls2d, dtype=torch.int64)
                            if gt_cls2d
                            else torch.zeros((0,), dtype=torch.int64)
                        )

                        if gt_boxes_t.shape[0] == 0:
                            tp2d = np.zeros((n_pred, self.det_iouv.numel()), dtype=bool)
                        else:
                            iou2d = box_iou(gt_boxes_t, pred_boxes_t).T  # NxM
                            # Use BaseValidator matching but with det_iouv.
                            # IMPORTANT: guard with try/finally so self.iouv is always restored (prevents leaking 10 IoUs
                            # into the 3D metrics path where tp/fp are shape (N, 2)).
                            old_iouv = self.iouv
                            try:
                                self.iouv = self.det_iouv
                                correct = (
                                    self.match_predictions(pred_cls_t, gt_cls_t, iou2d)
                                    .cpu()
                                    .numpy()
                                )
                            finally:
                                self.iouv = old_iouv
                            tp2d = correct

                        conf2d = np.asarray(pred_conf2d, dtype=np.float32)
                        pred_cls_np = np.asarray(pred_cls2d, dtype=np.int64)

                    target_cls_np = np.asarray(gt_cls2d, dtype=np.int64)
                    self.det_metrics.update_stats(
                        {
                            "tp": tp2d,
                            "conf": conf2d,
                            "pred_cls": pred_cls_np,
                            "target_cls": target_cls_np,
                            "target_img": np.unique(target_cls_np),
                        }
                    )
                except Exception as e:
                    LOGGER.debug("bbox metrics update failed (sample %d): %s", si, e)

                # Handle empty predictions or ground truth
                if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                    continue

                # Compute 3D IoU matrix using vectorized batch computation
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    # Match predictions to ground truth using 3D IoU (vectorized)
                    try:
                        iou_matrix = compute_3d_iou_batch(pred_boxes, gt_boxes)
                    except Exception as e:
                        LOGGER.warning(
                            "Error computing 3D IoU batch: %s, falling back to individual computation",
                            e,
                        )
                        # Fallback to individual computation if batch fails
                        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
                        for i, pred_box in enumerate(pred_boxes):
                            for j, gt_box in enumerate(gt_boxes):
                                if pred_box.class_id == gt_box.class_id:
                                    try:
                                        iou = compute_3d_iou(pred_box, gt_box)
                                        iou_matrix[i, j] = iou
                                    except Exception as e2:
                                        LOGGER.warning("Error computing 3D IoU: %s", e2)
                                        iou_matrix[i, j] = 0.0

                    # Match predictions to ground truth (greedy matching)
                    matched_gt = set()
                    tp = np.zeros((len(pred_boxes), self.niou), dtype=bool)
                    fp = np.zeros((len(pred_boxes), self.niou), dtype=bool)

                    # Sort predictions by confidence
                    pred_indices = sorted(
                        range(len(pred_boxes)),
                        key=lambda i: pred_boxes[i].confidence,
                        reverse=True,
                    )

                    for pred_idx in pred_indices:
                        pred_box = pred_boxes[pred_idx]
                        best_iou = 0.0
                        best_gt_idx = -1

                        # Find best matching ground truth
                        for gt_idx, gt_box in enumerate(gt_boxes):
                            if gt_idx in matched_gt:
                                continue
                            if pred_box.class_id != gt_box.class_id:
                                continue
                            if iou_matrix[pred_idx, gt_idx] > best_iou:
                                best_iou = iou_matrix[pred_idx, gt_idx]
                                best_gt_idx = gt_idx

                        # Check if match exceeds IoU thresholds
                        for iou_idx, iou_thresh in enumerate(self.iouv):
                            if best_iou >= iou_thresh.item():
                                tp[pred_idx, iou_idx] = True
                                if best_gt_idx >= 0:
                                    matched_gt.add(best_gt_idx)
                            else:
                                fp[pred_idx, iou_idx] = True

                else:
                    # No matches possible
                    tp = np.zeros((len(pred_boxes), self.niou), dtype=bool)
                    fp = (
                        np.ones((len(pred_boxes), self.niou), dtype=bool)
                        if len(pred_boxes) > 0
                        else np.zeros((0, self.niou), dtype=bool)
                    )

                # Extract statistics
                conf = (
                    np.array([box.confidence for box in pred_boxes])
                    if pred_boxes
                    else np.array([])
                )
                pred_cls = (
                    np.array([box.class_id for box in pred_boxes])
                    if pred_boxes
                    else np.array([], dtype=int)
                )
                target_cls = (
                    np.array([box.class_id for box in gt_boxes])
                    if gt_boxes
                    else np.array([], dtype=int)
                )

                # DIAGNOSTIC START
                # self._diagnostic_log_statistics_extraction(conf, pred_cls, target_cls, si)
                # DIAGNOSTIC END

                # Update metrics
                self.metrics.update_stats(
                    {
                        "tp": tp,
                        "fp": fp,
                        "conf": conf,
                        "pred_cls": pred_cls,
                        "target_cls": target_cls,
                        "boxes3d_pred": pred_boxes,
                        "boxes3d_target": gt_boxes,
                    }
                )

                # DIAGNOSTIC START
                # self.metrics._diagnostic_log_statistics_accumulation(self.metrics.stats, self.batch_i if hasattr(self, 'batch_i') else 0)
                # DIAGNOSTIC END

                # Update progress bar with intermediate metrics (every batch for real-time feedback)
            if (
                hasattr(self, "_progress_bar")
                and self._progress_bar is not None
                and RANK in {-1, 0}
            ):
                # Update progress bar periodically to avoid performance impact
                if hasattr(self, "_batch_count"):
                    self._batch_count += 1
                else:
                    self._batch_count = 1

                # Update every 5 batches or if we're near the end (more frequent than before)
                if self._batch_count % 5 == 0 or (
                    hasattr(self, "_total_batches")
                    and self._batch_count >= self._total_batches - 1
                ):
                    try:
                        metrics_str = self._format_progress_metrics()
                        if metrics_str:
                            self._progress_bar.set_description(metrics_str)
                    except Exception as e:
                        LOGGER.debug("Error updating progress bar: %s", e)

            # Generate visualization images if plots enabled.
            # NOTE: This stereo validator saves 1 file per sample, so keep defaults conservative to avoid generating
            # thousands of images when using large validation batch sizes.
            #
            # Default to 3 batches (matching Detect task style), but can be overridden via `max_plot_batches`.
            # Additionally cap samples per batch (default=1) via `max_plot_samples`.
            max_plot_batches = getattr(self.args, "max_plot_batches", 3)
            if (
                self.args.plots
                and hasattr(self, "batch_i")
                and self.batch_i < max_plot_batches
                and RANK in {-1, 0}
            ):
                try:
                    self.plot_validation_samples(batch, preds, self.batch_i)
                except Exception as e:
                    LOGGER.warning("Error generating validation visualizations: %s", e)

    def get_desc(self) -> str:
        """Return a formatted string summarizing validation metrics header for progress bar.

        Returns:
            Formatted header string matching the progress bar format.
        """
        # Format: class name (22 chars), then Images (11 chars), Instances (11 chars), then 6 metric columns (11 chars each)
        # This matches the data row format: "%22s" + "%11i" * 2 + "%11.3g" * 6
        return ("%22s" + "%11s" * 8) % (
            "",  # Empty class name column (22 chars)
            "Images".rjust(11),
            "Instances".rjust(11),
            "AP3D@0.5".rjust(11),
            "AP3D@0.7".rjust(11),
            "Precision".rjust(11),
            "Recall".rjust(11),
            "mAP50".rjust(11),
            "mAP50-95".rjust(11),
        )

    def finalize_metrics(self) -> None:
        """Finalize metrics computation."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.det_metrics.speed = self.speed
        self.det_metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Returns:
            Dictionary containing metrics results.
        """
        self.metrics.process(
            save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot
        )
        self.det_metrics.process(
            save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot
        )
        # Merge so training logs/CSV show both 3D and 2D bbox metrics.
        return {**self.metrics.results_dict, **self.det_metrics.results_dict}

    def plot_validation_samples(
        self,
        batch: dict[str, Any],
        pred_boxes3d: list[list[Box3D]],
        batch_idx: int,
    ) -> None:
        """Generate and save validation visualization images with 3D bounding boxes in up-down layout.

        Creates simple up-down layout: predictions on top, ground truth on bottom.
        Uses original image sizes (not resized/letterboxed).

        Args:
            batch: Batch dictionary containing images, labels, and calibration data.
            pred_boxes3d: List of predicted Box3D lists (one per image).
            batch_idx: Batch index for file naming.
        """
        if not self.args.plots:
            return

        try:
            import cv2

            labels_list = batch.get("labels", [])
            calibs = batch.get("calib", [])
            im_files = batch.get("im_file", [])

            if not im_files:
                LOGGER.warning("No image files in batch for visualization")
                return

            batch_size = len(im_files)
            # TEMPORARY: Modified to generate ALL validation samples for error analysis
            max_samples = getattr(self.args, "max_plot_samples", 4)
            num_samples = min(batch_size, max_samples)

            for si in range(num_samples):
                im_file = im_files[si] if si < len(im_files) else None
                if not im_file:
                    continue

                # Load original images from file paths
                left_path = Path(im_file)
                if not left_path.exists():
                    LOGGER.debug(
                        "Left image not found: %s, skipping visualization", left_path
                    )
                    continue

                # Get right image path (same filename, different directory)
                # im_file format: images/{split}/left/{image_id}.png
                # right path: images/{split}/right/{image_id}.png
                right_path = left_path.parent.parent / "right" / left_path.name
                if not right_path.exists():
                    LOGGER.debug(
                        "Right image not found: %s, skipping visualization", right_path
                    )
                    continue

                # Load original images (BGR format from OpenCV)
                left_img = cv2.imread(str(left_path))
                right_img = cv2.imread(str(right_path))

                if left_img is None or right_img is None:
                    LOGGER.debug("Failed to load images for %s, skipping", left_path)
                    continue

                # Get predictions and ground truth for this sample
                pred_boxes = pred_boxes3d[si] if si < len(pred_boxes3d) else []
                labels = labels_list[si] if si < len(labels_list) else []
                calib = (
                    calibs[si]
                    if si < len(calibs) and isinstance(calibs[si], dict)
                    else None
                )

                # Skip visualization if no calibration available
                if calib is None:
                    continue

                # Get actual image dimensions and compute letterbox parameters
                # Calibration in batch is for letterboxed images, but we're visualizing original images
                actual_h, actual_w = left_img.shape[:2]
                imgsz = getattr(self.args, "imgsz", 384)

                # Compute letterbox parameters
                letterbox_scale, pad_left, pad_top = _compute_letterbox_params(
                    actual_h, actual_w, imgsz
                )

                # Reverse letterbox transformation on calibration to get original calibration
                # Original: fx_orig = fx_letterbox / scale, cx_orig = (cx_letterbox - pad_left) / scale
                if isinstance(calib, dict):
                    calib_orig = calib.copy()
                    calib_orig["fx"] = calib["fx"] / letterbox_scale
                    calib_orig["fy"] = calib["fy"] / letterbox_scale
                    calib_orig["cx"] = (calib["cx"] - pad_left) / letterbox_scale
                    calib_orig["cy"] = (calib["cy"] - pad_top) / letterbox_scale
                    # baseline is unchanged (in meters)
                    calib_orig["image_width"] = actual_w
                    calib_orig["image_height"] = actual_h
                else:
                    # For CalibrationParameters object, create a dict with reversed transformation

                    if isinstance(calib, CalibrationParameters):
                        calib_orig = {
                            "fx": calib.fx / letterbox_scale,
                            "fy": calib.fy / letterbox_scale,
                            "cx": (calib.cx - pad_left) / letterbox_scale,
                            "cy": (calib.cy - pad_top) / letterbox_scale,
                            "baseline": calib.baseline,  # unchanged
                            "image_width": actual_w,
                            "image_height": actual_h,
                        }
                    else:
                        calib_orig = calib

                # Get letterboxed input size
                if isinstance(imgsz, int):
                    in_h, in_w = imgsz, imgsz
                else:
                    in_h, in_w = int(imgsz[0]), int(imgsz[1])

                # Convert labels to Box3D for ground truth
                # Use original calibration, pass letterbox parameters for bbox_2d conversion
                gt_boxes = []
                if labels:
                    gt_boxes = _labels_to_box3d_list(
                        labels,
                        calib_orig,
                        names=self.names,
                        letterbox_scale=letterbox_scale,
                        pad_left=pad_left,
                        pad_top=pad_top,
                        in_h=in_h,
                        in_w=in_w,
                    )

                # Filter out predictions with confidence == 0 or below threshold before visualization
                if pred_boxes:
                    conf_threshold = self.args.conf
                    if conf_threshold < 0.1:
                        LOGGER.warning(
                            f"The prediction conf threshold is less than 0.1, you can set the conf through CLI."
                        )
                    pred_boxes = [
                        box
                        for box in pred_boxes
                        if hasattr(box, "confidence")
                        and box.confidence > conf_threshold
                    ]

                # Generate visualization with predictions only (top image)
                # Use original calibration (not letterboxed) since images are original size
                left_pred, right_pred, combined_pred = plot_stereo3d_boxes(
                    left_img=left_img.copy(),
                    right_img=right_img.copy(),
                    pred_boxes3d=pred_boxes,
                    gt_boxes3d=[],  # No ground truth for prediction visualization
                    left_calib=calib_orig,  # Use original calibration
                    letterbox_scale=None,  # No letterboxing - using original size
                    letterbox_pad_left=None,
                    letterbox_pad_top=None,
                )

                # Generate visualization with ground truth only (bottom image)
                # Use original calibration (not letterboxed) since images are original size
                left_gt, right_gt, combined_gt = plot_stereo3d_boxes(
                    left_img=left_img.copy(),
                    right_img=right_img.copy(),
                    pred_boxes3d=[],  # No predictions for ground truth visualization
                    gt_boxes3d=gt_boxes,
                    left_calib=calib_orig,  # Use original calibration
                    letterbox_scale=None,  # No letterboxing - using original size
                    letterbox_pad_left=None,
                    letterbox_pad_top=None,
                )
                # Stack vertically: predictions on top, ground truth on bottom
                # Use left image only for simplicity (or combine left+right horizontally first)
                h_pred, w_pred = combined_pred.shape[:2]
                h_gt, w_gt = combined_gt.shape[:2]

                # Ensure both images have the same width
                if w_pred != w_gt:
                    target_w = max(w_pred, w_gt)
                    if w_pred < target_w:
                        combined_pred = cv2.resize(
                            combined_pred,
                            (target_w, h_pred),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    if w_gt < target_w:
                        combined_gt = cv2.resize(
                            combined_gt,
                            (target_w, h_gt),
                            interpolation=cv2.INTER_LINEAR,
                        )

                # Stack vertically
                stacked = np.vstack([combined_pred, combined_gt])

                # Add labels
                label_height = 30
                stacked_with_labels = np.zeros(
                    (stacked.shape[0] + label_height * 2, stacked.shape[1], 3),
                    dtype=np.uint8,
                )
                stacked_with_labels[
                    label_height : label_height + stacked.shape[0], :, :
                ] = stacked

                # Add text labels
                cv2.putText(
                    stacked_with_labels,
                    "Predictions",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    stacked_with_labels,
                    "Ground Truth",
                    (10, label_height + stacked.shape[0] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Save individual image (one file per sample)
                image_id = left_path.stem
                save_path = (
                    self.save_dir / f"val_batch{batch_idx}_sample{si}_{image_id}.jpg"
                )
                cv2.imwrite(str(save_path), stacked_with_labels)
                if self.on_plot:
                    self.on_plot(save_path)

        except Exception as e:
            LOGGER.warning("Error in plot_validation_samples: %s", e)

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        if not self.metrics.stats:
            LOGGER.warning(
                f"no labels found in {self.args.task} set, can not compute metrics without labels"
            )
            return

        # Count ground truth objects per class
        all_target_cls = (
            np.concatenate(
                [
                    s["target_cls"]
                    for s in self.metrics.stats
                    if len(s["target_cls"]) > 0
                ],
                axis=0,
            )
            if self.metrics.stats
            else np.array([], dtype=int)
        )
        if len(all_target_cls) == 0:
            LOGGER.warning(
                f"no labels found in {self.args.task} set, can not compute metrics without labels"
            )
            return

        nt_per_class = (
            np.bincount(all_target_cls.astype(int), minlength=self.metrics.nc)
            if len(all_target_cls) > 0
            else np.zeros(self.metrics.nc, dtype=int)
        )
        total_gt = int(nt_per_class.sum())

        # Compute images per class (how many images contain each class)
        # Use det_metrics if available (already computed), otherwise compute from stats
        if hasattr(self, "det_metrics") and hasattr(self.det_metrics, "nt_per_image"):
            nt_per_image = self.det_metrics.nt_per_image
        else:
            # Compute from stats: count unique images (by stat index) that contain each class
            nt_per_image = np.zeros(self.metrics.nc, dtype=int)
            for si, stat in enumerate(self.metrics.stats):
                if len(stat.get("target_cls", [])) > 0:
                    unique_classes = np.unique(stat["target_cls"].astype(int))
                    for cls_id in unique_classes:
                        if 0 <= cls_id < self.metrics.nc:
                            nt_per_image[cls_id] += 1

        # Get mean metrics
        maps3d_50 = self.metrics.maps3d_50
        maps3d_70 = self.metrics.maps3d_70

        # Get precision and recall (flatten nested dicts to get mean values)
        precision_mean = 0.0
        recall_mean = 0.0
        if isinstance(self.metrics.precision, dict) and self.metrics.precision:
            all_precisions = []
            for iou_dict in self.metrics.precision.values():
                if isinstance(iou_dict, dict):
                    all_precisions.extend(
                        [v for v in iou_dict.values() if isinstance(v, (int, float))]
                    )
            precision_mean = float(np.mean(all_precisions)) if all_precisions else 0.0

        if isinstance(self.metrics.recall, dict) and self.metrics.recall:
            all_recalls = []
            for iou_dict in self.metrics.recall.values():
                if isinstance(iou_dict, dict):
                    all_recalls.extend(
                        [v for v in iou_dict.values() if isinstance(v, (int, float))]
                    )
            recall_mean = float(np.mean(all_recalls)) if all_recalls else 0.0

        # Get 2D bbox mAP50 and mAP50-95 metrics (for main summary line)
        box_map50 = 0.0
        box_map5095 = 0.0
        try:
            det_res = (
                self.det_metrics.results_dict if hasattr(self, "det_metrics") else {}
            )
            box_map50 = det_res.get("metrics/mAP50(B)", 0.0)
            box_map5095 = det_res.get("metrics/mAP50-95(B)", 0.0)
        except Exception as e:
            LOGGER.debug("Failed to get bbox2d metrics for main summary: %s", e)

        # Print format: class name, images, instances, AP3D@0.5, AP3D@0.7, precision, recall, mAP50, mAP50-95
        # Matches detect task format: "Class", "Images", "Instances", ...
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 6
        LOGGER.info(
            pf
            % (
                "all",
                self.seen,
                total_gt,
                maps3d_50,
                maps3d_70,
                precision_mean,
                recall_mean,
                box_map50,
                box_map5095,
            )
        )

        # Print results per class if verbose and multiple classes
        if self.args.verbose and self.metrics.nc > 1 and self.metrics.ap3d_50:
            # Get per-class 2D bbox metrics for printing
            class_map50_dict = {}
            class_map5095_dict = {}
            try:
                if (
                    hasattr(self, "det_metrics")
                    and hasattr(self.det_metrics, "class_result")
                    and hasattr(self.det_metrics, "ap_class_index")
                ):
                    # Get per-class mAP50 and mAP50-95 from DetMetrics
                    # class_result(i) returns (p, r, map50, map) for index i in ap_class_index
                    # ap_class_index[i] gives the class_id at index i
                    for i, class_id in enumerate(self.det_metrics.ap_class_index):
                        try:
                            class_result = self.det_metrics.class_result(i)
                            if len(class_result) >= 4:
                                class_map50_dict[class_id] = float(
                                    class_result[2]
                                )  # mAP50
                                class_map5095_dict[class_id] = float(
                                    class_result[3]
                                )  # mAP50-95
                        except (IndexError, AttributeError) as e:
                            LOGGER.debug(
                                "Failed to get metrics for class %d at index %d: %s",
                                class_id,
                                i,
                                e,
                            )
                            continue
            except Exception as e:
                LOGGER.debug("Failed to get per-class bbox2d metrics: %s", e)

            for class_id, class_name in self.metrics.names.items():
                ap3d_50_class = self.metrics.ap3d_50.get(class_name, 0.0)
                ap3d_70_class = self.metrics.ap3d_70.get(class_name, 0.0)

                # Get class-specific precision and recall (average across IoU thresholds)
                prec_class = 0.0
                recall_class = 0.0
                if isinstance(self.metrics.precision, dict):
                    prec_values = []
                    for iou_dict in self.metrics.precision.values():
                        if isinstance(iou_dict, dict) and class_id in iou_dict:
                            prec_values.append(iou_dict[class_id])
                    prec_class = float(np.mean(prec_values)) if prec_values else 0.0

                if isinstance(self.metrics.recall, dict):
                    recall_values = []
                    for iou_dict in self.metrics.recall.values():
                        if isinstance(iou_dict, dict) and class_id in iou_dict:
                            recall_values.append(iou_dict[class_id])
                    recall_class = (
                        float(np.mean(recall_values)) if recall_values else 0.0
                    )

                # Get class-specific mAP50 and mAP50-95
                class_map50 = class_map50_dict.get(class_id, 0.0)
                class_map5095 = class_map5095_dict.get(class_id, 0.0)

                nt_class = (
                    int(nt_per_class[class_id]) if class_id < len(nt_per_class) else 0
                )
                nt_images = (
                    int(nt_per_image[class_id]) if class_id < len(nt_per_image) else 0
                )
                # Use same format as main summary: class name, images, instances, then metrics
                LOGGER.info(
                    pf
                    % (
                        class_name,
                        nt_images,  # number of images containing this class
                        nt_class,  # number of ground truth instances for this class
                        ap3d_50_class,
                        ap3d_70_class,
                        prec_class,
                        recall_class,
                        class_map50,
                        class_map5095,
                    )
                )

    def _format_progress_metrics(self) -> str:
        """Format current metrics for progress bar display.

        Returns:
            Formatted string with key metrics in training-style format.
        """
        if not hasattr(self.metrics, "stats") or len(self.metrics.stats) == 0:
            return ("%11i" + "%11s" * 6) % (
                int(self.seen),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            )

        # Compute intermediate metrics on accumulated stats
        try:
            # Save current stats
            saved_stats = self.metrics.stats.copy()
            # Process to get metrics
            temp_results = self.metrics.process(save_dir=self.save_dir, plot=False)
            # Also compute bbox metrics if available
            det_temp = (
                self.det_metrics.process(save_dir=self.save_dir, plot=False)
                if hasattr(self, "det_metrics")
                else {}
            )
            # Restore stats for final processing
            self.metrics.stats = saved_stats

            if not temp_results:
                return ("%11i" + "%11s" * 6) % (
                    int(self.seen),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                )

            # Get AP3D metrics (use mean values)
            ap50 = temp_results.get("maps3d_50", 0.0)
            if isinstance(ap50, dict):
                ap50 = (
                    float(
                        np.mean(
                            [v for v in ap50.values() if isinstance(v, (int, float))]
                        )
                    )
                    if ap50
                    else 0.0
                )
            ap70 = temp_results.get("maps3d_70", 0.0)
            if isinstance(ap70, dict):
                ap70 = (
                    float(
                        np.mean(
                            [v for v in ap70.values() if isinstance(v, (int, float))]
                        )
                    )
                    if ap70
                    else 0.0
                )

            # Get precision and recall (flatten nested dicts)
            precision = temp_results.get("precision", 0.0)
            if isinstance(precision, dict):
                all_precisions = []
                for iou_dict in precision.values():
                    if isinstance(iou_dict, dict):
                        all_precisions.extend(
                            [
                                v
                                for v in iou_dict.values()
                                if isinstance(v, (int, float))
                            ]
                        )
                precision = float(np.mean(all_precisions)) if all_precisions else 0.0

            recall = temp_results.get("recall", 0.0)
            if isinstance(recall, dict):
                all_recalls = []
                for iou_dict in recall.values():
                    if isinstance(iou_dict, dict):
                        all_recalls.extend(
                            [
                                v
                                for v in iou_dict.values()
                                if isinstance(v, (int, float))
                            ]
                        )
                recall = float(np.mean(all_recalls)) if all_recalls else 0.0

            # Get bbox mAPs (DetMetrics uses 'metrics/mAP50(B)' keys)
            map50 = (
                det_temp.get("metrics/mAP50(B)", 0.0)
                if isinstance(det_temp, dict)
                else 0.0
            )
            map5095 = (
                det_temp.get("metrics/mAP50-95(B)", 0.0)
                if isinstance(det_temp, dict)
                else 0.0
            )

            # Format: Images, AP3D@0.5, AP3D@0.7, Precision, Recall, mAP50(B), mAP50-95(B)
            # Use same width format as training for consistency (matches get_desc header)
            # Use %11i for Images (integer count) and %11.4g for float metrics
            return ("%11i" + "%11.4g" * 6) % (
                int(self.seen),  # Images (integer)
                ap50,  # AP3D@0.5
                ap70,  # AP3D@0.7
                precision,  # Precision
                recall,  # Recall
                float(map50),
                float(map5095),
            )
        except Exception as e:
            LOGGER.debug("Error formatting progress metrics: %s", e)
            return ("%11i" + "%11s" * 6) % (
                int(self.seen),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            )

    def build_dataset(
        self,
        img_path: str | dict[str, Any],
        mode: str = "val",
        batch: int | None = None,
    ) -> torch.utils.data.Dataset:
        """Build Stereo3DDetDataset for validation.

        Args:
            img_path: Path to dataset root directory, or a descriptor dict from self.data.get(split).
            mode: 'train' or 'val' mode.
            batch: Batch size (unused, kept for compatibility).

        Returns:
            Stereo3DDetDataset: Dataset instance for validation.
        """
        from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetDataset

        # img_path should be a dir

        # means it's a file instead of the path, return it's parent directory
        if isinstance(img_path, str) and not os.path.isdir(img_path):
            img_path = Path(img_path).parent

        # Handle descriptor dict from self.data.get(self.args.split)
        desc = (
            img_path
            if isinstance(img_path, dict)
            else self.data.get(mode) if hasattr(self, "data") else None
        )

        if isinstance(desc, dict) and desc.get("type") == "kitti_stereo":
            # Get image size from args, default to 384
            imgsz = getattr(self.args, "imgsz", 384)
            if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                imgsz = (int(imgsz[0]), int(imgsz[1]))  # (H, W)
            elif isinstance(imgsz, (list, tuple)):
                imgsz = (int(imgsz[0]), int(imgsz[0]))  # Fallback to square
            else:
                imgsz = (int(imgsz), int(imgsz))  # Int to square

            # Get max_samples from args if available (for profiling/testing)
            max_samples = getattr(self.args, "max_samples", None)

            # Compute output_size from imgsz with default stride (8x for P3)
            # This can be overridden if model is available later, but default works for most cases
            output_size = None  # Will use dataset default (imgsz // 8)

            # Get mean_dims from dataset config if available
            mean_dims = self.data.get("mean_dims") if hasattr(self, "data") else None
            std_dims = self.data.get("std_dims") if hasattr(self, "data") else None

            return Stereo3DDetDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", mode)),
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
                max_samples=max_samples,
                output_size=output_size,
                mean_dims=mean_dims,
                std_dims=std_dims,
            )

        # Fallback: if img_path is a string, try to use it directly
        if isinstance(img_path, str) or isinstance(img_path, Path):
            imgsz = getattr(self.args, "imgsz", 384)
            if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                imgsz = (int(imgsz[0]), int(imgsz[1]))  # (H, W)
            elif isinstance(imgsz, (list, tuple)):
                imgsz = (int(imgsz[0]), int(imgsz[0]))  # Fallback to square
            else:
                imgsz = (int(imgsz), int(imgsz))  # Int to square

            return Stereo3DDetDataset(
                root=img_path,
                split=mode,
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
                output_size=None,  # Will use dataset default
                mean_dims=self.data.get("mean_dims") if hasattr(self, "data") else None,
                std_dims=self.data.get("std_dims") if hasattr(self, "data") else None,
            )

        # If we can't determine the dataset, raise an error
        raise ValueError(
            f"Cannot build dataset from img_path={img_path} (type: {type(img_path)}). "
            f"Expected a string path or a descriptor dict with type='kitti_stereo'."
        )

    def get_dataloader(
        self, dataset_path: str | dict[str, Any], batch_size: int
    ) -> torch.utils.data.DataLoader:
        """Construct and return dataloader for validation.

        Args:
            dataset_path: Path to the dataset, or a descriptor dict from self.data.get(split).
            batch_size: Size of each batch.

        Returns:
            torch.utils.data.DataLoader: Dataloader for validation.
        """
        from ultralytics.data import build_dataloader

        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")

        # build_dataloader automatically uses dataset.collate_fn if available
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=False,  # No shuffling for validation
            rank=-1,  # Single GPU validation
            drop_last=False,  # Don't drop last batch in validation
            pin_memory=self.device.type == "cuda" if hasattr(self, "device") else True,
        )
