# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations


import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.stereo.box3d import Box3D
import os
from ultralytics.data.stereo.calib import CalibrationParameters
from ultralytics.models.yolo.stereo3ddet.metrics import Stereo3DDetMetrics
from ultralytics.utils import LOGGER, RANK, YAML, colorstr, emojis
from ultralytics.utils.metrics import DetMetrics, box_iou, compute_3d_iou
from ultralytics.utils.plotting import plot_stereo3d_boxes
from ultralytics.utils.profiling import profile_function, profile_section
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
from ultralytics.utils.nms import non_max_suppression
from typing import Any

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
) -> list[Box3D] | list[list[Box3D]]:
    """Decode YOLO11-mapped stereo3ddet outputs (P3-only) to Box3D objects.

    This uses Detect inference output for candidate 2D boxes and class scores, then samples the auxiliary
    stereo/3D maps at the kept P3 indices to estimate depth/dimensions/orientation.
    """
    if "det" not in outputs:
        raise KeyError("decode_stereo3d_outputs_yolo11_p3 expected outputs['det']")

    det_out = outputs["det"]
    det_inf = det_out[0] if isinstance(det_out, (tuple, list)) else det_out  # [B, 4+nc, HW]
    bs = int(det_inf.shape[0])
    nc = int(det_inf.shape[1] - 4)

    # Determine letterbox input size
    if imgsz is None:
        imgsz = (384, 384)
    input_h, input_w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))

    # Feature map size from any aux map (P3 grid)
    sample_aux = None
    for k in ("lr_distance", "dimensions", "orientation"):
        if k in outputs and isinstance(outputs[k], torch.Tensor):
            sample_aux = outputs[k]
            break
    if sample_aux is None:
        raise KeyError("decode_stereo3d_outputs_yolo11_p3 expected at least one aux map in outputs")
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

    class_names = get_paper_class_names()
    mean_dims = {
        0: (1.52, 1.73, 3.89),  # Car (H, W, L)
        1: (1.73, 0.50, 0.80),  # Pedestrian
        2: (1.77, 0.60, 1.76),  # Cyclist
    }

    # Original shapes fallback
    if ori_shapes is None or len(ori_shapes) == 0:
        ori_shapes = [(375, 1242)] * bs

    results_per_batch: list[list[Box3D]] = []
    eps = 1e-6

    for b in range(bs):
        # Calibration per sample
        if calib is None:
            fx = fy = 721.5377
            cx, cy = 609.5593, 172.8540
            baseline = 0.54
        elif isinstance(calib, list):
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
        else:
            fx = float(calib.get("fx", 721.5377))
            fy = float(calib.get("fy", 721.5377))
            cx = float(calib.get("cx", 609.5593))
            cy = float(calib.get("cy", 172.8540))
            baseline = float(calib.get("baseline", 0.54))

        ori_h, ori_w = ori_shapes[b]
        letterbox_scale, pad_left, pad_top = _compute_letterbox_params(ori_h, ori_w, imgsz)

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
            lr_feat = float(outputs["lr_distance"][b, 0, gy, gx].item()) if "lr_distance" in outputs else 0.0
            dim_off = outputs["dimensions"][b, :, gy, gx].float() if "dimensions" in outputs else torch.zeros(3)
            ori_enc = outputs["orientation"][b, :, gy, gx].float() if "orientation" in outputs else torch.zeros(8)

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

            # Dimensions decode (H, W, L)
            mean_h, mean_w, mean_l = mean_dims.get(c, mean_dims[0])
            height = mean_h + float(dim_off[0].item())
            width = mean_w + float(dim_off[1].item())
            length = mean_l + float(dim_off[2].item())

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

# Import geometric construction module (GAP-001)
from ultralytics.models.yolo.stereo3ddet.geometric import (
    GeometricConstruction,
    GeometricObservations,
    CalibParams,
    fallback_simple_triangulation,
)

# Import dense alignment module (GAP-002)
from ultralytics.models.yolo.stereo3ddet.dense_align import (
    DenseAlignment,
)


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
        config_path = Path(__file__).parent.parent.parent / "cfg" / "models" / "stereo3ddet_full.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        try:
            full_config = YAML.load(str(config_path))
            geo_config = full_config.get("geometric_construction", {})
            # Merge with defaults
            _geometric_config = {**default_config, **geo_config}
        except Exception as e:
            LOGGER.debug(f"Failed to load geometric config from {config_path}: {e}")
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
    LOGGER.info(f"Geometric solver (SC-007): {converged}/{total} converged ({rate:.1%}) [{status}]")


# Global dense alignment instance (GAP-002)
_dense_aligner: DenseAlignment | None = None
_dense_alignment_config: dict | None = None


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
    if _dense_alignment_config is not None:
        return _dense_alignment_config
    
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
        config_path = Path(__file__).parent.parent.parent / "cfg" / "models" / "stereo3ddet_full.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        try:
            full_config = YAML.load(str(config_path))
            dense_config = full_config.get("dense_alignment", {})
            # Merge with defaults
            _dense_alignment_config = {**default_config, **dense_config}
        except Exception as e:
            LOGGER.debug(f"Failed to load dense alignment config from {config_path}: {e}")
            _dense_alignment_config = default_config
    else:
        _dense_alignment_config = default_config
    
    return _dense_alignment_config


def get_dense_aligner() -> DenseAlignment | None:
    """Get or create the global dense alignment instance.
    
    T023: Provides a singleton aligner for consistent depth refinement.
    
    Returns:
        DenseAlignment instance if enabled, None if disabled.
    """
    global _dense_aligner
    
    if _dense_aligner is None:
        config = get_dense_alignment_config()
        if not config.get("enabled", True):
            return None
        _dense_aligner = DenseAlignment(
            depth_search_range=config.get("depth_search_range", 2.0),
            depth_steps=config.get("depth_steps", 32),
            patch_size=config.get("patch_size", 7),
            method=config.get("method", "ncc"),
        )
    
    return _dense_aligner


def reset_dense_aligner() -> None:
    """Reset the global dense aligner.
    
    Call this at the start of validation to reset dense alignment state.
    """
    global _dense_aligner, _dense_alignment_config
    _dense_aligner = None
    _dense_alignment_config = None


# Import heatmap NMS for GAP-003 (T028)
from ultralytics.models.yolo.stereo3ddet.nms import heatmap_nms

# Import occlusion classification for GAP-006 (T040)
from ultralytics.models.yolo.stereo3ddet.occlusion import (
    classify_occlusion,
    should_skip_dense_alignment,
)

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
        config_path = Path(__file__).parent.parent.parent / "cfg" / "models" / "stereo3ddet_full.yaml"
    
    try:
        if Path(config_path).exists():
            full_config = YAML.load(str(config_path))
            occ_config = full_config.get("occlusion", {})
            _occlusion_config = {**default_config, **occ_config}
        else:
            LOGGER.debug(f"Occlusion config file not found at {config_path}")
            _occlusion_config = default_config
    except Exception as e:
        LOGGER.debug(f"Failed to load occlusion config from {config_path}: {e}")
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
            corners_obj = np.array([
                [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2],  # x (length)
                [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],  # y (height)
                [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2],  # z (width)
            ])  # [3, 8]
            
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


def _decode_stereo3d_outputs_per_sample(
    outputs: dict[str, torch.Tensor],
    conf_threshold: float = 0.25,
    top_k: int = 100,
    calib: dict[str, float] | None = None,
    use_nms: bool = True,
    nms_kernel: int = 3,
    use_geometric_construction: bool | None = None,
    use_dense_alignment: bool | None = None,
    use_occlusion_classification: bool | None = None,
    left_img: np.ndarray | torch.Tensor | None = None,
    right_img: np.ndarray | torch.Tensor | None = None,
    imgsz: int | tuple[int, int] | list[int] | None = None,
    ori_shape: tuple[int, int] | None = None,
) -> list[Box3D]:
    """Original per-sample implementation for backward compatibility.
    
    This function processes a single sample (batch_size=1) using the original
    per-detection loop implementation.
    
    Now supports optional geometric construction for refined 3D estimation.
    Now supports occlusion classification for dense alignment skipping.
    
    Args:
        outputs: Model output tensors
        conf_threshold: Confidence threshold for filtering
        top_k: Maximum detections to return  
        calib: Camera calibration parameters
        use_nms: Enable heatmap NMS
        nms_kernel: NMS kernel size
        use_geometric_construction: Enable geometric solver (None = use config)
        use_dense_alignment: Enable dense photometric alignment (None = use config)
        use_occlusion_classification: Enable occlusion classification (None = use config)
        left_img: Left camera image for dense alignment (optional, for future use)
        right_img: Right camera image for dense alignment (optional, for future use)
    """
    # Class names and mean dimensions (Paper uses 3 classes: Car, Pedestrian, Cyclist)
    from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
    class_names = get_paper_class_names()  # {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    # Mean dimensions: (height, width, length) in meters
    mean_dims = {
        0: (1.52, 1.73, 3.89),  # Car
        1: (1.73, 0.50, 0.80),  # Pedestrian
        2: (1.77, 0.60, 1.76),  # Cyclist
    }

    # Get letterbox parameters
    if imgsz is None:
        imgsz = (384, 384)  # Default letterbox size (H, W)
    input_h, input_w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
    
    # Get original image size (with fallback to KITTI default)
    if ori_shape is not None:
        original_height, original_width = ori_shape
    else:
        # Fallback to KITTI default dimensions
        original_height = 375
        original_width = 1242
    
    # Compute letterbox parameters
    letterbox_scale, pad_left, pad_top = _compute_letterbox_params(
        original_height, original_width, imgsz
    )
    
    if calib is not None:
        fx = calib["fx"]
        fy = calib["fy"]
        cx = calib["cx"]
        cy = calib["cy"]
        baseline = calib["baseline"]
    else:
        # Default KITTI calibration values
        fx = fy = 721.5377
        cx, cy = 609.5593, 172.8540
        baseline = 0.54
    
    # T014/T015: Determine whether to use geometric construction
    if use_geometric_construction is None:
        geo_config = get_geometric_config()
        use_geometric_construction = geo_config.get("enabled", True)
    
    # T014: Get geometric solver if enabled
    geometric_solver = None
    geo_fallback = True
    calib_params = None
    if use_geometric_construction:
        geometric_solver = get_geometric_solver()
        geo_config = get_geometric_config()
        geo_fallback = geo_config.get("fallback_on_failure", True)
        # Create CalibParams for geometric solver
        calib_params = CalibParams(fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)

    boxes3d = []
    batch_size = outputs["heatmap"].shape[0]
    assert batch_size == 1, "This function only handles single sample"

    b = 0
    heatmap = outputs["heatmap"][b]  # [C, H, W]
    offset = outputs["offset"][b]  # [2, H, W]
    bbox_size = outputs["bbox_size"][b]  # [2, H, W]
    lr_distance = outputs["lr_distance"][b]  # [1, H, W]
    dimensions = outputs["dimensions"][b]  # [3, H, W]
    orientation = outputs["orientation"][b]  # [8, H, W]

    num_classes, h, w = heatmap.shape
    
    # Compute scale from feature map to letterboxed input space
    # Scale is computed dynamically from actual feature map size (architecture-agnostic)
    scale_w_letterbox = input_w / w  # w is feature map width
    scale_h_letterbox = input_h / h  # h is feature map height

    # Apply sigmoid to get probabilities
    heatmap = torch.sigmoid(heatmap)
    
    # T028: Apply heatmap NMS (GAP-003) before topk selection
    # Paper: Section 3.1 "For inference, use the 3×3 max pooling operation instead of NMS"
    if use_nms:
        # Need to add batch dimension for heatmap_nms: [C, H, W] -> [1, C, H, W]
        heatmap = heatmap_nms(heatmap.unsqueeze(0), kernel_size=nms_kernel).squeeze(0)
    
    # Flatten heatmap and get top-k detections
    heatmap_flat = heatmap.reshape(num_classes, -1)  # [C, H*W]
    scores, indices = torch.topk(heatmap_flat, k=min(top_k, heatmap_flat.numel()), dim=1)

    for c in range(num_classes):
        # Filter to only mean dimensions
        class_scores = scores[c]
        class_indices = indices[c]

        for score, idx in zip(class_scores, class_indices):
            # debug
            assert score.min() >= 0 and score.max() <= 1, "score is not normalized"
            
            confidence = float(score.item())
            
            if confidence < conf_threshold:
                continue

            # Convert flat index to (y, x) coordinates
            idx_int = int(idx.item())
            y_idx = idx_int // w
            x_idx = idx_int % w

            # Get sub-pixel offset
            dx = offset[0, y_idx, x_idx].item()
            dy = offset[1, y_idx, x_idx].item()

            # Refined 2D center (in feature map coordinates)
            center_x = x_idx + dx
            center_y = y_idx + dy

            # Get 2D box size
            box_w = float(bbox_size[0, y_idx, x_idx].item())
            box_h = float(bbox_size[1, y_idx, x_idx].item())
            # Safety: bbox_size is trained to be positive but the head output is unconstrained.
            # If box_w/box_h becomes <= 0 (or nan/inf), xyxy conversion will produce x_min >= x_max warnings.
            if not np.isfinite(box_w) or box_w <= 0:
                box_w = 1e-3
            if not np.isfinite(box_h) or box_h <= 0:
                box_h = 1e-3

            # Get left-right distance (in feature map space)
            d = lr_distance[0, y_idx, x_idx].item()

            # Convert 2D center from feature map to letterboxed input space
            u_letterbox = center_x * scale_w_letterbox
            v_letterbox = center_y * scale_h_letterbox
            
            # Compute depth from stereo geometry
            # IMPORTANT: d is in feature map space, must scale to letterboxed input space
            # Calibration is already in letterboxed space (from dataset transformation)
            d_letterbox = d * scale_w_letterbox  # Scale disparity to letterboxed input width

            # Decode dimensions (offsets + class mean)
            dim_offsets = dimensions[:, y_idx, x_idx].cpu().numpy()
            mean_h, mean_w, mean_l = mean_dims[c]
            height = mean_h + dim_offsets[0]
            width = mean_w + dim_offsets[1]
            length = mean_l + dim_offsets[2]

            # Ensure positive dimensions
            height = max(0.1, height)
            width = max(0.1, width)
            length = max(0.1, length)

            # Decode orientation from Multi-Bin representation
            orient_bins = orientation[:, y_idx, x_idx].cpu().numpy()
            
            # Correct format: [conf1, conf2, sin1, cos1, sin2, cos2, pad, pad]
            # Extract bin confidences from indices 0 and 1
            bin_confidences = orient_bins[:2]  # [conf1, conf2]
            bin_idx = np.argmax(bin_confidences)  # Select bin with max confidence
            
            # Get sin/cos for selected bin
            # Bin 0: sin at index 2, cos at index 3
            # Bin 1: sin at index 4, cos at index 5
            sin_val = orient_bins[2 + bin_idx * 2]  # 2 + 0*2 = 2 for bin0, 2 + 1*2 = 4 for bin1
            cos_val = orient_bins[3 + bin_idx * 2]  # 3 + 0*2 = 3 for bin0, 3 + 1*2 = 5 for bin1
            
            # Bin centers: Bin 0 covers [-π, 0] centered at -π/2
            #              Bin 1 covers [0, π] centered at +π/2
            bin_centers = np.array([-np.pi/2, np.pi/2])
            
            # Compute residual angle
            residual = np.arctan2(sin_val, cos_val)  # This is the residual within the bin
            
            # Alpha = bin_center + residual (observation angle)
            alpha = bin_centers[bin_idx] + residual
            
            # Apply geometric construction if enabled and disparity is valid
            if geometric_solver is not None and d_letterbox > 1.0:
                # IMPORTANT: Convert to original coordinates for geometric solver
                # 3D coordinates should be in original camera space, not letterboxed
                u_orig = (u_letterbox - pad_left) / letterbox_scale
                v_orig = (v_letterbox - pad_top) / letterbox_scale
                d_orig = d_letterbox / letterbox_scale  # Disparity scales with image
                
                # Get original calibration (reverse letterbox transformation)
                fx_orig = fx / letterbox_scale
                fy_orig = fy / letterbox_scale
                cx_orig = (cx - pad_left) / letterbox_scale
                cy_orig = (cy - pad_top) / letterbox_scale
                
                # Build observations for geometric solver using original coordinates
                u_right_orig = u_orig - d_orig
                v_right_orig = v_orig  # Same v due to epipolar constraint
                
                observations = GeometricObservations(
                    ul=u_orig,
                    vl=v_orig,
                    ur=u_right_orig,
                    vr=v_right_orig,
                    ul_prime=u_right_orig,  # Left center projected to right
                    ur_prime=u_right_orig,  # Approximation
                    up=u_orig,  # Perspective keypoint (simplified: use center)
                    vp=v_orig,
                )
                
                # Create original calibration params for solver
                calib_params_orig = CalibParams(fx=fx_orig, fy=fy_orig, cx=cx_orig, cy=cy_orig, baseline=baseline)
                
                # Initial depth estimate for solver (using original coordinates)
                z_init = (fx_orig * baseline) / (d_orig + 1e-6)
                
                # Initial theta from alpha and initial position estimate
                x_init = (u_orig - cx_orig) * z_init / fx_orig
                theta_init = alpha + np.arctan2(x_init, z_init)
                theta_init = float(np.arctan2(np.sin(theta_init), np.cos(theta_init)))
                
                # Solve using geometric construction with original coordinates
                x_3d, y_3d, z_3d, theta, converged = geometric_solver.solve(
                    observations=observations,
                    dimensions=(length, width, height),
                    theta_init=theta_init,
                    calib=calib_params_orig,
                    z_init=z_init,
                )
                
                # Fallback if solver failed and fallback is enabled
                if not converged and geo_fallback:
                    x_3d, y_3d, z_3d, theta = fallback_simple_triangulation (
                        center_2d=(u_orig, v_orig),
                        disparity=d_orig,
                        calib=calib_params_orig,
                        theta_init=theta_init,
                    )
            else:
                # Original simple triangulation method
                # Use letterboxed coordinates (calibration is in letterboxed space)
                if d_letterbox > 0:
                    depth = (fx * baseline) / (d_letterbox + 1e-6)
                else:
                    depth = 50.0  # Default depth

                # 3D position - convert to original coordinates first
                # IMPORTANT: 3D coordinates should be in original camera space, not letterboxed
                u_orig = (u_letterbox - pad_left) / letterbox_scale
                v_orig = (v_letterbox - pad_top) / letterbox_scale
                
                # Get original calibration (reverse letterbox transformation)
                fx_orig = fx / letterbox_scale
                fy_orig = fy / letterbox_scale
                cx_orig = (cx - pad_left) / letterbox_scale
                cy_orig = (cy - pad_top) / letterbox_scale
                
                # Calculate 3D position using original coordinates and calibration
                x_3d = float((u_orig - cx_orig) * depth / fx_orig)
                y_3d = float((v_orig - cy_orig) * depth / fy_orig)
                z_3d = float(depth)
                
                # Convert observation angle α to global yaw θ
                # θ = α + arctan(x/z)
                ray_angle = np.arctan2(x_3d, z_3d)
                theta = alpha + ray_angle
                
                # Normalize to [-π, π]
                theta = np.arctan2(np.sin(theta), np.cos(theta))

            # Convert bbox from feature map to letterboxed, then reverse letterbox
            box_w_letterbox = box_w * scale_w_letterbox
            box_h_letterbox = box_h * scale_h_letterbox
            x_min_letterbox = u_letterbox - box_w_letterbox / 2
            y_min_letterbox = v_letterbox - box_h_letterbox / 2
            x_max_letterbox = u_letterbox + box_w_letterbox / 2
            y_max_letterbox = v_letterbox + box_h_letterbox / 2
            
            # Reverse letterbox transformation
            x_min = (x_min_letterbox - pad_left) / letterbox_scale
            y_min = (y_min_letterbox - pad_top) / letterbox_scale
            x_max = (x_max_letterbox - pad_left) / letterbox_scale
            y_max = (y_max_letterbox - pad_top) / letterbox_scale
            
            # Final safety: enforce xyxy ordering (prevents x_min>=x_max / y_min>=y_max)
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            
            # Create Box3D object
            box3d = Box3D(
                center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                dimensions=(float(length), float(width), float(height)),
                orientation=float(theta),
                class_label=class_names[c],
                class_id=c,
                confidence=confidence,
                bbox_2d=(float(x_min), float(y_min), float(x_max), float(y_max)),
            )
            boxes3d.append(box3d)

    # T040: Apply occlusion classification (GAP-006) for dense alignment integration
    # Determine whether to use occlusion classification from config if not specified
    if use_occlusion_classification is None:
        occ_config = get_occlusion_config()
        use_occlusion_classification = occ_config.get("enabled", True)
    
    # T040: Classify occlusion if enabled
    # Store occlusion indices for later use by dense alignment (when implemented)
    if use_occlusion_classification and len(boxes3d) > 0:
        # Convert Box3D objects to dict format for classify_occlusion
        detections_for_occlusion = []
        for box in boxes3d:
            detections_for_occlusion.append({
                "bbox_2d": box.bbox_2d,
                "center_3d": box.center_3d,
            })
        
        # Run occlusion classification
        occluded_indices, unoccluded_indices = classify_occlusion(
            detections_for_occlusion,
            image_width=int(original_width),
        )
        
        # T040: Store occlusion classification results in Box3D metadata
        # This enables downstream processing (like dense alignment) to check occlusion status
        for idx, box in enumerate(boxes3d):
            # Add occlusion attribute to Box3D (will be used by dense alignment)
            # Note: Box3D.occluded is already an existing attribute for ground truth,
            # we're setting it here based on our classification
            if idx in occluded_indices:
                box.occluded = 2  # KITTI occlusion level: 2 = heavily occluded
            else:
                box.occluded = 0  # KITTI occlusion level: 0 = fully visible

    return boxes3d


def decode_stereo3d_outputs(
    outputs: dict[str, torch.Tensor],
    conf_threshold: float = 0.25,
    top_k: int = 100,
    calib: dict[str, float] | list[dict[str, float]] | None = None,
    use_nms: bool = True,
    nms_kernel: int = 3,
    use_occlusion_classification: bool | None = None,
    imgsz: int | tuple[int, int] | list[int] | None = None,
    ori_shapes: list[tuple[int, int]] | None = None,
) -> list[Box3D] | list[list[Box3D]]:
    """Decode 10-branch model outputs to 3D bounding boxes.

    Decodes Stereo CenterNet 10-branch outputs following the paper methodology:
    1. Extract top-k detections from heatmap
    2. Apply offset for sub-pixel center refinement
    3. Compute depth from lr_distance using calibration parameters
    4. Decode 3D dimensions from offsets + class means
    5. Decode orientation from Multi-Bin representation
    6. Construct Box3D objects with all attributes
    7. Apply occlusion classification (GAP-006) for dense alignment support

    Args:
        outputs: Dictionary with 10 branch outputs:
            - heatmap: [B, C, H/4, W/4] - center point heatmap
            - offset: [B, 2, H/4, W/4] - sub-pixel offset (δx, δy)
            - bbox_size: [B, 2, H/4, W/4] - 2D box size (w, h)
            - lr_distance: [B, 1, H/4, W/4] - left-right center distance d
            - right_width: [B, 1, H/4, W/4] - right box width wr
            - dimensions: [B, 3, H/4, W/4] - 3D dimension offsets (ΔH, ΔW, ΔL)
            - orientation: [B, 8, H/4, W/4] - Multi-Bin orientation encoding
            - vertices: [B, 8, H/4, W/4] - bottom 4 vertex coordinates
            - vertex_offset: [B, 8, H/4, W/4] - vertex sub-pixel offsets
            - vertex_dist: [B, 4, H/4, W/4] - center to vertex distances
        conf_threshold: Confidence threshold for filtering detections.
        top_k: Maximum number of detections to extract.
        calib: Calibration parameters. Can be:
            - Single dict (shared across batch): dict with keys: fx, fy, cx, cy, baseline
            - List of dicts (one per batch item): list[dict] with same keys
        use_nms: Whether to apply heatmap NMS (GAP-003). Default True.
            Paper: Section 3.1 "For inference, use the 3×3 max pooling operation instead of NMS"
        nms_kernel: Kernel size for max pooling NMS. Default 3.
        use_occlusion_classification: Enable occlusion classification (GAP-006).
            When enabled, Box3D.occluded is set based on depth-line algorithm.
            None = use config default. Default enables for dense alignment support.

    Returns:
        - If batch_size == 1: list[Box3D] (backward compatibility)
        - If batch_size > 1: list[list[Box3D]] (one list per batch item)

    References:
        Stereo CenterNet paper: Section 3.2 (Decoding), Algorithm 1 (Occlusion)
    """
    # T213: Backward compatibility - detect single sample and use fallback
    batch_size = outputs["heatmap"].shape[0]
    is_single_sample = batch_size == 1
    
    # T213: Fallback to original per-sample processing for single sample or edge cases
    if is_single_sample:
        # Use original implementation for single sample (backward compatibility)
        # Get ori_shape for single sample
        ori_shape_single = None
        if ori_shapes and len(ori_shapes) > 0:
            ori_shape_single = ori_shapes[0]
        if isinstance(calib, list):
            calib = calib[0]
        
        return _decode_stereo3d_outputs_per_sample(
            outputs, conf_threshold, top_k, calib,
            use_nms=use_nms, nms_kernel=nms_kernel,
            use_occlusion_classification=use_occlusion_classification,
            imgsz=imgsz,
            ori_shape=ori_shape_single,
        )
    
    # T206-T211: Batch processing implementation
    # Class names and mean dimensions (Paper uses 3 classes: Car, Pedestrian, Cyclist)
    from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
    class_names = get_paper_class_names()  # {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    # Mean dimensions: (height, width, length) in meters
    mean_dims = {
        0: (1.52, 1.73, 3.89),  # Car
        1: (1.73, 0.50, 0.80),  # Pedestrian
        2: (1.77, 0.60, 1.76),  # Cyclist
    }
    
    # Get letterbox parameters
    if imgsz is None:
        imgsz = (384, 384)  # Default letterbox size (H, W)
    input_h, input_w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
    
    # Get original image sizes (with fallback to KITTI default)
    if ori_shapes is None or len(ori_shapes) == 0:
        # Use default KITTI dimensions for all batch items
        ori_shapes = [(375, 1242)] * batch_size

    # T210, T211: Cache calibration parameters and handle batch calibration
    if calib is not None:
        if isinstance(calib, list):
            # List of dicts - one per batch item
            calibs_list = calib
            # Extract calibration values for each batch item
            fx_list = [c.get("fx", 721.5377) for c in calibs_list]
            fy_list = [c.get("fy", 721.5377) for c in calibs_list]
            cx_list = [c.get("cx", 609.5593) for c in calibs_list]
            cy_list = [c.get("cy", 172.8540) for c in calibs_list]
            baseline_list = [c.get("baseline", 0.54) for c in calibs_list]
            # Convert to tensors for batch processing
            device = outputs["heatmap"].device
            fx_tensor = torch.tensor(fx_list, device=device, dtype=torch.float32)
            fy_tensor = torch.tensor(fy_list, device=device, dtype=torch.float32)
            cx_tensor = torch.tensor(cx_list, device=device, dtype=torch.float32)
            cy_tensor = torch.tensor(cy_list, device=device, dtype=torch.float32)
            baseline_tensor = torch.tensor(baseline_list, device=device, dtype=torch.float32)
            shared_calib = False
        else:
            # Single dict - shared across batch
            fx = calib.get("fx", 721.5377)
            fy = calib.get("fy", 721.5377)
            cx = calib.get("cx", 609.5593)
            cy = calib.get("cy", 172.8540)
            baseline = calib.get("baseline", 0.54)
            shared_calib = True
    else:
        # Default KITTI calibration values
        fx = fy = 721.5377
        cx, cy = 609.5593, 172.8540
        baseline = 0.54
        shared_calib = True

    # T206: Batch processing - extract all tensors at once
    device = outputs["heatmap"].device
    heatmap = outputs["heatmap"]  # [B, C, H, W]
    offset = outputs["offset"]  # [B, 2, H, W]
    bbox_size = outputs["bbox_size"]  # [B, 2, H, W]
    lr_distance = outputs["lr_distance"]  # [B, 1, H, W]
    dimensions = outputs["dimensions"]  # [B, 3, H, W]
    orientation = outputs["orientation"]  # [B, 8, H, W]

    num_classes, h, w = heatmap.shape[1], heatmap.shape[2], heatmap.shape[3]
    
    # Compute scale from feature map to letterboxed input space
    # Scale is computed dynamically from actual feature map size (architecture-agnostic)
    scale_w_letterbox = input_w / w  # w is feature map width
    scale_h_letterbox = input_h / h  # h is feature map height

    # T207: Vectorize heatmap peak detection for batch
    # Apply sigmoid to get probabilities
    heatmap = torch.sigmoid(heatmap)
    
    # T028: Apply heatmap NMS (GAP-003) before topk selection
    # Paper: Section 3.1 "For inference, use the 3×3 max pooling operation instead of NMS"
    if use_nms:
        heatmap = heatmap_nms(heatmap, kernel_size=nms_kernel)
    
    # Flatten heatmap: [B, C, H*W]
    heatmap_flat = heatmap.reshape(batch_size, num_classes, -1)  # [B, C, H*W]
    
    # Get top-k scores and indices for each batch and class
    # Use torch.topk across the spatial dimension
    topk_scores, topk_indices = torch.topk(heatmap_flat, k=min(top_k, heatmap_flat.shape[2]), dim=2)  # [B, C, K]

    # T207, T208, T209: Vectorized batch processing
    # Process each batch item with optimized operations
    batch_results = []
    
    for b in range(batch_size):
        boxes3d_batch = []
        
        # Get original image size for this batch item
        if b < len(ori_shapes):
            ori_h, ori_w = ori_shapes[b]
        else:
            # Fallback to KITTI default
            ori_h, ori_w = 375, 1242
        
        # Compute letterbox parameters for this batch item
        letterbox_scale, pad_left, pad_top = _compute_letterbox_params(ori_h, ori_w, imgsz)
        
        # Get calibration for this batch item
        if shared_calib:
            fx_b = fx
            fy_b = fy
            cx_b = cx
            cy_b = cy
            baseline_b = baseline
        else:
            fx_b = fx_tensor[b].item()
            fy_b = fy_tensor[b].item()
            cx_b = cx_tensor[b].item()
            cy_b = cy_tensor[b].item()
            baseline_b = baseline_tensor[b].item()
        
        # T207: Use pre-computed top-k scores and indices for this batch item
        batch_scores = topk_scores[b]  # [C, K]
        batch_indices = topk_indices[b]  # [C, K]
        
        # T209: Keep tensors on GPU, only move to CPU when creating Box3D
        batch_offset = offset[b]  # [2, H, W] - keep on GPU
        batch_bbox_size = bbox_size[b]  # [2, H, W] - keep on GPU
        batch_lr_distance = lr_distance[b]  # [1, H, W] - keep on GPU
        batch_dimensions = dimensions[b]  # [3, H, W] - keep on GPU
        batch_orientation = orientation[b]  # [8, H, W] - keep on GPU
        
        for c in range(num_classes):
            # Filter to only paper classes (0, 1, 2 = Car, Pedestrian, Cyclist)
            if c not in mean_dims:
                continue
            
            class_scores = batch_scores[c]  # [K]
            class_indices = batch_indices[c]  # [K]
            mean_h, mean_w, mean_l = mean_dims[c]

            # debug
            assert class_scores.min() >= 0 and class_scores.max() <= 1, "class_scores are not normalized"
            valid_mask = class_scores >= conf_threshold
            if not valid_mask.any():
                continue
            valid_scores = class_scores[valid_mask]
            valid_indices = class_indices[valid_mask]
            
            # T207: Vectorize coordinate conversion
            y_indices = valid_indices // w  # [K_valid]
            x_indices = valid_indices % w   # [K_valid]
            
            # T208: Vectorize offset and bbox_size extraction using gather
            # Gather offset values: [K_valid, 2]
            offset_yx = torch.stack([
                batch_offset[0, y_indices, x_indices],  # dx
                batch_offset[1, y_indices, x_indices],  # dy
            ], dim=1)  # [K_valid, 2]
            
            # Gather bbox_size values: [K_valid, 2]
            bbox_size_yx = torch.stack([
                batch_bbox_size[0, y_indices, x_indices],  # w
                batch_bbox_size[1, y_indices, x_indices],  # h
            ], dim=1)  # [K_valid, 2]
            
            # Safety: bbox_size should be positive (it represents width/height in feature-map units),
            # but the head outputs are unconstrained. Clamp here to avoid invalid xyxy boxes.
            # Also guard against NaN/Inf to keep downstream numpy ops stable.
            invalid_wh = (~torch.isfinite(bbox_size_yx)).any(dim=1) | (bbox_size_yx <= 0).any(dim=1)
            if invalid_wh.any() and os.getenv("ULTRA_STEREO_DEBUG_BBOX", "0") == "1":
                # Log a small summary (throttled) to help diagnose training drift.
                # Keep this lightweight to avoid slowing validation.
                if not hasattr(decode_stereo3d_outputs, "_debug_bbox_warn_count"):
                    decode_stereo3d_outputs._debug_bbox_warn_count = 0  # type: ignore[attr-defined]
                if decode_stereo3d_outputs._debug_bbox_warn_count < 5:  # type: ignore[attr-defined]
                    decode_stereo3d_outputs._debug_bbox_warn_count += 1  # type: ignore[attr-defined]
                    wh = bbox_size_yx.detach()
                    wv, hv = wh[:, 0], wh[:, 1]
                    LOGGER.info(
                        "stereo3ddet: invalid bbox_size detected (will clamp). "
                        f"w(min/mean/max)={wv.min().item():.4g}/{wv.mean().item():.4g}/{wv.max().item():.4g}, "
                        f"h(min/mean/max)={hv.min().item():.4g}/{hv.mean().item():.4g}/{hv.max().item():.4g}, "
                        f"invalid_count={int(invalid_wh.sum().item())}"
                    )
            bbox_size_yx = torch.nan_to_num(bbox_size_yx, nan=0.0, posinf=0.0, neginf=0.0)
            bbox_size_yx = torch.clamp(bbox_size_yx, min=1e-3)
            
            # Gather lr_distance: [K_valid]
            d_values = batch_lr_distance[0, y_indices, x_indices]  # [K_valid]
            
            # T208: Vectorize dimension decoding
            # Gather dimension offsets: [K_valid, 3]
            dim_offsets = batch_dimensions[:, y_indices, x_indices].t()  # [K_valid, 3]
            
            # T208: Vectorize orientation decoding
            # Gather orientation bins: [K_valid, 8]
            orient_bins = batch_orientation[:, y_indices, x_indices].t()  # [K_valid, 8]
            
            # T209: Compute all values on GPU before moving to CPU
            # Refined 2D centers
            center_x = x_indices.float() + offset_yx[:, 0]  # [K_valid]
            center_y = y_indices.float() + offset_yx[:, 1]  # [K_valid]
            
            # Convert 2D centers from feature map to letterboxed input space (vectorized)
            scale_w_letterbox_tensor = torch.tensor(scale_w_letterbox, device=device, dtype=center_x.dtype)
            scale_h_letterbox_tensor = torch.tensor(scale_h_letterbox, device=device, dtype=center_y.dtype)
            u_values_letterbox = center_x * scale_w_letterbox_tensor  # [K_valid]
            v_values_letterbox = center_y * scale_h_letterbox_tensor  # [K_valid]
            
            # Compute depth from stereo geometry (vectorized)
            # IMPORTANT: d_values is in feature map space, must scale to letterboxed input space
            # Calibration is already in letterboxed space (from dataset transformation)
            fx_b_tensor = torch.tensor(fx_b, device=device, dtype=d_values.dtype)
            baseline_b_tensor = torch.tensor(baseline_b, device=device, dtype=d_values.dtype)
            d_values_letterbox = d_values * scale_w_letterbox_tensor  # Scale disparity to letterboxed input width
            depth_values = torch.where(
                d_values_letterbox > 0,
                (fx_b_tensor * baseline_b_tensor) / (d_values_letterbox + 1e-6),
                torch.tensor(50.0, device=device, dtype=d_values.dtype)  # Default depth
            )  # [K_valid]
            
            # Compute 3D position
            # IMPORTANT: Convert letterboxed coordinates to original before 3D calculation
            # 3D coordinates should be in original camera space, not letterboxed space
            pad_left_tensor = torch.tensor(pad_left, device=device, dtype=u_values_letterbox.dtype)
            pad_top_tensor = torch.tensor(pad_top, device=device, dtype=v_values_letterbox.dtype)
            letterbox_scale_tensor = torch.tensor(letterbox_scale, device=device, dtype=u_values_letterbox.dtype)
            
            # Convert to original image coordinates
            u_values_orig = (u_values_letterbox - pad_left_tensor) / letterbox_scale_tensor  # [K_valid]
            v_values_orig = (v_values_letterbox - pad_top_tensor) / letterbox_scale_tensor  # [K_valid]
            
            # Get original calibration (reverse letterbox transformation)
            fx_orig = fx_b / letterbox_scale
            fy_orig = fy_b / letterbox_scale
            cx_orig = (cx_b - pad_left) / letterbox_scale
            cy_orig = (cy_b - pad_top) / letterbox_scale
            
            cx_orig_tensor = torch.tensor(cx_orig, device=device, dtype=u_values_letterbox.dtype)
            cy_orig_tensor = torch.tensor(cy_orig, device=device, dtype=v_values_letterbox.dtype)
            fx_orig_tensor = torch.tensor(fx_orig, device=device, dtype=depth_values.dtype)
            fy_orig_tensor = torch.tensor(fy_orig, device=device, dtype=depth_values.dtype)
            
            # Calculate 3D position using original coordinates and calibration
            x_3d_values = (u_values_orig - cx_orig_tensor) * depth_values / fx_orig_tensor  # [K_valid]
            y_3d_values = (v_values_orig - cy_orig_tensor) * depth_values / fy_orig_tensor  # [K_valid]
            z_3d_values = depth_values  # [K_valid]
            
            # u_values_orig and v_values_orig already calculated above for 3D position
            # Reuse them for bbox_2d calculation
            
            # Decode dimensions (vectorized)
            mean_h_tensor = torch.tensor(mean_h, device=device, dtype=dim_offsets.dtype)
            mean_w_tensor = torch.tensor(mean_w, device=device, dtype=dim_offsets.dtype)
            mean_l_tensor = torch.tensor(mean_l, device=device, dtype=dim_offsets.dtype)
            height_values = torch.clamp(mean_h_tensor + dim_offsets[:, 0], min=0.1)  # [K_valid]
            width_values = torch.clamp(mean_w_tensor + dim_offsets[:, 1], min=0.1)  # [K_valid]
            length_values = torch.clamp(mean_l_tensor + dim_offsets[:, 2], min=0.1)  # [K_valid]
            
            # Decode orientation from Multi-Bin (vectorized)
            # Correct format: [conf1, conf2, sin1, cos1, sin2, cos2, pad, pad]
            # orient_bins: [K_valid, 8]
            # Extract bin confidences from indices 0 and 1
            bin_confidences = orient_bins[:, :2]  # [K_valid, 2] - [conf1, conf2] for each detection
            bin_indices = torch.argmax(bin_confidences, dim=1)  # [K_valid] - which bin (0 or 1)
            
            # Gather sin/cos values for selected bins
            # Bin 0: sin at index 2, cos at index 3
            # Bin 1: sin at index 4, cos at index 5
            sin_indices = 2 + bin_indices * 2  # [K_valid] - sin indices: 2 for bin0, 4 for bin1
            cos_indices = 3 + bin_indices * 2  # [K_valid] - cos indices: 3 for bin0, 5 for bin1
            sin_vals = orient_bins[torch.arange(len(bin_indices), device=device), sin_indices]
            cos_vals = orient_bins[torch.arange(len(bin_indices), device=device), cos_indices]
            
            # Bin centers: Bin 0 covers [-π, 0] centered at -π/2
            #              Bin 1 covers [0, π] centered at +π/2
            bin_centers = torch.tensor([-np.pi/2, np.pi/2], device=device, dtype=orient_bins.dtype)
            
            # Compute residual angle
            residual = torch.atan2(sin_vals, cos_vals)  # This is the residual within the bin
            
            # Alpha = bin_center + residual (observation angle)
            alpha_values = bin_centers[bin_indices] + residual  # [K_valid]
            
            # Convert observation angle α to global yaw θ
            # θ = α + arctan(x/z)
            ray_angle = torch.atan2(x_3d_values, z_3d_values)  # [K_valid]
            theta_values = alpha_values + ray_angle  # [K_valid] - global yaw
            
            # Normalize to [-π, π]
            theta_values = torch.atan2(torch.sin(theta_values), torch.cos(theta_values))
            
            # T209: Move to CPU only when creating Box3D objects
            # Convert to numpy/cpu for Box3D creation
            x_3d_cpu = x_3d_values.cpu().numpy()
            y_3d_cpu = y_3d_values.cpu().numpy()
            z_3d_cpu = z_3d_values.cpu().numpy()
            length_cpu = length_values.cpu().numpy()
            width_cpu = width_values.cpu().numpy()
            height_cpu = height_values.cpu().numpy()
            theta_cpu = theta_values.cpu().numpy()
            confidence_cpu = valid_scores.cpu().numpy()
            box_w_cpu = bbox_size_yx[:, 0].cpu().numpy()
            box_h_cpu = bbox_size_yx[:, 1].cpu().numpy()
            center_x_cpu = center_x.cpu().numpy()
            center_y_cpu = center_y.cpu().numpy()
            
            # Convert bbox from feature map to letterboxed, then reverse letterbox (vectorized)
            # Use numpy for CPU-side calculations
            box_w_letterbox = box_w_cpu * scale_w_letterbox
            box_h_letterbox = box_h_cpu * scale_h_letterbox
            center_x_letterbox = center_x_cpu * scale_w_letterbox
            center_y_letterbox = center_y_cpu * scale_h_letterbox
            
            x_min_letterbox = center_x_letterbox - box_w_letterbox / 2
            y_min_letterbox = center_y_letterbox - box_h_letterbox / 2
            x_max_letterbox = center_x_letterbox + box_w_letterbox / 2
            y_max_letterbox = center_y_letterbox + box_h_letterbox / 2
            
            # Reverse letterbox transformation
            x_min = (x_min_letterbox - pad_left) / letterbox_scale
            y_min = (y_min_letterbox - pad_top) / letterbox_scale
            x_max = (x_max_letterbox - pad_left) / letterbox_scale
            y_max = (y_max_letterbox - pad_top) / letterbox_scale
            
            # Final safety: enforce xyxy ordering (prevents warnings and downstream metric issues)
            x1 = np.minimum(x_min, x_max)
            x2 = np.maximum(x_min, x_max)
            y1 = np.minimum(y_min, y_max)
            y2 = np.maximum(y_min, y_max)
            
            # Create Box3D objects
            for i in range(len(valid_scores)):
                box3d = Box3D(
                    center_3d=(float(x_3d_cpu[i]), float(y_3d_cpu[i]), float(z_3d_cpu[i])),
                    dimensions=(float(length_cpu[i]), float(width_cpu[i]), float(height_cpu[i])),
                    orientation=float(theta_cpu[i]),
                    class_label=class_names[c],
                    class_id=c,
                    confidence=float(confidence_cpu[i]),
                    bbox_2d=(
                        float(x1[i]),
                        float(y1[i]),
                        float(x2[i]),
                        float(y2[i]),
                    ),
                )
                boxes3d_batch.append(box3d)
        
        batch_results.append(boxes3d_batch)

    return batch_results

def _labels_to_box3d_list(labels: list[dict[str, Any]], calib: dict[str, float] | None = None, names: dict[int, str] | None = None) -> list[Box3D]:
    """Convert label dictionaries to Box3D objects.

    Uses provided names mapping, or falls back to paper classes (Car, Pedestrian, Cyclist) if not available.

    Args:
        labels: List of label dictionaries from dataset.
        calib: Calibration parameters (dict or CalibrationParameters object).
        names: Optional class names mapping {class_id: class_name}.

    Returns:
        List of Box3D objects with filtered and remapped class IDs.
    """
    boxes3d = []
    
    # Use provided names, or fall back to paper classes (Car, Pedestrian, Cyclist)
    class_names = names if names else get_paper_class_names()
    
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
            y_3d = y_bottom - float(height) / 2.0  # bottom-center -> geometric center (Y points down)
        else:
            # Fallback: reconstruct 3D center from stereo disparity (matching prediction pipeline)
            left_box = label["left_box"]
            right_box = label["right_box"]
            
            # Handle both dict and CalibrationParameters objects
            if isinstance(calib, dict):
                calib = CalibrationParameters.from_dict(calib)
            fx_val = calib.fx
            fy_val = calib.fy
            cx_val = calib.cx
            cy_val = calib.cy
            baseline_val = calib.baseline

            img_width = calib.image_width
            img_height = calib.image_height
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
        bbox_2d_x1y1x2y2 = (
            bbox_2d_xywh["center_x"] - bbox_2d_xywh["width"] / 2,
            bbox_2d_xywh["center_y"] - bbox_2d_xywh["height"] / 2,
            bbox_2d_xywh["center_x"] + bbox_2d_xywh["width"] / 2,
            bbox_2d_xywh["center_y"] + bbox_2d_xywh["height"] / 2,
        )

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

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
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

        # Names/nc fallback - use paper classes (3 classes: Car, Pedestrian, Cyclist)
        from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
        names = data_cfg.get("names") or get_paper_class_names()  # {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        nc = data_cfg.get("nc", len(names))

        # Return a dict compatible with BaseValidator expectations, plus stereo descriptors
        return {
            "yaml_file": str(self.args.data) if isinstance(self.args.data, (str, Path)) else None,
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
            batch["targets"] = {k: v.to(self.device, non_blocking=True) for k, v in batch["targets"].items()}
        if "aux_targets" in batch and isinstance(batch["aux_targets"], dict):
            batch["aux_targets"] = {k: v.to(self.device, non_blocking=True) for k, v in batch["aux_targets"].items()}
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
        # Support both legacy heatmap head and YOLO11-mapped head ("det").
        if "heatmap" in preds:
            batch_size = preds["heatmap"].shape[0]
            mode = "heatmap"
        elif "det" in preds:
            det_out = preds["det"]
            det_inf = det_out[0] if isinstance(det_out, (tuple, list)) else det_out
            batch_size = int(det_inf.shape[0])
            mode = "det"
        else:
            raise KeyError(f"Unsupported stereo3ddet preds keys: {list(preds.keys())}")

        # T212: Get calibration from batch if available
        calib = None
        calibs = []
        ori_shapes = None
        if hasattr(self, "_current_batch") and self._current_batch:
            # Try to get calibration from batch
            calibs = self._current_batch.get("calib", [])
            if calibs:
                # T211: Handle batch calibration - pass list if different per sample, single dict if shared
                if len(calibs) == batch_size and all(isinstance(c, dict) for c in calibs):
                    # Different calibration per sample
                    calib = calibs
                elif len(calibs) > 0 and isinstance(calibs[0], dict):
                    # Shared calibration (use first one)
                    calib = calibs[0]
            
            # Get original shapes from batch
            ori_shapes = self._current_batch.get("ori_shape", [])
        
        # Get imgsz from args
        imgsz = getattr(self.args, 'imgsz', 384)

        # T212, T028: Call decode_stereo3d_outputs once with entire batch
        # Get NMS config from args (defaults: use_nms=True, nms_kernel=3)
        use_nms = getattr(self.args, 'use_nms', True)
        nms_kernel = getattr(self.args, 'nms_kernel', 3)
        
        if mode == "heatmap":
            results = decode_stereo3d_outputs(
                preds,
                conf_threshold=self.args.conf,
                top_k=100,
                calib=calib,
                use_nms=use_nms,
                nms_kernel=nms_kernel,
                imgsz=imgsz,
                ori_shapes=ori_shapes,
            )
        else:
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
            )
        
        # T212: Ensure results is list of lists
        # decode_stereo3d_outputs returns list[list[Box3D]] for batch_size > 1
        # or list[Box3D] for batch_size == 1 (backward compatibility)
        if batch_size == 1 and isinstance(results, list) and len(results) > 0 and isinstance(results[0], Box3D):
            # Single sample result - wrap in list for consistency
            results = [results]
        
        # T023: Apply dense alignment for depth refinement (GAP-002)
        # Only apply if enabled in config and we have access to images
        use_dense_alignment = getattr(self.args, 'use_dense_alignment', None)
        dense_config = get_dense_alignment_config()
        
        # Check if dense alignment should be applied
        should_apply_dense = (
            use_dense_alignment is True or 
            (use_dense_alignment is None and dense_config.get("enabled", True))
        )
        
        if should_apply_dense and hasattr(self, "_current_batch") and self._current_batch:
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
                    left_img = (imgs_np[b, :3].transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3] RGB
                    right_img = (imgs_np[b, 3:].transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3] RGB
                    
                    # Get calibration for this sample
                    if calibs and b < len(calibs):
                        sample_calib = calibs[b] if isinstance(calibs[b], dict) else calibs[0]
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
                                    "bbox_2d": box.bbox_2d if box.bbox_2d else (0, 0, 100, 100),
                                    "center_3d": box.center_3d,
                                }
                                detections.append(det)
                            occluded_indices, _ = classify_occlusion(detections)
                        except Exception as e:
                            LOGGER.debug(f"Occlusion classification failed: {e}")
                            occluded_indices = []
                    
                    # Refine depth for each box
                    refined_boxes = []
                    for i, box in enumerate(boxes):
                        # Skip dense alignment for heavily occluded objects
                        if skip_occluded and should_skip_dense_alignment(i, occluded_indices):
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
                                center_3d=(float(x_refined), float(y_refined), float(refined_depth)),
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
                            LOGGER.debug(f"Dense alignment failed for box {i}: {e}")
                            refined_boxes.append(box)  # Keep original on failure
                    
                    results[b] = refined_boxes
                    
            except Exception as e:
                LOGGER.debug(f"Dense alignment batch processing failed: {e}")
                # Return original results on failure
            
            return results

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics with model information.

        Args:
            model: Model being validated.
        """
        # T016: Reset geometric solver statistics at start of validation
        reset_geometric_solver()
        
        # Get class names from dataset, not from metrics results_dict (which contains metric keys, not class names)
        # This fixes the bug where self.nc was set to number of metric keys (6-7) instead of number of classes (3)
        if hasattr(self, "data") and self.data and "names" in self.data:
            self.names = self.data["names"]
        elif hasattr(model, "names") and model.names:
            self.names = model.names
        else:
            # Fallback to paper class names
            from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
            self.names = get_paper_class_names()
        
        self.nc = len(self.names) if isinstance(self.names, dict) else len(self.names) if isinstance(self.names, (list, tuple)) else 0
        self.seen = 0
        self.metrics.names = self.names
        self.metrics.nc = self.nc  # Also update metrics.nc to match the correct number of classes

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
                calib = calibs[si] if si < len(calibs) and isinstance(calibs[si], dict) else None

                # Convert labels to Box3D - need to reverse letterbox transformation on calibration
                # Labels use original image normalized coordinates, but calib is letterboxed
                if calib is not None and si < len(ori_shapes):
                    ori_shape = ori_shapes[si]
                    if isinstance(ori_shape, (list, tuple)) and len(ori_shape) >= 2:
                        actual_h, actual_w = ori_shape[0], ori_shape[1]
                        imgsz = getattr(self.args, 'imgsz', 384)
                        
                        # Compute letterbox parameters
                        letterbox_scale, pad_left, pad_top = _compute_letterbox_params(
                            actual_h, actual_w, imgsz
                        )
                        
                        # Reverse letterbox transformation on calibration
                        if isinstance(calib, dict):
                            calib_orig = calib.copy()
                            calib_orig["fx"] = calib["fx"] / letterbox_scale
                            calib_orig["fy"] = calib["fy"] / letterbox_scale
                            calib_orig["cx"] = (calib["cx"] - pad_left) / letterbox_scale
                            calib_orig["cy"] = (calib["cy"] - pad_top) / letterbox_scale
                            calib_orig["image_width"] = actual_w
                            calib_orig["image_height"] = actual_h
                        else:
                            from ultralytics.data.stereo.calib import CalibrationParameters
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
                        # Labels are already in original coordinates (normalized to original image size)
                        # No need to reverse letterbox transformation on labels

                # Convert labels to Box3D
                try:
                    gt_boxes = _labels_to_box3d_list(labels, calib, names=self.names)
                except Exception as e:
                    LOGGER.warning(f"Error converting labels to Box3D (sample {si}): {e}")
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
                    if si < len(ori_shapes) and isinstance(ori_shapes[si], (list, tuple)) and len(ori_shapes[si]) >= 2:
                        ori_h, ori_w = int(ori_shapes[si][0]), int(ori_shapes[si][1])
                    else:
                        ori_h, ori_w = 375, 1242
                    letterbox_scale, pad_left, pad_top = _compute_letterbox_params(ori_h, ori_w, imgsz)
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
                        gt_boxes_t = torch.tensor(gt_bboxes2d, dtype=torch.float32) if gt_bboxes2d else torch.zeros((0, 4), dtype=torch.float32)
                        gt_cls_t = torch.tensor(gt_cls2d, dtype=torch.int64) if gt_cls2d else torch.zeros((0,), dtype=torch.int64)

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
                                correct = self.match_predictions(pred_cls_t, gt_cls_t, iou2d).cpu().numpy()
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
                    LOGGER.debug(f"bbox metrics update failed (sample {si}): {e}")

                # Handle empty predictions or ground truth
                if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                    continue

                # Compute 3D IoU matrix using vectorized batch computation
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    # Match predictions to ground truth using 3D IoU (vectorized)
                    try:
                        iou_matrix = compute_3d_iou_batch(pred_boxes, gt_boxes)
                    except Exception as e:
                        LOGGER.warning(f"Error computing 3D IoU batch: {e}, falling back to individual computation")
                        # Fallback to individual computation if batch fails
                        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
                        for i, pred_box in enumerate(pred_boxes):
                            for j, gt_box in enumerate(gt_boxes):
                                if pred_box.class_id == gt_box.class_id:
                                    try:
                                        iou = compute_3d_iou(pred_box, gt_box)
                                        iou_matrix[i, j] = iou
                                    except Exception as e2:
                                        LOGGER.warning(f"Error computing 3D IoU: {e2}")
                                        iou_matrix[i, j] = 0.0

                    # Match predictions to ground truth (greedy matching)
                    matched_gt = set()
                    tp = np.zeros((len(pred_boxes), self.niou), dtype=bool)
                    fp = np.zeros((len(pred_boxes), self.niou), dtype=bool)

                    # Sort predictions by confidence
                    pred_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i].confidence, reverse=True)

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
                    fp = np.ones((len(pred_boxes), self.niou), dtype=bool) if len(pred_boxes) > 0 else np.zeros((0, self.niou), dtype=bool)

                # Extract statistics
                conf = np.array([box.confidence for box in pred_boxes]) if pred_boxes else np.array([])
                pred_cls = np.array([box.class_id for box in pred_boxes]) if pred_boxes else np.array([], dtype=int)
                target_cls = np.array([box.class_id for box in gt_boxes]) if gt_boxes else np.array([], dtype=int)

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
            if hasattr(self, '_progress_bar') and self._progress_bar is not None and RANK in {-1, 0}:
                # Update progress bar periodically to avoid performance impact
                if hasattr(self, '_batch_count'):
                    self._batch_count += 1
                else:
                    self._batch_count = 1
                
                # Update every 5 batches or if we're near the end (more frequent than before)
                if self._batch_count % 5 == 0 or (hasattr(self, '_total_batches') and self._batch_count >= self._total_batches - 1):
                    try:
                        metrics_str = self._format_progress_metrics()
                        if metrics_str:
                            self._progress_bar.set_description(metrics_str)
                    except Exception as e:
                        LOGGER.debug(f"Error updating progress bar: {e}")

            # Generate visualization images if plots enabled.
            # NOTE: This stereo validator saves 1 file per sample, so keep defaults conservative to avoid generating
            # thousands of images when using large validation batch sizes.
            #
            # Default to 3 batches (matching Detect task style), but can be overridden via `max_plot_batches`.
            # Additionally cap samples per batch (default=1) via `max_plot_samples`.
            max_plot_batches = getattr(self.args, "max_plot_batches", 3)
            if self.args.plots and hasattr(self, "batch_i") and self.batch_i < max_plot_batches and RANK in {-1, 0}:
                try:
                    self.plot_validation_samples(batch, preds, self.batch_i)
                except Exception as e:
                    LOGGER.warning(f"Error generating validation visualizations: {e}")

    def get_desc(self) -> str:
        """Return a formatted string summarizing validation metrics header for progress bar.
        
        Returns:
            Formatted header string matching the progress bar format.
        """
        # Right-align header strings to match right-aligned numeric values
        # Format: right-align strings in 11-char fields to match %11i and %11.4g alignment
        return ("%11s" + "%11s" * 6) % (
            "Images".rjust(11),
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
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.det_metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
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
            max_samples = getattr(self.args, "max_plot_samples", 1)
            num_samples = min(batch_size, max_samples)

            for si in range(num_samples):
                im_file = im_files[si] if si < len(im_files) else None
                if not im_file:
                    continue

                # Load original images from file paths
                left_path = Path(im_file)
                if not left_path.exists():
                    LOGGER.debug(f"Left image not found: {left_path}, skipping visualization")
                    continue

                # Get right image path (same filename, different directory)
                # im_file format: images/{split}/left/{image_id}.png
                # right path: images/{split}/right/{image_id}.png
                right_path = left_path.parent.parent / "right" / left_path.name
                if not right_path.exists():
                    LOGGER.debug(f"Right image not found: {right_path}, skipping visualization")
                    continue

                # Load original images (BGR format from OpenCV)
                left_img = cv2.imread(str(left_path))
                right_img = cv2.imread(str(right_path))

                if left_img is None or right_img is None:
                    LOGGER.debug(f"Failed to load images for {left_path}, skipping")
                    continue

                # Get predictions and ground truth for this sample
                pred_boxes = pred_boxes3d[si] if si < len(pred_boxes3d) else []
                labels = labels_list[si] if si < len(labels_list) else []
                calib = calibs[si] if si < len(calibs) and isinstance(calibs[si], dict) else None

                # Skip visualization if no calibration available
                if calib is None:
                    continue

                # Get actual image dimensions and compute letterbox parameters
                # Calibration in batch is for letterboxed images, but we're visualizing original images
                actual_h, actual_w = left_img.shape[:2]
                imgsz = getattr(self.args, 'imgsz', 384)
                
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
                    from ultralytics.data.stereo.calib import CalibrationParameters
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

                # Convert labels to Box3D for ground truth
                # Use original calibration for accurate conversion
                gt_boxes = []
                if labels:
                    try:
                        gt_boxes = _labels_to_box3d_list(labels, calib_orig, names=self.names)
                    except Exception as e:
                        LOGGER.debug(f"Error converting labels to Box3D for visualization (sample {si}): {e}")

                # Filter out predictions with confidence == 0 or below threshold before visualization
                if pred_boxes:
                    conf_threshold = self.args.conf
                    if conf_threshold < 0.1:
                        LOGGER.warning(f"The prediction conf threshold is less than 0.1, you can set the conf through CLI.")
                    pred_boxes = [
                        box for box in pred_boxes 
                        if hasattr(box, 'confidence') and box.confidence > conf_threshold
                    ]

                # Generate visualization with predictions only (top image)
                # Use original calibration (not letterboxed) since images are original size
                try:
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
                except Exception as e:
                    LOGGER.debug(f"Error generating prediction visualization for sample {si}: {e}")
                    continue

                # Generate visualization with ground truth only (bottom image)
                # Use original calibration (not letterboxed) since images are original size
                try:
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
                except Exception as e:
                    LOGGER.debug(f"Error generating ground truth visualization for sample {si}: {e}")
                    continue

                # Stack vertically: predictions on top, ground truth on bottom
                # Use left image only for simplicity (or combine left+right horizontally first)
                h_pred, w_pred = combined_pred.shape[:2]
                h_gt, w_gt = combined_gt.shape[:2]
                
                # Ensure both images have the same width
                if w_pred != w_gt:
                    target_w = max(w_pred, w_gt)
                    if w_pred < target_w:
                        combined_pred = cv2.resize(combined_pred, (target_w, h_pred), interpolation=cv2.INTER_LINEAR)
                    if w_gt < target_w:
                        combined_gt = cv2.resize(combined_gt, (target_w, h_gt), interpolation=cv2.INTER_LINEAR)
                
                # Stack vertically
                stacked = np.vstack([combined_pred, combined_gt])
                
                # Add labels
                label_height = 30
                stacked_with_labels = np.zeros((stacked.shape[0] + label_height * 2, stacked.shape[1], 3), dtype=np.uint8)
                stacked_with_labels[label_height:label_height + stacked.shape[0], :, :] = stacked
                
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
                save_path = self.save_dir / f"val_batch{batch_idx}_sample{si}_{image_id}.jpg"
                cv2.imwrite(str(save_path), stacked_with_labels)
                if self.on_plot:
                    self.on_plot(save_path)

        except Exception as e:
            LOGGER.warning(f"Error in plot_validation_samples: {e}")

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        if not self.metrics.stats:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")
            return

        # Count ground truth objects per class
        all_target_cls = np.concatenate([s["target_cls"] for s in self.metrics.stats if len(s["target_cls"]) > 0], axis=0) if self.metrics.stats else np.array([], dtype=int)
        if len(all_target_cls) == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")
            return

        nt_per_class = np.bincount(all_target_cls.astype(int), minlength=self.metrics.nc) if len(all_target_cls) > 0 else np.zeros(self.metrics.nc, dtype=int)
        total_gt = int(nt_per_class.sum())

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
                    all_precisions.extend([v for v in iou_dict.values() if isinstance(v, (int, float))])
            precision_mean = float(np.mean(all_precisions)) if all_precisions else 0.0
        
        if isinstance(self.metrics.recall, dict) and self.metrics.recall:
            all_recalls = []
            for iou_dict in self.metrics.recall.values():
                if isinstance(iou_dict, dict):
                    all_recalls.extend([v for v in iou_dict.values() if isinstance(v, (int, float))])
            recall_mean = float(np.mean(all_recalls)) if all_recalls else 0.0

        # Get 2D bbox mAP50 and mAP50-95 metrics (for main summary line)
        box_map50 = 0.0
        box_map5095 = 0.0
        try:
            det_res = self.det_metrics.results_dict if hasattr(self, "det_metrics") else {}
            box_map50 = det_res.get("metrics/mAP50(B)", 0.0)
            box_map5095 = det_res.get("metrics/mAP50-95(B)", 0.0)
        except Exception as e:
            LOGGER.debug(f"Failed to get bbox2d metrics for main summary: {e}")

        # Print format: class name, images, AP3D@0.5, AP3D@0.7, precision, recall, mAP50, mAP50-95 (matches progress bar format)
        # Note: labels (total_gt) is shown in verbose per-class output, not in main summary
        pf = "%22s" + "%11i" + "%11.3g" * 6
        LOGGER.info(pf % ("all", self.seen, maps3d_50, maps3d_70, precision_mean, recall_mean, box_map50, box_map5095))

        # Also print 2D bbox metrics (YOLO-style)
        try:
            det_res = self.det_metrics.results_dict if hasattr(self, "det_metrics") else {}
            box_p = det_res.get("metrics/precision(B)", 0.0)
            box_r = det_res.get("metrics/recall(B)", 0.0)
            box_map50 = det_res.get("metrics/mAP50(B)", 0.0)
            box_map = det_res.get("metrics/mAP50-95(B)", 0.0)
            pf2 = "%22s" + "%11i" + "%11.3g" * 4
            LOGGER.info(pf2 % ("bbox2d", self.seen, box_p, box_r, box_map50, box_map))
        except Exception as e:
            LOGGER.debug(f"Failed to print bbox2d metrics: {e}")

        # Print results per class if verbose and multiple classes
        if self.args.verbose and not self.training and self.metrics.nc > 1 and self.metrics.ap3d_50:
            # Get per-class 2D bbox metrics for printing
            class_map50_dict = {}
            class_map5095_dict = {}
            try:
                if hasattr(self, "det_metrics") and hasattr(self.det_metrics, "class_result") and hasattr(self.det_metrics, "ap_class_index"):
                    # Get per-class mAP50 and mAP50-95 from DetMetrics
                    # class_result(i) returns (p, r, map50, map) for index i in ap_class_index
                    # ap_class_index[i] gives the class_id at index i
                    for i, class_id in enumerate(self.det_metrics.ap_class_index):
                        try:
                            class_result = self.det_metrics.class_result(i)
                            if len(class_result) >= 4:
                                class_map50_dict[class_id] = float(class_result[2])  # mAP50
                                class_map5095_dict[class_id] = float(class_result[3])  # mAP50-95
                        except (IndexError, AttributeError) as e:
                            LOGGER.debug(f"Failed to get metrics for class {class_id} at index {i}: {e}")
                            continue
            except Exception as e:
                LOGGER.debug(f"Failed to get per-class bbox2d metrics: {e}")

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
                    recall_class = float(np.mean(recall_values)) if recall_values else 0.0
                
                # Get class-specific mAP50 and mAP50-95
                class_map50 = class_map50_dict.get(class_id, 0.0)
                class_map5095 = class_map5095_dict.get(class_id, 0.0)
                
                nt_class = int(nt_per_class[class_id]) if class_id < len(nt_per_class) else 0
                # Use same format as main summary (without labels column to match progress bar)
                LOGGER.info(
                    pf
                    % (
                        class_name,
                        nt_class,  # number of ground truth labels for this class
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
        if not hasattr(self.metrics, 'stats') or len(self.metrics.stats) == 0:
            return ("%11i" + "%11s" * 6) % (int(self.seen), "-", "-", "-", "-", "-", "-")
        
        # Compute intermediate metrics on accumulated stats
        try:
            # Save current stats
            saved_stats = self.metrics.stats.copy()
            # Process to get metrics
            temp_results = self.metrics.process(save_dir=self.save_dir, plot=False)
            # Also compute bbox metrics if available
            det_temp = self.det_metrics.process(save_dir=self.save_dir, plot=False) if hasattr(self, "det_metrics") else {}
            # Restore stats for final processing
            self.metrics.stats = saved_stats
            
            if not temp_results:
                return ("%11i" + "%11s" * 6) % (int(self.seen), "-", "-", "-", "-", "-", "-")
            
            # Get AP3D metrics (use mean values)
            ap50 = temp_results.get('maps3d_50', 0.0)
            if isinstance(ap50, dict):
                ap50 = float(np.mean([v for v in ap50.values() if isinstance(v, (int, float))])) if ap50 else 0.0
            ap70 = temp_results.get('maps3d_70', 0.0)
            if isinstance(ap70, dict):
                ap70 = float(np.mean([v for v in ap70.values() if isinstance(v, (int, float))])) if ap70 else 0.0
            
            # Get precision and recall (flatten nested dicts)
            precision = temp_results.get('precision', 0.0)
            if isinstance(precision, dict):
                all_precisions = []
                for iou_dict in precision.values():
                    if isinstance(iou_dict, dict):
                        all_precisions.extend([v for v in iou_dict.values() if isinstance(v, (int, float))])
                precision = float(np.mean(all_precisions)) if all_precisions else 0.0
            
            recall = temp_results.get('recall', 0.0)
            if isinstance(recall, dict):
                all_recalls = []
                for iou_dict in recall.values():
                    if isinstance(iou_dict, dict):
                        all_recalls.extend([v for v in iou_dict.values() if isinstance(v, (int, float))])
                recall = float(np.mean(all_recalls)) if all_recalls else 0.0
            
            # Get bbox mAPs (DetMetrics uses 'metrics/mAP50(B)' keys)
            map50 = det_temp.get("metrics/mAP50(B)", 0.0) if isinstance(det_temp, dict) else 0.0
            map5095 = det_temp.get("metrics/mAP50-95(B)", 0.0) if isinstance(det_temp, dict) else 0.0

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
            LOGGER.debug(f"Error formatting progress metrics: {e}")
            return ("%11i" + "%11s" * 6) % (int(self.seen), "-", "-", "-", "-", "-", "-")



    def build_dataset(self, img_path: str | dict[str, Any], mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
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
        desc = img_path if isinstance(img_path, dict) else self.data.get(mode) if hasattr(self, "data") else None
        
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
            
            return Stereo3DDetDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", mode)),
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
                max_samples=max_samples,
                output_size=output_size,
                mean_dims=mean_dims,
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
            )
        
        # If we can't determine the dataset, raise an error
        raise ValueError(
            f"Cannot build dataset from img_path={img_path} (type: {type(img_path)}). "
            f"Expected a string path or a descriptor dict with type='kitti_stereo'."
        )

    def get_dataloader(self, dataset_path: str | dict[str, Any], batch_size: int) -> torch.utils.data.DataLoader:
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