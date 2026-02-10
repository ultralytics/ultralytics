# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Unified preprocessing and postprocessing utilities for stereo3ddet.

This module provides shared preprocessing and postprocessing functions used by
the trainer, validator, and predictor to ensure consistent behavior across
the entire stereo 3D detection pipeline.

Key Functions:
    - preprocess_stereo_batch: Unified preprocessing for train/val batches from dataset
    - preprocess_stereo_images: Unified preprocessing for prediction (raw images)
    - compute_letterbox_params: Compute letterbox scale and padding
    - decode_and_refine_predictions: Unified decode + geometric + dense alignment pipeline
    - get_refinement_config: Load geometric and dense alignment config from YAML
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.nms import non_max_suppression


# =============================================================================
# Configuration Loading (simplified with lru_cache)
# =============================================================================

@lru_cache(maxsize=1)
def _load_stereo3ddet_config() -> dict:
    """Load stereo3ddet config from YAML file.

    Returns:
        Full config dict from stereo3ddet_full.yaml, or empty dict if not found.
    """
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "cfg"
        / "models"
        / "stereo3ddet_full.yaml"
    )
    if config_path.exists():
        try:
            return YAML.load(str(config_path))
        except Exception as e:
            LOGGER.debug("Failed to load stereo3ddet config: %s", e)
    return {}


def get_geometric_config() -> dict:
    """Get geometric construction configuration.

    Returns:
        Dict with geometric_construction settings:
        - enabled: bool (default True)
        - max_iterations: int (default 10)
        - tolerance: float (default 1e-6)
        - damping: float (default 1e-3)
        - fallback_on_failure: bool (default True)
    """
    defaults = {
        "enabled": True,
        "max_iterations": 10,
        "tolerance": 1e-6,
        "damping": 1e-3,
        "fallback_on_failure": True,
    }
    full_config = _load_stereo3ddet_config()
    return {**defaults, **full_config.get("geometric_construction", {})}


def get_dense_alignment_config() -> dict:
    """Get dense alignment configuration.

    Returns:
        Dict with dense_alignment settings:
        - enabled: bool (default True)
        - method: "ncc" or "sad" (default "ncc")
        - depth_search_range: float in meters (default 2.0)
        - depth_steps: int (default 32)
        - patch_size: int in pixels (default 7)
        - skip_dense_for_occluded: bool (default True)
    """
    defaults = {
        "enabled": True,
        "method": "ncc",
        "depth_search_range": 2.0,
        "depth_steps": 32,
        "patch_size": 7,
        "skip_dense_for_occluded": True,
    }
    full_config = _load_stereo3ddet_config()
    return {**defaults, **full_config.get("dense_alignment", {})}


def clear_config_cache() -> None:
    """Clear the config cache. Call this to reload config from disk."""
    _load_stereo3ddet_config.cache_clear()


# =============================================================================
# Letterbox Utilities
# =============================================================================

def compute_letterbox_params(
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


# =============================================================================
# Decoding Functions
# =============================================================================



def decode_stereo3d_outputs(
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
    """Decode stereo3ddet outputs to Box3D objects.

    Uses Detect inference output for candidate 2D boxes and class scores, then samples
    the auxiliary stereo/3D maps at the kept P3 indices to estimate depth/dimensions/orientation.

    Args:
        outputs: Model outputs dictionary.
        conf_threshold: Confidence threshold for filtering detections.
        top_k: Maximum number of detections to extract.
        calib: Calibration parameters (dict or list of dicts).
        imgsz: Input image size.
        ori_shapes: Original image shapes per batch item.
        iou_thres: IoU threshold for NMS.
        mean_dims: Mean dimensions per class (class ID -> (H, W, L) in meters).
        std_dims: Standard deviation of dimensions per class.
        class_names: Mapping from class ID to class name.
    """
    if "det" not in outputs:
        raise KeyError("decode_stereo3d_outputs expected outputs['det']")

    det_out = outputs["det"]
    det_inf = det_out[0] if isinstance(det_out, (tuple, list)) else det_out
    bs = int(det_inf.shape[0])
    nc = int(det_inf.shape[1] - 4)

    # Determine letterbox input size
    if imgsz is None:
        imgsz = (384, 384)
    input_h, input_w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))

    # NMS on Detect inference output (BCN format)
    dets, keepi = non_max_suppression(
        det_inf,
        conf_thres=conf_threshold,
        iou_thres=iou_thres,
        max_det=top_k,
        nc=nc,
        return_idxs=True,
    )

    # Validate required parameters
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
        if calib is None or len(calib) == 0:
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
        letterbox_scale, pad_left, pad_top = compute_letterbox_params(ori_h, ori_w, imgsz)

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

            # Map kept index -> flat index for sampling aux maps [B, C, HW_total]
            hw_total = outputs["lr_distance"].shape[2] if "lr_distance" in outputs else 0
            if idx_b is not None and j < idx_b.numel():
                flat_idx = int(idx_b[j].item())
                flat_idx = max(0, min(hw_total - 1, flat_idx))
            else:
                # Fallback: compute flat index from bbox center (approximate)
                u_letterbox = float(((x1_l + x2_l) / 2.0).item())
                v_letterbox = float(((y1_l + y2_l) / 2.0).item())
                # Use P3 grid as approximation for flat index
                gx = int(max(0, min(input_w // 8 - 1, u_letterbox / 8)))
                gy = int(max(0, min(input_h // 8 - 1, v_letterbox / 8)))
                flat_idx = gy * (input_w // 8) + gx
                flat_idx = max(0, min(hw_total - 1, flat_idx))

            # Sample aux predictions (3D: [B, C, HW_total])
            lr_log = float(outputs["lr_distance"][b, 0, flat_idx].item()) if "lr_distance" in outputs else -4.0
            depth_log = float(outputs["depth"][b, 0, flat_idx].item()) if "depth" in outputs else None
            dim_off = outputs["dimensions"][b, :, flat_idx].float() if "dimensions" in outputs else torch.zeros(3)
            ori_pred = outputs["orientation"][b, :, flat_idx].float() if "orientation" in outputs else None

            # Depth from disparity (lr_distance is in log-space, exp() to get normalized disparity)
            disparity_letterbox = math.exp(max(lr_log, -10.0)) * input_w
            disparity_orig = disparity_letterbox / letterbox_scale
            z_from_disp = (fx_orig * baseline) / max(disparity_orig, eps)

            # Combine disparity-derived depth with direct depth prediction (geometric mean)
            if depth_log is not None:
                z_from_direct = math.exp(max(depth_log, 0.0))  # clamp min ~1m
                z_3d = math.sqrt(z_from_disp * z_from_direct)  # geometric mean
            else:
                z_3d = z_from_disp

            # Use bbox center as (u,v)
            u_letterbox = float(((x1_l + x2_l) / 2.0).item())
            v_letterbox = float(((y1_l + y2_l) / 2.0).item())
            u_orig = (u_letterbox - pad_left) / letterbox_scale
            v_orig = (v_letterbox - pad_top) / letterbox_scale

            x_3d = (u_orig - cx_orig) * z_3d / fx_orig
            y_3d = (v_orig - cy_orig) * z_3d / fy_orig

            # Dimensions decode: offset [ΔH, ΔW, ΔL] -> actual dims
            # mean_dims/std_dims from validator are (H, W, L) format, dim_off is [ΔH, ΔW, ΔL]
            mean_h, mean_w, mean_l = mean_dims.get(c, mean_dims[0])
            std_h, std_w, std_l = std_dims.get(c, std_dims[0])
            height = mean_h + float(dim_off[0].item()) * std_h
            width = mean_w + float(dim_off[1].item()) * std_w
            length = mean_l + float(dim_off[2].item()) * std_l

            # Orientation: decode from predicted sin/cos or fallback to alpha=0
            ray_angle = math.atan2(x_3d, z_3d)
            if ori_pred is not None:
                alpha = math.atan2(float(ori_pred[0].item()), float(ori_pred[1].item()))
                theta = alpha + ray_angle
            else:
                theta = ray_angle  # fallback: alpha=0

            box3d = Box3D(
                center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                dimensions=(float(length), float(width), float(height)),
                orientation=float(theta),
                class_label=class_names.get(c, str(c)),
                class_id=c,
                confidence=confidence,
            )
            boxes3d.append(box3d)

        results_per_batch.append(boxes3d)

    # Match legacy return format
    if bs == 1:
        return results_per_batch[0]
    return results_per_batch


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_stereo_batch(
    batch: dict[str, Any],
    device: torch.device,
    half: bool = False,
) -> dict[str, Any]:
    """Unified preprocessing for train/val batches from dataset.

    Normalizes 6-channel images to float [0,1] and moves targets to device.
    Targets are generated in the dataset's collate_fn, so this just moves
    them to the device if they're not already there.

    Args:
        batch: Batch dictionary from dataloader containing 'img' tensor and targets.
        device: Target device for tensors.
        half: If True, convert images to half precision (FP16).

    Returns:
        Preprocessed batch dictionary with images normalized and on device.
    """
    imgs = batch["img"].to(device, non_blocking=True)
    batch["img"] = (imgs.half() if half else imgs.float()) / 255.0

    # Move optional dict targets to device (generated by dataset)
    if "targets" in batch and isinstance(batch["targets"], dict):
        batch["targets"] = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
    if "aux_targets" in batch and isinstance(batch["aux_targets"], dict):
        batch["aux_targets"] = {k: v.to(device, non_blocking=True) for k, v in batch["aux_targets"].items()}
    for k in ("batch_idx", "cls", "bboxes"):
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=True)

    return batch


def preprocess_stereo_images(
    images: list[np.ndarray] | torch.Tensor,
    imgsz: tuple[int, int],
    device: torch.device,
    half: bool = False,
    letterbox: LetterBox | None = None,
) -> torch.Tensor:
    """Unified preprocessing for prediction (raw images).

    Applies letterbox resizing, BGR to RGB conversion, normalization, and
    converts numpy arrays to tensors on the target device.

    Args:
        images: List of 6-channel stereo images [H, W, 6] in BGR format, or tensor.
        imgsz: Target image size as (H, W).
        device: Target device for tensors.
        half: If True, convert images to half precision (FP16).
        letterbox: Optional LetterBox transformer. If None, a default one is created.

    Returns:
        Preprocessed tensor of shape (N, 6, H, W) normalized to [0, 1].
    """
    if isinstance(images, torch.Tensor):
        # Already a tensor, just move to device and normalize
        images = images.to(device)
        images = images.half() if half else images.float()
        if images.dtype == torch.uint8:
            images = images / 255.0
        return images

    # Create letterbox if not provided
    if letterbox is None:
        letterbox = LetterBox(new_shape=imgsz, auto=False, scale_fill=False, scaleup=True, stride=32)

    # Apply letterbox to each stereo image (same as dataset)
    # Each image is [H, W, 6] (stereo pair)
    letterboxed = []
    for stereo_img in images:
        letterboxed_img = letterbox(image=stereo_img)
        letterboxed.append(letterboxed_img)

    # Convert list of letterboxed numpy arrays to tensor
    im = np.stack(letterboxed)  # [N, H, W, 6]
    im = im[..., ::-1]  # BGR to RGB for all 6 channels
    im = im.transpose((0, 3, 1, 2))  # [N, H, W, 6] -> [N, 6, H, W]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im)

    im = im.to(device)
    im = im.half() if half else im.float()
    im /= 255  # 0-255 to 0.0-1.0

    return im


# =============================================================================
# Postprocessing Functions
# =============================================================================

def decode_and_refine_predictions(
    preds: dict[str, torch.Tensor],
    batch: dict[str, Any] | None = None,
    args: Any = None,
    use_geometric: bool | None = None,
    use_dense_alignment: bool | None = None,
    conf_threshold: float = 0.25,
    top_k: int = 100,
    iou_thres: float = 0.45,
    imgsz: int | tuple[int, int] | None = None,
    mean_dims: dict[int, tuple[float, float, float]] | None = None,
    std_dims: dict[int, tuple[float, float, float]] | None = None,
    class_names: dict[int, str] | None = None,
) -> list[list[Box3D]]:
    """Unified decode + refine pipeline for val and predict.

    Decodes raw model outputs to Box3D objects and optionally applies
    geometric construction and dense alignment refinements.

    Args:
        preds: Dictionary of model outputs.
        batch: Optional batch dictionary with calibration, images, and original shapes.
        args: Optional args object with configuration (conf, iou, imgsz, etc.).
        use_geometric: If True, apply geometric construction refinement.
            If None, uses config default (enabled).
        use_dense_alignment: If True, apply dense alignment refinement.
            If None, uses config default (enabled).
        conf_threshold: Confidence threshold for filtering detections.
        top_k: Maximum number of detections to extract.
        iou_thres: IoU threshold for NMS.
        imgsz: Input image size for letterbox calculations.
        mean_dims: Mean dimensions per class (class ID -> (H, W, L) in meters).
        std_dims: Standard deviation of dimensions per class.
        class_names: Mapping from class ID to class name.

    Returns:
        List of Box3D lists (one per batch item).
    """
    # Get parameters from args if provided
    if args is not None:
        conf_threshold = getattr(args, "conf", conf_threshold)
        iou_thres = getattr(args, "iou", iou_thres)
        if imgsz is None:
            imgsz = getattr(args, "imgsz", 384)

    if imgsz is None:
        imgsz = 384

    # Extract calibration and original shapes from batch
    calibs = []
    ori_shapes = []
    calib = None
    if batch is not None:
        calibs = batch.get("calib", [])
        ori_shapes = batch.get("ori_shape", [])

        # Get batch size from predictions
        det_out = preds.get("det")
        if det_out is not None:
            det_inf = det_out[0] if isinstance(det_out, (tuple, list)) else det_out
            batch_size = int(det_inf.shape[0])
        else:
            batch_size = 1

        # Handle batch calibration
        if calibs:
            if len(calibs) == batch_size and all(isinstance(c, dict) for c in calibs):
                calib = calibs
            elif len(calibs) > 0 and isinstance(calibs[0], dict):
                calib = calibs[0]

    # Decode predictions
    results = decode_stereo3d_outputs(
        preds,
        conf_threshold=conf_threshold,
        top_k=top_k,
        calib=calib,
        imgsz=imgsz,
        ori_shapes=ori_shapes if ori_shapes else None,
        iou_thres=iou_thres,
        mean_dims=mean_dims,
        std_dims=std_dims,
        class_names=class_names,
    )

    # Ensure results is list of lists
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], Box3D):
        results = [results]

    # Get batch size from results
    batch_size = len(results)

    # Get configs
    geo_config = get_geometric_config()
    dense_config = get_dense_alignment_config()

    # Apply geometric construction refinement
    # NOTE: Geometric construction is disabled by default in postprocess to match
    # the original behavior. It can be enabled explicitly with use_geometric=True.
    # The geometric construction was present in the codebase but not called from
    # the validator's postprocess method.
    should_apply_geo = use_geometric is True  # Only apply if explicitly enabled
    if should_apply_geo and batch is not None:
        results = _apply_geometric_construction(results, calibs, batch_size, geo_config)

    # Apply dense alignment refinement (enabled by default, matching old behavior)
    should_apply_dense = use_dense_alignment is True or (
        use_dense_alignment is None and dense_config.get("enabled", True)
    )
    if should_apply_dense and batch is not None:
        results = _apply_dense_alignment(results, calibs, batch, batch_size, dense_config)

    return results


def _apply_geometric_construction(
    results: list[list[Box3D]],
    calibs: list[dict],
    batch_size: int,
    config: dict,
) -> list[list[Box3D]]:
    """Apply geometric construction to refine initial 3D estimates.

    Refines 3D box center (x, y, z) and orientation using Gauss-Newton
    optimization with geometric constraint equations.

    Args:
        results: List of Box3D lists (one per batch item).
        calibs: List of calibration dictionaries.
        batch_size: Number of images in batch.
        config: Geometric construction config dict.

    Returns:
        Results with geometrically refined 3D estimates.
    """
    from ultralytics.models.yolo.stereo3ddet.geometric import solve_geometric_batch

    if not config.get("enabled", True):
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
            # Project 3D box to 2D to get center for geometric solver
            bbox_2d = box.project_to_2d(sample_calib)
            if bbox_2d[2] <= bbox_2d[0] or bbox_2d[3] <= bbox_2d[1]:
                continue
            u_left = (bbox_2d[0] + bbox_2d[2]) / 2.0
            v_left = (bbox_2d[1] + bbox_2d[3]) / 2.0

            # Extract center_3d and compute disparity
            _, _, z_3d = box.center_3d
            if sample_calib and z_3d > 0:
                fx = sample_calib.get("fx", 721.5)
                baseline = sample_calib.get("baseline", 0.54)
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
            fallback_on_failure=config.get("fallback_on_failure", True),
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
                    truncated=box.truncated,
                    occluded=box.occluded,
                )
                refined_boxes.append(refined_box)
            else:
                refined_boxes.append(box)

        results[b] = refined_boxes

    return results


def _apply_dense_alignment(
    results: list[list[Box3D]],
    calibs: list[dict],
    batch: dict[str, Any],
    batch_size: int,
    config: dict,
) -> list[list[Box3D]]:
    """Apply dense photometric alignment to refine depth estimates.

    Refines the depth estimates of detected 3D boxes using photometric
    matching between left and right stereo images.

    Args:
        results: List of Box3D lists (one per batch item).
        calibs: List of calibration dictionaries.
        batch: Batch dictionary containing images.
        batch_size: Number of images in batch.
        config: Dense alignment config dict.

    Returns:
        Results with refined depth values.
    """
    from ultralytics.models.yolo.stereo3ddet.dense_align_optimized import (
        create_dense_alignment_optimized,
    )
    from ultralytics.models.yolo.stereo3ddet.occlusion import (
        classify_occlusion,
        should_skip_dense_alignment,
    )

    if not config.get("enabled", True):
        return results

    # Create aligner on-demand
    aligner = create_dense_alignment_optimized(config)

    # Get images from batch
    imgs = batch.get("img", None)
    if imgs is None:
        return results

    # Images are [B, 6, H, W] tensor - split into left [B, 3, H, W] and right [B, 3, H, W]
    # Convert to numpy HWC format for dense alignment
    try:
        # Move to CPU and convert to numpy
        imgs_np = imgs.cpu().numpy()  # [B, 6, H, W]

        # Get occlusion config for skipping heavily occluded objects
        skip_occluded = config.get("skip_dense_for_occluded", True)

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
                    detections = []
                    for box in boxes:
                        bbox_2d = box.project_to_2d(sample_calib)
                        det = {
                            "bbox_2d": tuple(bbox_2d) if bbox_2d[2] > bbox_2d[0] else (0, 0, 100, 100),
                            "center_3d": box.center_3d,
                        }
                        detections.append(det)
                    occluded_indices, _ = classify_occlusion(detections)
                except Exception as e:
                    LOGGER.debug("Occlusion classification failed: %s", e)
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
