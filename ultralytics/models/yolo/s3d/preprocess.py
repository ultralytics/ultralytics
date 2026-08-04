# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unified preprocessing and postprocessing utilities for s3d.

This module provides shared preprocessing and postprocessing functions used by
the trainer, validator, and predictor to ensure consistent behavior across
the entire stereo 3D detection pipeline.

Key Functions:
    - preprocess_stereo_batch: Unified preprocessing for train/val batches from dataset
    - preprocess_stereo_images: Unified preprocessing for prediction (raw images)
    - compute_letterbox_params: Compute letterbox scale and padding
    - decode_and_refine_predictions: Shared decode pipeline for val and predict
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.models.yolo.s3d.head import AUX_SPECS
from ultralytics.models.yolo.s3d.orientation import decode_orientation
from ultralytics.utils.nms import non_max_suppression

# Head outputs carried per anchor as [B, C, HW_total]. Derived from the head's own spec so a new aux
# branch is covered automatically; the two split-out channels are named separately by forward_head.
PER_ANCHOR_AUX_KEYS = (*AUX_SPECS, "lr_logvar", "depth_bins")

# =============================================================================
# Configuration Defaults
# =============================================================================


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
    new_unpad_w = round(ori_w * scale)
    new_unpad_h = round(ori_h * scale)
    dw = out_w - new_unpad_w
    dh = out_h - new_unpad_h
    pad_left = dw // 2
    pad_top = dh // 2
    return scale, pad_left, pad_top


# =============================================================================
# Decoding Functions
# =============================================================================


def _dfl_variance(outputs: dict[str, torch.Tensor], b: int, idx: int) -> float:
    """Return the DFL depth-distribution spread Σpᵢ(bᵢ-μ)² at (b, idx), else 1.0 (high variance).

    The bin grid comes from the head, which is the only thing that knows it: DepthDFL._set_range()
    retargets the grid to each dataset's depth range, so rebuilding it here from DEPTH_MIN/DEPTH_MAX
    would evaluate the logits on a different axis than the one they were trained and decoded on — and
    this variance is the fusion weight for the depth cue against the disparity cue.

    Args:
        outputs: Model outputs dictionary, may contain raw "depth_bins" logits [B, n_bins, HW] and the
            matching "depth_bin_values" grid [n_bins].
        b: Batch index.
        idx: Flat spatial index into the aux maps.

    Returns:
        Variance of the softmax-weighted depth-bin distribution in log-depth space, or 1.0 when the
        bins or their grid are absent.
    """
    if "depth_bins" not in outputs or "depth_bin_values" not in outputs:
        return 1.0

    logits = outputs["depth_bins"][b, :, idx].float()
    bin_values = outputs["depth_bin_values"].to(logits.device).float()
    probs = torch.softmax(logits, dim=0)
    mu = (probs * bin_values).sum()
    return float((probs * (bin_values - mu) ** 2).sum().item())


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
    score_k: float = 0.5,
    depth_var_scale: float = 1.0,
    calib_letterboxed: bool = False,
) -> list[Box3D] | list[list[Box3D]]:
    """Decode s3d outputs to Box3D objects.

    Uses Detect inference output for candidate 2D boxes and class scores, then samples the auxiliary stereo/3D maps at
    the kept P3 indices to estimate depth/dimensions/orientation.

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
        score_k: Decay rate for the uncertainty-based confidence weighting (confidence is scaled by exp(-score_k *
            sigma), sigma = predicted lr-distance std-dev, when "lr_logvar" is present).
        depth_var_scale: Multiplier on the direct-depth cue's variance in the inverse-variance depth fusion. 1.0
            (default) is an exact no-op; <1 shrinks the variance and so gives the direct cue more fusion weight,
            which is how a higher bin count re-weights the fusion without changing the bin count.
        calib_letterboxed: If True, reverse-letterbox the per-sample calib (fx/fy/cx/cy) to original-image coords before
            back-projection (the production caller sets this).

    Notes:
        The projected-center offset ("proj_offset"), inverse-variance depth fusion, and
        uncertainty score-weighting are applied unconditionally whenever the corresponding
        head outputs ("proj_offset"/"lr_logvar") are present; a checkpoint lacking them
        degrades gracefully to the 2D-box-center / geometric-mean path.
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

    # Default KITTI dimensions (H, W, L) when not provided by dataset config
    _DEFAULT_DIMS = {0: (1.53, 1.63, 3.88), 1: (1.73, 0.60, 0.80), 2: (1.73, 0.60, 1.76)}
    _DEFAULT_STD = {0: (0.15, 0.10, 0.42), 1: (0.12, 0.08, 0.20), 2: (0.15, 0.10, 0.25)}
    if mean_dims is None:
        mean_dims = _DEFAULT_DIMS
    if std_dims is None:
        std_dims = _DEFAULT_STD
    if class_names is None:
        class_names = {0: "Object"}

    # Original shapes fallback
    if ori_shapes is None or len(ori_shapes) == 0:
        ori_shapes = [(375, 1242)] * bs

    results_per_batch: list[list[Box3D]] = []
    eps = 1e-6

    # Anchor count for sampling the aux maps, which are all [B, C, HW_total]. Taken from the known
    # per-anchor keys rather than "whatever key comes first": the head also emits maps on a single
    # scale's grid (cv_disparity is [B, 1, H/8, W/8]), and reading shape[2] off one of those yields the
    # grid HEIGHT, which then clamps every detection's flat index into the first few anchors and zeroes
    # every 3D metric while leaving the 2D detections and all training losses untouched.
    hw_total = next(
        (outputs[k].shape[2] for k in PER_ANCHOR_AUX_KEYS if k in outputs and outputs[k].ndim == 3),
        0,
    )

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

        # When the caller supplies calib in letterbox-input space (fx,cx scaled by letterbox_scale,
        # principal point shifted by padding), reverse it to original-image coords so depth (fx) and the
        # 2D-center back-projection (cx,cy) are imgsz-invariant. u_orig/v_orig below are already in
        # original coords, so cx/cy/fx must match. Mirrors val.py's _reverse_letterbox_calib on the GT
        # side; without this, predictions land in a different frame than GT — correct only at
        # aspect-preserving imgsz (scale~=1) and badly wrong under a square imgsz. Tests that pass
        # original-space calib leave this False (the default).
        if calib_letterboxed:
            fx = fx / letterbox_scale
            fy = fy / letterbox_scale
            cx = (cx - pad_left) / letterbox_scale
            cy = (cy - pad_top) / letterbox_scale

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
            has_lr = "lr_distance" in outputs
            has_depth = "depth" in outputs
            lr_log = float(outputs["lr_distance"][b, 0, flat_idx].item()) if has_lr else None
            depth_log = float(outputs["depth"][b, 0, flat_idx].item()) if has_depth else None
            lr_logvar = (
                min(float(outputs["lr_logvar"][b, 0, flat_idx].item()), 20.0) if "lr_logvar" in outputs else None
            )
            if lr_logvar is not None:
                confidence *= math.exp(-score_k * math.sqrt(math.exp(lr_logvar)))
            dim_off = outputs["dimensions"][b, :, flat_idx].float() if "dimensions" in outputs else torch.zeros(3)
            ori_pred = outputs["orientation"][b, :, flat_idx].float() if "orientation" in outputs else None

            # Depth from disparity (lr_distance is in log-space, exp() to get normalized disparity)
            z_from_disp = None
            disparity_letterbox = None
            disparity_orig = None
            if lr_log is not None:
                disparity_letterbox = math.exp(max(lr_log, -10.0)) * input_w
                disparity_orig = disparity_letterbox / letterbox_scale  # original-image pixels
                z_from_disp = (fx * baseline) / max(disparity_orig, eps)  # original-image fx (imgsz-invariant)

            z_from_direct = None
            if depth_log is not None:
                z_from_direct = math.exp(max(depth_log, -10.0))

            # Combine depth sources
            if z_from_disp is not None and z_from_direct is not None:
                if lr_logvar is not None:  # inverse-variance fusion of the two depth cues
                    var_disp = math.exp(lr_logvar)
                    # Spread of the depth bins (else 1.0), rescaled: bin width sets this spread, so the knob
                    # reproduces a bin count's fusion re-weighting at a fixed bin count. 1.0 is exact.
                    var_direct = _dfl_variance(outputs, b, flat_idx) * depth_var_scale
                    w_disp, w_direct = 1.0 / max(var_disp, eps), 1.0 / max(var_direct, eps)
                    log_z = (w_disp * math.log(z_from_disp) + w_direct * math.log(z_from_direct)) / (w_disp + w_direct)
                    z_3d = math.exp(log_z)
                else:
                    z_3d = math.sqrt(z_from_disp * z_from_direct)  # geometric mean (no-uncertainty fallback)
            elif z_from_disp is not None:
                z_3d = z_from_disp
            elif z_from_direct is not None:
                z_3d = z_from_direct
            else:
                z_3d = 30.0  # fallback (shouldn't happen with valid config)

            # Use bbox center as (u,v)
            u_letterbox = float(((x1_l + x2_l) / 2.0).item())
            v_letterbox = float(((y1_l + y2_l) / 2.0).item())
            if "proj_offset" in outputs:  # decode the 3D center from the projected-center offset
                off = outputs["proj_offset"][b, :, flat_idx].float()
                u_letterbox += float(off[0]) * input_w
                v_letterbox += float(off[1]) * input_h
            u_orig = (u_letterbox - pad_left) / letterbox_scale
            v_orig = (v_letterbox - pad_top) / letterbox_scale

            x_3d = (u_orig - cx) * z_3d / fx
            y_3d = (v_orig - cy) * z_3d / fy

            # Dimensions decode: offset [ΔH, ΔW, ΔL] -> actual dims
            # mean_dims/std_dims from validator are (H, W, L) format, dim_off is [ΔH, ΔW, ΔL]
            mean_h, mean_w, mean_l = mean_dims.get(c, mean_dims[0])
            std_h, std_w, std_l = std_dims.get(c, std_dims[0])
            height = max(mean_h + float(dim_off[0].item()) * std_h, 0.01)
            width = max(mean_w + float(dim_off[1].item()) * std_w, 0.01)
            length = max(mean_l + float(dim_off[2].item()) * std_l, 0.01)

            # Orientation: decode MultiBin prediction (argmax bin + residual) → alpha, else alpha=0
            ray_angle = math.atan2(x_3d, z_3d)
            if ori_pred is not None:
                alpha = decode_orientation([float(v) for v in ori_pred.tolist()])
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

    Normalizes 6-channel images to float [0,1] and moves targets to device. Targets are generated in the dataset's
    collate_fn, so this just moves them to the device if they're not already there.

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

    Applies letterbox resizing, BGR to RGB conversion, normalization, and converts numpy arrays to tensors on the target
    device.

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
    # Convert BGR->RGB per view while preserving stereo order [left, right].
    left_rgb = im[..., :3][..., ::-1]
    right_rgb = im[..., 3:6][..., ::-1]
    im = np.concatenate([left_rgb, right_rgb], axis=3)
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
    conf_threshold: float = 0.25,
    top_k: int = 100,
    iou_thres: float = 0.45,
    imgsz: int | tuple[int, int] | None = None,
    mean_dims: dict[int, tuple[float, float, float]] | None = None,
    std_dims: dict[int, tuple[float, float, float]] | None = None,
    class_names: dict[int, str] | None = None,
    score_k: float = 0.5,
    depth_var_scale: float = 1.0,
) -> list[list[Box3D]]:
    """Shared decode pipeline for val and predict.

    Decodes raw model outputs to Box3D objects, pulling calibration and original shapes off the batch so both
    entry points reverse the letterbox identically.

    Args:
        preds: Dictionary of model outputs.
        batch: Optional batch dictionary with calibration, images, and original shapes.
        args: Optional args object with configuration (conf, iou, imgsz, etc.).
        conf_threshold: Confidence threshold for filtering detections.
        top_k: Maximum number of detections to extract.
        iou_thres: IoU threshold for NMS.
        imgsz: Input image size for letterbox calculations.
        mean_dims: Mean dimensions per class (class ID -> (H, W, L) in meters).
        std_dims: Standard deviation of dimensions per class.
        class_names: Mapping from class ID to class name.
        score_k: Decay rate for the uncertainty-based confidence weighting (see decode_stereo3d_outputs).
        depth_var_scale: Multiplier on the direct-depth cue's fusion variance (see decode_stereo3d_outputs).

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
        score_k=score_k,
        depth_var_scale=depth_var_scale,
        calib_letterboxed=True,  # batch calib is in letterbox-input space; decode reverses it to original
    )

    # Ensure results is list of lists
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], Box3D):
        results = [results]

    return results
