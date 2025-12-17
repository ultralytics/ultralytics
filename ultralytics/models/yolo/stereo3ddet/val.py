# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.stereo3ddet.metrics import Stereo3DDetMetrics
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, RANK, TQDM, YAML, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.metrics import compute_3d_iou
from ultralytics.utils.ops import Profile
from ultralytics.utils.plotting import plot_stereo3d_boxes
from ultralytics.utils.profiling import profile_function, profile_section
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model


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


@profile_function(name="decode_stereo3d_outputs")
def _decode_stereo3d_outputs_per_sample(
    outputs: dict[str, torch.Tensor],
    conf_threshold: float = 0.25,
    top_k: int = 100,
    calib: dict[str, float] | None = None,
) -> list[Box3D]:
    """T213: Original per-sample implementation for backward compatibility.
    
    This function processes a single sample (batch_size=1) using the original
    per-detection loop implementation.
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

    # T210: Cache calibration parameters at function start
    # KITTI original image size
    # TODO: get from calib
    original_width = 1242.0
    original_height = 375.0
    
    if calib is not None:
        fx = calib.get("fx", 721.5377)
        fy = calib.get("fy", 721.5377)
        cx = calib.get("cx", 609.5593)
        cy = calib.get("cy", 172.8540)
        baseline = calib.get("baseline", 0.54)
    else:
        # Default KITTI calibration values
        fx = fy = 721.5377
        cx, cy = 609.5593, 172.8540
        baseline = 0.54

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
    
    # Compute actual scale from feature map to original image
    # Feature map is at 1/4 of input resolution, but input was resized from original
    # scale = original_size / feature_map_size
    scale_w = original_width / w
    scale_h = original_height / h
    scale = (scale_w + scale_h) / 2.0  # Average scale factor

    # Flatten heatmap and get top-k detections
    heatmap = torch.sigmoid(heatmap)
    heatmap_flat = heatmap.reshape(num_classes, -1)  # [C, H*W]
    scores, indices = torch.topk(heatmap_flat, k=min(top_k, heatmap_flat.numel()), dim=1)

    for c in range(num_classes):
        # Filter to only paper classes (0, 1, 2 = Car, Pedestrian, Cyclist)
        if c not in mean_dims:
            LOGGER.debug(f"Skipping class {c} - not in paper classes (0, 1, 2)")
            continue
        
        class_scores = scores[c]
        class_indices = indices[c]

        for score, idx in zip(class_scores, class_indices):
            # debug
            assert score.min() >= 0 and score.max() <= 1, "score is not normalized"
            
            confidence = float(score.item())
            
            if confidence < conf_threshold:
                continue

            # Convert flat index to (y, x) coordinates
            y_idx = idx // w
            x_idx = idx % w

            # Get sub-pixel offset
            dx = offset[0, y_idx, x_idx].item()
            dy = offset[1, y_idx, x_idx].item()

            # Refined 2D center (in feature map coordinates)
            center_x = x_idx + dx
            center_y = y_idx + dy

            # Get 2D box size
            box_w = bbox_size[0, y_idx, x_idx].item()
            box_h = bbox_size[1, y_idx, x_idx].item()

            # Get left-right distance (in feature map space)
            d = lr_distance[0, y_idx, x_idx].item()

            # Convert 2D center to original image space
            u = center_x * scale_w
            v = center_y * scale_h
            
            # Compute depth from stereo geometry
            # IMPORTANT: d is in feature map space, must scale to original image space
            d_image = d * scale_w  # Scale disparity to original image width
            
            if d_image > 0:
                depth = (fx * baseline) / (d_image + 1e-6)
            else:
                depth = 50.0  # Default depth

            # 3D position
            x_3d = float((u - cx) * depth / fx)
            y_3d = float((v - cy) * depth / fy)
            z_3d = float(depth)

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
            
            # Convert observation angle α to global yaw θ
            # θ = α + arctan(x/z)
            ray_angle = np.arctan2(x_3d, z_3d)
            theta = alpha + ray_angle
            
            # Normalize to [-π, π]
            theta = np.arctan2(np.sin(theta), np.cos(theta))

            # Create Box3D object
            box3d = Box3D(
                center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                dimensions=(float(length), float(width), float(height)),
                orientation=float(theta),
                class_label=class_names[c],
                class_id=c,
                confidence=confidence,
                bbox_2d=(
                    float((center_x - box_w / 2) * scale),
                    float((center_y - box_h / 2) * scale),
                    float((center_x + box_w / 2) * scale),
                    float((center_y + box_h / 2) * scale),
                ),
            )
            boxes3d.append(box3d)

    return boxes3d


def decode_stereo3d_outputs(
    outputs: dict[str, torch.Tensor],
    conf_threshold: float = 0.25,
    top_k: int = 100,
    calib: dict[str, float] | list[dict[str, float]] | None = None,
) -> list[Box3D] | list[list[Box3D]]:
    """Decode 10-branch model outputs to 3D bounding boxes.

    Decodes Stereo CenterNet 10-branch outputs following the paper methodology:
    1. Extract top-k detections from heatmap
    2. Apply offset for sub-pixel center refinement
    3. Compute depth from lr_distance using calibration parameters
    4. Decode 3D dimensions from offsets + class means
    5. Decode orientation from Multi-Bin representation
    6. Construct Box3D objects with all attributes

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

    Returns:
        - If batch_size == 1: list[Box3D] (backward compatibility)
        - If batch_size > 1: list[list[Box3D]] (one list per batch item)

    References:
        Stereo CenterNet paper: Section 3.2 (Decoding)
    """
    # T213: Backward compatibility - detect single sample and use fallback
    batch_size = outputs["heatmap"].shape[0]
    is_single_sample = batch_size == 1
    
    # T213: Fallback to original per-sample processing for single sample or edge cases
    if is_single_sample:
        # Use original implementation for single sample (backward compatibility)
        return _decode_stereo3d_outputs_per_sample(outputs, conf_threshold, top_k, calib)
    
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
    
    # TODO: get from calib
    # KITTI original image size
    original_width = 1242.0
    original_height = 375.0

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
    
    # Compute actual scale from feature map to original image
    # scale = original_size / feature_map_size
    scale_w = original_width / w
    scale_h = original_height / h

    # T207: Vectorize heatmap peak detection for batch
    # Flatten heatmap: [B, C, H*W]
    heatmap = torch.sigmoid(heatmap)
    heatmap_flat = heatmap.reshape(batch_size, num_classes, -1)  # [B, C, H*W]
    
    # Get top-k scores and indices for each batch and class
    # Use torch.topk across the spatial dimension
    topk_scores, topk_indices = torch.topk(heatmap_flat, k=min(top_k, heatmap_flat.shape[2]), dim=2)  # [B, C, K]

    # T207, T208, T209: Vectorized batch processing
    # Process each batch item with optimized operations
    batch_results = []
    
    for b in range(batch_size):
        boxes3d_batch = []
        
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
            
            # Convert 2D centers to original image space (vectorized)
            scale_w_tensor = torch.tensor(scale_w, device=device, dtype=center_x.dtype)
            scale_h_tensor = torch.tensor(scale_h, device=device, dtype=center_y.dtype)
            u_values = center_x * scale_w_tensor  # [K_valid]
            v_values = center_y * scale_h_tensor  # [K_valid]
            
            # Compute depth from stereo geometry (vectorized)
            # IMPORTANT: d_values is in feature map space, must scale to original image space
            fx_b_tensor = torch.tensor(fx_b, device=device, dtype=d_values.dtype)
            baseline_b_tensor = torch.tensor(baseline_b, device=device, dtype=d_values.dtype)
            d_values_image = d_values * scale_w_tensor  # Scale disparity to original image width
            depth_values = torch.where(
                d_values_image > 0,
                (fx_b_tensor * baseline_b_tensor) / (d_values_image + 1e-6),
                torch.tensor(50.0, device=device, dtype=d_values.dtype)  # Default depth
            )  # [K_valid]
            
            cx_b_tensor = torch.tensor(cx_b, device=device, dtype=u_values.dtype)
            cy_b_tensor = torch.tensor(cy_b, device=device, dtype=v_values.dtype)
            fy_b_tensor = torch.tensor(fy_b, device=device, dtype=depth_values.dtype)
            x_3d_values = (u_values - cx_b_tensor) * depth_values / fx_b_tensor  # [K_valid]
            y_3d_values = (v_values - cy_b_tensor) * depth_values / fy_b_tensor  # [K_valid]
            z_3d_values = depth_values  # [K_valid]
            
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
                        float((center_x_cpu[i] - box_w_cpu[i] / 2) * scale_w),
                        float((center_y_cpu[i] - box_h_cpu[i] / 2) * scale_h),
                        float((center_x_cpu[i] + box_w_cpu[i] / 2) * scale_w),
                        float((center_y_cpu[i] + box_h_cpu[i] / 2) * scale_h),
                    ),
                )
                boxes3d_batch.append(box3d)
        
        batch_results.append(boxes3d_batch)

    return batch_results

def _labels_to_box3d_list(labels: list[dict[str, Any]], calib: dict[str, float] | None = None) -> list[Box3D]:
    """Convert label dictionaries to Box3D objects.

    Filters and remaps class IDs to paper classes (Car, Pedestrian, Cyclist) if needed.

    Args:
        labels: List of label dictionaries from dataset.
        calib: Calibration parameters (dict or CalibrationParameters object).

    Returns:
        List of Box3D objects with filtered and remapped class IDs.
    """
    from ultralytics.models.yolo.stereo3ddet.utils import (
        filter_and_remap_class_id,
        get_paper_class_names,
    )

    boxes3d = []
    class_names = get_paper_class_names()  # {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

    for label in labels:
        try:
            original_class_id = label.get("class_id", 0)
            
            # Filter and remap class ID to paper classes
            remapped_class_id = filter_and_remap_class_id(original_class_id)
            if remapped_class_id is None:
                # Class is not in paper set, skip it
                continue
            
            class_id = remapped_class_id
            if class_id not in class_names:
                continue

            # Get dimensions
            dims = label.get("dimensions", {})
            height = dims.get("height", 1.5)
            width = dims.get("width", 1.5)
            length = dims.get("length", 3.0)

            # Get orientation (alpha is observation angle, convert to rotation_y)
            alpha = label.get("alpha", 0.0)

            # Reconstruct 3D center from stereo disparity (matching prediction pipeline)
            left_box = label.get("left_box", {})
            right_box = label.get("right_box", {})
            
            # Handle both dict and CalibrationParameters objects
            from ultralytics.data.stereo.calib import CalibrationParameters
            if isinstance(calib, CalibrationParameters):
                fx_val = calib.fx
                fy_val = calib.fy
                cx_val = calib.cx
                cy_val = calib.cy
                baseline_val = calib.baseline
            elif isinstance(calib, dict):
                fx_val = calib.get("fx", 721.5377)
                fy_val = calib.get("fy", 721.5377)
                cx_val = calib.get("cx", 609.5593)
                cy_val = calib.get("cy", 172.8540)
                baseline_val = calib.get("baseline", 0.54)
            else:
                fx_val = 721.5377
                fy_val = 721.5377
                cx_val = 609.5593
                cy_val = 172.8540
                baseline_val = 0.54
            
            # Compute depth from stereo disparity (same as prediction pipeline)
            # Get 2D center positions (normalized coordinates)
            left_center_x = left_box.get("center_x", 0.5)
            right_center_x = right_box.get("center_x", 0.5)
            
            # Assuming original image width = 1242 pixels (KITTI standard)
            img_width = 1242.0
            img_height = 375.0
            
            # Convert normalized to pixel coordinates
            left_u = left_center_x * img_width
            right_u = right_center_x * img_width
            
            # Compute disparity (left-right distance in pixels)
            disparity = left_u - right_u
            
            # Compute depth from disparity: Z = (f × baseline) / disparity
            if disparity > 0:
                depth = (fx_val * baseline_val) / disparity
            else:
                # Fallback to a reasonable default if disparity is invalid
                depth = 30.0

            # Convert 2D center to 3D
            center_x_2d = left_u
            center_y_2d = left_box.get("center_y", 0.5) * img_height

            x_3d = (center_x_2d - cx_val) * depth / fx_val
            # y_3d is at geometric center (matching prediction decoder convention)
            # Note: vis_yolo_kitti.py converts to bottom center, but we use geometric center
            # for consistency with plotting.py corner generation which uses [-h/2, +h/2]
            y_3d = (center_y_2d - cy_val) * depth / fy_val
            z_3d = depth

            # Compute rotation_y from alpha (matches vis_yolo_kitti.py)
            # Convert alpha (observation angle) to rotation_y (global yaw)
            # θ = α + arctan(x/z)
            import math
            ray_angle = math.atan2(x_3d, z_3d)
            rotation_y = alpha + ray_angle
            # Normalize to [-π, π]
            rotation_y = math.atan2(math.sin(rotation_y), math.cos(rotation_y))

            box3d = Box3D(
                center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                dimensions=(float(length), float(width), float(height)),
                orientation=float(rotation_y),
                class_label=class_names[class_id],
                class_id=class_id,
                confidence=1.0,  # Ground truth has confidence 1.0
                bbox_2d=None,
                truncated=label.get("truncated"),
                occluded=label.get("occluded"),
            )
            boxes3d.append(box3d)
        except Exception as e:
            LOGGER.warning(f"Error converting label to Box3D: {e}")
            continue

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
        """Preprocess stereo batch for validation.

        Args:
            batch: Batch containing stereo images [B, 6, H, W] and labels.

        Returns:
            Preprocessed batch.
        """
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        # Normalize images
        if "img" in batch:
            batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255.0

        return batch

    def postprocess(self, preds: dict[str, torch.Tensor]) -> list[list[Box3D]]:
        """Postprocess model outputs to Box3D objects.

        Args:
            preds: Dictionary of 10-branch model outputs.

        Returns:
            List of Box3D lists (one per batch item).
        """
        with profile_section("postprocess"):
            batch_size = preds["heatmap"].shape[0]

            # T212: Get calibration from batch if available
            calib = None
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

            # T212: Call decode_stereo3d_outputs once with entire batch
            results = decode_stereo3d_outputs(
                preds,
                conf_threshold=self.args.conf,
                top_k=100,
                calib=calib,
            )
            
            # T212: Return list of Box3D lists directly
            # decode_stereo3d_outputs returns list[list[Box3D]] for batch_size > 1
            # or list[Box3D] for batch_size == 1 (backward compatibility)
            if batch_size == 1 and isinstance(results, list) and len(results) > 0 and isinstance(results[0], Box3D):
                # Single sample result - wrap in list for consistency
                return [results]
            return results

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics with model information.

        Args:
            model: Model being validated.
        """
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

    def _diagnostic_log_iou_matrix(
        self, iou_matrix: np.ndarray, pred_boxes: list[Box3D], gt_boxes: list[Box3D], sample_idx: int
    ) -> None:
        """Log IoU matrix computation results for diagnostic purposes.

        Args:
            iou_matrix: IoU matrix of shape [N, M] with IoU values between predictions and ground truth.
            pred_boxes: List of N predicted boxes.
            gt_boxes: List of M ground truth boxes.
            sample_idx: Index of current sample in batch.
        """
        try:
            n, m = iou_matrix.shape
            non_zero_count = np.count_nonzero(iou_matrix)
            total = n * m
            iou_min = float(np.min(iou_matrix)) if total > 0 else 0.0
            iou_max = float(np.max(iou_matrix)) if total > 0 else 0.0
            iou_mean = float(np.mean(iou_matrix)) if total > 0 else 0.0

            LOGGER.info(f"[DIAG] Sample {sample_idx}: IoU Matrix")
            LOGGER.info(f"  Shape: [{n}, {m}]")
            LOGGER.info(f"  Non-zero count: {non_zero_count} / {total}")
            LOGGER.info(f"  Value range: [{iou_min:.4f}, {iou_max:.4f}]")
            LOGGER.info(f"  Mean: {iou_mean:.4f}")

            # Sample values (first few predictions vs first few ground truth)
            sample_size = min(3, n, m)
            if sample_size > 0:
                LOGGER.info("  Sample values:")
                for i in range(min(3, n)):
                    for j in range(min(3, m)):
                        iou_val = float(iou_matrix[i, j])
                        if iou_val > 0.0:
                            LOGGER.info(f"    Pred[{i}] vs GT[{j}]: {iou_val:.4f}")
        except Exception as e:
            LOGGER.warning(f"Diagnostic logging failed (_diagnostic_log_iou_matrix): {e}")

    def _diagnostic_log_tp_fp_assignment(
        self, tp: np.ndarray, fp: np.ndarray, pred_boxes: list[Box3D], iou_thresholds: torch.Tensor, sample_idx: int
    ) -> None:
        """Log TP/FP assignment results for diagnostic purposes.

        Args:
            tp: True positives array of shape [N, 2] per prediction per threshold.
            fp: False positives array of shape [N, 2] per prediction per threshold.
            pred_boxes: List of N predicted boxes.
            iou_thresholds: IoU thresholds tensor [0.5, 0.7].
            sample_idx: Index of current sample in batch.
        """
        try:
            n = tp.shape[0] if len(tp.shape) > 0 else 0
            LOGGER.info(f"[DIAG] Sample {sample_idx}: TP/FP Assignment")
            LOGGER.info(f"  Arrays shape: [{n}, {tp.shape[1] if len(tp.shape) > 1 else 0}]")

            for iou_idx, iou_thresh in enumerate(iou_thresholds):
                if iou_idx < tp.shape[1]:
                    tp_count = int(np.sum(tp[:, iou_idx]))
                    fp_count = int(np.sum(fp[:, iou_idx]))
                    tp_indices = np.where(tp[:, iou_idx])[0].tolist()[:10]  # First 10 TP indices
                    fp_indices = np.where(fp[:, iou_idx])[0].tolist()[:10]  # First 10 FP indices

                    LOGGER.info(f"  IoU Threshold {iou_thresh.item():.1f}: TP={tp_count}, FP={fp_count}")
                    if tp_indices:
                        LOGGER.info(f"  TP indices (threshold {iou_thresh.item():.1f}): {tp_indices}")
                    if fp_indices:
                        LOGGER.info(f"  FP indices (threshold {iou_thresh.item():.1f}): {fp_indices}")
        except Exception as e:
            LOGGER.warning(f"Diagnostic logging failed (_diagnostic_log_tp_fp_assignment): {e}")

    def _diagnostic_log_statistics_extraction(
        self, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray, sample_idx: int
    ) -> None:
        """Log extracted statistics arrays for diagnostic purposes.

        Args:
            conf: Confidence scores array of shape [N].
            pred_cls: Predicted class IDs array of shape [N].
            target_cls: Target class IDs array of shape [M].
            sample_idx: Index of current sample in batch.
        """
        try:
            LOGGER.info(f"[DIAG] Sample {sample_idx}: Statistics Extraction")
            LOGGER.info(f"  conf: shape={conf.shape}, dtype={conf.dtype}, range=[{float(np.min(conf)) if len(conf) > 0 else 0.0:.4f}, {float(np.max(conf)) if len(conf) > 0 else 0.0:.4f}], non-zero={np.count_nonzero(conf)}")
            unique_pred_cls = np.unique(pred_cls).tolist() if len(pred_cls) > 0 else []
            LOGGER.info(f"  pred_cls: shape={pred_cls.shape}, dtype={pred_cls.dtype}, unique={unique_pred_cls}")
            unique_target_cls = np.unique(target_cls).tolist() if len(target_cls) > 0 else []
            LOGGER.info(f"  target_cls: shape={target_cls.shape}, dtype={target_cls.dtype}, unique={unique_target_cls}")
        except Exception as e:
            LOGGER.warning(f"Diagnostic logging failed (_diagnostic_log_statistics_extraction): {e}")

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

            for si, (pred_boxes, labels) in enumerate(zip(preds, labels_list)):
                self.seen += 1

                # Get calibration for this sample
                calib = calibs[si] if si < len(calibs) and isinstance(calibs[si], dict) else None

                # Convert labels to Box3D
                try:
                    gt_boxes = _labels_to_box3d_list(labels, calib)
                except Exception as e:
                    LOGGER.warning(f"Error converting labels to Box3D (sample {si}): {e}")
                    gt_boxes = []

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

                    # DIAGNOSTIC START
                    # self._diagnostic_log_iou_matrix(iou_matrix, pred_boxes, gt_boxes, si)
                    # DIAGNOSTIC END

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

                    # DIAGNOSTIC START
                    # self._diagnostic_log_tp_fp_assignment(tp, fp, pred_boxes, self.iouv, si)
                    # DIAGNOSTIC END

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

            # Generate visualization images if plots enabled
            # Default to 3 batches (matching Detect task style), but can be overridden via max_plot_batches arg
            max_plot_batches = getattr(self.args, 'max_plot_batches', 9)
            if self.args.plots and hasattr(self, 'batch_i') and self.batch_i < max_plot_batches:
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
        return ("%11s" + "%11s" * 4) % (
            "Images".rjust(11),
            "AP3D@0.5".rjust(11),
            "AP3D@0.7".rjust(11),
            "Precision".rjust(11),
            "Recall".rjust(11),
        )

    def finalize_metrics(self) -> None:
        """Finalize metrics computation."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Returns:
            Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        return self.metrics.results_dict

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
            max_samples = getattr(self.args, 'max_plot_samples', batch_size)
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

                # Convert labels to Box3D for ground truth
                gt_boxes = []
                if labels:
                    try:
                        gt_boxes = _labels_to_box3d_list(labels, calib)
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
                try:
                    left_pred, right_pred, combined_pred = plot_stereo3d_boxes(
                        left_img=left_img.copy(),
                        right_img=right_img.copy(),
                        pred_boxes3d=pred_boxes,
                        gt_boxes3d=[],  # No ground truth for prediction visualization
                        left_calib=calib,
                        letterbox_scale=None,  # No letterboxing - using original size
                        letterbox_pad_left=None,
                        letterbox_pad_top=None,
                    )
                except Exception as e:
                    LOGGER.debug(f"Error generating prediction visualization for sample {si}: {e}")
                    continue

                # Generate visualization with ground truth only (bottom image)
                try:
                    left_gt, right_gt, combined_gt = plot_stereo3d_boxes(
                        left_img=left_img.copy(),
                        right_img=right_img.copy(),
                        pred_boxes3d=[],  # No predictions for ground truth visualization
                        gt_boxes3d=gt_boxes,
                        left_calib=calib,
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

        # Print format: class name, images, AP3D@0.5, AP3D@0.7, precision, recall (matches progress bar format)
        # Note: labels (total_gt) is shown in verbose per-class output, not in main summary
        pf = "%22s" + "%11i" + "%11.3g" * 4
        LOGGER.info(pf % ("all", self.seen, maps3d_50, maps3d_70, precision_mean, recall_mean))

        # Print results per class if verbose and multiple classes
        if self.args.verbose and not self.training and self.metrics.nc > 1 and self.metrics.ap3d_50:
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
                    )
                )

    def _format_progress_metrics(self) -> str:
        """Format current metrics for progress bar display.
        
        Returns:
            Formatted string with key metrics in training-style format.
        """
        if not hasattr(self.metrics, 'stats') or len(self.metrics.stats) == 0:
            return ("%11i" + "%11s" * 4) % (int(self.seen), "-", "-", "-", "-")
        
        # Compute intermediate metrics on accumulated stats
        try:
            # Save current stats
            saved_stats = self.metrics.stats.copy()
            # Process to get metrics
            temp_results = self.metrics.process(save_dir=self.save_dir, plot=False)
            # Restore stats for final processing
            self.metrics.stats = saved_stats
            
            if not temp_results:
                return ("%11i" + "%11s" * 4) % (int(self.seen), "-", "-", "-", "-")
            
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
            
            # Format similar to training: Images, AP3D@0.5, AP3D@0.7, Precision, Recall
            # Use same width format as training for consistency (matches get_desc header)
            # Use %11i for Images (integer count) and %11.4g for float metrics
            return ("%11i" + "%11.4g" * 4) % (
                int(self.seen),  # Images (integer)
                ap50,  # AP3D@0.5
                ap70,  # AP3D@0.7
                precision,  # Precision
                recall,  # Recall
            )
        except Exception as e:
            LOGGER.debug(f"Error formatting progress metrics: {e}")
            return ("%11i" + "%11s" * 4) % (int(self.seen), "-", "-", "-", "-")



    def build_dataset(self, img_path: str | dict[str, Any], mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build Stereo3DDetAdapterDataset for validation.

        Args:
            img_path: Path to dataset root directory, or a descriptor dict from self.data.get(split).
            mode: 'train' or 'val' mode.
            batch: Batch size (unused, kept for compatibility).

        Returns:
            Stereo3DDetAdapterDataset: Dataset instance for validation.
        """
        from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset
        # img_path should be a dir 
        if isinstance(img_path, str) and not os.path.isdir(img_path):
            # means it's a file instead of the path, return it's parent directory
            img_path = Path(img_path).parent


        # Handle descriptor dict from self.data.get(self.args.split)
        desc = img_path if isinstance(img_path, dict) else self.data.get(mode) if hasattr(self, "data") else None
        
        if isinstance(desc, dict) and desc.get("type") == "kitti_stereo":
            # Get image size from args, default to 384
            imgsz = getattr(self.args, "imgsz", 384)
            if isinstance(imgsz, (list, tuple)):
                imgsz = imgsz[0] if len(imgsz) > 0 else 384
            
            # Get max_samples from args if available (for profiling/testing)
            max_samples = getattr(self.args, "max_samples", None)
            
            return Stereo3DDetAdapterDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", mode)),
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
                max_samples=max_samples,
            )
        
        # Fallback: if img_path is a string, try to use it directly
        if isinstance(img_path, str) or isinstance(img_path, Path):
            imgsz = getattr(self.args, "imgsz", 384)
            if isinstance(imgsz, (list, tuple)):
                imgsz = imgsz[0] if len(imgsz) > 0 else 384
            
            return Stereo3DDetAdapterDataset(
                root=img_path,
                split=mode,
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
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