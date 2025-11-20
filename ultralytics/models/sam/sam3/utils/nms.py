# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging

import numpy as np
import torch


try:
    from torch_generic_nms import generic_nms as generic_nms_cuda

    GENERIC_NMS_AVAILABLE = True
except ImportError:
    logging.debug(
        "Falling back to triton or CPU mask NMS implementation -- please install `torch_generic_nms` via\n\t"
        'pip uninstall -y torch_generic_nms; TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/ronghanghu/torch_generic_nms'
    )
    GENERIC_NMS_AVAILABLE = False


def nms_masks(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Args:
      - pred_probs: (num_det,) float Tensor, containing the score (probability) of each detection
      - pred_masks: (num_det, H_mask, W_mask) float Tensor, containing the binary segmentation mask of each detection
      - prob_threshold: float, score threshold to prefilter detections (NMS is performed on detections above threshold)
      - iou_threshold: float, mask IoU threshold for NMS

    Returns:
     - keep: (num_det,) bool Tensor, indicating whether each detection is kept after score thresholding + NMS
    """
    # prefilter the detections with prob_threshold ("valid" are those above prob_threshold)
    is_valid = pred_probs > prob_threshold  # (num_det,)
    probs = pred_probs[is_valid]  # (num_valid,)
    masks_binary = pred_masks[is_valid] > 0  # (num_valid, H_mask, W_mask)
    if probs.numel() == 0:
        return is_valid  # no valid detection, return empty keep mask

    ious = mask_iou(masks_binary, masks_binary)  # (num_valid, num_valid)
    kept_inds = generic_nms(ious, probs, iou_threshold)

    # valid_inds are the indices among `probs` of valid detections before NMS (or -1 for invalid)
    valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1, -1)  # (num_det,)
    keep = torch.isin(valid_inds, kept_inds)  # (num_det,)
    return keep


def generic_nms(ious: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5) -> torch.Tensor:
    """A generic version of `torchvision.ops.nms` that takes a pairwise IoU matrix."""

    assert ious.dim() == 2 and ious.size(0) == ious.size(1)
    assert scores.dim() == 1 and scores.size(0) == ious.size(0)

    if ious.is_cuda:
        if GENERIC_NMS_AVAILABLE:
            return generic_nms_cuda(ious, scores, iou_threshold, use_iou_matrix=True)
        else:
            from sam3.perflib.triton.nms import nms_triton

            return nms_triton(ious, scores, iou_threshold)

    return generic_nms_cpu(ious, scores, iou_threshold)


def generic_nms_cpu(ious: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5) -> torch.Tensor:
    """
    A generic version of `torchvision.ops.nms` that takes a pairwise IoU matrix. (CPU implementation
    based on https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/nms/nms_cpu.py)
    """
    ious_np = ious.float().detach().cpu().numpy()
    scores_np = scores.float().detach().cpu().numpy()
    order = scores_np.argsort()[::-1]
    kept_inds = []
    while order.size > 0:
        i = order.item(0)
        kept_inds.append(i)
        inds = np.where(ious_np[i, order[1:]] <= iou_threshold)[0]
        order = order[inds + 1]

    return torch.tensor(kept_inds, dtype=torch.int64, device=scores.device)


def masks_to_boxes(masks: torch.Tensor, obj_ids: list[int]):
    with torch.autograd.profiler.record_function("perflib: masks_to_boxes"):
        # Sanity check based on callsite for replacement
        assert masks.shape[0] == len(obj_ids)
        assert masks.dim() == 3

        # Based on torchvision masks_to_boxes
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        N, H, W = masks.shape
        device = masks.device
        y = torch.arange(H, device=device).view(1, H)
        x = torch.arange(W, device=device).view(1, W)

        masks_with_obj = masks != 0  # N, H, W
        masks_with_obj_x = masks_with_obj.amax(dim=1)  # N, H (which columns have objects)
        masks_with_obj_y = masks_with_obj.amax(dim=2)  # N, W (which rows have objects)
        masks_without_obj_x = ~masks_with_obj_x
        masks_without_obj_y = ~masks_with_obj_y

        bounding_boxes_0 = torch.amin((masks_without_obj_x * W) + (masks_with_obj_x * x), dim=1)
        bounding_boxes_1 = torch.amin((masks_without_obj_y * H) + (masks_with_obj_y * y), dim=1)
        bounding_boxes_2 = torch.amax(masks_with_obj_x * x, dim=1)
        bounding_boxes_3 = torch.amax(masks_with_obj_y * y, dim=1)

        bounding_boxes = torch.stack(
            [bounding_boxes_0, bounding_boxes_1, bounding_boxes_2, bounding_boxes_3],
            dim=1,
        ).to(dtype=torch.float)
        assert bounding_boxes.shape == (N, 4)
        assert bounding_boxes.device == masks.device
        assert bounding_boxes.dtype == torch.float
        return bounding_boxes


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU (Intersection over Union) between predicted masks and ground truth masks.
    Args:
      - pred_masks: (N, H, W) bool Tensor, containing binary predicted segmentation masks
      - gt_masks: (M, H, W) bool Tensor, containing binary ground truth segmentation masks
    Returns:
      - ious: (N, M) float Tensor, containing IoUs for each pair of predicted and ground truth masks
    """
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape

    # Flatten masks: (N, 1, H*W) and (1, M, H*W)
    pred_flat = pred_masks.view(N, 1, H * W)
    gt_flat = gt_masks.view(1, M, H * W)

    # Compute intersection and union: (N, M)
    intersection = (pred_flat & gt_flat).sum(dim=2).float()
    union = (pred_flat | gt_flat).sum(dim=2).float()
    ious = intersection / union.clamp(min=1)
    return ious  # shape: (N, M)
