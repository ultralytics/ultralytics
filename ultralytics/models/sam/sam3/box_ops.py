# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""

from typing import Tuple

import torch


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [(x), (y), (x + w), (y + h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_cxcywh(x):
    x, y, w, h = x.unbind(-1)
    b = [(x + 0.5 * w), (y + 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x, y, X, Y = x.unbind(-1)
    b = [(x), (y), (X - x), (Y - y)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Batched version of box area. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes: Tensor of shape (..., 4)

    Returns:
    - areas: Tensor of shape (...,)
    """
    x0, y0, x1, y1 = boxes.unbind(-1)
    return (x1 - x0) * (y1 - y0)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0] + 1
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0] + 1
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    boxes = torch.stack([x_min, y_min, x_max, y_max], 1)
    # Invalidate boxes corresponding to empty masks.
    boxes = boxes * masks.flatten(-2).any(-1)
    return boxes


def box_iou(boxes1, boxes2):
    """
    Batched version of box_iou. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes1: Tensor of shape (..., N, 4)
    - boxes2: Tensor of shape (..., M, 4)

    Returns:
    - iou, union: Tensors of shape (..., N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # boxes1: (..., N, 4) -> (..., N, 1, 2)
    # boxes2: (..., M, 4) -> (..., 1, M, 2)
    lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

    wh = (rb - lt).clamp(min=0)  # (..., N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (..., N, M)

    union = area1[..., None] + area2[..., None, :] - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Batched version of Generalized IoU from https://giou.stanford.edu/

    Boxes should be in [x0, y0, x1, y1] format

    Inputs:
    - boxes1: Tensor of shape (..., N, 4)
    - boxes2: Tensor of shape (..., M, 4)

    Returns:
    - giou: Tensor of shape (..., N, M)
    """
    iou, union = box_iou(boxes1, boxes2)

    # boxes1: (..., N, 4) -> (..., N, 1, 2)
    # boxes2: (..., M, 4) -> (..., 1, M, 2)
    lt = torch.min(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.max(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

    wh = (rb - lt).clamp(min=0)  # (..., N, M, 2)
    area = wh[..., 0] * wh[..., 1]  # (..., N, M)

    return iou - (area - union) / area


@torch.jit.script
def fast_diag_generalized_box_iou(boxes1, boxes2):
    assert len(boxes1) == len(boxes2)
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]
    # assert (box1_xy >= box1_XY).all()
    # assert (box2_xy >= box2_XY).all()
    area1 = (box1_xy - box1_XY).prod(-1)
    area2 = (box2_xy - box2_XY).prod(-1)

    lt = torch.max(box1_XY, box2_XY)  # [N,2]
    lt2 = torch.min(box1_XY, box2_XY)
    rb = torch.min(box1_xy, box2_xy)  # [N,2]
    rb2 = torch.max(box1_xy, box2_xy)

    inter = (rb - lt).clamp(min=0).prod(-1)
    tot_area = (rb2 - lt2).clamp(min=0).prod(-1)

    union = area1 + area2 - inter

    iou = inter / union

    return iou - (tot_area - union) / tot_area


@torch.jit.script
def fast_diag_box_iou(boxes1, boxes2):
    assert len(boxes1) == len(boxes2)
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]
    # assert (box1_xy >= box1_XY).all()
    # assert (box2_xy >= box2_XY).all()
    area1 = (box1_xy - box1_XY).prod(-1)
    area2 = (box2_xy - box2_XY).prod(-1)

    lt = torch.max(box1_XY, box2_XY)  # [N,2]
    rb = torch.min(box1_xy, box2_xy)  # [N,2]

    inter = (rb - lt).clamp(min=0).prod(-1)

    union = area1 + area2 - inter

    iou = inter / union

    return iou


def box_xywh_inter_union(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Asuumes boxes in xywh format
    assert boxes1.size(-1) == 4 and boxes2.size(-1) == 4
    boxes1 = box_xywh_to_xyxy(boxes1)
    boxes2 = box_xywh_to_xyxy(boxes2)
    box1_tl_xy = boxes1[..., :2]
    box1_br_xy = boxes1[..., 2:]
    box2_tl_xy = boxes2[..., :2]
    box2_br_xy = boxes2[..., 2:]
    area1 = (box1_br_xy - box1_tl_xy).prod(-1)
    area2 = (box2_br_xy - box2_tl_xy).prod(-1)

    assert (area1 >= 0).all() and (area2 >= 0).all()
    tl = torch.max(box1_tl_xy, box2_tl_xy)
    br = torch.min(box1_br_xy, box2_br_xy)

    inter = (br - tl).clamp(min=0).prod(-1)
    union = area1 + area2 - inter

    return inter, union
