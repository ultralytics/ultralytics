# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Box operation utilities for DETR-family detection models."""

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format, clamping w/h to >= 0."""
    x_c, y_c, w, h = boxes.unbind(-1)
    w = w.clamp(min=0.0)
    h = h.clamp(min=0.0)
    return torch.stack((x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h), dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)), dim=-1)


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise IoU between all pairs of xyxy boxes, returning (iou, union) of shape (N, M)."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def pairwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Generalized IoU between all pairs of xyxy boxes, returning matrix of shape (N, M)."""
    iou, union = pairwise_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area


def _upcast_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Upcast low-precision floating boxes before area multiplications."""
    return boxes.float() if boxes.is_floating_point() and boxes.dtype not in (torch.float32, torch.float64) else boxes


def _aligned_inter_union(
    boxes1: torch.Tensor, boxes2: torch.Tensor, xywh: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute aligned intersection and union, returning upcast boxes for follow-up geometry."""
    boxes1 = _upcast_boxes(boxes1)
    boxes2 = _upcast_boxes(boxes2)
    if xywh:
        boxes1 = box_cxcywh_to_xyxy(boxes1)
        boxes2 = box_cxcywh_to_xyxy(boxes2)

    inter_lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    inter_rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    union = box_area(boxes1) + box_area(boxes2) - inter
    return inter, union, boxes1, boxes2


def aligned_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7, xywh: bool = False
) -> torch.Tensor:
    """Compute element-wise IoU for N matched box pairs, returning vector of shape (N,)."""
    inter, union, _, _ = _aligned_inter_union(boxes1, boxes2, xywh=xywh)
    return inter / union.clamp(min=eps)


def aligned_giou(boxes1: torch.Tensor, boxes2: torch.Tensor, xywh: bool = False) -> torch.Tensor:
    """Compute element-wise Generalized IoU for matched box pairs."""
    if xywh:
        boxes1 = box_cxcywh_to_xyxy(boxes1)
        boxes2 = box_cxcywh_to_xyxy(boxes2)

    inter_lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    inter_rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    union = box_area(boxes1) + box_area(boxes2) - inter
    iou = inter / union

    cover_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    cover_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    cover_wh = (cover_rb - cover_lt).clamp(min=0)
    cover_area = cover_wh[:, 0] * cover_wh[:, 1]
    return iou - (cover_area - union) / cover_area
