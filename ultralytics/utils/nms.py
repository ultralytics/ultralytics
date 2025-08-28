# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.utils.metrics import batch_probiou


class FastNMS:
    """Ultralytics custom NMS implementation optimized for YOLO11."""
    
    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Fast vectorized NMS implementation using pure PyTorch."""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        
        # Sort by scores in descending order
        sorted_indices = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        
        # Compute all IoUs at once using vectorized operations
        x1 = boxes[:, 0]
        y1 = boxes[:, 1] 
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        
        # Expand dimensions for broadcasting
        x1_exp = x1.unsqueeze(1)
        y1_exp = y1.unsqueeze(1)
        x2_exp = x2.unsqueeze(1)
        y2_exp = y2.unsqueeze(1)
        areas_exp = areas.unsqueeze(1)
        
        # Compute intersection coordinates
        xx1 = torch.maximum(x1_exp, x1)
        yy1 = torch.maximum(y1_exp, y1)
        xx2 = torch.minimum(x2_exp, x2)
        yy2 = torch.minimum(y2_exp, y2)
        
        # Compute intersection area
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        intersection = w * h
        
        # Compute IoU
        union = areas_exp + areas - intersection
        iou = intersection / (union + 1e-7)
        
        # Use upper triangular matrix to avoid self-comparison
        triu_mask = torch.triu(torch.ones_like(iou, dtype=torch.bool), diagonal=1)
        iou = iou * triu_mask
        
        # Find boxes to suppress
        suppress_mask = (iou > iou_threshold).any(dim=1)
        keep_indices = torch.nonzero(~suppress_mask, as_tuple=False).squeeze(1)
        
        return sorted_indices[keep_indices]

    @staticmethod
    def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Batched NMS for class-aware suppression."""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        
        # Strategy: offset boxes by class index to prevent cross-class suppression
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        
        return FastNMS.nms(boxes_for_nms, scores, iou_threshold)


def nms_rotated(boxes, scores, threshold: float = 0.45, use_triu: bool = True):
    """
    Perform NMS on oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (torch.Tensor): Rotated bounding boxes with shape (N, 5) in xywhr format.
        scores (torch.Tensor): Confidence scores with shape (N,).
        threshold (float): IoU threshold for NMS.
        use_triu (bool): Whether to use torch.triu operator for upper triangular matrix operations.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    if use_triu:
        ious = ious.triu_(diagonal=1)
        # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
        pick = torch.nonzero((ious >= threshold).sum(0) <= 0).squeeze_(-1)
    else:
        n = boxes.shape[0]
        row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
        col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
        upper_mask = row_idx < col_idx
        ious = ious * upper_mask
        # Zeroing these scores ensures the additional indices would not affect the final results
        scores[~((ious >= threshold).sum(0) <= 0)] = 0
        # NOTE: return indices with fixed length to avoid TFLite reshape error
        pick = torch.topk(scores, scores.shape[0]).indices
    return sorted_idx[pick]
