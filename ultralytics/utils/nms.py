# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.utils.metrics import batch_probiou, box_iou


class FastNMS:
    """
    Ultralytics custom NMS implementation optimized for YOLO.

    This class provides static methods for performing non-maximum suppression (NMS) operations on bounding boxes,
    including both standard NMS and batched NMS for multi-class scenarios.

    Methods:
        nms: Optimized NMS with early termination that matches torchvision behavior exactly.
        batched_nms: Batched NMS for class-aware suppression.

    Examples:
        Perform standard NMS on boxes and scores
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = FastNMS.nms(boxes, scores, 0.5)
    """

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, use_triu: bool = True) -> torch.Tensor:
        """
        Optimized NMS with early termination that matches torchvision behavior exactly.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = FastNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = box_iou(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            # Zeroing these scores ensures the additional indices would not affect the final results
            scores[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            # NOTE: return indices with fixed length to avoid TFLite reshape error
            pick = torch.topk(scores, scores.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def batched_nms(
        boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
    ) -> torch.Tensor:
        """
        Batched NMS for class-aware suppression.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            idxs (torch.Tensor): Class indices with shape (N,).
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply batched NMS across multiple classes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> idxs = torch.tensor([0, 1])
            >>> keep = FastNMS.batched_nms(boxes, scores, idxs, 0.5)
        """
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

    Examples:
        Apply rotated NMS to oriented boxes
        >>> boxes = torch.tensor([[100, 100, 50, 30, 0.5], [120, 120, 40, 25, 0.3]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = nms_rotated(boxes, scores, 0.45)
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
