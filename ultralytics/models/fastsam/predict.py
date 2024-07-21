# Ultralytics YOLO 🚀, AGPL-3.0 license
from typing import List

import torch
from ultralytics.models.yolo.segment import SegmentationPredictor

from ultralytics.models.fastsam.utils import bbox_iou


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks in Ultralytics
    YOLO framework.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM.
    It adjusts post-processing steps to incorporate mask prediction and non-max suppression while optimizing
    for single-class segmentation.
    """

    def _process_full_box(self, p: List[torch.Tensor], img: torch.Tensor) -> List[torch.Tensor]:
        """
        Process the full box and update the predictions based on critical IoU index.

        Args:
            p (List[torch.Tensor]): The predictions after non-max suppression.
            img (torch.Tensor): The processed image tensor.

        Returns:
            List[torch.Tensor]: Updated predictions.
        """
        full_box = torch.zeros(p[0].shape[1], device=p[0].device)
        full_box[2] = img.shape[3]  # Image width
        full_box[3] = img.shape[2]  # Image height
        full_box[4] = 1.0  # Confidence score
        full_box[6:] = 1.0  # Additional attributes
        full_box = full_box.view(1, -1)

        critical_iou_index = bbox_iou(full_box[0][:4], p[0][:, :4], iou_thres=0.9, image_shape=img.shape[2:])
        if critical_iou_index.numel() != 0:
            full_box[0][4] = p[0][critical_iou_index][:, 4]
            full_box[0][6:] = p[0][critical_iou_index][:, 6:]
            p[0][critical_iou_index] = full_box

        return p
