# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MDE (Monocular Depth Estimation) validation module.

This module provides validation functionality specifically for MDE models,
handling the unique output format with depth estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DetMetrics
# from ultralytics.utils.torch_utils import de_parallel  # Not used
from ultralytics.utils.nms import non_max_suppression


class MDEValidator(BaseValidator):
    """
    MDE Validator for YOLO models with depth estimation.
    
    This validator handles the unique output format of MDE models that includes
    bounding boxes, class probabilities, and depth values.
    
    Attributes:
        model: The MDE model to validate.
        dataloader: Validation data loader.
        device: Device to run validation on.
        metrics: Detection metrics.
        
    Methods:
        __call__: Run validation on the model.
        postprocess: Post-process MDE model predictions.
    """
    
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """
        Initialize MDE validator with necessary variables and settings.
        
        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (SimpleNamespace, optional): Arguments for validation.
            _callbacks (dict, optional): Callbacks for validation.
        """
        # Initialize base validator
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = 'detect'  # Ensure task is set for detection metrics
        self.metrics = DetMetrics()
        
    def __call__(self):
        """Run validation on the model."""
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                # Move batch to device
                if isinstance(batch, dict):
                    img = batch['img'].to(self.device)
                    targets = batch.get('targets', None)
                else:
                    img = batch[0].to(self.device)
                    targets = batch[1] if len(batch) > 1 else None
                
                # Forward pass
                preds = self.model(img)
                
                # Post-process predictions
                preds = self.postprocess(preds)
                
                # Update metrics
                if targets is not None:
                    self.metrics.update(preds, targets)
        
        return self.metrics.results_dict
    
    def postprocess(self, preds):
        """
        Post-process MDE model predictions.
        
        Args:
            preds: Raw model predictions.
            
        Returns:
            Post-processed predictions with NMS applied.
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]  # Take first element if tuple/list
        
        # Handle MDE model outputs (70 features: 4 box + 5 cls + 64 dfl + 1 depth)
        if preds.shape[-1] == 70:  # MDE format
            # For MDE models, we need to handle the 70-feature output
            # Split into components: [4, 5, 64, 1] = [box, cls, dfl, depth]
            bs = preds.shape[0]
            outputs = []
            
            for i in range(bs):
                pred = preds[i]  # [num_anchors, 70]
                
                # Split into components
                box_pred = pred[:, :4]  # [num_anchors, 4] - bounding box coordinates
                cls_pred = pred[:, 4:9]  # [num_anchors, 5] - class probabilities
                dfl_pred = pred[:, 9:73]  # [num_anchors, 64] - DFL regression
                depth_pred = pred[:, 73:74]  # [num_anchors, 1] - depth values
                
                # Apply NMS with MDE-specific handling
                # We need to create a standard format for NMS: [num_anchors, 6] = [x1, y1, x2, y2, conf, cls]
                
                # Get max class confidence
                max_conf, max_cls = torch.max(cls_pred, dim=1)
                
                # Combine box coordinates with confidence and class
                # For now, use the raw box coordinates (they should be processed by DFL in training)
                detections = torch.cat([
                    box_pred,  # [num_anchors, 4] - x1, y1, x2, y2
                    max_conf.unsqueeze(1),  # [num_anchors, 1] - confidence
                    max_cls.unsqueeze(1).float(),  # [num_anchors, 1] - class
                    depth_pred  # [num_anchors, 1] - depth
                ], dim=1)  # [num_anchors, 7] = [x1, y1, x2, y2, conf, cls, depth]
                
                # Apply NMS (we'll use a simplified version for now)
                # Filter by confidence threshold
                conf_thres = 0.25
                keep = max_conf > conf_thres
                detections = detections[keep]
                
                if len(detections) > 0:
                    # Simple NMS implementation for MDE
                    detections = self._simple_nms(detections)
                
                outputs.append(detections)
            
            return outputs
        else:
            # Standard YOLO format - use regular NMS
            return non_max_suppression(preds)
    
    def _simple_nms(self, detections, iou_threshold=0.5):
        """
        Simple NMS implementation for MDE detections.
        
        Args:
            detections: Detection tensor [N, 7] = [x1, y1, x2, y2, conf, cls, depth]
            iou_threshold: IoU threshold for NMS.
            
        Returns:
            Filtered detections after NMS.
        """
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        _, indices = torch.sort(detections[:, 4], descending=True)
        detections = detections[indices]
        
        keep = []
        while len(detections) > 0:
            # Take the detection with highest confidence
            keep.append(detections[0])
            
            if len(detections) == 1:
                break
            
            # Calculate IoU with remaining detections
            ious = self._calculate_iou(detections[0:1, :4], detections[1:, :4])
            
            # Keep detections with IoU below threshold
            keep_mask = ious < iou_threshold
            detections = detections[1:][keep_mask]
        
        if keep:
            return torch.stack(keep)
        else:
            return torch.empty(0, 7, device=detections.device)
    
    def _calculate_iou(self, box1, boxes):
        """
        Calculate IoU between one box and multiple boxes.
        
        Args:
            box1: Single box [1, 4] = [x1, y1, x2, y2]
            boxes: Multiple boxes [N, 4] = [x1, y1, x2, y2]
            
        Returns:
            IoU values [N]
        """
        # Calculate intersection
        x1 = torch.max(box1[0, 0], boxes[:, 0])
        y1 = torch.max(box1[0, 1], boxes[:, 1])
        x2 = torch.min(box1[0, 2], boxes[:, 2])
        y2 = torch.min(box1[0, 3], boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        area1 = (box1[0, 2] - box1[0, 0]) * (box1[0, 3] - box1[0, 1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Calculate union
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        return iou
