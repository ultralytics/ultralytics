# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
MDE (Monocular Depth Estimation) Metrics

This module provides metrics for evaluating depth estimation performance
in YOLO models with depth estimation capabilities.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any


def calculate_depth_error(pred_depths: torch.Tensor, gt_depths: torch.Tensor) -> torch.Tensor:
    """
    Calculate depth error rate (Equation 8 from paper).
    
    Args:
        pred_depths: Predicted depth values [N]
        gt_depths: Ground truth depth values [N]
        
    Returns:
        torch.Tensor: Depth error rate as percentage
    """
    # Avoid division by zero
    mask = gt_depths > 1e-6
    if not mask.any():
        return torch.tensor(0.0, device=pred_depths.device)
    
    pred_depths = pred_depths[mask]
    gt_depths = gt_depths[mask]
    
    errors = torch.abs(gt_depths - pred_depths) / gt_depths
    return errors.mean() * 100  # Percentage


def calculate_absolute_depth_error(pred_depths: torch.Tensor, gt_depths: torch.Tensor) -> torch.Tensor:
    """
    Calculate absolute depth error in meters.
    
    Args:
        pred_depths: Predicted depth values [N]
        gt_depths: Ground truth depth values [N]
        
    Returns:
        torch.Tensor: Mean absolute depth error in meters
    """
    return torch.abs(pred_depths - gt_depths).mean()


def calculate_squared_depth_error(pred_depths: torch.Tensor, gt_depths: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared depth error (RMSE).
    
    Args:
        pred_depths: Predicted depth values [N]
        gt_depths: Ground truth depth values [N]
        
    Returns:
        torch.Tensor: Root mean squared depth error in meters
    """
    return torch.sqrt(torch.mean((pred_depths - gt_depths) ** 2))


def calculate_depth_accuracy(pred_depths: torch.Tensor, gt_depths: torch.Tensor, 
                           thresholds: List[float] = [1.25, 1.25**2, 1.25**3]) -> Dict[str, float]:
    """
    Calculate depth accuracy metrics (δ < threshold).
    
    Args:
        pred_depths: Predicted depth values [N]
        gt_depths: Ground truth depth values [N]
        thresholds: List of accuracy thresholds
        
    Returns:
        Dict[str, float]: Accuracy metrics for each threshold
    """
    # Avoid division by zero
    mask = gt_depths > 1e-6
    if not mask.any():
        return {f"δ < {th:.3f}": 0.0 for th in thresholds}
    
    pred_depths = pred_depths[mask]
    gt_depths = gt_depths[mask]
    
    # Calculate ratios
    ratio1 = pred_depths / gt_depths
    ratio2 = gt_depths / pred_depths
    ratio = torch.max(ratio1, ratio2)
    
    accuracies = {}
    for th in thresholds:
        acc = (ratio < th).float().mean().item()
        accuracies[f"δ < {th:.3f}"] = acc * 100  # Percentage
    
    return accuracies


def match_predictions(preds: List[torch.Tensor], gt_labels: List[Dict], 
                     iou_threshold: float = 0.5) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Match predictions with ground truth labels for depth evaluation.
    
    Args:
        preds: List of prediction tensors [batch_size, num_detections, 6] (x1,y1,x2,y2,conf,depth)
        gt_labels: List of ground truth labels for each image
        iou_threshold: IoU threshold for matching
        
    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: List of (pred_depth, gt_depth) pairs
    """
    matched_pairs = []
    
    for pred, gt_label in zip(preds, gt_labels):
        if len(pred) == 0 or len(gt_label) == 0:
            continue
            
        # Extract boxes and depths
        pred_boxes = pred[:, :4]  # x1, y1, x2, y2
        pred_depths = pred[:, 5]  # depth
        pred_confs = pred[:, 4]   # confidence
        
        gt_boxes = gt_label['boxes']  # x1, y1, x2, y2
        gt_depths = gt_label['depths']  # depth values
        
        # Calculate IoU between all prediction and ground truth boxes
        ious = calculate_iou(pred_boxes, gt_boxes)
        
        # Find best matches
        for i, pred_box in enumerate(pred_boxes):
            if pred_confs[i] < 0.5:  # Skip low confidence predictions
                continue
                
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                iou = ious[i, j]
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                matched_pairs.append((pred_depths[i], gt_depths[best_gt_idx]))
    
    return matched_pairs


def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        boxes1: First set of boxes [N, 4] (x1, y1, x2, y2)
        boxes2: Second set of boxes [M, 4] (x1, y1, x2, y2)
        
    Returns:
        torch.Tensor: IoU matrix [N, M]
    """
    # Calculate intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate union
    union = area1[:, None] + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def evaluate_mde(model, dataloader, device='cuda'):
    """
    Evaluate model with AP and depth error metrics.
    
    Args:
        model: Trained YOLO MDE model
        dataloader: DataLoader with validation data
        device: Device to run evaluation on
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    model.eval()
    depth_errors = []
    absolute_errors = []
    squared_errors = []
    accuracy_metrics = {f"δ < {th:.3f}": [] for th in [1.25, 1.25**2, 1.25**3]}
    
    print("🔍 Evaluating MDE model...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"   Processing batch {batch_idx}/{len(dataloader)}")
            
            # Move batch to device
            images = batch['img'].to(device)
            
            # Get predictions
            preds = model(images)
            
            # Process each image in the batch
            for i, pred in enumerate(preds):
                if pred.boxes is not None and len(pred.boxes) > 0:
                    # Extract predictions
                    pred_data = pred.boxes.data  # [num_detections, 6] (x1,y1,x2,y2,conf,depth)
                    
                    # Get ground truth for this image
                    gt_boxes = batch['bboxes'][i]  # [num_gt, 4]
                    gt_depths = batch['depths'][i]  # [num_gt]
                    
                    if len(gt_boxes) > 0 and len(gt_depths) > 0:
                        # Match predictions with ground truth
                        matched_pairs = match_predictions(
                            [pred_data], 
                            [{'boxes': gt_boxes, 'depths': gt_depths}]
                        )
                        
                        # Calculate metrics for matched pairs
                        for pred_depth, gt_depth in matched_pairs:
                            # Depth error rate
                            error_rate = calculate_depth_error(pred_depth, gt_depth)
                            depth_errors.append(error_rate.item())
                            
                            # Absolute error
                            abs_error = calculate_absolute_depth_error(pred_depth, gt_depth)
                            absolute_errors.append(abs_error.item())
                            
                            # Squared error
                            sq_error = calculate_squared_depth_error(pred_depth, gt_depth)
                            squared_errors.append(sq_error.item())
                            
                            # Accuracy metrics
                            acc_metrics = calculate_depth_accuracy(pred_depth, gt_depth)
                            for key, value in acc_metrics.items():
                                accuracy_metrics[key].append(value)
    
    # Calculate final metrics
    metrics = {}
    
    if depth_errors:
        metrics['mean_depth_error_rate'] = np.mean(depth_errors)
        metrics['std_depth_error_rate'] = np.std(depth_errors)
    
    if absolute_errors:
        metrics['mean_absolute_error'] = np.mean(absolute_errors)
        metrics['std_absolute_error'] = np.std(absolute_errors)
    
    if squared_errors:
        metrics['rmse'] = np.sqrt(np.mean(squared_errors))
        metrics['std_rmse'] = np.std(squared_errors)
    
    for key, values in accuracy_metrics.items():
        if values:
            metrics[f"{key}_mean"] = np.mean(values)
            metrics[f"{key}_std"] = np.std(values)
    
    # Print results
    print("\n📊 MDE Evaluation Results:")
    print("=" * 50)
    
    if 'mean_depth_error_rate' in metrics:
        print(f"Mean Depth Error Rate: {metrics['mean_depth_error_rate']:.2f}% ± {metrics['std_depth_error_rate']:.2f}%")
    
    if 'mean_absolute_error' in metrics:
        print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.2f}m ± {metrics['std_absolute_error']:.2f}m")
    
    if 'rmse' in metrics:
        print(f"Root Mean Squared Error: {metrics['rmse']:.2f}m ± {metrics['std_rmse']:.2f}m")
    
    for key, values in accuracy_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.2f}% ± {std_val:.2f}%")
    
    print("=" * 50)
    
    return metrics


def print_depth_statistics(depths: torch.Tensor, name: str = "Depth"):
    """
    Print depth statistics for analysis.
    
    Args:
        depths: Depth values tensor
        name: Name for the statistics
    """
    depths_np = depths.cpu().numpy()
    
    print(f"\n📈 {name} Statistics:")
    print(f"   Count: {len(depths_np)}")
    print(f"   Min: {depths_np.min():.2f}m")
    print(f"   Max: {depths_np.max():.2f}m")
    print(f"   Mean: {depths_np.mean():.2f}m")
    print(f"   Std: {depths_np.std():.2f}m")
    print(f"   Median: {np.median(depths_np):.2f}m")
    print(f"   25th percentile: {np.percentile(depths_np, 25):.2f}m")
    print(f"   75th percentile: {np.percentile(depths_np, 75):.2f}m")
