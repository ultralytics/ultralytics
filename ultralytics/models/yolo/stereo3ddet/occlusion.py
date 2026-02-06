# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Occlusion Classification Module

This module implements occlusion classification using the depth-line algorithm as described
in the Stereo CenterNet paper. The algorithm determines which objects are occluded by
analyzing depth discontinuities along horizontal scan lines.

Paper Reference: Algorithm 1 "3D Object Classification Strategy"
    "We classify objects as occluded or unoccluded by analyzing the depth-line,
    which encodes the nearest object at each horizontal position. Objects whose
    boundaries are not visible in the depth-line are classified as occluded."

The algorithm works in two passes:
    1. First pass: Build depth line from bounding boxes
       - For each pixel column, track the depth of the closest (frontmost) object
       - This creates a 1D array representing visible depth across the image width
    
    2. Second pass: Classify occlusion by boundary visibility
       - For each object, check if its left and right boundaries are visible
       - An object is occluded if BOTH boundaries are hidden by closer objects
       - An object is unoccluded if at least one boundary is visible

The depth-line serves as a visibility buffer:
    - If depth_line[x] < object_depth: position x is blocked by a closer object
    - If depth_line[x] >= object_depth: position x is visible from the camera

Usage:
    Heavily occluded objects should skip dense photometric alignment, as the
    occluding objects will corrupt the photometric error measurement.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

import numpy as np


def _build_depth_line(
    detections: List[Dict[str, Any]],
    image_width: int = 1242,
) -> np.ndarray:
    """Build depth line from bounding boxes (Algorithm 1, First Pass).
    
    The depth line is a 1D array where each element represents the depth of the
    closest (frontmost) object at that horizontal pixel position. This serves as
    a visibility buffer for occlusion classification.
    
    Paper Reference: Algorithm 1, lines 1-7
        "For each column j from 0 to image_width:
            depth_line[j] = depth of closest object covering column j"
    
    Args:
        detections: List of detection dicts with keys:
            - bbox_2d: (x1, y1, x2, y2) 2D bounding box coordinates
            - center_3d: (x, y, z) 3D center where z is depth
        image_width: Width of the image in pixels (default: 1242 for KITTI).
    
    Returns:
        np.ndarray: Depth line array of shape (image_width,) where each element
            contains the depth of the closest object at that column. Zero values
            indicate no object coverage at that position.
    
    Note:
        - Objects with invalid bbox_2d or center_3d are skipped
        - Depth blending is used when multiple objects overlap at similar depths
    """
    # Initialize depth line with zeros (no objects)
    depth_line = np.zeros(image_width, dtype=np.float32)
    
    # Process each detection
    for det in detections:
        # Extract bounding box
        bbox_2d = det.get("bbox_2d")
        if bbox_2d is None:
            continue
        
        # Handle different bbox_2d formats
        if hasattr(bbox_2d, '__iter__') and not isinstance(bbox_2d, str):
            if len(bbox_2d) >= 4:
                x1, y1, x2, y2 = bbox_2d[:4]
            else:
                continue
        else:
            continue
        
        # Get depth (z coordinate)
        center_3d = det.get("center_3d")
        if center_3d is None:
            continue
        
        if hasattr(center_3d, '__iter__') and len(center_3d) >= 3:
            z = center_3d[2]  # depth is the third coordinate
        else:
            continue
        
        # Skip invalid depths
        if z <= 0:
            continue
        
        # Clamp bounding box to image bounds
        x1 = max(0, min(int(x1), image_width - 1))
        x2 = max(0, min(int(x2), image_width))
        
        # Skip if box is outside image or has zero width
        if x1 >= x2:
            continue
        
        # Update depth line for this object's horizontal extent
        for j in range(x1, x2):
            if depth_line[j] == 0:
                # No previous object at this position
                depth_line[j] = z
            elif z < depth_line[j]:
                # This object is closer - blend depths for smooth transitions
                # Blending helps with objects at similar depths
                depth_line[j] = (z + depth_line[j]) / 2.0
            # If z >= depth_line[j], the existing object is closer or same depth
            # so we keep the current depth_line value unchanged
    
    return depth_line


def _classify_by_boundary_visibility(
    detections: List[Dict[str, Any]],
    depth_line: np.ndarray,
    image_width: int = 1242,
    depth_tolerance: float = 1.0,
) -> Tuple[List[int], List[int]]:
    """Classify objects by boundary visibility (Algorithm 1, Second Pass).
    
    Determines which objects are occluded by checking if their left and right
    boundaries are visible in the depth line. An object is considered occluded
    if BOTH boundaries are hidden by closer objects.
    
    Paper Reference: Algorithm 1, lines 8-14
        "For each object k:
            Check if left boundary (x1) is visible: depth_line[x1] >= z_k
            Check if right boundary (x2) is visible: depth_line[x2] >= z_k
            If both hidden -> occluded, else -> unoccluded"
    
    Args:
        detections: List of detection dicts with bbox_2d and center_3d keys.
        depth_line: Pre-computed depth line array from _build_depth_line().
        image_width: Width of the image in pixels.
        depth_tolerance: Tolerance in meters for depth comparison. Objects within
            this tolerance of the depth line value are considered visible.
            Default is 1.0 meter to handle depth estimation errors.
    
    Returns:
        Tuple of (occluded_indices, unoccluded_indices):
            - occluded_indices: List of detection indices that are occluded
            - unoccluded_indices: List of detection indices that are unoccluded
    
    Note:
        - Objects with zero or missing depth_line coverage are treated as visible
        - The depth_tolerance helps handle noisy depth estimates
    """
    occluded = []
    unoccluded = []
    
    for k, det in enumerate(detections):
        # Extract bounding box
        bbox_2d = det.get("bbox_2d")
        if bbox_2d is None:
            # No bbox, treat as unoccluded (can't determine occlusion)
            unoccluded.append(k)
            continue
        
        # Handle different bbox_2d formats
        if hasattr(bbox_2d, '__iter__') and not isinstance(bbox_2d, str):
            if len(bbox_2d) >= 4:
                x1, y1, x2, y2 = bbox_2d[:4]
            else:
                unoccluded.append(k)
                continue
        else:
            unoccluded.append(k)
            continue
        
        # Get depth (z coordinate)
        center_3d = det.get("center_3d")
        if center_3d is None:
            unoccluded.append(k)
            continue
        
        if hasattr(center_3d, '__iter__') and len(center_3d) >= 3:
            z = center_3d[2]
        else:
            unoccluded.append(k)
            continue
        
        # Skip invalid depths
        if z <= 0:
            unoccluded.append(k)
            continue
        
        # Get boundary positions (clamped to image bounds)
        x1_clamped = max(0, min(int(x1), image_width - 1))
        x2_clamped = max(0, min(int(x2) - 1, image_width - 1))  # x2 is exclusive
        
        # Ensure x2_clamped >= x1_clamped
        if x2_clamped < x1_clamped:
            x2_clamped = x1_clamped
        
        # Check if boundaries are visible
        # A boundary is visible if:
        #   1. The depth_line at that position is zero (no coverage), OR
        #   2. The depth_line value is >= object depth (within tolerance)
        
        depth_at_left = depth_line[x1_clamped]
        depth_at_right = depth_line[x2_clamped]
        
        # Left boundary visible?
        left_visible = (
            depth_at_left == 0 or  # No coverage means visible
            depth_at_left >= z - depth_tolerance  # Object is at or in front of depth line
        )
        
        # Right boundary visible?
        right_visible = (
            depth_at_right == 0 or
            depth_at_right >= z - depth_tolerance
        )
        
        # Classify based on boundary visibility
        if not left_visible and not right_visible:
            # Both boundaries hidden -> heavily occluded
            occluded.append(k)
        else:
            # At least one boundary visible -> not occluded (or partially occluded)
            unoccluded.append(k)
    
    return occluded, unoccluded


def classify_occlusion(
    detections: List[Dict[str, Any]],
    image_width: int = 1242,
    depth_tolerance: float = 1.0,
) -> Tuple[List[int], List[int]]:
    """Classify objects as occluded or unoccluded using depth-line algorithm.
    
    This is the main entry point for occlusion classification. It implements
    Algorithm 1 from the Stereo CenterNet paper using a two-pass approach:
    
    1. First pass: Build a depth line that tracks the closest object at each
       horizontal pixel position across the image width.
    
    2. Second pass: For each object, check if its left and right boundaries
       are visible in the depth line. Objects with both boundaries hidden
       are classified as occluded.
    
    Paper Reference: Algorithm 1 "3D Object Classification Strategy"
    
    Args:
        detections: List of detection dicts, each containing:
            - bbox_2d: (x1, y1, x2, y2) 2D bounding box coordinates in pixels.
                       Can also be a Box3D object with bbox_2d attribute.
            - center_3d: (x, y, z) 3D center coordinates where z is depth in meters.
                         Can also be a Box3D object with center_3d attribute.
        image_width: Width of the image in pixels. Default is 1242 (KITTI standard).
        depth_tolerance: Tolerance in meters for depth visibility comparison.
            Objects within this tolerance of the depth line are considered visible.
            Default is 1.0 meter to handle depth estimation uncertainty.
    
    Returns:
        Tuple[List[int], List[int]]: Two lists of detection indices:
            - occluded_indices: Indices of detections that are heavily occluded
              (both boundaries hidden by closer objects)
            - unoccluded_indices: Indices of detections that are not occluded
              (at least one boundary is visible)
    
    Example:
        >>> detections = [
        ...     {"bbox_2d": (100, 50, 200, 150), "center_3d": (1.0, 1.0, 20.0)},
        ...     {"bbox_2d": (150, 60, 250, 160), "center_3d": (1.5, 1.0, 30.0)},
        ... ]
        >>> occluded, unoccluded = classify_occlusion(detections)
        >>> print(f"Occluded: {occluded}, Unoccluded: {unoccluded}")
    
    Note:
        - Empty detection lists return ([], [])
        - Detections with missing or invalid data are classified as unoccluded
        - The algorithm assumes all detections are from the same image
    """
    if len(detections) == 0:
        return [], []
    
    # Normalize detections to dict format (handle Box3D objects)
    normalized_detections = []
    for det in detections:
        if isinstance(det, dict):
            normalized_detections.append(det)
        else:
            # Assume it's a Box3D-like object with attributes
            norm_det = {}
            if hasattr(det, "center_3d"):
                norm_det["center_3d"] = det.center_3d
            # bbox_2d may come from dict callers; Box3D no longer has this field
            if hasattr(det, "bbox_2d"):
                norm_det["bbox_2d"] = det.bbox_2d
            normalized_detections.append(norm_det)
    
    # First pass: Build depth line
    depth_line = _build_depth_line(normalized_detections, image_width)
    
    # Second pass: Classify by boundary visibility
    occluded, unoccluded = _classify_by_boundary_visibility(
        normalized_detections,
        depth_line,
        image_width,
        depth_tolerance,
    )
    
    return occluded, unoccluded


def should_skip_dense_alignment(
    detection_idx: int,
    occluded_indices: List[int],
) -> bool:
    """Check if dense alignment should be skipped for a detection.
    
    Dense photometric alignment can produce incorrect results for heavily occluded
    objects because the occluding object's appearance will contaminate the
    photometric error measurement. This helper function determines whether
    to skip dense alignment based on occlusion classification.
    
    Args:
        detection_idx: Index of the detection to check.
        occluded_indices: List of detection indices classified as occluded
            (typically from classify_occlusion()).
    
    Returns:
        bool: True if dense alignment should be skipped for this detection
            (i.e., the detection is heavily occluded), False otherwise.
    
    Example:
        >>> occluded, unoccluded = classify_occlusion(detections)
        >>> for i, det in enumerate(detections):
        ...     if should_skip_dense_alignment(i, occluded):
        ...         # Use geometric construction depth only
        ...         refined_depth = det["center_3d"][2]
        ...     else:
        ...         # Apply dense photometric alignment
        ...         refined_depth = aligner.refine_depth(...)
    
    Note:
        This is a simple lookup function. The main occlusion classification
        logic is in classify_occlusion().
    """
    return detection_idx in occluded_indices


def get_occlusion_stats(
    detections: List[Dict[str, Any]],
    image_width: int = 1242,
) -> Dict[str, Any]:
    """Get detailed occlusion statistics for a set of detections.
    
    Useful for debugging, visualization, and logging of occlusion classification
    results. Returns statistics about the depth line and occlusion distribution.
    
    Args:
        detections: List of detection dicts with bbox_2d and center_3d.
        image_width: Width of the image in pixels.
    
    Returns:
        Dict containing:
            - total_detections: Total number of detections
            - num_occluded: Number of occluded detections
            - num_unoccluded: Number of unoccluded detections
            - occlusion_rate: Fraction of detections that are occluded
            - depth_line_coverage: Fraction of image width with depth values
            - depth_range: (min_depth, max_depth) of valid depth values
    """
    if len(detections) == 0:
        return {
            "total_detections": 0,
            "num_occluded": 0,
            "num_unoccluded": 0,
            "occlusion_rate": 0.0,
            "depth_line_coverage": 0.0,
            "depth_range": (0.0, 0.0),
        }
    
    # Normalize detections
    normalized_detections = []
    for det in detections:
        if isinstance(det, dict):
            normalized_detections.append(det)
        else:
            norm_det = {}
            if hasattr(det, "center_3d"):
                norm_det["center_3d"] = det.center_3d
            if hasattr(det, "bbox_2d"):
                norm_det["bbox_2d"] = det.bbox_2d
            normalized_detections.append(norm_det)
    
    # Get depth line and classification
    depth_line = _build_depth_line(normalized_detections, image_width)
    occluded, unoccluded = classify_occlusion(detections, image_width)
    
    # Compute statistics
    non_zero_mask = depth_line > 0
    coverage = non_zero_mask.sum() / image_width
    
    if non_zero_mask.any():
        min_depth = float(depth_line[non_zero_mask].min())
        max_depth = float(depth_line[non_zero_mask].max())
    else:
        min_depth = max_depth = 0.0
    
    total = len(detections)
    num_occluded = len(occluded)
    
    return {
        "total_detections": total,
        "num_occluded": num_occluded,
        "num_unoccluded": len(unoccluded),
        "occlusion_rate": num_occluded / total if total > 0 else 0.0,
        "depth_line_coverage": float(coverage),
        "depth_range": (min_depth, max_depth),
    }
