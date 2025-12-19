# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Perspective Keypoint Selection Module (GAP-004)

This module implements the perspective keypoint selection algorithm from the Stereo CenterNet paper.
The algorithm selects 2 perspective keypoints from the 4 predicted bottom vertices based on the
object's orientation angle, following quadrant-based selection rules.

Paper Reference: Section 3.2 "The perspective key points are the same as the bottom vertex
                 closest to the camera"

Vertex numbering convention (bottom corners, viewed from above):
    3 ─────── 2
    │   car   │  → heading direction
    0 ─────── 1

    Vertex 0: front-left corner
    Vertex 1: front-right corner
    Vertex 2: rear-right corner
    Vertex 3: rear-left corner

Selection rules based on orientation θ (yaw angle):
    θ ∈ [-π, -π/2)  : vertices 0, 1 (front face visible - front-left to front-right)
    θ ∈ [-π/2, 0)   : vertices 1, 2 (right face visible - front-right to rear-right)
    θ ∈ [0, π/2)    : vertices 2, 3 (rear face visible - rear-right to rear-left)
    θ ∈ [π/2, π]    : vertices 3, 0 (left face visible - rear-left to front-left)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Union, Tuple

# Type aliases for clarity
ArrayLike = Union[np.ndarray, torch.Tensor]
Point2D = Union[np.ndarray, torch.Tensor]


def select_perspective_keypoints(
    vertices: ArrayLike,
    orientation: Union[float, torch.Tensor],
) -> Tuple[Point2D, Point2D]:
    """Select 2 perspective keypoints based on object orientation.
    
    This function implements the perspective keypoint selection algorithm from Section 3.2
    of the Stereo CenterNet paper. Given the 4 bottom vertices of a 3D bounding box and
    the object's orientation (yaw angle), it selects the 2 vertices that are most visible
    from the camera viewpoint.
    
    Paper Reference: Section 3.2 Figure 4 - "The perspective key points are the same as 
                     the bottom vertex closest to the camera"
    
    Args:
        vertices: [4, 2] array/tensor - 4 bottom vertex coordinates (x, y) in image space.
                  Vertex ordering: 0=front-left, 1=front-right, 2=rear-right, 3=rear-left
        orientation: Object yaw angle θ in radians, range [-π, π].
                     θ=0 means facing away from camera (rear visible)
                     θ=π/-π means facing toward camera (front visible)
                     θ=π/2 means left side visible
                     θ=-π/2 means right side visible
        
    Returns:
        (kp1, kp2): Tuple of two keypoint coordinates, each as [2] array/tensor.
                    These are the two visible bottom corners based on orientation.
                    
    Example:
        >>> vertices = np.array([[100, 200], [150, 200], [150, 250], [100, 250]])
        >>> orientation = np.pi / 4  # 45 degrees, rear-right quadrant
        >>> kp1, kp2 = select_perspective_keypoints(vertices, orientation)
        >>> # Returns vertices 2 and 3 (rear-right and rear-left)
    """
    is_tensor = isinstance(vertices, torch.Tensor)
    
    if is_tensor:
        # Normalize orientation to [-π, π] using atan2 for numerical stability
        theta = torch.atan2(torch.sin(orientation), torch.cos(orientation))
        pi = torch.tensor(np.pi, device=vertices.device, dtype=vertices.dtype)
    else:
        # Convert to numpy array if needed
        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)
        if not isinstance(orientation, (int, float)):
            orientation = float(orientation)
        
        # Normalize orientation to [-π, π]
        theta = np.arctan2(np.sin(orientation), np.cos(orientation))
        pi = np.pi
    
    # Quadrant-based selection logic per paper Section 3.2 Figure 4:
    # The visible face determines which 2 bottom vertices are selected
    #
    # Quadrant 1: θ ∈ [-π, -π/2) - Front face visible (facing toward camera)
    #             Select vertices 0 (front-left) and 1 (front-right)
    # Quadrant 2: θ ∈ [-π/2, 0) - Right face visible
    #             Select vertices 1 (front-right) and 2 (rear-right)
    # Quadrant 3: θ ∈ [0, π/2) - Rear face visible (facing away from camera)
    #             Select vertices 2 (rear-right) and 3 (rear-left)
    # Quadrant 4: θ ∈ [π/2, π] - Left face visible
    #             Select vertices 3 (rear-left) and 0 (front-left)
    
    if is_tensor:
        # For single tensor values, we use Python comparisons
        theta_val = theta.item() if theta.ndim == 0 else float(theta)
        pi_val = float(pi)
    else:
        theta_val = float(theta)
        pi_val = pi
    
    if -pi_val <= theta_val < -pi_val / 2:
        # Quadrant 1: Front face visible
        idx1, idx2 = 0, 1
    elif -pi_val / 2 <= theta_val < 0:
        # Quadrant 2: Right face visible
        idx1, idx2 = 1, 2
    elif 0 <= theta_val < pi_val / 2:
        # Quadrant 3: Rear face visible
        idx1, idx2 = 2, 3
    else:  # pi/2 <= theta <= pi
        # Quadrant 4: Left face visible
        idx1, idx2 = 3, 0
    
    return vertices[idx1], vertices[idx2]


def select_perspective_keypoints_batch(
    vertices: torch.Tensor,
    orientations: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized batch version of perspective keypoint selection.
    
    This function performs the same keypoint selection as `select_perspective_keypoints`
    but efficiently handles batches of objects in a single forward pass without Python loops.
    
    Args:
        vertices: [N, 4, 2] tensor - N objects, 4 bottom vertices each, (x, y) coordinates.
                  Vertex ordering per object: 0=front-left, 1=front-right, 2=rear-right, 3=rear-left
        orientations: [N] tensor - Yaw angles in radians for each object.
        
    Returns:
        (kp1, kp2): Tuple of two tensors, each [N, 2] containing the selected keypoints
                    for each object in the batch.
                    
    Example:
        >>> vertices = torch.randn(100, 4, 2)  # 100 objects, 4 vertices each
        >>> orientations = torch.rand(100) * 2 * np.pi - np.pi  # Random orientations
        >>> kp1, kp2 = select_perspective_keypoints_batch(vertices, orientations)
        >>> print(kp1.shape, kp2.shape)  # torch.Size([100, 2]) torch.Size([100, 2])
    """
    N = vertices.shape[0]
    device = vertices.device
    dtype = vertices.dtype
    
    # Handle empty batch
    if N == 0:
        return torch.zeros(0, 2, device=device, dtype=dtype), torch.zeros(0, 2, device=device, dtype=dtype)
    
    # Normalize orientations to [-π, π]
    theta = torch.atan2(torch.sin(orientations), torch.cos(orientations))
    
    # Define pi as tensor for consistent comparisons
    pi = torch.tensor(np.pi, device=device, dtype=dtype)
    
    # Create masks for each quadrant
    # Quadrant 1: θ ∈ [-π, -π/2) → indices (0, 1)
    # Quadrant 2: θ ∈ [-π/2, 0)  → indices (1, 2)
    # Quadrant 3: θ ∈ [0, π/2)   → indices (2, 3)
    # Quadrant 4: θ ∈ [π/2, π]   → indices (3, 0)
    
    mask_q1 = (theta >= -pi) & (theta < -pi / 2)
    mask_q2 = (theta >= -pi / 2) & (theta < 0)
    mask_q3 = (theta >= 0) & (theta < pi / 2)
    # mask_q4 covers the remaining case: (theta >= pi/2) | (theta <= pi)
    # Using complement of other masks for numerical stability at boundaries
    mask_q4 = ~(mask_q1 | mask_q2 | mask_q3)
    
    # Initialize index tensors
    idx1 = torch.zeros(N, dtype=torch.long, device=device)
    idx2 = torch.ones(N, dtype=torch.long, device=device)
    
    # Assign indices based on quadrant masks
    # Quadrant 1: front face visible → vertices (0, 1)
    idx1[mask_q1] = 0
    idx2[mask_q1] = 1
    
    # Quadrant 2: right face visible → vertices (1, 2)
    idx1[mask_q2] = 1
    idx2[mask_q2] = 2
    
    # Quadrant 3: rear face visible → vertices (2, 3)
    idx1[mask_q3] = 2
    idx2[mask_q3] = 3
    
    # Quadrant 4: left face visible → vertices (3, 0)
    idx1[mask_q4] = 3
    idx2[mask_q4] = 0
    
    # Gather selected keypoints using advanced indexing
    batch_indices = torch.arange(N, device=device)
    kp1 = vertices[batch_indices, idx1]  # [N, 2]
    kp2 = vertices[batch_indices, idx2]  # [N, 2]
    
    return kp1, kp2


def get_visible_face_indices(orientation: Union[float, torch.Tensor]) -> Tuple[int, int]:
    """Get the indices of the two visible vertices based on orientation.
    
    This is a utility function that returns just the vertex indices without
    requiring the actual vertex coordinates. Useful for debugging and testing.
    
    Args:
        orientation: Object yaw angle θ in radians.
        
    Returns:
        (idx1, idx2): Tuple of two vertex indices (0-3) for the visible face.
    """
    if isinstance(orientation, torch.Tensor):
        theta = torch.atan2(torch.sin(orientation), torch.cos(orientation)).item()
    else:
        theta = float(np.arctan2(np.sin(orientation), np.cos(orientation)))
    
    pi = np.pi
    
    if -pi <= theta < -pi / 2:
        return 0, 1  # Front face
    elif -pi / 2 <= theta < 0:
        return 1, 2  # Right face
    elif 0 <= theta < pi / 2:
        return 2, 3  # Rear face
    else:
        return 3, 0  # Left face


def get_quadrant_name(orientation: Union[float, torch.Tensor]) -> str:
    """Get human-readable name of the visible face based on orientation.
    
    Useful for debugging and visualization.
    
    Args:
        orientation: Object yaw angle θ in radians.
        
    Returns:
        Name of the visible face: "front", "right", "rear", or "left".
    """
    idx1, idx2 = get_visible_face_indices(orientation)
    
    quadrant_names = {
        (0, 1): "front",
        (1, 2): "right",
        (2, 3): "rear",
        (3, 0): "left",
    }
    
    return quadrant_names.get((idx1, idx2), "unknown")

