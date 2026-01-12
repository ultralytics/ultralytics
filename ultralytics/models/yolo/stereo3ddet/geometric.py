# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Geometric Construction Module (GAP-001)

This module implements the Gauss-Newton geometric construction solver from the Stereo CenterNet paper.
The solver estimates 3D bounding box parameters (x, y, z, θ) from 2D stereo observations using
7 geometric constraint equations.

Paper Reference: Section 3.2, Equation 9

The geometric construction provides more accurate 3D localization compared to simple triangulation
by jointly optimizing over all stereo observations using nonlinear least squares.

Algorithm Overview:
1. Initialize 3D state from simple triangulation
2. Compute residuals from 7 geometric constraint equations
3. Compute Jacobian via numerical differentiation
4. Apply Gauss-Newton update with Levenberg-Marquardt damping
5. Iterate until convergence or max iterations
6. Fall back to simple triangulation if convergence fails
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple, Union, Optional

import numpy as np
import torch

from ultralytics.utils import LOGGER


class GeometricObservations(NamedTuple):
    """2D observations for geometric construction from stereo detection.
    
    These are the measured 2D coordinates extracted from the detection pipeline:
    - Left image center and keypoints
    - Right image center (from stereo matching)
    - Perspective keypoint from visible face
    
    All coordinates are in pixel space (original image resolution).
    
    Attributes:
        ul: Left image box center u-coordinate (horizontal)
        vl: Left image box center v-coordinate (vertical)
        ur: Right image box center u-coordinate 
        vr: Right image box center v-coordinate
        ul_prime: Left box center projected to right image (for stereo matching)
        ur_prime: Right box center projected to right image
        up: Perspective keypoint u-coordinate (from select_perspective_keypoints)
        vp: Perspective keypoint v-coordinate
    """
    ul: float  # left box center u
    vl: float  # left box center v 
    ur: float  # right box center u
    vr: float  # right box center v
    ul_prime: float  # left box center u in right image (disparity reference)
    ur_prime: float  # right box center u in right image
    up: float  # perspective keypoint u
    vp: float  # perspective keypoint v (added for completeness)


class CalibParams(NamedTuple):
    """Camera calibration parameters for stereo geometry.
    
    Standard pinhole camera model with stereo baseline for depth estimation.
    
    Attributes:
        fx: Focal length in x-direction (pixels)
        fy: Focal length in y-direction (pixels)
        cx: Principal point x-coordinate (pixels)
        cy: Principal point y-coordinate (pixels)
        baseline: Stereo baseline (meters) - distance between camera centers
    """
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float


class GeometricConstruction:
    """Solve 3D box parameters from 2D stereo observations using Gauss-Newton.
    
    This class implements the geometric construction algorithm from Section 3.2 of
    the Stereo CenterNet paper. Given 2D observations from stereo detection, it
    solves for the 3D box center (x, y, z) and orientation (θ) using 7 geometric
    constraint equations and Gauss-Newton optimization.
    
    The constraint equations relate:
    - 2D center positions in left/right images
    - Stereo disparity
    - Perspective keypoint positions
    - 3D box geometry (dimensions, orientation)
    
    Paper Reference: Section 3.2, Equation 9
    
    Attributes:
        max_iterations: Maximum number of Gauss-Newton iterations (default: 10)
        tolerance: Convergence tolerance on residual norm (default: 1e-6)
        damping: Levenberg-Marquardt damping factor for numerical stability (default: 1e-3)
        
    Example:
        >>> solver = GeometricConstruction(max_iterations=10, tolerance=1e-6)
        >>> calib = CalibParams(fx=721.5, fy=721.5, cx=609.5, cy=172.8, baseline=0.54)
        >>> obs = GeometricObservations(ul=500, vl=180, ur=480, vr=180, 
        ...                              ul_prime=450, ur_prime=430, up=520, vp=200)
        >>> dims = (3.89, 1.73, 1.52)  # length, width, height
        >>> x, y, z, theta, converged = solver.solve(obs, dims, theta_init=0.0, calib=calib)
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        damping: float = 1e-3,
    ) -> None:
        """Initialize the geometric construction solver.
        
        Args:
            max_iterations: Maximum number of Gauss-Newton iterations.
                           Higher values may improve accuracy but increase computation.
            tolerance: Convergence threshold on residual L2 norm.
                      Solver stops when ||residuals|| < tolerance.
            damping: Levenberg-Marquardt damping factor λ for numerical stability.
                    Larger values make updates more conservative (like gradient descent).
                    Smaller values approach pure Gauss-Newton (faster convergence near optimum).
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        
        # Statistics for convergence monitoring (SC-007)
        self._total_solves = 0
        self._converged_solves = 0
        
    @property
    def convergence_rate(self) -> float:
        """Get the convergence rate for SC-007 validation.
        
        Returns:
            Fraction of solve() calls that converged (0.0 to 1.0).
        """
        if self._total_solves == 0:
            return 1.0  # No solves yet, assume success
        return self._converged_solves / self._total_solves
    
    def reset_statistics(self) -> None:
        """Reset convergence statistics."""
        self._total_solves = 0
        self._converged_solves = 0
    
    def solve(
        self,
        observations: GeometricObservations,
        dimensions: tuple[float, float, float],  # (length, width, height)
        theta_init: float,
        calib: CalibParams,
        z_init: Optional[float] = None,
    ) -> tuple[float, float, float, float, bool]:
        """Solve for 3D box center and orientation using Gauss-Newton optimization.
        
        This is the main entry point for the geometric construction algorithm.
        It initializes the 3D state from simple triangulation (or provided z_init),
        then iteratively refines using Gauss-Newton with Levenberg-Marquardt damping.
        
        Args:
            observations: 2D observations from stereo detection pipeline.
            dimensions: (length, width, height) of 3D box in meters.
                       Length is along the forward direction, width is lateral,
                       height is vertical.
            theta_init: Initial orientation estimate (yaw angle in radians).
                       Usually from Multi-Bin orientation decoder.
            calib: Camera calibration parameters.
            z_init: Optional initial depth estimate. If None, computed from
                   simple stereo triangulation using disparity.
                   
        Returns:
            Tuple of (x, y, z, theta, converged):
            - x: 3D center x-coordinate in camera frame (meters, right is positive)
            - y: 3D center y-coordinate in camera frame (meters, down is positive)
            - z: 3D center z-coordinate/depth (meters, forward is positive)
            - theta: Refined orientation/yaw angle (radians)
            - converged: True if solver converged within tolerance
            
        Note:
            If the solver fails to converge, it returns the best estimate found
            and sets converged=False. The caller can then fall back to simple
            triangulation using the fallback_simple_triangulation() function.
        """
        self._total_solves += 1
        
        l, w, h = dimensions
        fx, fy, cx, cy, b = calib
        
        # Initialize depth from simple triangulation if not provided
        if z_init is None:
            disparity = observations.ul - observations.ul_prime
            if disparity > 1.0:  # Valid disparity
                z_init = (fx * b) / disparity
            else:
                z_init = 30.0  # Default depth for invalid disparity
        
        # Initialize 3D position from 2D center and depth
        x_init = (observations.ul - cx) * z_init / fx
        y_init = (observations.vl - cy) * z_init / fy
        
        # Initial state vector: [x, y, z, θ]
        state = np.array([x_init, y_init, z_init, theta_init], dtype=np.float64)
        
        converged = False
        best_state = state.copy()
        best_residual_norm = float('inf')
        
        for iteration in range(self.max_iterations):
            # Compute residuals from 7 constraint equations
            residuals = self._compute_residuals(state, observations, dimensions, calib)
            residual_norm = np.linalg.norm(residuals)
            
            # Track best solution
            if residual_norm < best_residual_norm:
                best_residual_norm = residual_norm
                best_state = state.copy()
            
            # Check convergence
            if residual_norm < self.tolerance:
                converged = True
                break
            
            # Compute Jacobian via numerical differentiation
            jacobian = self._compute_jacobian(state, observations, dimensions, calib)
            
            # Gauss-Newton update with Levenberg-Marquardt damping
            # Normal equations: (J^T J + λI) δ = -J^T r
            JtJ = jacobian.T @ jacobian
            JtJ += self.damping * np.eye(4)  # Add damping for stability
            Jtr = jacobian.T @ residuals
            
            try:
                delta = np.linalg.solve(JtJ, -Jtr)
            except np.linalg.LinAlgError:
                # Singular matrix - stop iteration
                LOGGER.debug(f"Geometric solver: singular matrix at iteration {iteration}")
                break
            
            # Update state
            state = state + delta
            
            # Normalize theta to [-π, π]
            state[3] = math.atan2(math.sin(state[3]), math.cos(state[3]))
            
            # Check for numerical issues
            if not np.all(np.isfinite(state)):
                LOGGER.debug(f"Geometric solver: non-finite state at iteration {iteration}")
                state = best_state  # Revert to best
                break
        
        # Use best state found
        x, y, z, theta = best_state
        
        # Ensure depth is positive
        z = max(z, 1.0)
        
        if converged:
            self._converged_solves += 1
        
        return float(x), float(y), float(z), float(theta), converged
    
    def _compute_residuals(
        self,
        state: np.ndarray,
        obs: GeometricObservations,
        dims: tuple[float, float, float],
        calib: CalibParams,
    ) -> np.ndarray:
        """Compute residual vector from 7 geometric constraint equations.
        
        The constraints relate the 2D observations to the 3D box parameters:
        1-2: Left image center (u, v) from 3D center projection
        3-4: Right image center (u, v) from 3D center projection  
        5: Disparity constraint (left center in right image)
        6: Right center projected to right image
        7: Perspective keypoint projection
        
        Paper Reference: Section 3.2, Equation 9
        
        Args:
            state: Current state [x, y, z, θ]
            obs: 2D observations
            dims: (length, width, height) in meters
            calib: Camera calibration
            
        Returns:
            Residual vector [7] with prediction errors
        """
        x, y, z, theta = state
        l, w, h = dims
        fx, fy, cx, cy, b = calib
        
        # Prevent division by zero
        z = max(z, 0.1)
        
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # ===== Constraint 1-2: Left image center =====
        # Project 3D center to left image
        u_left_pred = fx * x / z + cx
        v_left_pred = fy * y / z + cy
        
        # ===== Constraint 3-4: Right image center =====
        # Right camera is offset by baseline along x-axis
        # x_right = x - baseline
        u_right_pred = fx * (x - b) / z + cx
        v_right_pred = fy * y / z + cy  # Same v (epipolar constraint)
        
        # ===== Constraint 5: Disparity (left center in right image) =====
        # This is the stereo correspondence constraint
        ul_prime_pred = fx * (x - b) / z + cx
        
        # ===== Constraint 6: Right box center in right image =====
        # For a symmetric box, right edge center offset
        # Simplified: use disparity-shifted center
        ur_prime_pred = u_right_pred  # Approximation
        
        # ===== Constraint 7: Perspective keypoint =====
        # The perspective keypoint is one of the visible bottom corners
        # Based on orientation, select the appropriate corner offset
        
        # Compute corner offsets based on orientation
        # For simplicity, use the front-right corner offset
        # In practice, this should use select_perspective_keypoints
        
        # Front-right corner offset from center (in camera coordinates)
        dx = (l / 2) * cos_t + (w / 2) * sin_t  # x offset
        dz = -(l / 2) * sin_t + (w / 2) * cos_t  # z offset
        
        # Corner position
        x_corner = x + dx
        z_corner = z + dz
        y_corner = y + h / 2  # Bottom corner (positive y is down)
        
        # Prevent division by zero for corner projection
        z_corner = max(z_corner, 0.1)
        
        # Project corner to image
        up_pred = fx * x_corner / z_corner + cx
        vp_pred = fy * y_corner / z_corner + cy
        
        # Compute residuals (observed - predicted)
        residuals = np.array([
            obs.ul - u_left_pred,      # 1: Left center u
            obs.vl - v_left_pred,      # 2: Left center v
            obs.ur - u_right_pred,     # 3: Right center u
            obs.vr - v_right_pred,     # 4: Right center v
            obs.ul_prime - ul_prime_pred,  # 5: Disparity
            obs.ur_prime - ur_prime_pred,  # 6: Right in right image
            obs.up - up_pred,          # 7: Perspective keypoint u
        ], dtype=np.float64)
        
        return residuals
    
    def _compute_jacobian(
        self,
        state: np.ndarray,
        obs: GeometricObservations,
        dims: tuple[float, float, float],
        calib: CalibParams,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute Jacobian matrix via numerical differentiation.
        
        Uses central differences for better accuracy:
        J[i,j] = (r(x + ε*e_j) - r(x - ε*e_j)) / (2ε)
        
        Args:
            state: Current state [x, y, z, θ]
            obs: 2D observations
            dims: (length, width, height)
            calib: Camera calibration
            eps: Step size for finite differences
            
        Returns:
            Jacobian matrix [7, 4] where entry (i,j) is ∂r_i/∂x_j
        """
        jacobian = np.zeros((7, 4), dtype=np.float64)
        
        for j in range(4):
            # Perturb state in j-th dimension
            state_plus = state.copy()
            state_plus[j] += eps
            
            state_minus = state.copy()
            state_minus[j] -= eps
            
            # Compute residuals at perturbed states
            r_plus = self._compute_residuals(state_plus, obs, dims, calib)
            r_minus = self._compute_residuals(state_minus, obs, dims, calib)
            
            # Central difference
            jacobian[:, j] = (r_plus - r_minus) / (2 * eps)
        
        return jacobian


def fallback_simple_triangulation(
    center_2d: tuple[float, float],
    disparity: float,
    calib: CalibParams,
    theta_init: float,
) -> tuple[float, float, float, float]:
    """Simple triangulation fallback when geometric solver fails.
    
    This provides a basic 3D estimate using only stereo disparity,
    without the full geometric constraint optimization.
    
    Args:
        center_2d: (u, v) center coordinates in left image (pixels)
        disparity: Stereo disparity (left_u - right_u in pixels)
        calib: Camera calibration parameters
        theta_init: Initial orientation estimate (passed through unchanged)
        
    Returns:
        Tuple of (x, y, z, theta) - 3D center and orientation
    """
    u, v = center_2d
    fx, fy, cx, cy, b = calib
    
    # Compute depth from disparity
    if disparity > 1.0:
        z = (fx * b) / disparity
    else:
        z = 50.0  # Default depth for invalid disparity
    
    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return float(x), float(y), float(z), float(theta_init)


def solve_geometric_batch(
    detections: list[dict],
    calib: Union[CalibParams, dict],
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    damping: float = 1e-3,
    fallback_on_failure: bool = True,
) -> tuple[list[dict], float]:
    """Apply geometric construction to a batch of detections.
    
    This is a convenience function that processes multiple detections,
    applying the geometric solver to each and optionally falling back
    to simple triangulation for failed cases.
    
    Args:
        detections: List of detection dicts with keys:
            - center_2d: (u, v) in left image
            - center_2d_right: (u, v) in right image (or disparity)
            - lr_distance: stereo disparity (if center_2d_right not available)
            - dimensions: (l, w, h) in meters
            - orientation: initial theta
            - bbox_2d: optional (x1, y1, x2, y2)
            - perspective_keypoint: optional (u, v) from keypoint selection
        calib: Camera calibration (CalibParams or dict with fx, fy, cx, cy, baseline)
        max_iterations: Max Gauss-Newton iterations per detection
        tolerance: Convergence tolerance
        damping: Levenberg-Marquardt damping
        fallback_on_failure: If True, use simple triangulation when solver fails
        
    Returns:
        Tuple of (refined_detections, convergence_rate):
        - refined_detections: List of detection dicts with updated center_3d, 
          orientation, and geometric_converged flag
        - convergence_rate: Fraction of detections that converged (0.0 to 1.0)
    """
    if len(detections) == 0:
        return [], 1.0
    
    # Convert calib dict to CalibParams if needed
    if isinstance(calib, dict):
        calib = CalibParams(
            fx=calib.get("fx", 721.5377),
            fy=calib.get("fy", 721.5377),
            cx=calib.get("cx", 609.5593),
            cy=calib.get("cy", 172.8540),
            baseline=calib.get("baseline", 0.54),
        )
    
    solver = GeometricConstruction(
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    
    refined = []
    
    for det in detections:
        # Extract 2D center
        center_2d = det.get("center_2d", (0, 0))
        u_left, v_left = center_2d
        
        # Get right image center or compute from disparity
        if "center_2d_right" in det:
            u_right, v_right = det["center_2d_right"]
        else:
            # Use lr_distance as disparity
            disparity = det.get("lr_distance", 0)
            u_right = u_left - disparity
            v_right = v_left  # Epipolar constraint
        
        # Compute disparity for ul_prime
        disparity = u_left - u_right
        
        # Get perspective keypoint (optional)
        if "perspective_keypoint" in det:
            up, vp = det["perspective_keypoint"]
        else:
            # Default to center if no keypoint available
            up, vp = u_left, v_left
        
        # Build observations
        observations = GeometricObservations(
            ul=u_left,
            vl=v_left,
            ur=u_right,
            vr=v_right,
            ul_prime=u_left - disparity,  # Left center in right image
            ur_prime=u_right - disparity,  # Approximation
            up=up,
            vp=vp,
        )
        
        # Get dimensions and orientation
        dimensions = det.get("dimensions", (3.89, 1.73, 1.52))  # Default car dims
        theta_init = det.get("orientation", 0.0)
        
        # Solve
        x, y, z, theta, converged = solver.solve(
            observations=observations,
            dimensions=dimensions,
            theta_init=theta_init,
            calib=calib,
        )
        
        # Fallback if needed
        if not converged and fallback_on_failure:
            x, y, z, theta = fallback_simple_triangulation(
                center_2d=center_2d,
                disparity=disparity,
                calib=calib,
                theta_init=theta_init,
            )
        
        # Update detection
        refined_det = det.copy()
        refined_det["center_3d"] = (x, y, z)
        refined_det["orientation"] = theta
        refined_det["geometric_converged"] = converged
        refined.append(refined_det)
    
    convergence_rate = solver.convergence_rate
    
    return refined, convergence_rate


# Convenience function for single detection
def solve_geometric_single(
    center_2d: tuple[float, float],
    disparity: float,
    dimensions: tuple[float, float, float],
    theta_init: float,
    calib: Union[CalibParams, dict],
    perspective_keypoint: Optional[tuple[float, float]] = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    damping: float = 1e-3,
) -> tuple[float, float, float, float, bool]:
    """Solve geometric construction for a single detection.
    
    Simplified interface for single detection processing.
    
    Args:
        center_2d: (u, v) center in left image
        disparity: Stereo disparity in pixels
        dimensions: (length, width, height) in meters
        theta_init: Initial orientation estimate
        calib: Camera calibration
        perspective_keypoint: Optional (u, v) keypoint
        max_iterations: Max solver iterations
        tolerance: Convergence tolerance
        damping: LM damping factor
        
    Returns:
        (x, y, z, theta, converged) tuple
    """
    if isinstance(calib, dict):
        calib = CalibParams(
            fx=calib.get("fx", 721.5377),
            fy=calib.get("fy", 721.5377),
            cx=calib.get("cx", 609.5593),
            cy=calib.get("cy", 172.8540),
            baseline=calib.get("baseline", 0.54),
        )
    
    u_left, v_left = center_2d
    u_right = u_left - disparity
    v_right = v_left
    
    if perspective_keypoint is None:
        up, vp = u_left, v_left
    else:
        up, vp = perspective_keypoint
    
    observations = GeometricObservations(
        ul=u_left,
        vl=v_left,
        ur=u_right,
        vr=v_right,
        ul_prime=u_right,
        ur_prime=u_right,
        up=up,
        vp=vp,
    )
    
    solver = GeometricConstruction(
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
    )
    
    return solver.solve(
        observations=observations,
        dimensions=dimensions,
        theta_init=theta_init,
        calib=calib,
    )
