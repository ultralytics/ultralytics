# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""3D geometry utilities for stereo vision."""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.data.stereo.calib import CalibrationParameters


def estimate_depth(disparity: float | torch.Tensor, calib: CalibrationParameters) -> float | torch.Tensor:
    """Estimate depth from stereo disparity.

    Uses formula: Z = (f × B) / disparity

    Args:
        disparity: Stereo disparity (x_left - x_right) in pixels.
        calib: Calibration parameters containing focal length and baseline.

    Returns:
        float | torch.Tensor: Depth Z in meters.
    """
    if isinstance(disparity, torch.Tensor):
        depth = (calib.fx * calib.baseline) / (disparity + 1e-8)  # Add small epsilon to avoid division by zero
        return depth
    else:
        depth = (calib.fx * calib.baseline) / (disparity + 1e-8)
        return depth


def convert_to_3d(
    u: float | torch.Tensor,
    v: float | torch.Tensor,
    depth: float | torch.Tensor,
    calib: CalibrationParameters,
) -> tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor]:
    """Convert pixel coordinates to 3D camera coordinates.

    Uses formulas:
        X = (u - cx) × Z / fx
        Y = (v - cy) × Z / fy
        Z = depth

    Args:
        u: Pixel x coordinate.
        v: Pixel y coordinate.
        depth: Depth Z in meters.
        calib: Calibration parameters.

    Returns:
        tuple: 3D coordinates (X, Y, Z) in meters.
    """
    if isinstance(u, torch.Tensor):
        X = (u - calib.cx) * depth / calib.fx
        Y = (v - calib.cy) * depth / calib.fy
        return X, Y, depth
    else:
        X = (u - calib.cx) * depth / calib.fx
        Y = (v - calib.cy) * depth / calib.fy
        return X, Y, depth

