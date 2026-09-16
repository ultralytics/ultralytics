# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth encoding for the 3D pose task.

YOLO label files are validated with `assert lb.min() >= -0.01`, so a raw signed root-relative depth in metres
cannot be written to disk. Both depth channels are therefore stored encoded to roughly [0, 1], the same range the
x/y columns already live in, and decoded back to metres only at the edges of the system (validation, prediction).

The root joint (last keypoint) carries absolute depth; every other joint carries depth relative to it.
"""

from __future__ import annotations

import numpy as np
import torch

Z_REL_RANGE = 2.0  # root-relative depth is clipped to +/- this many metres
Z_ROOT_MAX = 50.0  # absolute root depth is clipped to this many metres


def encode_z(z_rel, z_root):
    """Encode metric depths to the [0, 1] range stored in label files.

    Args:
        z_rel (np.ndarray | torch.Tensor): Root-relative depth in metres, any shape.
        z_root (np.ndarray | torch.Tensor): Absolute root depth in metres, any shape.

    Returns:
        (tuple): Encoded (z_rel, z_root), clipped to [0, 1].
    """
    lib = torch if isinstance(z_rel, torch.Tensor) else np
    return (
        lib.clip((z_rel + Z_REL_RANGE) / (2 * Z_REL_RANGE), 0.0, 1.0),
        lib.clip(z_root / Z_ROOT_MAX, 0.0, 1.0),
    )


def decode_z(z_rel, z_root):
    """Decode [0, 1] depth channels back to metres. Inverse of `encode_z`."""
    return z_rel * (2 * Z_REL_RANGE) - Z_REL_RANGE, z_root * Z_ROOT_MAX


def keypoints_to_camera(kpts, focal: float, cx: float, cy: float):
    """Lift decoded keypoints to metric camera coordinates.

    Args:
        kpts (torch.Tensor): Keypoints of shape (..., nkpt, 4) as (x_px, y_px, visible, z_encoded), with the last
            keypoint being the root.
        focal (float): Focal length in pixels.
        cx (float): Principal point x in pixels.
        cy (float): Principal point y in pixels.

    Returns:
        (torch.Tensor): Shape (..., nkpt, 3) of metric (X, Y, Z) in camera space.
    """
    z_rel, z_root = decode_z(kpts[..., :-1, 3], kpts[..., -1:, 3])
    z = torch.cat([z_rel + z_root, z_root], dim=-1)  # absolute depth per joint
    x = (kpts[..., 0] - cx) * z / focal
    y = (kpts[..., 1] - cy) * z / focal
    return torch.stack([x, y, z], dim=-1)


def procrustes_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Similarity-align `pred` onto `gt` (rotation, scale, translation), the standard PA-MPJPE transform.

    Args:
        pred (torch.Tensor): Predicted joints, shape (N, J, 3).
        gt (torch.Tensor): Ground-truth joints, shape (N, J, 3).

    Returns:
        (torch.Tensor): `pred` mapped onto `gt`, shape (N, J, 3).
    """
    mu_p, mu_g = pred.mean(1, keepdim=True), gt.mean(1, keepdim=True)
    p, g = (pred - mu_p).double(), (gt - mu_g).double()
    var_p = (p**2).sum((1, 2)).clamp_min(1e-12)  # (N,)
    k = p.transpose(1, 2) @ g  # (N, 3, 3)
    u, _, vh = torch.linalg.svd(k)
    v = vh.transpose(1, 2)
    # Reflection guard: force det(R) = +1 so a mirrored pose is not scored as a perfect fit.
    z = torch.eye(3, dtype=p.dtype, device=p.device).expand(len(p), 3, 3).clone()
    z[:, -1, -1] = torch.sign(torch.linalg.det(u @ v.transpose(1, 2)))
    r = v @ z @ u.transpose(1, 2)  # (N, 3, 3)
    scale = torch.diagonal(r @ k, dim1=1, dim2=2).sum(-1) / var_p
    return (scale.view(-1, 1, 1) * (p @ r.transpose(1, 2)) + mu_g.double()).to(pred.dtype)
