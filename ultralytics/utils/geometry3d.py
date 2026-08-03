# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Shared projective-geometry helpers for monocular 3D detection."""

from __future__ import annotations

import math

import numpy as np
import torch

from ultralytics.utils.torch_utils import autocast

DEFAULT_KITTI_P2 = np.array(
    [[721.5377, 0.0, 609.5593, 0.0], [0.0, 721.5377, 172.854, 0.0], [0.0, 0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def image_transform_matrix(
    scale_x: float = 1.0, scale_y: float = 1.0, translate_x: float = 0.0, translate_y: float = 0.0
) -> np.ndarray:
    """Return a 3x3 image-plane scale/translation homography."""
    return np.array([[scale_x, 0.0, translate_x], [0.0, scale_y, translate_y], [0.0, 0.0, 1.0]], dtype=np.float64)


def transform_projection(p2: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply an image-plane homography to a 3x4 camera projection matrix."""
    p2 = np.asarray(p2, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if p2.shape != (3, 4) or transform.shape != (3, 3):
        raise ValueError(f"Expected P2=(3, 4) and H=(3, 3), got {p2.shape} and {transform.shape}")
    return transform @ p2


def project_points(points: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Project Nx3 camera-coordinate points with the complete 3x4 projection matrix."""
    points = np.asarray(points, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    points_h = np.concatenate((points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)), axis=-1)
    projected = points_h @ p2.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[..., :2] / projected[..., 2:3]


def backproject_points(uv: np.ndarray, depth: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Recover camera XYZ from image coordinates and camera-axis Z using a complete 3x4 P2 matrix."""
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    p2 = np.asarray(p2, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) != len(depth) or p2.shape != (3, 4):
        raise ValueError(f"Expected uv=(N,2), depth=(N,), P2=(3,4), got {uv.shape}, {depth.shape}, {p2.shape}")
    u, v = uv[:, 0], uv[:, 1]
    a = np.stack(
        (
            np.stack((p2[0, 0] - u * p2[2, 0], p2[0, 1] - u * p2[2, 1]), axis=1),
            np.stack((p2[1, 0] - v * p2[2, 0], p2[1, 1] - v * p2[2, 1]), axis=1),
        ),
        axis=1,
    )
    b = np.stack(
        (
            -(p2[0, 2] - u * p2[2, 2]) * depth - (p2[0, 3] - u * p2[2, 3]),
            -(p2[1, 2] - v * p2[2, 2]) * depth - (p2[1, 3] - v * p2[2, 3]),
        ),
        axis=1,
    )
    xy = np.linalg.solve(a, b[..., None]).squeeze(-1)
    return np.column_stack((xy, depth))


def project_points_torch(points: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`project_points`, supporting batched points and one P2 per sample."""
    original_dtype = points.dtype
    # Explicit casts alone are insufficient inside an outer AMP context: bmm would still autocast these operands back
    # to FP16, where ordinary KITTI terms such as focal_length * 100m exceed 65504 before perspective division.
    with autocast(enabled=False, device=points.device.type):
        points_f = points.float()
        p2_f = p2.float()
        if p2_f.ndim == 2:
            p2_f = p2_f.unsqueeze(0).expand(points_f.shape[0], -1, -1)
        points_h = torch.cat((points_f, torch.ones_like(points_f[..., :1])), dim=-1)
        if points_h.ndim == 2:
            projected = torch.bmm(p2_f, points_h.unsqueeze(-1)).squeeze(-1)
        else:
            projected = torch.matmul(points_h, p2_f.transpose(-1, -2))
        uv = projected[..., :2] / projected[..., 2:3].clamp_min(1e-12)
    return uv.to(original_dtype)


def backproject_points_torch(uv: torch.Tensor, depth: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`backproject_points`, supporting one P2 per point."""
    original_dtype = uv.dtype
    with autocast(enabled=False, device=uv.device.type):
        uv_f, depth_f, p2_f = uv.float(), depth.float().reshape(-1), p2.float()
        if p2_f.ndim == 2:
            p2_f = p2_f.unsqueeze(0).expand(uv_f.shape[0], -1, -1)
        u, v = uv_f[:, 0], uv_f[:, 1]
        a = torch.stack(
            (
                torch.stack((p2_f[:, 0, 0] - u * p2_f[:, 2, 0], p2_f[:, 0, 1] - u * p2_f[:, 2, 1]), 1),
                torch.stack((p2_f[:, 1, 0] - v * p2_f[:, 2, 0], p2_f[:, 1, 1] - v * p2_f[:, 2, 1]), 1),
            ),
            1,
        )
        b = torch.stack(
            (
                -(p2_f[:, 0, 2] - u * p2_f[:, 2, 2]) * depth_f - (p2_f[:, 0, 3] - u * p2_f[:, 2, 3]),
                -(p2_f[:, 1, 2] - v * p2_f[:, 2, 2]) * depth_f - (p2_f[:, 1, 3] - v * p2_f[:, 2, 3]),
            ),
            1,
        )
        xy = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        xyz = torch.cat((xy, depth_f[:, None]), 1)
    return xyz.to(original_dtype)


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles in radians to the half-open interval [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def encode_alpha_multibin(alpha: torch.Tensor, num_bins: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode observation angle ``alpha`` as a nearest-bin class and a bounded residual."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    bin_size = 2.0 * math.pi / num_bins
    wrapped = torch.remainder(alpha, 2.0 * math.pi)
    bin_index = torch.floor(torch.remainder(wrapped + bin_size * 0.5, 2.0 * math.pi) / bin_size).long()
    bin_center = bin_index.to(alpha.dtype) * bin_size
    residual = wrap_angle_torch(alpha - bin_center)
    return bin_index, residual


def decode_alpha_multibin(bin_logits: torch.Tensor, residual_logits: torch.Tensor) -> torch.Tensor:
    """Decode MultiBin logits using the winning bin and a tanh-bounded within-bin residual."""
    if bin_logits.shape != residual_logits.shape or bin_logits.shape[-1] < 2:
        raise ValueError(
            f"Expected matching (..., num_bins>=2) tensors, got {bin_logits.shape} and {residual_logits.shape}"
        )
    num_bins = bin_logits.shape[-1]
    bin_size = 2.0 * math.pi / num_bins
    bin_index = bin_logits.argmax(-1, keepdim=True)
    residual = residual_logits.gather(-1, bin_index).squeeze(-1).tanh() * (bin_size * 0.5)
    alpha = bin_index.squeeze(-1).to(bin_logits.dtype) * bin_size + residual
    return wrap_angle_torch(alpha)


def boxes3d_corners_torch(
    centers: torch.Tensor, dimensions_hwl: torch.Tensor, rotation_y: torch.Tensor
) -> torch.Tensor:
    """Return eight camera-coordinate corners for boxes parameterized by center, (h,w,l), and KITTI rotation_y."""
    if centers.shape[-1] != 3 or dimensions_hwl.shape[-1] != 3:
        raise ValueError(f"Expected centers/dimensions ending in 3, got {centers.shape} and {dimensions_hwl.shape}")
    if centers.shape[:-1] != dimensions_hwl.shape[:-1]:
        raise ValueError(f"Center and dimension batch shapes differ: {centers.shape} vs {dimensions_hwl.shape}")

    h, w, length = dimensions_hwl.unbind(-1)
    signs_x = centers.new_tensor((1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0))
    signs_y = centers.new_tensor((-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0))
    signs_z = centers.new_tensor((1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0))
    local_x = length[..., None] * 0.5 * signs_x
    local_y = h[..., None] * 0.5 * signs_y
    local_z = w[..., None] * 0.5 * signs_z

    rotation_y = rotation_y.squeeze(-1) if rotation_y.ndim == centers.ndim else rotation_y
    cos_y, sin_y = rotation_y.cos()[..., None], rotation_y.sin()[..., None]
    rotated_x = local_x * cos_y + local_z * sin_y
    rotated_z = -local_x * sin_y + local_z * cos_y
    return torch.stack(
        (
            rotated_x + centers[..., 0:1],
            local_y + centers[..., 1:2],
            rotated_z + centers[..., 2:3],
        ),
        dim=-1,
    )


def paired_boxes3d_iou_torch(
    centers_a: torch.Tensor,
    dims_a: torch.Tensor,
    yaws_a: torch.Tensor,
    centers_b: torch.Tensor,
    dims_b: torch.Tensor,
    yaws_b: torch.Tensor,
) -> torch.Tensor:
    """Return exact rotated 3D IoU for aligned camera-coordinate boxes using only Torch operations.

    Boxes use geometric centers, KITTI ``(h, w, l)`` dimensions, and yaw around the camera Y axis. The BEV intersection
    is the convex hull of at most 24 fixed candidates: eight contained corners and sixteen pairwise edge intersections.
    Computation remains FP32 inside autocast so the result is suitable as a stable quality target. Invalid or
    non-positive boxes receive IoU zero.
    """
    if centers_a.shape != dims_a.shape or centers_b.shape != dims_b.shape or centers_a.shape != centers_b.shape:
        raise ValueError(
            f"Expected matching (..., 3) center/dimension tensors, got {centers_a.shape}, {dims_a.shape}, "
            f"{centers_b.shape}, and {dims_b.shape}"
        )
    if centers_a.ndim < 1 or centers_a.shape[-1] != 3:
        raise ValueError(f"Expected centers and dimensions ending in 3, got {centers_a.shape}")
    prefix = centers_a.shape[:-1]
    if yaws_a.shape == (*prefix, 1):
        yaws_a = yaws_a.squeeze(-1)
    if yaws_b.shape == (*prefix, 1):
        yaws_b = yaws_b.squeeze(-1)
    if yaws_a.shape != prefix or yaws_b.shape != prefix:
        raise ValueError(f"Expected yaw shape {prefix} or {(*prefix, 1)}, got {yaws_a.shape} and {yaws_b.shape}")
    tensors = (centers_a, dims_a, yaws_a, centers_b, dims_b, yaws_b)
    if any(tensor.device != centers_a.device for tensor in tensors[1:]):
        raise ValueError("All paired 3D box tensors must be on the same device")

    output_shape = prefix
    with autocast(enabled=False, device=centers_a.device.type):
        ca, da = centers_a.float().reshape(-1, 3), dims_a.float().reshape(-1, 3)
        cb, db = centers_b.float().reshape(-1, 3), dims_b.float().reshape(-1, 3)
        ya, yb = yaws_a.float().reshape(-1), yaws_b.float().reshape(-1)

        valid_a = torch.isfinite(ca).all(-1) & torch.isfinite(da).all(-1) & torch.isfinite(ya) & (da > 0).all(-1)
        valid_b = torch.isfinite(cb).all(-1) & torch.isfinite(db).all(-1) & torch.isfinite(yb) & (db > 0).all(-1)
        valid = valid_a & valid_b
        ca = torch.where(valid_a[:, None], ca, torch.zeros_like(ca))
        cb = torch.where(valid_b[:, None], cb, torch.zeros_like(cb))
        da = torch.where(valid_a[:, None], da, torch.ones_like(da))
        db = torch.where(valid_b[:, None], db, torch.ones_like(db))
        ya = torch.where(valid_a, ya, torch.zeros_like(ya))
        yb = torch.where(valid_b, yb, torch.zeros_like(yb))

        def bev_corners(centers: torch.Tensor, dims: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
            """Build counter-clockwise footprint corners in camera (x, z)."""
            signs_x = centers.new_tensor((-1.0, 1.0, 1.0, -1.0))
            signs_z = centers.new_tensor((-1.0, -1.0, 1.0, 1.0))
            local_x = dims[:, 2:3] * 0.5 * signs_x
            local_z = dims[:, 1:2] * 0.5 * signs_z
            cosine, sine = yaws.cos()[:, None], yaws.sin()[:, None]
            return torch.stack(
                (
                    local_x * cosine + local_z * sine + centers[:, 0:1],
                    -local_x * sine + local_z * cosine + centers[:, 2:3],
                ),
                -1,
            )

        def contained(
            points: torch.Tensor,
            centers: torch.Tensor,
            dims: torch.Tensor,
            yaws: torch.Tensor,
        ) -> torch.Tensor:
            """Test footprint points against paired rotated rectangles, including their boundary."""
            delta_x = points[..., 0] - centers[:, None, 0]
            delta_z = points[..., 1] - centers[:, None, 2]
            cosine, sine = yaws.cos()[:, None], yaws.sin()[:, None]
            local_x = delta_x * cosine - delta_z * sine
            local_z = delta_x * sine + delta_z * cosine
            tolerance = 1e-6 * dims[:, 1:].amax(-1, keepdim=True).clamp_min(1.0)
            return (local_x.abs() <= dims[:, None, 2] * 0.5 + tolerance) & (
                local_z.abs() <= dims[:, None, 1] * 0.5 + tolerance
            )

        corners_a, corners_b = bev_corners(ca, da, ya), bev_corners(cb, db, yb)
        inside_a, inside_b = (
            contained(corners_a, cb, db, yb),
            contained(corners_b, ca, da, ya),
        )

        start_a, start_b = corners_a[:, :, None, :], corners_b[:, None, :, :]
        edge_a = (corners_a.roll(-1, 1) - corners_a)[:, :, None, :]
        edge_b = (corners_b.roll(-1, 1) - corners_b)[:, None, :, :]
        offset = start_b - start_a

        def cross_2d(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            """Return the scalar 2D cross product over the final coordinate axis."""
            return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

        denominator = cross_2d(edge_a, edge_b)
        scale = (edge_a.square().sum(-1).sqrt() * edge_b.square().sum(-1).sqrt()).clamp_min(1.0)
        non_parallel = denominator.abs() > 1e-7 * scale
        safe_denominator = torch.where(non_parallel, denominator, torch.ones_like(denominator))
        parameter_a = cross_2d(offset, edge_b) / safe_denominator
        parameter_b = cross_2d(offset, edge_a) / safe_denominator
        intersects = (
            non_parallel
            & (parameter_a >= -1e-6)
            & (parameter_a <= 1.0 + 1e-6)
            & (parameter_b >= -1e-6)
            & (parameter_b <= 1.0 + 1e-6)
        )
        crossings = start_a + parameter_a[..., None] * edge_a

        candidates = torch.cat((corners_a, corners_b, crossings.flatten(1, 2)), 1)
        candidate_mask = torch.cat((inside_a, inside_b, intersects.flatten(1)), 1) & valid[:, None]
        count = candidate_mask.sum(1)
        centroid = (candidates * candidate_mask[..., None]).sum(1) / count.clamp_min(1)[:, None]
        angles = torch.atan2(
            candidates[..., 1] - centroid[:, None, 1],
            candidates[..., 0] - centroid[:, None, 0],
        )
        angles = torch.where(candidate_mask, angles, angles.new_full(angles.shape, 4.0 * math.pi))
        order = angles.argsort(1)
        ordered = candidates.gather(1, order[..., None].expand(-1, -1, 2))

        positions = torch.arange(candidates.shape[1], device=candidates.device)[None]
        next_positions = torch.where(positions == count[:, None] - 1, 0, positions + 1).clamp_max(
            candidates.shape[1] - 1
        )
        following = ordered.gather(1, next_positions[..., None].expand(-1, -1, 2))
        polygon_area = 0.5 * (cross_2d(ordered, following) * (positions < count[:, None])).sum(1).abs()

        lower = torch.maximum(ca[:, 1] - da[:, 0] * 0.5, cb[:, 1] - db[:, 0] * 0.5)
        upper = torch.minimum(ca[:, 1] + da[:, 0] * 0.5, cb[:, 1] + db[:, 0] * 0.5)
        intersection = polygon_area * (upper - lower).clamp_min(0.0)
        union = da.prod(-1) + db.prod(-1) - intersection
        iou = (intersection / union.clamp_min(torch.finfo(torch.float32).eps)).clamp(0.0, 1.0)
        iou = torch.where(valid & torch.isfinite(iou) & (union > 0.0), iou, torch.zeros_like(iou))
    return iou.reshape(output_shape)
