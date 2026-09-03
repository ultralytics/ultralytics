# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = "inverse_sigmoid", "multi_scale_deformable_attn_pytorch"


def _get_clones(module, n):
    """Create a list of cloned modules from the given module.

    Args:
        module (nn.Module): The module to be cloned.
        n (int): Number of clones to create.

    Returns:
        (nn.ModuleList): A ModuleList containing n clones of the input module.

    Examples:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value.

    This function calculates the bias initialization value based on a prior probability using the inverse sigmoid
    (logit)
    function. It's commonly used in object detection models to initialize classification layers with a specific positive
    prediction probability.

    Args:
        prior_prob (float, optional): Prior probability for bias initialization.

    Returns:
        (float): Bias initialization value calculated from the prior probability.

    Examples:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Bias initialization value: {bias:.4f}")
        Bias initialization value: -4.5951
    """
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor.

    This function applies the inverse of the sigmoid function to a tensor, which is useful in various neural network
    operations, particularly in attention mechanisms and coordinate transformations.

    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1].
        eps (float, optional): Small epsilon value to prevent numerical instability.

    Returns:
        (torch.Tensor): Tensor after applying the inverse sigmoid function.

    Examples:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: list,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int] | None = None,
) -> torch.Tensor:
    """Implement multi-scale deformable attention in PyTorch.

    Folds the (num_levels, num_points) axes into a single num_total_points axis so every traced tensor stays at rank <=
    5, the maximum rank supported by CoreML's MIL converter. Numerically equivalent to the rank-6 reference
    implementation on CUDA and CPU. When num_points_list is provided, supports variable points-per-level used by
    DEIM/D-FINE-style decoders (e.g. ndp=[3, 6, 3]).

    Args:
        value (torch.Tensor): Value tensor with shape (bs, num_keys, num_heads, embed_dims).
        value_spatial_shapes (list): Per-level spatial shapes as [(H_0, W_0), ..., (H_{L-1}, W_{L-1})].
        sampling_locations (torch.Tensor): Sampling locations with shape (bs, num_queries, num_heads, num_total_points,
            2). For uniform points-per-level, num_total_points = num_levels * num_points; for variable, it
            equals sum(num_points_list).
        attention_weights (torch.Tensor): Attention weights with shape (bs, num_queries, num_heads, num_total_points).
        num_points_list (list[int], optional): Number of sampling points per level. When None, points-per-level is
            inferred as num_total_points // num_levels (uniform split). When given, enables variable splits.

    Returns:
        (torch.Tensor): Output tensor with shape (bs, num_queries, num_heads * embed_dims).

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_total_points, _ = sampling_locations.shape

    # (bs, num_keys, num_heads, embed_dims) -> (bs*num_heads, embed_dims, H*W)
    value = value.permute(0, 2, 3, 1).flatten(0, 1)
    # Map to grid_sample coords in [-1, 1]: (bs*num_heads, num_queries, num_total_points, 2)
    sampling_grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    if len(value_spatial_shapes) == 1:
        value_list, sampling_grids = (value,), (sampling_grids,)
    else:
        split_sizes = num_points_list if num_points_list is not None else num_total_points // len(value_spatial_shapes)
        value_list = value.split([h * w for h, w in value_spatial_shapes], dim=-1)
        sampling_grids = sampling_grids.split(split_sizes, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * num_heads, embed_dims, h, w)
        sampling_value_list.append(
            F.grid_sample(value_l, sampling_grids[level], mode="bilinear", padding_mode="zeros", align_corners=False)
        )
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * num_heads, 1, num_queries, num_total_points)
    output = (
        (torch.cat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# D-FINE distribution-based box regression helpers, used by the DeimDecoder head and DfineLoss.
def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        x (torch.Tensor): Boxes in xyxy format with shape (..., 4).

    Returns:
        (torch.Tensor): Boxes in xywh format with shape (..., 4).

    Notes:
        Kept separate from ultralytics.utils.ops.xyxy2xywh, which allocates with empty_like and assigns by index.
        This builds the result with unbind and stack instead, which stays traceable for CoreML export.
    """
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """Generate the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (torch.Tensor): Controls upper bounds of the sequence, where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function. Larger values result in flatter weights
            near the central axis W(reg_max/2)=0 and steeper weights at both ends.
        deploy (bool): If True, uses deployment mode settings.

    Returns:
        (torch.Tensor): Non-uniform bin centers with shape (reg_max + 1,).
    References:
        https://github.com/Peterande/D-FINE
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.cat(values, 0)


def translate_gt(gt, reg_max, reg_scale, up):
    """Decode bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used for regression tasks in
    object detection models. It calculates the indices of the closest bins to each GT value and assigns interpolation
    weights to these bins based on their proximity to the GT value.

    Args:
        gt (torch.Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (torch.Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        indices (torch.Tensor): Index of the left bin closest to each GT value, shape (N,).
        weight_right (torch.Tensor): Weight assigned to the right bin, shape (N,).
        weight_left (torch.Tensor): Weight assigned to the left bin, shape (N,).
    References:
        https://github.com/Peterande/D-FINE
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    invalid_idx_mask_neg = (indices < 0)
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = (indices >= reg_max)
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(points, distance, reg_scale):
    """Decode edge-distances into bounding box coordinates.

    Args:
        points (torch.Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h], where (x, y) is the center and (w,
            h) are width and height.
        distance (torch.Tensor): (B, N, 4) or (N, 4), representing distances from the point to the left, top, right, and
            bottom boundaries.
        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        (torch.Tensor): Boxes in xywh format with the same leading dimensions as points.
    References:
        https://github.com/Peterande/D-FINE
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """Convert bounding box coordinates to distances from a reference point.

    Args:
        points (torch.Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (torch.Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (torch.Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        four_lens (torch.Tensor): Flattened bin targets clamped to [0, reg_max - eps].
        weight_right (torch.Tensor): Weight assigned to the right bin.
        weight_left (torch.Tensor): Weight assigned to the left bin.
    References:
        https://github.com/Peterande/D-FINE
    """
    reg_scale = abs(reg_scale)
    left   = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top    = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right  = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max-eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
