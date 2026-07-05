# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

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


def linear_init(module):
    """Initialize the weights and biases of a linear module.

    This function initializes the weights of a linear module using a uniform distribution within bounds calculated from
    the output dimension. If the module has a bias, it is also initialized.

    Args:
        module (nn.Module): Linear module to initialize.

    Examples:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


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
) -> torch.Tensor:
    """Implement multi-scale deformable attention in PyTorch.

    Folds the (num_levels, num_points) axes into a single num_total_points axis so every traced tensor stays at rank <=
    5, the maximum rank supported by CoreML's MIL converter. Numerically equivalent to the rank-6 reference
    implementation on CUDA and CPU.

    Args:
        value (torch.Tensor): Value tensor with shape (bs, num_keys, num_heads, embed_dims).
        value_spatial_shapes (list): Per-level spatial shapes as [(H_0, W_0), ..., (H_{L-1}, W_{L-1})].
        sampling_locations (torch.Tensor): Sampling locations with shape (bs, num_queries, num_heads, num_levels *
            num_points, 2).
        attention_weights (torch.Tensor): Attention weights with shape (bs, num_queries, num_heads, num_levels *
            num_points).

    Returns:
        (torch.Tensor): Output tensor with shape (bs, num_queries, num_heads * embed_dims).

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_total_points, _ = sampling_locations.shape
    num_points = num_total_points // len(value_spatial_shapes)

    # (bs, num_keys, num_heads, embed_dims) -> tuple of (bs*num_heads, embed_dims, H*W) per level
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split([h * w for h, w in value_spatial_shapes], dim=-1)
    # Map to grid_sample coords in [-1, 1] and split per level: tuple of (bs*num_heads, num_queries, num_points, 2)
    sampling_grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(num_points, dim=-2)

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
