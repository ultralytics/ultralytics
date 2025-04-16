# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

__all__ = "multi_scale_deformable_attn_pytorch", "inverse_sigmoid"


def _get_clones(module, n):
    """
    Create a list of cloned modules from the given module.

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
    """
    Initialize conv/fc bias value according to a given probability value.

    This function calculates the bias initialization value based on a prior probability using the inverse error function.
    It's commonly used in object detection models to initialize classification layers with a specific positive prediction
    probability.

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
    """
    Initialize the weights and biases of a linear module.

    This function initializes the weights of a linear module using a uniform distribution within bounds calculated
    from the input dimension. If the module has a bias, it is also initialized.

    Args:
        module (nn.Module): Linear module to initialize.

    Returns:
        (nn.Module): The initialized module.

    Examples:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> initialized_linear = linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """
    Calculate the inverse sigmoid function for a tensor.

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
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list=None,
    method="default",
    value_shape="default",
) -> torch.Tensor:
    """
    Implement multi-scale deformable attention in PyTorch.

    This function performs deformable attention across multiple feature map scales, allowing the model to attend to
    different spatial locations with learned offsets.

    Args:
        value (torch.Tensor): The value tensor with shape (bs, num_keys, num_heads, embed_dims).
        value_spatial_shapes (torch.Tensor): Spatial shapes of the value tensor with shape (num_levels, 2).
        sampling_locations (torch.Tensor): The sampling locations with shape
            (bs, num_queries, num_heads, num_levels, num_points, 2).
        attention_weights (torch.Tensor): The attention weights with shape
            (bs, num_queries, num_heads, num_levels, num_points).

    Returns:
        (torch.Tensor): The output tensor with shape (bs, num_queries, embed_dims).

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, n_head, c = value.shape
    len_q = sampling_locations.shape[1]
    # (bs, len_v, n_head, c) -> (bs, n_head, c, len_v) -> (bs*n_head, c, len_v)
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split([h * w for h, w in value_spatial_shapes], dim=-1)

    sampling_grids = 2 * sampling_locations - 1
    # (bs, len_q, n_head, n_levels*n_points, 2) ->
    # (bs, n_head, len_q, n_levels*n_points, 2) ->
    # (bs*n_head, len_q, n_levels*n_points, 2)
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]
        # bs*n_head, embed_dims, num_queries, num_points
        sampling_value_list.append(
            F.grid_sample(value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
        )
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * n_head, 1, len_q, sum(num_points_list))
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, n_head * c, len_q)
    )
    return output.transpose(1, 2).contiguous()
