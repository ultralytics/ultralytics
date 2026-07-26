# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    import MultiScaleDeformableAttention as MSDA
except ImportError:
    MSDA = None

__all__ = (
    "inverse_sigmoid",
    "multi_scale_deformable_attn_pytorch",
    "deformable_attention_core_func_v2",
    "MSDeformAttnFunction",
)


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


def gen_sineembed_for_position(pos_tensor, length=128, temperature=10000):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(length, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / length)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 3:
        z_embed = pos_tensor[:, :, 2] * scale
        pos_z = z_embed[:, :, None] / dim_t
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_z), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


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


def deformable_attention_core_func_v2(
    value: torch.Tensor | tuple[torch.Tensor, ...],
    value_spatial_shapes: torch.Tensor | list,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int],
    method: str = "default",
    value_shape: str = "default",
) -> torch.Tensor:
    """Alternative deformable attention core used by D-FINE/DEIM-style attention.

    Args:
        value (torch.Tensor | tuple[torch.Tensor, ...]): Either per-level tensors or a flattened RT-DETR tensor.
        value_spatial_shapes (torch.Tensor | list): Spatial shapes per feature level.
        sampling_locations (torch.Tensor): Shape [bs, query_length, n_head, n_levels * n_points, 2].
        attention_weights (torch.Tensor): Shape [bs, query_length, n_head, n_levels * n_points].
        num_points_list (list[int]): Number of sampling points per level.
        method (str): Sampling method. Only "default" is used here.
        value_shape (str): "default" for per-level tensors, "reshape" for flattened RT-DETR value tensors.

    Returns:
        (torch.Tensor): Output tensor with shape [bs, query_length, n_head * head_dim].
    """
    if value_shape == "default":
        bs, n_head, c, _ = value[0].shape
    elif value_shape == "reshape":
        bs, _, n_head, c = value.shape
        split_shape = [h * w for h, w in value_spatial_shapes]
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    else:
        raise ValueError(f"Unsupported value_shape: {value_shape}")

    _, len_q, _, _, _ = sampling_locations.shape

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError(f"Unsupported deformable attention method: {method}")

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        else:
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, len_q * num_points_list[level], 2)
            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, len_q)
    return output.permute(0, 2, 1)


class MSDeformAttnFunction(Function):
    @staticmethod
    def symbolic(
        g,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        spatial_shapes = g.op("Cast", value_spatial_shapes, to_i=6)
        level_start_index = g.op("Cast", value_level_start_index, to_i=6)
        output = g.op(
            "com.nvidia::MultiscaleDeformableAttnPlugin_TRT",
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            plugin_version_s="1",
            plugin_namespace_s="",
        )
        value_sizes = value.type().sizes()
        loc_sizes = sampling_locations.type().sizes()
        n = value_sizes[0] if value_sizes else None
        lq = loc_sizes[1] if loc_sizes and len(loc_sizes) > 1 else None
        m = value_sizes[2] if value_sizes and len(value_sizes) > 2 else None
        d = value_sizes[3] if value_sizes and len(value_sizes) > 3 else None
        c = m * d if isinstance(m, int) and isinstance(d, int) else None
        output.setType(value.type().with_sizes([n, lq, c]))
        return output

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index,
                              sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n 

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )


def freeze_batch_norm2d(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.BatchNorm2d):
        frozen = FrozenBatchNorm2d(module.num_features, eps=module.eps)
        frozen.load_state_dict(module.state_dict(), strict=True)
        module = frozen
    else:
        for name, child in module.named_children():
            _child = freeze_batch_norm2d(child)
            if _child is not child:
                setattr(module, name, _child)
    return module


def freeze_norm_layers(module: nn.Module) -> nn.Module:
    """Freeze all normalization layers (BatchNorm2d, LayerNorm, GroupNorm) in a module.

    BatchNorm2d layers are replaced with FrozenBatchNorm2d (frozen running stats + params).
    LayerNorm/GroupNorm layers have their affine parameters frozen via requires_grad_(False).
    """
    if isinstance(module, nn.BatchNorm2d):
        frozen = FrozenBatchNorm2d(module.num_features, eps=module.eps)
        frozen.load_state_dict(module.state_dict(), strict=True)
        return frozen

    if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        module.requires_grad_(False)
        return module

    for name, child in module.named_children():
        _child = freeze_norm_layers(child)
        if _child is not child:
            setattr(module, name, _child)
    return module
