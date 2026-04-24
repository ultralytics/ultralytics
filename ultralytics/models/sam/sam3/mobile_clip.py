# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Modified for EfficientSAM3 and ported into Ultralytics by SimonZeng7108528.

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample during training."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize DropPath with the given drop probability."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth by randomly zeroing whole samples."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            noise.div_(keep_prob)
        return x * noise


# ==============================================================================
# MobileOneBlock (from mobileclip/modules/common/mobileone.py)
# ==============================================================================


class SEBlock(nn.Module):
    """Squeeze-and-Excite channel attention block.

    Args:
        in_channels (int): Number of input channels.
        rd_ratio (float): Reduction ratio for the bottleneck. Default is 0.0625.
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Initialize SqueezeExcitation reduction layers."""
        super().__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation channel recalibration to the input feature map."""
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne re-parameterisable building block.

    At training time the block maintains multiple branches (identity skip, scale branch, and ``num_conv_branches``
    parallel conv-BN branches). After calling :meth:`reparameterize` all branches are folded into a single
    ``reparam_conv`` for fast inference.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | tuple): Kernel size for the convolution.
        stride (int): Convolution stride. Default is 1.
        padding (int): Convolution padding. Default is 0.
        dilation (int): Convolution dilation. Default is 1.
        groups (int): Convolution groups. Default is 1.
        inference_mode (bool): If True, uses a single fused conv (no branches). Default is False.
        use_se (bool): Whether to attach a SEBlock. Default is False.
        use_act (bool): Whether to apply activation. Default is True.
        use_scale_branch (bool): Whether to add a 1×1 scale branch. Default is True.
        num_conv_branches (int): Number of parallel conv-BN branches. Default is 1.
        activation (nn.Module): Activation function instance. Default is ``nn.GELU()``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """Initialize MobileOneBlock with parallel conv-BN branches."""
        super().__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.activation = activation if use_act else nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_skip = (
                nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            )

            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None

            self.rbr_scale = None
            ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            if (ks > 1) and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MobileOneBlock; uses fused reparam conv in inference mode."""
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self) -> None:
        """Fold all branches into a single ``reparam_conv`` for inference."""
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch: Union[nn.Sequential, nn.BatchNorm2d]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_size = self.kernel_size
                if isinstance(self.kernel_size, int):
                    kernel_size = (self.kernel_size, self.kernel_size)

                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, kernel_size[0], kernel_size[1]),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


# ==============================================================================
# Transformer Utils (from mobileclip/modules/common/transformer.py)
# ==============================================================================


class LayerNormFP32(nn.LayerNorm):
    """LayerNorm that casts input to float32 before normalisation and back to the original dtype.

    Args:
        normalized_shape (int | list | torch.Size): Input shape from an expected input.
        eps (float): Epsilon for numerical stability. Default is 1e-5.
        elementwise_affine (bool): Whether to learn per-element affine parameters. Default is True.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs,
    ):
        """Initialize LayerNorm2d_fp32 forwarding parameters to nn.LayerNorm."""
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            *args,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute layer normalization in float32 precision, then cast back."""
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(inp_dtype)


def get_normalization_layer(norm_type: str, num_features: int) -> nn.Module:
    """Return a normalization layer by name.

    Args:
        norm_type (str): One of ``"layer_norm"`` or ``"layer_norm_fp32"``.
        num_features (int): Number of features (channels) to normalise.

    Returns:
        (nn.Module): The requested normalization layer.
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(num_features)
    elif norm_type == "layer_norm_fp32":
        return LayerNormFP32(num_features)
    else:
        raise NotImplementedError(f"Option: {norm_type} not supported.")


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embedding with optional interpolation.

    Args:
        num_embeddings (int): Maximum sequence length (number of positions).
        embedding_dim (int): Embedding dimension per position.
        padding_idx (int | None): If set, the embedding at this index is kept zero. Default is None.
        interpolation_mode (str | None): Interpolation mode used when the requested ``seq_len`` differs from
            ``num_embeddings``. Default is ``"bilinear"``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        """Initialize learnable positional embedding parameters."""
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, num_embeddings, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.interpolation_mode = interpolation_mode
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset positional embedding weights with truncated-normal initialization."""
        nn.init.trunc_normal_(self.pos_embed, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.pos_embed[:, :, self.padding_idx, ...] = 0.0

    def forward(self, seq_len: int, *args, **kwargs) -> torch.Tensor:
        """Return positional embeddings for the given sequence length."""
        pos_embed = self.pos_embed
        if self.padding_idx is not None:
            with torch.no_grad():
                pos_embed[:, :, self.padding_idx, ...] = 0.0

        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode=self.interpolation_mode,
            )
        return pos_embed.reshape(1, seq_len, self.embedding_dim)


class PositionalEmbedding(nn.Module):
    """Wrapper that delegates to :class:`LearnablePositionalEmbedding`.

    Args:
        num_embeddings (int): Maximum sequence length.
        embedding_dim (int): Embedding dimension.
        padding_idx (int | None): Index whose embedding is kept zero. Default is None.
        is_learnable (bool): Unused, retained for API compatibility. Default is False.
        interpolation_mode (str | None): Interpolation mode for length mismatch. Default is ``"bilinear"``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        is_learnable: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        """Initialize SinusoidalLearnablePositionalEmbedding wrapper."""
        super().__init__()
        self.pos_embed = LearnablePositionalEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            interpolation_mode=interpolation_mode,
            *args,
            **kwargs,
        )

    def forward(self, seq_len: int, *args, **kwargs) -> torch.Tensor:
        """Return positional embeddings by delegating to the inner embedding module."""
        return self.pos_embed(seq_len, *args, **kwargs)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module used inside the MobileCLIP text transformer.

    Args:
        embed_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        attn_dropout (float): Dropout probability for attention weights. Default is 0.0.
        bias (bool): Whether to add bias to QKV and output projections. Default is True.
        output_dim (int | None): Output dimension. Defaults to ``embed_dim``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize MultiHeadAttention with QKV and output projection layers."""
        if output_dim is None:
            output_dim = embed_dim

        super().__init__()
        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=output_dim, bias=bias)
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute scaled dot-product multi-head attention."""
        b_sz, s_len, _ = x_q.shape

        qkv = self.qkv_proj(x_q).reshape(b_sz, s_len, 3, self.num_heads, -1)
        qkv = qkv.transpose(1, 3).contiguous()
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling
        key = key.transpose(-1, -2)
        attn = torch.matmul(query, key)
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = self.softmax(attn.float()).to(attn.dtype)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).reshape(b_sz, s_len, -1)
        return self.out_proj(out)


class TransformerEncoder(nn.Module):
    """Single transformer encoder layer with pre-norm, MHA, and FFN.

    Args:
        embed_dim (int): Model embedding dimension.
        ffn_latent_dim (int): Hidden dimension of the feed-forward network.
        num_heads (int): Number of attention heads. Default is 8.
        attn_dropout (float): Dropout on attention weights. Default is 0.0.
        dropout (float): Dropout on layer outputs. Default is 0.0.
        ffn_dropout (float): Dropout inside FFN. Default is 0.0.
        transformer_norm_layer (str): Normalization type (``"layer_norm"`` or ``"layer_norm_fp32"``). Default is
            ``"layer_norm"``.
        stochastic_dropout (float): Stochastic depth drop probability. Default is 0.0.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        stochastic_dropout: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:
        """Initialize TransformerLayer with pre-norm attention and FFN sub-layers."""
        super().__init__()
        attn_unit = MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True)
        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=embed_dim),
            attn_unit,
            nn.Dropout(p=dropout),
        )
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )
        self.drop_path = DropPath(stochastic_dropout) if stochastic_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        x_prev: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Apply pre-norm multi-head attention and feed-forward network with drop-path."""
        res = x
        x = self.pre_norm_mha[0](x)
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.drop_path(self.pre_norm_mha[2](x))
        x = x + res
        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x


# ==============================================================================
# RepMixer (from mobileclip/modules/text/repmixer.py)
# ==============================================================================


class ConvFFN(nn.Module):
    """Convolutional feed-forward network used inside :class:`RepMixerBlock`.

    Args:
        in_channels (int): Number of input channels.
        context_size (int): Context window size (kernel width for the depthwise conv).
        hidden_channels (int | None): Hidden dimension. Defaults to ``in_channels``.
        out_channels (int | None): Output channels. Defaults to ``in_channels``.
        act_layer (type): Activation class. Default is ``nn.GELU``.
        drop (float): Dropout probability. Default is 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        context_size: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialize ConvFFN with depthwise convolution and MLP layers."""
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, int(context_size)),
                padding=(0, int(context_size // 2)),
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise convolution followed by the MLP feed-forward network."""
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepMixer(nn.Module):
    """Re-parameterisable token mixing layer used inside :class:`RepMixerBlock`.

    Args:
        dim (int): Number of channels.
        kernel_size (int): Width of the 1D depthwise convolution kernel. Default is 3.
        use_layer_scale (bool): Whether to apply a learnable layer-scale parameter. Default is True.
        layer_scale_init_value (float): Initial value for the layer-scale. Default is 1e-5.
        inference_mode (bool): If True, uses a single fused conv. Default is False.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        """Initialize RepMixer; uses a single reparameterized conv in inference mode."""
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=(1, self.kernel_size),
                stride=1,
                padding=(0, self.kernel_size // 2),
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                (1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                (1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RepMixer token mixing; uses the fused reparam conv in inference mode."""
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)
        if self.use_layer_scale:
            x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
        else:
            x = x + self.mixer(x) - self.norm(x)
        return x

    def reparameterize(self) -> None:
        """Fold mixer and norm branches into a single conv for inference."""
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias)
        else:
            w = self.mixer.id_tensor + self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=(1, self.kernel_size),
            stride=1,
            padding=(0, self.kernel_size // 2),
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class RepMixerBlock(nn.Module):
    """Block combining :class:`RepMixer` token mixing with :class:`ConvFFN`.

    Args:
        dim (int): Number of channels.
        kernel_size (int): Kernel size for token mixing. Default is 11.
        mlp_ratio (float): Expansion ratio for :class:`ConvFFN`. Default is 4.0.
        act_layer (type): Activation class used in :class:`ConvFFN`. Default is ``nn.GELU``.
        drop (float): Dropout probability. Default is 0.0.
        drop_path (float): Stochastic depth rate. Default is 0.0.
        use_layer_scale (bool): Whether to use layer-scale. Default is True.
        layer_scale_init_value (float): Initial value for layer-scale. Default is 1e-5.
        inference_mode (bool): If True, uses reparameterised convs. Default is False.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 11,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize RepMixerBlock with token mixer, ConvFFN, and drop-path."""
        super().__init__()
        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            context_size=kernel_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply RepMixerBlock: token mixing and ConvFFN with optional drop-path."""
        if x.dim() == 3:
            # (B, C, D) -> (B, D, C) -> (B, D, 1, C)
            x = x.permute(0, 2, 1)
            x = torch.unsqueeze(x, dim=2)
        else:
            raise ValueError(f"Expected tensor of dim=3, obtained tensor of dim={x.dim()}")

        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))

        # Convert back: (B, D, 1, C) -> (B, C, D)
        x = x.squeeze(dim=2).permute(0, 2, 1)
        return x

    def reparameterize(self) -> None:
        """Fuse the token-mixer RepMixer branches for inference.

        Delegates to :meth:`RepMixer.reparameterize` on the inner ``token_mixer``.
        Should be called after weights are loaded and before inference.
        """
        self.token_mixer.reparameterize()


# ==============================================================================
# MobileCLIP Text Transformer (from mobileclip/text_encoder.py)
# ==============================================================================


class MobileCLIPTextTransformer(nn.Module):
    """Lightweight text transformer backbone from MobileCLIP for use as a SAM3 text encoder.

    Supports two variants controlled by ``cfg["model_name"]``:
    - ``"base"``: Pure multi-head-attention transformer.
    - ``"mct"``: MobileCLIP-T variant with RepMixer bookend blocks.

    Args:
        cfg (dict): Configuration dictionary with keys:
            - ``dim`` (int): Model dimension.
            - ``model_name`` (str): ``"base"`` or ``"mct"``.
            - ``n_transformer_layers`` (int): Number of transformer layers.
            - ``n_heads_per_layer`` (int | list[int]): Attention heads.
            - ``ffn_multiplier_per_layer`` (float | list[float]): FFN expansion factors.
            - ``norm_layer`` (str): Normalization type.
            - ``context_length`` (int): Maximum context length (positional embedding size).
            - ``vocab_size`` (int): Vocabulary size.
            - ``causal_masking`` (bool): Whether to apply causal attention masking.
            - ``no_scale_embedding`` (bool): Skip embedding scale factor. Default False.
            - ``no_pos_embedding`` (bool): Skip positional embeddings. Default False.
            - ``embed_dropout`` (float): Dropout on embeddings. Default 0.
        projection_dim (int): Dimension of the output projection layer.
        skip_embeddings (bool): If True, skip building the embedding layer and positional embedding
            (the caller supplies pre-computed embeddings). Default is False.
    """

    def __init__(self, cfg: dict, projection_dim: int, skip_embeddings: bool = False, *args, **kwargs) -> None:
        """Initialize MobileCLIPTextTransformer from a configuration dict."""
        super().__init__()

        model_dim = cfg["dim"]
        no_scale_embedding = cfg.get("no_scale_embedding", False)
        no_pos_embedding = cfg.get("no_pos_embedding", False)
        embed_dropout = cfg.get("embed_dropout", 0.0)
        norm_layer = cfg["norm_layer"]
        variant = cfg["model_name"]
        self.vocab_size = cfg["vocab_size"]
        self.projection_dim = projection_dim
        self.skip_embeddings = skip_embeddings

        if not skip_embeddings:
            self.embedding_layer = nn.Embedding(embedding_dim=model_dim, num_embeddings=self.vocab_size)
            self.embed_scale = 1.0 if no_scale_embedding else model_dim**-0.5
            context_length = cfg["context_length"]
            self.positional_embedding = (
                None
                if no_pos_embedding
                else PositionalEmbedding(num_embeddings=context_length, embedding_dim=model_dim)
            )
            self.embedding_dropout = nn.Dropout(p=embed_dropout)
        else:
            self.embedding_layer = None
            self.positional_embedding = None
            self.embedding_dropout = None

        n_transformer_layers = cfg["n_transformer_layers"]
        ffn_multipliers = cfg["ffn_multiplier_per_layer"]
        if isinstance(ffn_multipliers, (float, int)):
            ffn_multipliers = [ffn_multipliers] * n_transformer_layers

        ffn_dims = [int(math.ceil(model_dim * ffn_mult / 16.0) * 16.0) for ffn_mult in ffn_multipliers]

        mha_heads = cfg["n_heads_per_layer"]
        if isinstance(mha_heads, int):
            mha_heads = [mha_heads] * n_transformer_layers

        if variant == "base":
            self.transformer = nn.ModuleList(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
        elif variant == "mct":
            self.transformer = nn.ModuleList([RepMixerBlock(dim=model_dim)])
            self.transformer.extend(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
            self.transformer.extend([RepMixerBlock(dim=model_dim)])
        else:
            raise ValueError(f"Unrecognised text encoder variant: {variant!r}")

        self.final_layer_norm = get_normalization_layer(num_features=model_dim, norm_type=norm_layer)
        self.projection_layer = nn.Parameter(torch.empty(model_dim, self.projection_dim))
        self.model_dim = model_dim
        self.causal_masking = cfg["causal_masking"]

    def resize_pos_embed(self, new_length: int) -> None:
        """Truncate positional embeddings to *new_length* tokens.

        This is called after loading a checkpoint that was trained with a larger context length
        (e.g. 77) to shrink the embeddings to the operational context length (e.g. 16 or 32).

        Args:
            new_length (int): Target context length.
        """
        if self.positional_embedding is None:
            return

        pos_embed = self.positional_embedding.pos_embed.pos_embed
        current_length = pos_embed.shape[2]
        if new_length == current_length:
            return

        if new_length < current_length:
            new_pos_embed = pos_embed[:, :, :new_length, :].clone()
            self.positional_embedding.pos_embed.pos_embed = nn.Parameter(new_pos_embed)
            self.positional_embedding.pos_embed.num_embeddings = new_length

    def forward_embedding(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Compute token + positional embeddings.

        Args:
            text_tokens (torch.Tensor): Integer token ids of shape ``(B, seq_len)``.

        Returns:
            (torch.Tensor): Embeddings of shape ``(B, seq_len, dim)``.
        """
        token_emb = self.embedding_layer(text_tokens)
        seq_len = token_emb.shape[1]
        if self.positional_embedding is not None:
            token_emb = token_emb + self.positional_embedding(seq_len).to(token_emb.dtype)
        token_emb = self.embedding_dropout(token_emb)
        return token_emb

    def build_attention_mask(self, context_length: int, batch_size: int) -> torch.Tensor:
        """Build a causal attention mask.

        Args:
            context_length (int): Sequence length.
            batch_size (int): Batch size.

        Returns:
            (torch.Tensor): Mask of shape ``(batch_size, context_length, context_length)`` with ``-inf`` in the upper
                triangle.
        """
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask

    def encode_text(
        self,
        text: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
        input_is_embeddings: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encode text tokens or pre-computed embeddings.

        Args:
            text (torch.Tensor): Token ids ``(B, seq_len)`` or pre-computed embeddings ``(B, seq_len, dim)`` when
                ``input_is_embeddings=True``.
            key_padding_mask (torch.Tensor | None): Boolean mask ``(B, seq_len)`` where True means *ignored*.
            return_all_tokens (bool): If True return all token states; otherwise return the EOS-pooled embedding.
            input_is_embeddings (bool): If True, ``text`` is treated as pre-computed embeddings.
            *args: Additional positional arguments (unused, retained for API compatibility).
            **kwargs: Additional keyword arguments (unused, retained for API compatibility).

        Returns:
            (torch.Tensor): Encoded representations.
        """
        token_emb = text if input_is_embeddings else self.forward_embedding(text)

        attn_mask = None
        if self.causal_masking:
            attn_mask = self.build_attention_mask(context_length=token_emb.shape[1], batch_size=token_emb.shape[0])
            attn_mask = attn_mask.to(device=token_emb.device, dtype=token_emb.dtype)
            key_padding_mask = None

        for layer in self.transformer:
            token_emb = layer(token_emb, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        token_emb = self.final_layer_norm(token_emb)

        if return_all_tokens:
            return token_emb

        if input_is_embeddings:
            token_emb = token_emb[:, -1]
        else:
            token_emb = token_emb[torch.arange(text.shape[0]), text.argmax(dim=-1)]

        return token_emb @ self.projection_layer

    def reparameterize(self) -> None:
        """Fuse all re-parameterisable blocks in the transformer for faster inference.

        Only applies to ``"mct"`` (S0) checkpoints that contain :class:`RepMixerBlock` layers.
        Safe to call on ``"base"`` checkpoints — those layers are silently skipped.
        Should be called **after** weights are loaded and :meth:`set_context_length` has been applied.
        """
        for layer in self.transformer:
            if hasattr(layer, "reparameterize"):
                layer.reparameterize()

    def forward(
        self,
        text_tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
        input_is_embeddings: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encode text tokens by delegating to encode_text."""
        return self.encode_text(
            text=text_tokens,
            key_padding_mask=key_padding_mask,
            return_all_tokens=return_all_tokens,
            input_is_embeddings=input_is_embeddings,
        )
