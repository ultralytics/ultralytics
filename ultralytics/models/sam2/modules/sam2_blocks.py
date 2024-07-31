# Ultralytics YOLO 🚀, AGPL-3.0 license

import copy
import math
from functools import partial
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ultralytics.models.sam.modules.transformer import (
    Attention,
)
from ultralytics.models.sam.modules.transformer import (
    TwoWayAttentionBlock as SAMTwoWayAttentionBlock,
)
from ultralytics.models.sam.modules.transformer import (
    TwoWayTransformer as SAMTwoWayTransformer,
)
from ultralytics.nn.modules import MLP, LayerNorm2d

from .utils import apply_rotary_enc, compute_axial_cis, window_partition, window_unpartition


class DropPath(nn.Module):
    """Implements stochastic depth regularization for neural networks during training."""

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        """Initialize DropPath module with specified drop probability and scaling option."""
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Applies stochastic depth to input tensor during training, with optional scaling."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MaskDownSampler(nn.Module):
    """Downsamples and embeds masks using convolutional layers and layer normalization for efficient processing."""

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        """Initializes a mask downsampler module for progressive downsampling and channel expansion."""
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        """Downsamples and encodes input mask to embed_dim channels using convolutional layers and LayerNorm2d."""
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    """
    ConvNeXt Block for efficient feature extraction in convolutional neural networks.

    This block implements a modified version of the ConvNeXt architecture, offering two equivalent
    implementations for improved performance and flexibility.

    Attributes:
        dwconv (nn.Conv2d): Depthwise convolution layer.
        norm (LayerNorm2d): Layer normalization applied to channels.
        pwconv1 (nn.Linear): First pointwise convolution implemented as a linear layer.
        act (nn.GELU): GELU activation function.
        pwconv2 (nn.Linear): Second pointwise convolution implemented as a linear layer.
        gamma (nn.Parameter | None): Learnable scale parameter for layer scaling.
        drop_path (nn.Module): DropPath layer for stochastic depth regularization.

    Methods:
        forward: Processes the input tensor through the ConvNeXt block.

    Examples:
        >>> import torch
        >>> x = torch.randn(1, 64, 56, 56)
        >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        """
        Initialize a ConvNeXt Block.

        This block implements a ConvNeXt architecture with optional depthwise convolution, layer normalization,
        pointwise convolutions, and GELU activation.

        Args:
            dim (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel. Default is 7.
            padding (int): Padding size for the convolution. Default is 3.
            drop_path (float): Stochastic depth rate. Default is 0.0.
            layer_scale_init_value (float): Initial value for Layer Scale. Default is 1e-6.
            use_dwconv (bool): Whether to use depthwise convolution. Default is True.

        Attributes:
            dwconv (nn.Conv2d): Depthwise or standard 2D convolution layer.
            norm (LayerNorm2d): Layer normalization applied to the output of dwconv.
            pwconv1 (nn.Linear): First pointwise convolution implemented as a linear layer.
            act (nn.GELU): GELU activation function.
            pwconv2 (nn.Linear): Second pointwise convolution implemented as a linear layer.
            gamma (nn.Parameter | None): Learnable scale parameter for the residual path.

        Examples:
            >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
            >>> x = torch.randn(1, 64, 32, 32)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 64, 32, 32])
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Applies ConvNeXt block operations to input tensor, including convolutions and residual connection."""
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    """
    A module for fusing features through multiple layers of a neural network.

    This class applies a series of identical layers to an input tensor, optionally projecting the input first.

    Attributes:
        proj (nn.Module): An optional input projection layer. Identity if no projection is needed.
        layers (nn.ModuleList): A list of identical layers to be applied sequentially.

    Methods:
        forward: Applies the fuser to an input tensor.

    Examples:
        >>> layer = CXBlock(dim=256)
        >>> fuser = Fuser(layer, num_layers=3, dim=256, input_projection=True)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = fuser(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        """
        Initializes the Fuser module.

        This module creates a sequence of identical layers and optionally applies an input projection.

        Args:
            layer (nn.Module): The layer to be replicated in the fuser.
            num_layers (int): The number of times to replicate the layer.
            dim (int | None): The dimension for input projection, if used.
            input_projection (bool): Whether to use input projection.

        Attributes:
            proj (nn.Module): The input projection layer, or nn.Identity if not used.
            layers (nn.ModuleList): A list of replicated layers.

        Examples:
            >>> layer = nn.Linear(64, 64)
            >>> fuser = Fuser(layer, num_layers=3, dim=64, input_projection=True)
            >>> input_tensor = torch.randn(1, 64)
            >>> output = fuser(input_tensor)
        """
        super().__init__()
        self.proj = nn.Identity()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        """Applies a series of layers to the input tensor, optionally projecting it first."""
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class TwoWayAttentionBlock(SAMTwoWayAttentionBlock):
    """
    A two-way attention block for performing self-attention and cross-attention in both directions.

    This block extends the SAMTwoWayAttentionBlock and consists of four main components: self-attention on
    sparse inputs, cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and
    cross-attention from dense to sparse inputs.

    Attributes:
        self_attn (Attention): Self-attention layer for queries.
        norm1 (nn.LayerNorm): Layer normalization after the first attention block.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after the second attention block.
        mlp (MLP): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after the MLP block.
        norm4 (nn.LayerNorm): Layer normalization after the third attention block.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Flag to skip positional encoding in the first layer.

    Methods:
        forward: Processes input through the attention blocks and MLP.

    Examples:
        >>> block = TwoWayAttentionBlock(embedding_dim=256, num_heads=8)
        >>> sparse_input = torch.randn(1, 100, 256)
        >>> dense_input = torch.randn(1, 256, 16, 16)
        >>> sparse_output, dense_output = block(sparse_input, dense_input)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Initializes a TwoWayAttentionBlock for performing self-attention and cross-attention in two directions.

        This block consists of four main layers: self-attention on sparse inputs, cross-attention of sparse inputs
        to dense inputs, an MLP block on sparse inputs, and cross-attention of dense inputs to sparse inputs.

        Args:
            embedding_dim (int): The channel dimension of the embeddings.
            num_heads (int): The number of heads in the attention layers.
            mlp_dim (int): The hidden dimension of the MLP block.
            activation (Type[nn.Module]): The activation function of the MLP block.
            attention_downsample_rate (int): The downsample rate for attention computations.
            skip_first_layer_pe (bool): Whether to skip the positional encoding in the first layer.

        Attributes:
            self_attn (Attention): The self-attention layer for the queries.
            norm1 (nn.LayerNorm): Layer normalization following the first attention block.
            cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
            norm2 (nn.LayerNorm): Layer normalization following the second attention block.
            mlp (MLP): MLP block that transforms the query embeddings.
            norm3 (nn.LayerNorm): Layer normalization following the MLP block.
            norm4 (nn.LayerNorm): Layer normalization following the third attention block.
            cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
            skip_first_layer_pe (bool): Whether to skip the positional encoding in the first layer.

        Examples:
            >>> block = TwoWayAttentionBlock(embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> sparse_inputs = torch.randn(1, 100, 256)
            >>> dense_inputs = torch.randn(1, 256, 32, 32)
            >>> sparse_outputs, dense_outputs = block(sparse_inputs, dense_inputs)
        """
        super().__init__(embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate, skip_first_layer_pe)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, act=activation)


class TwoWayTransformer(SAMTwoWayTransformer):
    """
    A Two-Way Transformer module for simultaneous attention to image and query points.

    This class implements a specialized transformer decoder that attends to an input image using queries with
    supplied positional embeddings. It is particularly useful for tasks like object detection, image
    segmentation, and point cloud processing.

    Attributes:
        depth (int): Number of layers in the transformer.
        embedding_dim (int): Channel dimension for input embeddings.
        num_heads (int): Number of heads for multihead attention.
        mlp_dim (int): Internal channel dimension for the MLP block.
        layers (nn.ModuleList): List of TwoWayAttentionBlock layers comprising the transformer.
        final_attn_token_to_image (Attention): Final attention layer from queries to image.
        norm_final_attn (nn.LayerNorm): Layer normalization applied to final queries.

    Methods:
        forward: Processes input image embeddings and query embeddings through the transformer.

    Examples:
        >>> transformer = TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 64, 64)
        >>> query_embedding = torch.randn(1, 100, 256)
        >>> output = transformer(image_embedding, query_embedding)
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        Initializes a TwoWayTransformer instance.

        This transformer decoder attends to an input image using queries with supplied positional embeddings.
        It is designed for tasks like object detection, image segmentation, and point cloud processing.

        Args:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for the input embeddings.
            num_heads (int): Number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim (int): Channel dimension internal to the MLP block.
            activation (Type[nn.Module]): Activation function to use in the MLP block.
            attention_downsample_rate (int): Downsampling rate for attention computations.

        Attributes:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for the input embeddings.
            num_heads (int): Number of heads for multihead attention.
            mlp_dim (int): Internal channel dimension for the MLP block.
            layers (nn.ModuleList): List of TwoWayAttentionBlock layers comprising the transformer.
            final_attn_token_to_image (Attention): Final attention layer from queries to image.
            norm_final_attn (nn.LayerNorm): Layer normalization applied to the final queries.

        Examples:
            >>> transformer = TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> transformer
            TwoWayTransformer(
              (layers): ModuleList(
                (0-4): 5 x TwoWayAttentionBlock(...)
              )
              (final_attn_token_to_image): Attention(...)
              (norm_final_attn): LayerNorm(...)
            )
        """
        super().__init__(depth, embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )


class RoPEAttention(Attention):
    """Implements rotary position encoding for attention mechanisms in transformer architectures."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        """Initializes RoPEAttention with rotary position encoding for attention mechanisms."""
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
        """Applies rotary position encoding and computes attention between query, key, and value tensors."""
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    """Applies pooling and optional normalization to a tensor, handling permutations for spatial operations."""
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    """Implements multi-scale self-attention with optional query pooling for efficient feature extraction."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        """Initializes a multi-scale attention module with configurable query pooling and linear projections."""
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies multi-scale attention to input tensor, optionally downsampling query features."""
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """Multiscale attention block with window partitioning and query pooling for efficient vision transformers."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        """Initializes a multi-scale attention block with optional window partitioning and downsampling."""
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            act=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies multi-scale attention and MLP processing to input tensor, with optional windowing."""
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PositionEmbeddingSine(nn.Module):
    """Generates sinusoidal positional embeddings for 2D inputs like images."""

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        """Initializes sinusoidal position embeddings for 2D image inputs."""
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x, y):
        """Encodes 2D positions using sine and cosine functions for positional embeddings."""
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        """Encodes box coordinates and dimensions into positional embeddings for object detection tasks."""
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        """Encodes 2D point coordinates with sinusoidal positional embeddings and appends labels."""
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Generate sinusoidal position embeddings for 2D inputs."""
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos
