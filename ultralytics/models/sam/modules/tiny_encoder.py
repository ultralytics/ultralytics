# Ultralytics YOLO 🚀, AGPL-3.0 license

# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ultralytics.utils.instance import to_2tuple


class Conv2d_BN(torch.nn.Sequential):
    """A sequential container that performs 2D convolution followed by batch normalization."""

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        """Initializes the MBConv model with given input channels, output channels, expansion ratio, activation, and
        drop path.
        """
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class PatchEmbed(nn.Module):
    """Embeds images into patches and projects them into a specified embedding dimension."""

    def __init__(self, in_chans, embed_dim, resolution, activation):
        """Initialize the PatchMerging class with specified input, output dimensions, resolution and activation
        function.
        """
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        """Runs input tensor 'x' through the PatchMerging model's sequence of operations."""
        return self.seq(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) layer, part of the EfficientNet architecture."""

    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        """Initializes a convolutional layer with specified dimensions, input resolution, depth, and activation
        function.
        """
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        # NOTE: `DropPath` is needed only for training.
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x):
        """Implements the forward pass for the model architecture."""
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        return self.act3(x)


class PatchMerging(nn.Module):
    """Merges neighboring patches in the feature map and projects to a new dimension."""

    def __init__(self, input_resolution, dim, out_dim, activation):
        """Initializes the ConvLayer with specific dimension, input resolution, depth, activation, drop path, and other
        optional parameters.
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 1 if out_dim in {320, 448, 576} else 2
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        """Applies forward pass on the input utilizing convolution and activation layers, and returns the result."""
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x.flatten(2).transpose(1, 2)


class ConvLayer(nn.Module):
    """
    Convolutional Layer featuring multiple MobileNetV3-style inverted bottleneck convolutions (MBConv).

    Optionally applies downsample operations to the output, and provides support for gradient checkpointing.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):
        """
        Initializes the ConvLayer with the given dimensions and settings.

        Args:
            dim (int): The dimensionality of the input and output.
            input_resolution (Tuple[int, int]): The resolution of the input image.
            depth (int): The number of MBConv layers in the block.
            activation (Callable): Activation function applied after each convolution.
            drop_path (Union[float, List[float]]): Drop path rate. Single float or a list of floats for each MBConv.
            downsample (Optional[Callable]): Function for downsampling the output. None to skip downsampling.
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
            out_dim (Optional[int]): The dimensionality of the output. None means it will be the same as `dim`.
            conv_expand_ratio (float): Expansion ratio for the MBConv layers.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # Patch merging layer
        self.downsample = (
            None
            if downsample is None
            else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        )

    def forward(self, x):
        """Processes the input through a series of convolutional layers and returns the activated output."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)


class Mlp(nn.Module):
    """
    Multi-layer Perceptron (MLP) for transformer architectures.

    This layer takes an input with in_features, applies layer normalization and two fully-connected layers.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """Initializes Attention module with the given parameters including dimension, key_dim, number of heads, etc."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Applies operations on input x and returns modified x, runs downsample if not None."""
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(torch.nn.Module):
    """
    Multi-head attention module with support for spatial awareness, applying attention biases based on spatial
    resolution. Implements trainable attention biases for each unique offset between spatial positions in the resolution
    grid.

    Attributes:
        ab (Tensor, optional): Cached attention biases for inference, deleted during training.
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
    ):
        """
        Initializes the Attention module.

        Args:
            dim (int): The dimensionality of the input and output.
            key_dim (int): The dimensionality of the keys and queries.
            num_heads (int, optional): Number of attention heads. Default is 8.
            attn_ratio (float, optional): Attention ratio, affecting the dimensions of the value vectors. Default is 4.
            resolution (Tuple[int, int], optional): Spatial resolution of the input feature map. Default is (14, 14).

        Raises:
            AssertionError: If `resolution` is not a tuple of length 2.
        """
        super().__init__()

        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        """Sets the module in training mode and handles attribute 'ab' based on the mode."""
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x
        """Performs forward pass over the input tensor 'x' by applying normalization and querying keys/values."""
        B, N, _ = x.shape  # B, N, C

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        self.ab = self.ab.to(self.attention_biases.device)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        return self.proj(x)


class TinyViTBlock(nn.Module):
    """TinyViT Block that applies self-attention and a local convolution to the input."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=nn.GELU,
    ):
        """
        Initializes the TinyViTBlock.

        Args:
            dim (int): The dimensionality of the input and output.
            input_resolution (Tuple[int, int]): Spatial resolution of the input feature map.
            num_heads (int): Number of attention heads.
            window_size (int, optional): Window size for attention. Default is 7.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default is 4.
            drop (float, optional): Dropout rate. Default is 0.
            drop_path (float, optional): Stochastic depth rate. Default is 0.
            local_conv_size (int, optional): The kernel size of the local convolution. Default is 3.
            activation (torch.nn, optional): Activation function for MLP. Default is nn.GELU.

        Raises:
            AssertionError: If `window_size` is not greater than 0.
            AssertionError: If `dim` is not divisible by `num_heads`.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # NOTE: `DropPath` is needed only for training.
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        """Applies attention-based transformation or padding to input 'x' before passing it through a local
        convolution.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # Window partition
            x = (
                x.view(B, nH, self.window_size, nW, self.window_size, C)
                .transpose(2, 3)
                .reshape(B * nH * nW, self.window_size * self.window_size, C)
            )
            x = self.attn(x)
            # Window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size, C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) -> str:
        """Returns a formatted string representing the TinyViTBlock's parameters: dimension, input resolution, number of
        attentions heads, window size, and MLP ratio.
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """A basic TinyViT layer for one stage in a TinyViT architecture."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=nn.GELU,
        out_dim=None,
    ):
        """
        Initializes the BasicLayer.

        Args:
            dim (int): The dimensionality of the input and output.
            input_resolution (Tuple[int, int]): Spatial resolution of the input feature map.
            depth (int): Number of TinyViT blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default is 4.
            drop (float, optional): Dropout rate. Default is 0.
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default is 0.
            downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default is None.
            use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Default is False.
            local_conv_size (int, optional): Kernel size of the local convolution. Default is 3.
            activation (torch.nn, optional): Activation function for MLP. Default is nn.GELU.
            out_dim (int | None, optional): The output dimension of the layer. Default is None.

        Raises:
            ValueError: If `drop_path` is a list of float but its length doesn't match `depth`.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)
            ]
        )

        # Patch merging layer
        self.downsample = (
            None
            if downsample is None
            else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        )

    def forward(self, x):
        """Performs forward propagation on the input tensor and returns a normalized tensor."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)

    def extra_repr(self) -> str:
        """Returns a string representation of the extra_repr function with the layer's parameters."""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class LayerNorm2d(nn.Module):
    """A PyTorch implementation of Layer Normalization in 2D."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Initialize LayerNorm2d with the number of channels and an optional epsilon."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass, normalizing the input tensor."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class TinyViT(nn.Module):
    """
    The TinyViT architecture for vision tasks.

    Attributes:
        img_size (int): Input image size.
        in_chans (int): Number of input channels.
        num_classes (int): Number of classification classes.
        embed_dims (List[int]): List of embedding dimensions for each layer.
        depths (List[int]): List of depths for each layer.
        num_heads (List[int]): List of number of attention heads for each layer.
        window_sizes (List[int]): List of window sizes for each layer.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop_rate (float): Dropout rate for drop layers.
        drop_path_rate (float): Drop path rate for stochastic depth.
        use_checkpoint (bool): Use checkpointing for efficient memory usage.
        mbconv_expand_ratio (float): Expansion ratio for MBConv layer.
        local_conv_size (int): Local convolution kernel size.
        layer_lr_decay (float): Layer-wise learning rate decay.

    Note:
        This implementation is generalized to accept a list of depths, attention heads,
        embedding dimensions and window sizes, which allows you to create a
        "stack" of TinyViT models of varying configurations.
    """

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=1.0,
    ):
        """
        Initializes the TinyViT model.

        Args:
            img_size (int, optional): The input image size. Defaults to 224.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of classification classes. Defaults to 1000.
            embed_dims (List[int], optional): List of embedding dimensions for each layer. Defaults to [96, 192, 384, 768].
            depths (List[int], optional): List of depths for each layer. Defaults to [2, 2, 6, 2].
            num_heads (List[int], optional): List of number of attention heads for each layer. Defaults to [3, 6, 12, 24].
            window_sizes (List[int], optional): List of window sizes for each layer. Defaults to [7, 7, 14, 7].
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            drop_path_rate (float, optional): Drop path rate for stochastic depth. Defaults to 0.1.
            use_checkpoint (bool, optional): Whether to use checkpointing for efficient memory usage. Defaults to False.
            mbconv_expand_ratio (float, optional): Expansion ratio for MBConv layer. Defaults to 4.0.
            local_conv_size (int, optional): Local convolution kernel size. Defaults to 3.
            layer_lr_decay (float, optional): Layer-wise learning rate decay. Defaults to 1.0.
        """
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0], resolution=img_size, activation=activation
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                ),
                #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                #                     patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # Init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay):
        """Sets the learning rate decay for each layer in the TinyViT model."""
        decay_rate = layer_lr_decay

        # Layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            """Sets the learning rate scale for each layer in the model based on the layer's depth."""
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            """Checks if the learning rate scale attribute is present in module's parameters."""
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        """Initializes weights for linear layers and layer normalization in the given module."""
        if isinstance(m, nn.Linear):
            # NOTE: This initialization is needed only for training.
            # trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Returns a dictionary of parameter names where weight decay should not be applied."""
        return {"attention_biases"}

    def forward_features(self, x):
        """Runs the input through the model layers and returns the transformed output."""
        x = self.patch_embed(x)  # x input is (N, C, H, W)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        B, _, C = x.shape
        x = x.view(B, 64, 64, C)
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)

    def forward(self, x):
        """Executes a forward pass on the input tensor through the constructed model layers."""
        return self.forward_features(x)
