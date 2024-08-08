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

from ultralytics.nn.modules import LayerNorm2d
from ultralytics.utils.instance import to_2tuple


class Conv2d_BN(torch.nn.Sequential):
    """
    A sequential container that performs 2D convolution followed by batch normalization.

    Attributes:
        c (torch.nn.Conv2d): 2D convolution layer.
        1 (torch.nn.BatchNorm2d): Batch normalization layer.

    Methods:
        __init__: Initializes the Conv2d_BN with specified parameters.

    Args:
        a (int): Number of input channels.
        b (int): Number of output channels.
        ks (int): Kernel size for the convolution. Defaults to 1.
        stride (int): Stride for the convolution. Defaults to 1.
        pad (int): Padding for the convolution. Defaults to 0.
        dilation (int): Dilation factor for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        bn_weight_init (float): Initial value for batch normalization weight. Defaults to 1.

    Examples:
        >>> conv_bn = Conv2d_BN(3, 64, ks=3, stride=1, pad=1)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = conv_bn(input_tensor)
        >>> print(output.shape)
    """

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        """Initializes a sequential container with 2D convolution followed by batch normalization."""
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class PatchEmbed(nn.Module):
    """
    Embeds images into patches and projects them into a specified embedding dimension.

    Attributes:
        patches_resolution (Tuple[int, int]): Resolution of the patches after embedding.
        num_patches (int): Total number of patches.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of the embedding.
        seq (nn.Sequential): Sequence of convolutional and activation layers for patch embedding.

    Methods:
        forward: Processes the input tensor through the patch embedding sequence.

    Examples:
        >>> import torch
        >>> patch_embed = PatchEmbed(in_chans=3, embed_dim=96, resolution=224, activation=nn.GELU)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
    """

    def __init__(self, in_chans, embed_dim, resolution, activation):
        """Initializes patch embedding with convolutional layers for image-to-patch conversion and projection."""
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
        """Processes input tensor through patch embedding sequence, converting images to patch embeddings."""
        return self.seq(x)


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Conv (MBConv) layer, part of the EfficientNet architecture.

    Attributes:
        in_chans (int): Number of input channels.
        hidden_chans (int): Number of hidden channels.
        out_chans (int): Number of output channels.
        conv1 (Conv2d_BN): First convolutional layer.
        act1 (nn.Module): First activation function.
        conv2 (Conv2d_BN): Depthwise convolutional layer.
        act2 (nn.Module): Second activation function.
        conv3 (Conv2d_BN): Final convolutional layer.
        act3 (nn.Module): Third activation function.
        drop_path (nn.Module): Drop path layer (Identity for inference).

    Methods:
        forward: Performs the forward pass through the MBConv layer.

    Examples:
        >>> in_chans, out_chans = 32, 64
        >>> mbconv = MBConv(in_chans, out_chans, expand_ratio=4, activation=nn.ReLU, drop_path=0.1)
        >>> x = torch.randn(1, in_chans, 56, 56)
        >>> output = mbconv(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        """Initializes the MBConv layer with specified input/output channels, expansion ratio, and activation."""
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
        """Implements the forward pass of MBConv, applying convolutions and skip connection."""
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
    """
    Merges neighboring patches in the feature map and projects to a new dimension.

    This class implements a patch merging operation that combines spatial information and adjusts the feature
    dimension. It uses a series of convolutional layers with batch normalization to achieve this.

    Attributes:
        input_resolution (Tuple[int, int]): The input resolution (height, width) of the feature map.
        dim (int): The input dimension of the feature map.
        out_dim (int): The output dimension after merging and projection.
        act (nn.Module): The activation function used between convolutions.
        conv1 (Conv2d_BN): The first convolutional layer for dimension projection.
        conv2 (Conv2d_BN): The second convolutional layer for spatial merging.
        conv3 (Conv2d_BN): The third convolutional layer for final projection.

    Methods:
        forward: Applies the patch merging operation to the input tensor.

    Examples:
        >>> input_resolution = (56, 56)
        >>> patch_merging = PatchMerging(input_resolution, dim=64, out_dim=128, activation=nn.ReLU)
        >>> x = torch.randn(4, 64, 56, 56)
        >>> output = patch_merging(x)
        >>> print(output.shape)
    """

    def __init__(self, input_resolution, dim, out_dim, activation):
        """Initializes the PatchMerging module for merging and projecting neighboring patches in feature maps."""
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
        """Applies patch merging and dimension projection to the input feature map."""
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

    This layer optionally applies downsample operations to the output and supports gradient checkpointing.

    Attributes:
        dim (int): Dimensionality of the input and output.
        input_resolution (Tuple[int, int]): Resolution of the input image.
        depth (int): Number of MBConv layers in the block.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
        blocks (nn.ModuleList): List of MBConv layers.
        downsample (Optional[Callable]): Function for downsampling the output.

    Methods:
        forward: Processes the input through the convolutional layers.

    Examples:
        >>> input_tensor = torch.randn(1, 64, 56, 56)
        >>> conv_layer = ConvLayer(64, (56, 56), depth=3, activation=nn.ReLU)
        >>> output = conv_layer(input_tensor)
        >>> print(output.shape)
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

        This layer consists of multiple MobileNetV3-style inverted bottleneck convolutions (MBConv) and
        optionally applies downsampling to the output.

        Args:
            dim (int): The dimensionality of the input and output.
            input_resolution (Tuple[int, int]): The resolution of the input image.
            depth (int): The number of MBConv layers in the block.
            activation (Callable): Activation function applied after each convolution.
            drop_path (float | List[float]): Drop path rate. Single float or a list of floats for each MBConv.
            downsample (Optional[Callable]): Function for downsampling the output. None to skip downsampling.
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
            out_dim (Optional[int]): The dimensionality of the output. None means it will be the same as `dim`.
            conv_expand_ratio (float): Expansion ratio for the MBConv layers.

        Examples:
            >>> input_tensor = torch.randn(1, 64, 56, 56)
            >>> conv_layer = ConvLayer(64, (56, 56), depth=3, activation=nn.ReLU)
            >>> output = conv_layer(input_tensor)
            >>> print(output.shape)
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
        """Processes input through convolutional layers, applying MBConv blocks and optional downsampling."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)


class Mlp(nn.Module):
    """
    Multi-layer Perceptron (MLP) module for transformer architectures.

    This module applies layer normalization, two fully-connected layers with an activation function in between,
    and dropout. It is commonly used in transformer-based architectures.

    Attributes:
        norm (nn.LayerNorm): Layer normalization applied to the input.
        fc1 (nn.Linear): First fully-connected layer.
        fc2 (nn.Linear): Second fully-connected layer.
        act (nn.Module): Activation function applied after the first fully-connected layer.
        drop (nn.Dropout): Dropout layer applied after the activation function.

    Methods:
        forward: Applies the MLP operations on the input tensor.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> mlp = Mlp(in_features=256, hidden_features=512, out_features=256, act_layer=nn.GELU, drop=0.1)
        >>> x = torch.randn(32, 100, 256)
        >>> output = mlp(x)
        >>> print(output.shape)
        torch.Size([32, 100, 256])
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """Initializes a multi-layer perceptron with configurable input, hidden, and output dimensions."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Applies MLP operations: layer norm, FC layers, activation, and dropout to the input tensor."""
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(torch.nn.Module):
    """
    Multi-head attention module with spatial awareness and trainable attention biases.

    This module implements a multi-head attention mechanism with support for spatial awareness, applying
    attention biases based on spatial resolution. It includes trainable attention biases for each unique
    offset between spatial positions in the resolution grid.

    Attributes:
        num_heads (int): Number of attention heads.
        scale (float): Scaling factor for attention scores.
        key_dim (int): Dimensionality of the keys and queries.
        nh_kd (int): Product of num_heads and key_dim.
        d (int): Dimensionality of the value vectors.
        dh (int): Product of d and num_heads.
        attn_ratio (float): Attention ratio affecting the dimensions of the value vectors.
        norm (nn.LayerNorm): Layer normalization applied to input.
        qkv (nn.Linear): Linear layer for computing query, key, and value projections.
        proj (nn.Linear): Linear layer for final projection.
        attention_biases (nn.Parameter): Learnable attention biases.
        attention_bias_idxs (Tensor): Indices for attention biases.
        ab (Tensor): Cached attention biases for inference, deleted during training.

    Methods:
        train: Sets the module in training mode and handles the 'ab' attribute.
        forward: Performs the forward pass of the attention mechanism.

    Examples:
        >>> attn = Attention(dim=256, key_dim=64, num_heads=8, resolution=(14, 14))
        >>> x = torch.randn(1, 196, 256)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 196, 256])
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
        Initializes the Attention module for multi-head attention with spatial awareness.

        This module implements a multi-head attention mechanism with support for spatial awareness, applying
        attention biases based on spatial resolution. It includes trainable attention biases for each unique
        offset between spatial positions in the resolution grid.

        Args:
            dim (int): The dimensionality of the input and output.
            key_dim (int): The dimensionality of the keys and queries.
            num_heads (int): Number of attention heads. Default is 8.
            attn_ratio (float): Attention ratio, affecting the dimensions of the value vectors. Default is 4.
            resolution (Tuple[int, int]): Spatial resolution of the input feature map. Default is (14, 14).

        Raises:
            AssertionError: If 'resolution' is not a tuple of length 2.

        Examples:
            >>> attn = Attention(dim=256, key_dim=64, num_heads=8, resolution=(14, 14))
            >>> x = torch.randn(1, 196, 256)
            >>> output = attn(x)
            >>> print(output.shape)
            torch.Size([1, 196, 256])
        """
        super().__init__()

        assert isinstance(resolution, tuple) and len(resolution) == 2, "'resolution' argument not tuple of length 2"
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
        """Performs multi-head attention with spatial awareness and trainable attention biases."""
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x
        """Applies multi-head attention with spatial awareness and trainable attention biases."""
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
    """
    TinyViT Block that applies self-attention and a local convolution to the input.

    This block is a key component of the TinyViT architecture, combining self-attention mechanisms with
    local convolutions to process input features efficiently.

    Attributes:
        dim (int): The dimensionality of the input and output.
        input_resolution (Tuple[int, int]): Spatial resolution of the input feature map.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop_path (nn.Module): Stochastic depth layer, identity function during inference.
        attn (Attention): Self-attention module.
        mlp (Mlp): Multi-layer perceptron module.
        local_conv (Conv2d_BN): Depth-wise local convolution layer.

    Methods:
        forward: Processes the input through the TinyViT block.
        extra_repr: Returns a string with extra information about the block's parameters.

    Examples:
        >>> input_tensor = torch.randn(1, 196, 192)
        >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3)
        >>> output = block(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 196, 192])
    """

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
        Initializes a TinyViT block with self-attention and local convolution.

        This block is a key component of the TinyViT architecture, combining self-attention mechanisms with
        local convolutions to process input features efficiently.

        Args:
            dim (int): Dimensionality of the input and output features.
            input_resolution (Tuple[int, int]): Spatial resolution of the input feature map (height, width).
            num_heads (int): Number of attention heads.
            window_size (int): Size of the attention window. Must be greater than 0.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            drop (float): Dropout rate.
            drop_path (float): Stochastic depth rate.
            local_conv_size (int): Kernel size of the local convolution.
            activation (torch.nn.Module): Activation function for MLP.

        Raises:
            AssertionError: If window_size is not greater than 0.
            AssertionError: If dim is not divisible by num_heads.

        Examples:
            >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3)
            >>> input_tensor = torch.randn(1, 196, 192)
            >>> output = block(input_tensor)
            >>> print(output.shape)
            torch.Size([1, 196, 192])
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
        """Applies self-attention, local convolution, and MLP operations to the input tensor."""
        h, w = self.input_resolution
        b, hw, c = x.shape  # batch, height*width, channels
        assert hw == h * w, "input feature has wrong size"
        res_x = x
        if h == self.window_size and w == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = h + pad_b, w + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size

            # Window partition
            x = (
                x.view(b, nH, self.window_size, nW, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * nH * nW, self.window_size * self.window_size, c)
            )
            x = self.attn(x)

            # Window reverse
            x = x.view(b, nH, nW, self.window_size, self.window_size, c).transpose(2, 3).reshape(b, pH, pW, c)
            if padding:
                x = x[:, :h, :w].contiguous()

            x = x.view(b, hw, c)

        x = res_x + self.drop_path(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, hw).transpose(1, 2)

        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) -> str:
        """
        Returns a string representation of the TinyViTBlock's parameters.

        This method provides a formatted string containing key information about the TinyViTBlock, including its
        dimension, input resolution, number of attention heads, window size, and MLP ratio.

        Returns:
            (str): A formatted string containing the block's parameters.

        Examples:
            >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0)
            >>> print(block.extra_repr())
            dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """
    A basic TinyViT layer for one stage in a TinyViT architecture.

    This class represents a single layer in the TinyViT model, consisting of multiple TinyViT blocks
    and an optional downsampling operation.

    Attributes:
        dim (int): The dimensionality of the input and output features.
        input_resolution (Tuple[int, int]): Spatial resolution of the input feature map.
        depth (int): Number of TinyViT blocks in this layer.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
        blocks (nn.ModuleList): List of TinyViT blocks that make up this layer.
        downsample (nn.Module | None): Downsample layer at the end of the layer, if specified.

    Methods:
        forward: Processes the input through the layer's blocks and optional downsampling.
        extra_repr: Returns a string with the layer's parameters for printing.

    Examples:
        >>> input_tensor = torch.randn(1, 3136, 192)
        >>> layer = BasicLayer(dim=192, input_resolution=(56, 56), depth=2, num_heads=3, window_size=7)
        >>> output = layer(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 784, 384])
    """

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
        Initializes a BasicLayer in the TinyViT architecture.

        This layer consists of multiple TinyViT blocks and an optional downsampling operation. It is designed to
        process feature maps at a specific resolution and dimensionality within the TinyViT model.

        Args:
            dim (int): Dimensionality of the input and output features.
            input_resolution (Tuple[int, int]): Spatial resolution of the input feature map (height, width).
            depth (int): Number of TinyViT blocks in this layer.
            num_heads (int): Number of attention heads in each TinyViT block.
            window_size (int): Size of the local window for attention computation.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            drop (float): Dropout rate.
            drop_path (float | List[float]): Stochastic depth rate. Can be a float or a list of floats for each block.
            downsample (nn.Module | None): Downsampling layer at the end of the layer. None to skip downsampling.
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
            local_conv_size (int): Kernel size for the local convolution in each TinyViT block.
            activation (nn.Module): Activation function used in the MLP.
            out_dim (int | None): Output dimension after downsampling. None means it will be the same as `dim`.

        Raises:
            ValueError: If `drop_path` is a list and its length doesn't match `depth`.

        Examples:
            >>> layer = BasicLayer(dim=96, input_resolution=(56, 56), depth=2, num_heads=3, window_size=7)
            >>> x = torch.randn(1, 56*56, 96)
            >>> output = layer(x)
            >>> print(output.shape)
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
        """Processes input through TinyViT blocks and optional downsampling."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)

    def extra_repr(self) -> str:
        """Returns a string with the layer's parameters for printing."""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):
    """
    TinyViT: A compact vision transformer architecture for efficient image classification and feature extraction.

    This class implements the TinyViT model, which combines elements of vision transformers and convolutional
    neural networks for improved efficiency and performance on vision tasks.

    Attributes:
        img_size (int): Input image size.
        num_classes (int): Number of classification classes.
        depths (List[int]): Number of blocks in each stage.
        num_layers (int): Total number of layers in the network.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        patch_embed (PatchEmbed): Module for patch embedding.
        patches_resolution (Tuple[int, int]): Resolution of embedded patches.
        layers (nn.ModuleList): List of network layers.
        norm_head (nn.LayerNorm): Layer normalization for the classifier head.
        head (nn.Linear): Linear layer for final classification.
        neck (nn.Sequential): Neck module for feature refinement.

    Methods:
        set_layer_lr_decay: Sets layer-wise learning rate decay.
        _init_weights: Initializes weights for linear and normalization layers.
        no_weight_decay_keywords: Returns keywords for parameters that should not use weight decay.
        forward_features: Processes input through the feature extraction layers.
        forward: Performs a forward pass through the entire network.

    Examples:
        >>> model = TinyViT(img_size=224, num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = model.forward_features(x)
        >>> print(features.shape)
        torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=(96, 192, 384, 768),
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_sizes=(7, 7, 14, 7),
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

        This constructor sets up the TinyViT architecture, including patch embedding, multiple layers of
        attention and convolution blocks, and a classification head.

        Args:
            img_size (int): Size of the input image. Default is 224.
            in_chans (int): Number of input channels. Default is 3.
            num_classes (int): Number of classes for classification. Default is 1000.
            embed_dims (Tuple[int, int, int, int]): Embedding dimensions for each stage.
                Default is (96, 192, 384, 768).
            depths (Tuple[int, int, int, int]): Number of blocks in each stage. Default is (2, 2, 6, 2).
            num_heads (Tuple[int, int, int, int]): Number of attention heads in each stage.
                Default is (3, 6, 12, 24).
            window_sizes (Tuple[int, int, int, int]): Window sizes for each stage. Default is (7, 7, 14, 7).
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default is 4.0.
            drop_rate (float): Dropout rate. Default is 0.0.
            drop_path_rate (float): Stochastic depth rate. Default is 0.1.
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default is False.
            mbconv_expand_ratio (float): Expansion ratio for MBConv layer. Default is 4.0.
            local_conv_size (int): Kernel size for local convolutions. Default is 3.
            layer_lr_decay (float): Layer-wise learning rate decay factor. Default is 1.0.

        Examples:
            >>> model = TinyViT(img_size=224, num_classes=1000)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> output = model(x)
            >>> print(output.shape)
            torch.Size([1, 1000])
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
        """Sets layer-wise learning rate decay for the TinyViT model based on depth."""
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
        """Initializes weights for linear and normalization layers in the TinyViT model."""
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
        """Returns a set of keywords for parameters that should not use weight decay."""
        return {"attention_biases"}

    def forward_features(self, x):
        """Processes input through feature extraction layers, returning spatial features."""
        x = self.patch_embed(x)  # x input is (N, C, H, W)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        batch, _, channel = x.shape
        x = x.view(batch, 64, 64, channel)
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)

    def forward(self, x):
        """Performs the forward pass through the TinyViT model, extracting features from the input image."""
        return self.forward_features(x)
