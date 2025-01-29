# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391

    Args:
        c1 (int, optional): Number of channels per group. The total input channels will be 4 * c1. Defaults to 16.

    Shape:
        - Input: (batch_size, 4 * c1, num_anchors)
        - Output: (batch_size, 4, num_anchors)

    Attributes:
        conv (nn.Conv2d): Convolutional layer used for integral computation with fixed weights.

    Example:
        >>> module = DFL(16)
        >>> input_tensor = torch.randn(2, 64, 10)  # 4*c1=64 channels
        >>> output = module(input_tensor)
        >>> print(output.shape)
        torch.Size([2, 4, 10])
    """

    def __init__(self, c1=16):
        """
        Initializes a convolutional layer with fixed weight distribution.

        Args:
            c1 (int, optional): Number of channels per group. The total input channels will be 4 * c1. Defaults to 16.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """
        Performs integral transformation using softmax and weighted summation.

        Splits the input into 4 groups of `c1` channels each, applies softmax over each group,
        and computes the weighted sum using fixed convolution weights.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 4 * c1, num_anchors)

        Returns:
            (torch.Tensor): Output tensor with shape (batch_size, 4, num_anchors)
        """
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """
    YOLOv8 mask Proto module for segmentation models.

    This module processes feature maps to generate prototype masks for instance segmentation,
    using convolutional layers and transposed convolution for upsampling.

    Args:
        c1 (int): Number of input channels.
        c_ (int, optional): Number of intermediate prototype channels. Defaults to 256.
        c2 (int, optional): Number of output mask channels. Defaults to 32.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, 2H, 2W)  # Spatial dimensions doubled by upsampling

    Attributes:
        cv1 (Conv): First convolution layer reducing channel depth to c_
        upsample (nn.ConvTranspose2d): 2x upsampling layer using transposed convolution
        cv2 (Conv): Intermediate convolution layer
        cv3 (Conv): Final convolution layer producing c2 output channels

    Example:
        >>> model = Proto(c1=128)  # Create Proto module with 128 input channels
        >>> x = torch.randn(16, 128, 32, 32)  # Example input (batch=16, ch=128, 32x32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([16, 32, 64, 64])
    """

    def __init__(self, c1, c_=256, c2=32):
        """Initializes the YOLOv8 mask Proto module with specified channel dimensions."""
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """
        Performs feature processing and upsampling to generate segmentation masks.

        Args:
            x (torch.Tensor): Input features with shape (batch_size, c1, H, W)

        Returns:
            (torch.Tensor): Output masks with shape (batch_size, c2, 2H, 2W)
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

    Args:
        c1 (int): Input channels.
        cm (int): Intermediate channels (expanded in processing).
        c2 (int): Output channels.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H//4, W//4)  # Typical spatial reduction

    Attributes:
        stem1 (Conv): Initial 3x3 convolution with stride 2
        stem2a (Conv): First 2x2 convolution in parallel path
        stem2b (Conv): Second 2x2 convolution in parallel path
        stem3 (Conv): 3x3 convolution after feature concatenation
        stem4 (Conv): Final 1x1 projection convolution
        pool (nn.MaxPool2d): Pooling layer for parallel path

    Example:
        >>> model = HGStem(c1=3, cm=64, c2=128)
        >>> x = torch.randn(16, 3, 224, 224)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([16, 128, 56, 56])
    """

    def __init__(self, c1, cm, c2):
        """Initializes PPHGNetV2 StemBlock with channel configurations."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """
        Processes input through multi-path stem architecture with feature fusion.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor with spatial dimensions reduced by 4x
        """
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])  # Add right+bottom padding
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)  # Channel dimension concatenation
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

    Args:
        c1 (int): Input channels.
        cm (int): Intermediate channels (expanded through multiple convolutions).
        c2 (int): Output channels.
        k (int, optional): Kernel size for convolutions. Defaults to 3.
        n (int, optional): Number of repeated Conv/LightConv blocks. Defaults to 6.
        lightconv (bool, optional): Use LightConv instead of regular Conv. Defaults to False.
        shortcut (bool, optional): Enable residual shortcut connection. Defaults to False.
        act (nn.Module, optional): Activation function. Defaults to nn.ReLU().

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Maintains spatial dimensions

    Attributes:
        m (nn.ModuleList): Sequence of Conv/LightConv blocks
        sc (Conv): Squeeze convolution reducing channels to c2//2
        ec (Conv): Excitation convolution expanding to final c2 channels
        add (bool): Indicates if shortcut connection is active

    Example:
        >>> block = HGBlock(c1=64, cm=32, c2=128)
        >>> x = torch.randn(8, 64, 56, 56)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([8, 128, 56, 56])

        >>> # With shortcut connection (c1 == c2 required)
        >>> shortcut_block = HGBlock(c1=128, cm=64, c2=128, shortcut=True)
        >>> x = torch.randn(8, 128, 56, 56)
        >>> output = shortcut_block(x)
        >>> print(output.shape)
        torch.Size([8, 128, 56, 56])
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes HGBlock with configurable convolution patterns and optional shortcut."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Processes input through multi-stage convolutions with optional residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor where:
                - If self.add=True: (batch, c2, H, W) = y + x
                - Else: (batch, c2, H, W) = y
        """
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) layer implementing multi-scale pooling.

    Reference
    - https://arxiv.org/abs/1406.4729

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        k (tuple, optional): Pooling kernel sizes. Defaults to (5, 9, 13).

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Spatial dimensions preserved

    Attributes:
        cv1 (Conv): 1x1 convolution for channel reduction
        cv2 (Conv): 1x1 convolution for channel expansion
        m (nn.ModuleList): Max pooling layers with different kernel sizes

    Example:
        >>> spp = SPP(c1=64, c2=128)
        >>> input_tensor = torch.randn(2, 64, 224, 224)
        >>> output = spp(input_tensor)
        >>> print(output.shape)
        torch.Size([2, 128, 224, 224])
    """

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with channel configuration and pooling kernels."""
        super().__init__()
        c_ = c1 // 2  # Hidden channel dimension
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """
        Applies multi-scale pooling and concatenates features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor with concatenated pooled features
        """
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.

    A faster implementation of Spatial Pyramid Pooling using repeated single kernel max pooling,
    equivalent to SPP(k=(5, 9, 13)) but more efficient.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        k (int, optional): Pooling kernel size. Defaults to 5.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Spatial dimensions preserved

    Attributes:
        cv1 (Conv): 1x1 convolution for channel reduction
        cv2 (Conv): 1x1 convolution for channel expansion
        m (nn.MaxPool2d): Single max pooling layer reused for multiple pooling operations

    Example:
        >>> sppf = SPPF(c1=64, c2=128)
        >>> input_tensor = torch.randn(2, 64, 224, 224)
        >>> output = sppf(input_tensor)
        >>> print(output.shape)
        torch.Size([2, 128, 224, 224])
    """

    def __init__(self, c1, c2, k=5):
        """Initializes SPPF layer with channel configuration and pooling kernel."""
        super().__init__()
        c_ = c1 // 2  # Hidden channel dimension
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        Applies sequential max pooling operations and concatenates features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor with concatenated pooled features (batch, c2, H, W)
        """
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))  # 3 repeated pool operations
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """
    CSP Bottleneck with 1 initial convolution (Cross Stage Partial Network).

    This variant applies a single initial convolution followed by multiple residual-style 3x3 convolutions,
    maintaining the CSP design philosophy of partial feature processing.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int, optional): Number of 3x3 convolutions in main path. Defaults to 1.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)

    Attributes:
        cv1 (Conv): Initial 1x1 convolution (channel transformation)
        m (nn.Sequential): Main computational block containing sequential 3x3 convolutions

    Example:
        >>> model = C1(c1=64, c2=64, n=3)  # CSP Bottleneck with 3 convs
        >>> x = torch.randn(2, 64, 224, 224)
        >>> print(model(x).shape)
        torch.Size([2, 64, 224, 224])
    """

    def __init__(self, c1, c2, n=1):
        """Initializes CSP Bottleneck with channel configuration and convolution count."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)  # Initial pointwise convolution
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))  # Main computational block

    def forward(self, x):
        """
        Implements CSP-style forward pass with partial feature processing.

        Processing Flow:
        1. Initial channel transformation (cv1)
        2. Feature processing through main computational block (m)
        3. Residual connection with initial transformed features

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor where output = main_computational_block(cv1(x)) + cv1(x)
        """
        y = self.cv1(x)  # Initial transformation
        return self.m(y) + y  # CSP-style partial residual connection


class C2(nn.Module):
    """
    CSP Bottleneck with 2 convolutions and cross-stage partial connections.

    Implements the CSP architecture with feature split/concatenation strategy.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int, optional): Number of Bottleneck blocks in main computational block. Defaults to 1.
        shortcut (bool, optional): Enable residual shortcuts in Bottleneck blocks. Defaults to True.
        g (int, optional): Groups for 3x3 convolutions. Defaults to 1.
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)

    Attributes:
        cv1 (Conv): Initial 1x1 convolution that doubles hidden channels
        cv2 (Conv): Final 1x1 convolution for channel adjustment
        m (nn.Sequential): Main computational block containing sequential Bottleneck modules
        c (int): Hidden channels calculated as int(c2 * e)

    Example:
        >>> model = C2(c1=64, c2=64, n=3)
        >>> x = torch.randn(2, 64, 224, 224)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 224, 224])
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP Bottleneck with channel configuration and processing parameters."""
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Split into two branches
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(
            *(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        )  # Main computational block

    def forward(self, x):
        """
        Implements CSP forward pass with feature split-processing-concat pattern.

        Processing Flow:
        1. Channel expansion and split into two branches
        2. Main computational block processing through Bottleneck modules
        3. Concatenation with bypass branch
        4. Final channel adjustment

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor with shape (B, c2, H, W)
        """
        a, b = self.cv1(x).chunk(2, 1)  # Split channels
        return self.cv2(torch.cat((self.m(a), b), 1))  # Process through main block and concat


class C2f(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions and extended feature concatenation.

    Enhanced version of C2 with optimized feature aggregation through dynamic concatenation.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int, optional): Number of Bottleneck blocks in main computational block. Defaults to 1.
        shortcut (bool, optional): Enable residual shortcuts in Bottleneck blocks. Defaults to False.
        g (int, optional): Groups for 3x3 convolutions. Defaults to 1.
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)

    Attributes:
        cv1 (Conv): Initial 1x1 convolution that doubles hidden channels
        cv2 (Conv): Final 1x1 convolution for channel compression
        m (nn.ModuleList): Main computational block containing n Bottleneck modules
        c (int): Hidden channels calculated as int(c2 * e)

    Example:
        >>> model = C2f(c1=64, c2=128, n=3)
        >>> x = torch.randn(2, 64, 224, 224)

        # General usage
        >>> print(model(x).shape)
        torch.Size([2, 128, 224, 224])

        # Edge TPU deployment
        >>> model.forward = model.forward_split  # Override before export
        >>> torch.onnx.export(model, x, "c2f.onnx")
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes C2f layer with expanded feature aggregation capabilities."""
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial split convolution
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Final fusion convolution
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )  # Main computational block

    def forward(self, x):
        """
        Implements forward pass with chunk-based feature concatenation.

        Processing Flow:
        1. Initial channel expansion and chunk split
        2. Iterative feature processing through Bottleneck modules
        3. Progressive concatenation of intermediate features
        4. Final channel compression

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor of shape (B, c2, H, W)
        """
        y = list(self.cv1(x).chunk(2, 1))  # Initial split [c, c]
        y.extend(m(y[-1]) for m in self.m)  # Extend with processed features
        return self.cv2(torch.cat(y, 1))  # Concatenate all features

    def forward_split(self, x):
        """
        Edge TPU-friendly forward pass using split() for explicit tensor partitioning.

        Differences from forward():
        - Uses split() with exact channel counts instead of chunk()
        - Generates ONNX graph without FlexSplitV operations
        - Required for Google Coral Edge TPU compatibility

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor of shape (B, c2, H, W)
        """
        y = self.cv1(x).split((self.c, self.c), 1)  # Precise channel split
        y = [y[0], y[1]]  # Initialize feature list
        y.extend(m(y[-1]) for m in self.m)  # Process and collect features
        return self.cv2(torch.cat(y, 1))  # Final concatenation


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """
    C3 module with cross-convolutions for enhanced spatial feature capture.

    Inherits from C3 while replacing standard square convolutions with cross-convolutions
    (1x3 and 3x1 kernel combinations).

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int, optional): Number of Bottleneck blocks. Defaults to 1.
        shortcut (bool, optional): Enable residual connections. Defaults to True.
        g (int, optional): Groups for 3x1/1x3 convolutions. Defaults to 1.
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Maintains input spatial dimensions

    Attributes:
        m (nn.Sequential): Main computational block with cross-convolution Bottlenecks

    Example:
        >>> model = C3x(c1=64, c2=64, n=3)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(model(x).shape)
        torch.Size([2, 64, 56, 56])
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x with cross-convolution Bottleneck blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """
    Reparameterized CSP Bottleneck with 3 convolutions and structural re-parameterization.

    Designed for efficient deployment through parameter fusion in RepConv blocks.
    Employed in RT-DETR's neck architecture for real-time performance.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int, optional): Number of RepConv blocks in main computational block. Defaults to 3.
        e (float, optional): Hidden channels expansion ratio. Defaults to 1.0.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Spatial dimensions preserved

    Attributes:
        cv1 (Conv): First 1x1 convolution branch
        cv2 (Conv): Second 1x1 convolution branch (identity when c1=c2)
        m (nn.Sequential): Main computational block containing n RepConv modules
        cv3 (Conv | nn.Identity): Final projection convolution

    Example:
        >>> model = RepC3(c1=64, c2=128, n=3)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(model(x).shape)
        torch.Size([2, 128, 56, 56])
    """

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initializes reparameterizable CSP bottleneck with dual branches."""
        super().__init__()
        c_ = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # Main processing branch
        self.cv2 = Conv(c1, c_, 1, 1)  # Shortcut branch
        self.m = nn.Sequential(*(RepConv(c_, c_) for _ in range(n)))  # RepConv sequence
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()  # Optional projection

    def forward(self, x):
        """
        Processes input through reparameterizable dual paths.

        Processing Flow:
        1. Branch 1: cv1 → RepConv sequence
        2. Branch 2: cv2 (direct processing)
        3. Element-wise addition of both branches
        4. Final projection with cv3 (if needed)

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor of shape (B, c2, H, W)
        """
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """
    C3 module with TransformerBlock for hybrid CNN/Transformer architectures.

    Inherits from C3 and replaces standard Bottleneck blocks with TransformerBlocks.
    Part of the RT-DETR architecture for joint feature learning.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int, optional): Number of transformer layers. Defaults to 1.
        shortcut (bool, optional): Enable residual connection. Defaults to True.
        g (int, optional): Groups for convolution (unused in transformer). Defaults to 1.
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5.

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Maintains spatial dimensions

    Attributes:
        m (TransformerBlock): Transformer-based computational block replacing Bottleneck

    Example:
        >>> model = C3TR(c1=64, c2=64, n=3)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(model(x).shape)
        torch.Size([2, 64, 56, 56])
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes hybrid CNN/Transformer bottleneck."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)  # (hidden_dim, num_heads, num_layers)


class C3Ghost(C3):
    """
    C3 module with GhostBottleneck for efficient feature extraction.

    Inherits from C3 while replacing standard Bottleneck blocks with GhostBottleneck
    operations from GhostNet. Designed to reduce computational complexity while
    maintaining performance.

    Reference
    - https://arxiv.org/abs/1911.11907

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int, optional): Number of GhostBottleneck blocks. Defaults to 1
        shortcut (bool, optional): Enable residual connections. Defaults to True
        g (int, optional): Groups for grouped convolutions. Defaults to 1
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Maintains spatial dimensions

    Attributes:
        m (nn.Sequential): Sequence of GhostBottleneck modules

    Example:
        >>> model = C3Ghost(64, 64)
        >>> x = torch.randn(1, 64, 56, 56)
        >>> print(model(x).shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3Ghost module with GhostBottleneck blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck block from GhostNet for efficient feature transformation.

    Original implementation: https://github.com/huawei-noah/ghostnet

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int, optional): Kernel size for depthwise convolution. Defaults to 3
        s (int, optional): Stride for depthwise convolution. Defaults to 1

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H//s, W//s)  # Spatial downsampling when s=2

    Attributes:
        conv (nn.Sequential): Main processing path with GhostConv and DWConv
        shortcut (nn.Sequential): Skip connection path with optional downsampling


    Example:
        >>> block = GhostBottleneck(64, 128, s=2)
        >>> x = torch.randn(1, 64, 56, 56)
        >>> print(block(x).shape)
        torch.Size([1, 128, 28, 28])
    """

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck with channel config, kernel size, and stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # Pointwise expansion
            (DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity()),  # Optional DW downsampling
            GhostConv(c_, c2, 1, 1, act=False),  # Pointwise projection
        )
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False),  # Depthwise downsampling
                Conv(c1, c2, 1, 1, act=False),  # Channel alignment
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        """
        Applies feature transformation with residual connection.

        Processing Flow:
        1. Main path: GhostConv → (DWConv if stride=2) → GhostConv
        2. Shortcut path: (DWConv + Conv if stride=2) or identity
        3. Element-wise addition of both paths

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor with combined features
        """
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """
    Standard bottleneck block with optional residual connection.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        shortcut (bool, optional): Enable residual connection. Defaults to True
        g (int, optional): Groups for grouped convolution. Defaults to 1
        k (tuple, optional): Kernel sizes for two convolutional layers. Defaults to (3, 3)
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Maintains spatial dimensions when c1 == c2

    Attributes:
        cv1 (Conv): First convolutional layer with kernel size k[0]
        cv2 (Conv): Second convolutional layer with kernel size k[1]
        add (bool): Indicates if residual connection is active

    Example:
        >>> bottleneck = Bottleneck(64, 64)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> out = bottleneck(x)
        >>> print(out.shape)
        torch.Size([2, 64, 56, 56])

        >>> # Without residual connection
        >>> bottleneck_no_res = Bottleneck(64, 128, shortcut=False)
        >>> print(bottleneck_no_res(x).shape)
        torch.Size([2, 128, 56, 56])
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes bottleneck block with channel configuration and convolution parameters."""
        super().__init__()
        c_ = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Processes input through two convolutional layers with optional residual connection.

        Processing Flow:
        1. Feature transformation through cv1 → cv2
        2. Residual connection with input if conditions met

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor with combined features
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """
    CSP Bottleneck from Cross-Stage Partial Networks (CSPNet).

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int, optional): Number of Bottleneck blocks. Defaults to 1
        shortcut (bool, optional): Enable residual connection. Defaults to True
        g (int, optional): Groups for grouped convolution. Defaults to 1
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)  # Maintains spatial dimensions

    Attributes:
        cv1-4 (Conv | nn.Conv2d): Convolution layers
        bn (nn.BatchNorm2d): Batch normalization after feature concatenation
        m (nn.Sequential): Sequence of Bottleneck blocks

    Example:
        >>> csp = BottleneckCSP(64, 64)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(csp(x).shape)
        torch.Size([2, 64, 56, 56])
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with channel configuration and processing parameters."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Processes input through two parallel paths and merges features.

        Processing Flow:
        1. Main path: cv1 → Bottleneck blocks → cv3
        2. Shortcut path: cv2 (1x1 conv)
        3. Concatenate and normalize → cv4

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor with merged features
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """
    Standard ResNet Bottleneck Block with expansion.

    Reference
    - https://arxiv.org/abs/1512.03385

    Args:
        c1 (int): Input channels
        c2 (int): Base output channels
        s (int, optional): Stride for 3x3 convolution. Defaults to 1
        e (int, optional): Channel expansion ratio. Defaults to 4

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2*e, H//s, W//s)  # Spatial downsampling when s>1

    Attributes:
        cv1-3 (Conv): Convolution layers (1x1-3x3-1x1)
        shortcut (nn.Sequential): Skip connection path

    Example:
        >>> block = ResNetBlock(64, 64, s=2)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(block(x).shape)
        torch.Size([2, 256, 28, 28])
    """

    def __init__(self, c1, c2, s=1, e=4):
        """Initializes ResNet block with expansion configuration."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """
        Processes input through bottleneck with residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor with residual connection
        """
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """
    ResNet Layer containing multiple ResNet blocks.

    Args:
        c1 (int): Input channels
        c2 (int): Base output channels
        s (int, optional): Stride for first block. Defaults to 1
        is_first (bool, optional): If first layer in network. Defaults to False
        n (int, optional): Number of blocks in layer. Defaults to 1
        e (int, optional): Channel expansion ratio. Defaults to 4

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2*e, H//(s*2 if is_first else s), W//(s*2 if is_first else s))

    Attributes:
        layer (nn.Sequential): Sequence of Conv/MaxPool or ResNet blocks

    Example:
        >>> # Initial layer
        >>> layer = ResNetLayer(3, 64, is_first=True)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> print(layer(x).shape)
        torch.Size([2, 256, 56, 56])

        >>> # Subsequent layer
        >>> layer = ResNetLayer(256, 128, s=2, n=3)
        >>> x = torch.randn(2, 256, 56, 56)
        >>> print(layer(x).shape)
        torch.Size([2, 512, 28, 28])
    """

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes ResNet layer configuration."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Processes input through initial conv+pool or block sequence.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            (torch.Tensor): Processed output tensor
        """
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """
    Multi-head Max Sigmoid Attention Block for guided feature enhancement.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        nh (int, optional): Number of attention heads. Defaults to 1
        ec (int, optional): Embedding channels. Defaults to 128
        gc (int, optional): Guide vector channels. Defaults to 512
        scale (bool, optional): Enable learnable scale parameter. Defaults to False

    Shape:
        - Input: (batch_size, c1, H, W)
        - Guide: (batch_size, gc)
        - Output: (batch_size, c2, H, W)

    Attributes:
        ec (Conv | None): Embedding convolution
        gl (nn.Linear): Guide linear projection
        proj_conv (Conv): Final projection convolution
        scale (nn.Parameter | float): Attention scaling factor

    Example:
        >>> attn = MaxSigmoidAttnBlock(256, 256, nh=2)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> guide = torch.randn(2, 512)
        >>> print(attn(x, guide).shape)
        torch.Size([2, 256, 32, 32])
    """

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes attention block with configurable head and channel settings."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """
        Processes features through attention mechanism with guidance vector.

        Args:
            x (torch.Tensor): Input features (B, c1, H, W)
            guide (torch.Tensor): Guidance vector (B, gc)

        Returns:
            (torch.Tensor): Attention-weighted features (B, c2, H, W)
        """
        bs, _, h, w = x.shape
        # Process guide vector
        guide = self.gl(guide).view(bs, -1, self.nh, self.hc)
        # Embed features
        embed = self.ec(x) if self.ec else x
        embed = embed.view(bs, self.nh, self.hc, h, w)
        # Compute attention weights
        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide).max(-1)[0]
        aw = (aw / (self.hc**0.5) + self.bias[None, :, None, None]).sigmoid() * self.scale
        # Apply attention
        return (self.proj_conv(x).view(bs, self.nh, -1, h, w) * aw.unsqueeze(2)).view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """
    C2f module with integrated attention mechanism for enhanced feature interaction.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int, optional): Number of Bottleneck blocks. Defaults to 1
        ec (int, optional): Attention embedding channels. Defaults to 128
        nh (int, optional): Number of attention heads. Defaults to 1
        gc (int, optional): Guide vector channels. Defaults to 512
        shortcut (bool, optional): Enable residual connections. Defaults to False
        g (int, optional): Groups for convolution. Defaults to 1
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5

    Shape:
        - Input: (batch_size, c1, H, W)
        - Guide: (batch_size, gc)
        - Output: (batch_size, c2, H, W)

    Attributes:
        cv1 (Conv): Initial feature split convolution
        cv2 (Conv): Final feature fusion convolution
        m (nn.ModuleList): Bottleneck blocks
        attn (MaxSigmoidAttnBlock): Attention module

    Example:
        >>> model = C2fAttn(64, 128, n=3)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> guide = torch.randn(2, 512)
        >>> # Standard forward
        >>> print(model(x, guide).shape)
        torch.Size([2, 128, 56, 56])
        >>> # EdgeTPU-compatible forward
        >>> print(model.forward_split(x, guide).shape)
        torch.Size([2, 128, 56, 56])
    """

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f attention module with configurable parameters."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Processes features through multi-stage pipeline with attention."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """EdgeTPU-compatible forward with explicit channel splitting."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """
    Image-Aware Text Embedding Enhancement with Multi-Scale Pooling Attention.

    Enhances text embeddings with spatially pooled image features using multi-head attention.

    Args:
        ec (int, optional): Embedding channels. Defaults to 256
        ch (tuple): Input channel tuple from different feature levels
        ct (int, optional): Text embedding dimension. Defaults to 512
        nh (int, optional): Number of attention heads. Defaults to 8
        k (int, optional): Spatial pooling size. Defaults to 3 (k x k grid)
        scale (bool, optional): Enable learnable output scale. Defaults to False

    Shape:
        - Input:
            x (list[Tensor]): Multi-scale image features [(B, ch[i], H, W), ...]
            text (Tensor): Text embeddings (B, seq_len, ct)
        - Output: (B, seq_len, ct)  # Enhanced text embeddings

    Attributes:
        query (nn.Sequential): Text transformation pipeline
        key (nn.Sequential): Image key projection
        value (nn.Sequential): Image value projection
        projections (nn.ModuleList): Channel alignment convs for each feature level
        im_pools (nn.ModuleList): Spatial pooling adapters


    Example:
        >>> attn = ImagePoolingAttn(ch=(256, 512), ct=512)
        >>> features = [torch.randn(2, 256, 64, 64), torch.randn(2, 512, 32, 32)]
        >>> text = torch.randn(2, 10, 512)  # (B, seq_len, ct)
        >>> print(attn(features, text).shape)
        torch.Size([2, 10, 512])
    """

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes image-text attention with configurable pooling."""
        super().__init__()
        self.nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, 1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(self.nf)])
        self.ec = ec
        self.nh = nh
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """
        Fuses multi-scale image features with text embeddings using attention.

        Processing Flow:
        1. Project and pool multi-scale image features
        2. Compute attention between text queries and image keys/values
        3. Apply residual connection with original text embeddings

        Args:
            x (list[Tensor]): Multi-scale image features [(B, ch[i], H, W), ...]
            text (Tensor): Text embeddings (B, seq_len, ct)

        Returns:
            (Tensor): Vision-enhanced text embeddings (B, seq_len, ct)
        """
        bs = x[0].shape[0]
        # Process image features
        x = [pool(proj(x)).flatten(2) for x, proj, pool in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)  # (B, num_patches, ec)

        # Transform inputs
        q = self.query(text).view(bs, -1, self.nh, self.hc)
        k = self.key(x).view(bs, -1, self.nh, self.hc)
        v = self.value(x).view(bs, -1, self.nh, self.hc)

        # Compute attention
        aw = torch.einsum("bnmc,bkmc->bmnk", q, k) / (self.hc**0.5)
        x = torch.einsum("bmnk,bkmc->bnmc", F.softmax(aw, dim=-1), v)

        return self.proj(x.reshape(bs, -1, self.ec)) * self.scale + text


class ContrastiveHead(nn.Module):
    """
    Contrastive Learning Head for region-text similarity computation.

    Computes similarity scores between image regions and text embeddings,
    used in vision-language pretraining architectures.

    Shape:
        - Input x: Image features (batch_size, feat_dim, H, W)
        - Input w: Text embeddings (batch_size, num_texts, feat_dim)
        - Output: Similarity scores (batch_size, num_texts, H, W)

    Attributes:
        bias (nn.Parameter): Learnable bias term initialized at -10.0
        logit_scale (nn.Parameter): Learnable temperature scaling parameter

    Example:
        >>> head = ContrastiveHead()
        >>> img_feats = torch.randn(2, 256, 32, 32)  # (B, C, H, W)
        >>> text_embs = torch.randn(2, 5, 256)  # (B, num_texts, C)
        >>> similarity = head(img_feats, text_embs)
        >>> print(similarity.shape)
        torch.Size([2, 5, 32, 32])
    """

    def __init__(self):
        """Initializes contrastive head with bias and temperature parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """
        Computes region-text similarity scores through normalized cosine similarity.

        Args:
            x (torch.Tensor): Image region features (B, C, H, W)
            w (torch.Tensor): Text embeddings (B, N, C)

        Returns:
            (torch.Tensor): Similarity scores (B, N, H, W)
        """
        # Normalize features
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        # Compute scaled similarity
        return torch.einsum("bchw,bkc->bkhw", x, w) * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Normalized Contrastive Head for YOLO-World with modified normalization strategy.

    Args:
        embed_dims (int): Embedding dimensions for text and image features

    Shape:
        - Input x: Image features (batch_size, embed_dims, H, W)
        - Input w: Text embeddings (batch_size, num_texts, embed_dims)
        - Output: Similarity scores (batch_size, num_texts, H, W)

    Attributes:
        norm (nn.BatchNorm2d): Batch normalization layer for image features
        bias (nn.Parameter): Learnable bias term initialized at -10.0
        logit_scale (nn.Parameter): Temperature parameter initialized at -1.0

    Example:
        >>> head = BNContrastiveHead(256)
        >>> img_feats = torch.randn(2, 256, 32, 32)
        >>> text_embs = torch.randn(2, 5, 256)
        >>> print(head(img_feats, text_embs).shape)
        torch.Size([2, 5, 32, 32])
    """

    def __init__(self, embed_dims: int):
        """Initializes contrastive head with batch normalization."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Computes region-text similarity with batch-normalized image features."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        return torch.einsum("bchw,bkc->bkhw", x, w) * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """
    Reparameterizable Bottleneck block with enhanced inference efficiency.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        shortcut (bool, optional): Enable residual connection. Defaults to True
        g (int, optional): Groups for convolution. Defaults to 1
        k (tuple, optional): Kernel sizes for two conv layers. Defaults to (3, 3)
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5



    Example:
        >>> block = RepBottleneck(64, 64)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(block(x).shape)
        torch.Size([2, 64, 56, 56])
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes RepBottleneck with reparameterizable components."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k[0], 1)  # Replaced with RepConv


class RepCSP(C3):
    """
    Reparameterizable Cross Stage Partial Network for efficient feature extraction.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int, optional): Number of RepBottleneck blocks. Defaults to 1
        shortcut (bool, optional): Enable residual connections. Defaults to True
        g (int, optional): Groups for convolution. Defaults to 1
        e (float, optional): Hidden channels expansion ratio. Defaults to 0.5

    Shape:
        - Input: (batch_size, c1, H, W)
        - Output: (batch_size, c2, H, W)

    Example:
        >>> layer = RepCSP(64, 128, n=3)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(layer(x).shape)
        torch.Size([2, 128, 56, 56])
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP with reparameterizable bottlenecks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """
    Reparameterizable CSP-ELAN4 block with hierarchical feature integration.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        c3 (int): First expansion channels
        c4 (int): Secondary expansion channels
        n (int, optional): Number of RepCSP blocks per branch. Defaults to 1

    Attributes:
        cv1-4 (Conv): Convolutional transformation layers
        cv2-3 (nn.Sequential): Feature processing branches with RepCSP blocks

    Example:
        >>> layer = RepNCSPELAN4(64, 128, 96, 64)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(layer(x).shape)
        torch.Size([2, 128, 56, 56])
    """

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes hierarchical feature integration block."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x):
        """Processes input through multi-branch feature integration."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """EdgeTPU-compatible feature processing with explicit split."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """
    Lightweight ELAN variant without reparameterization.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        c3 (int): First expansion channels
        c4 (int): Secondary expansion channels

    Example:
        >>> layer = ELAN1(64, 128, 96, 64)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> print(layer(x).shape)
        torch.Size([2, 128, 56, 56])
    """

    def __init__(self, c1, c2, c3, c4):
        """Initializes non-reparameterized ELAN block."""
        super().__init__(c1, c2, c3, c4)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)


class AConv(nn.Module):
    """
    AConv module from YOLOv9.

    Reference
    - https://github.com/WongKinYiu/yolov9

    Args:
        c1 (int): Input channels
        c2 (int): Output channels

    Attributes:
        cv1 (Conv): 3x3 convolution with stride 2

    Example:
        >>> aconv = AConv(64, 128)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = aconv(x)
        >>> print(out.shape)
        torch.Size([1, 128, 16, 16])
    """

    def __init__(self, c1, c2):
        """Initializes pooling and convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Applies average pooling followed by convolution."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """
    ADown module from YOLOv9.

    Reference
    - https://github.com/WongKinYiu/yolov9

    Args:
        c1 (int): Input channels
        c2 (int): Output channels

    Attributes:
        c (int): Half of output channels
        cv1 (Conv): 3x3 convolution with stride 2 for first half
        cv2 (Conv): 1x1 convolution for second half

    Example:
        >>> adown = ADown(128, 256)
        >>> x = torch.randn(1, 128, 32, 32)
        >>> out = adown(x)
        >>> print(out.shape)
        torch.Size([1, 256, 16, 16])
    """

    def __init__(self, c1, c2):
        """Initializes channel split and processing layers."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Processes input through parallel pathways and concatenates results."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """
    ELAN Block with Spatial Pyramid Pooling (SPP-ELAN) from YOLOv9.

    Integrates Spatial Pyramid Pooling within an Efficient Layer Aggregation Network (ELAN) structure.
    Captures multi-scale features through parallel pooling paths while maintaining spatial resolution.

    Reference
    - https://github.com/WongKinYiu/yolov9

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        c3 (int): Intermediate channels for pyramid branches
        k (int, optional): Shared kernel size for pyramid pooling layers. Default: 5

    Attributes:
        cv1 (Conv): 1x1 channel reduction (c1 -> c3)
        cv2-4 (nn.MaxPool2d): Triple identical pooling layers (kernel=k)
        cv5 (Conv): 1x1 channel expansion (4*c3 -> c2)

    Example:
        >>> # Basic usage with 5x5 pooling
        >>> spp_elan = SPPELAN(64, 128, 32)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = spp_elan(x)
        >>> print(out.shape)
        torch.Size([2, 128, 32, 32])

        >>> # Verify feature scales
        >>> x = torch.zeros(1, 64, 64, 64)
        >>> out = spp_elan(x)
        >>> print(out[0, 0, 32, 32].item())  # Central feature response
        0.0
    """

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN components with configurable pyramid depth."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """
        Processes input through multi-scale pyramid aggregation.

        1. Initial feature projection
        2. Parallel pyramid processing
        3. Cross-scale concatenation
        4. Final feature projection

        Maintains spatial resolution through all operations.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """
    Composite Backbone Linear Adapter (CBLinear) for CBNet architectures.

    Implements feature routing between composite backbones in CBNet.
    Part of the Composite Backbone Network framework to enable cross-backbone feature interaction.

    CBNet Paper: https://arxiv.org/abs/1909.03625
    YOLOv9 Implementation: https://github.com/WongKinYiu/yolov9

    Args:
        c1 (int): Input channels from upstream backbone
        c2s (list[int]): Output channel distribution for downstream backbones
                         (e.g., [256, 512] = 2 backbones with 256ch and 512ch inputs)
        k (int, optional): Kernel size for cross-backbone fusion. Default: 1
        s (int, optional): Stride for fusion convolution. Default: 1
        p (int, optional): Padding. Auto-calculated if None. Default: None
        g (int, optional): Groups for grouped convolution. Default: 1

    Attributes:
        fusion_conv (nn.Conv2d): Cross-backbone fusion convolution
        c2s (list): Channel distribution for downstream backbone inputs

    Example:
        >>> # Connecting 2 composite backbones (512ch → [256ch, 256ch] branches)
        >>> adapter = CBLinear(512, [256, 256])
        >>> x = torch.randn(2, 512, 32, 32)
        >>> branch1, branch2 = adapter(x)
        >>> print(branch1.shape, branch2.shape)
        torch.Size([2, 256, 32, 32]) torch.Size([2, 256, 32, 32])

        >>> # 3-way split for multi-backbone architecture
        >>> cb_linear = CBLinear(1024, [256, 384, 384])
        >>> b1, b2, b3 = cb_linear(torch.randn(1, 1024, 16, 16))
        >>> print(b3.shape)
        torch.Size([1, 384, 16, 16])
    """

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes cross-backbone feature router."""
        super().__init__()
        self.c2s = c2s
        total_out = sum(c2s)
        self.fusion_conv = nn.Conv2d(c1, total_out, k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """
        Routes features to multiple downstream backbones.

        Args:
            x (Tensor): Features from upstream backbone (B, c1, H, W)

        Returns:
            (tuple[Tensor]): Split features for downstream backbones
        """
        fused = self.fusion_conv(x)
        return fused.split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """
    Composite Backbone Feature Fusion (CBFuse) module.

    This module fuses feature maps from multiple composite backbones by selecting, resizing,
    and summing features. The final fused feature map maintains the resolution of the
    last backbone's output.

    References:
        - CBNet: Luo et al., *CBNet: A Novel Composite Backbone Network Architecture for Object Detection*,
          ICCV 2019. https://arxiv.org/abs/1909.03625

    Args:
        idx (list[int]): Indices specifying which feature maps to select from
                         each upstream backbone.

    Attributes:
        idx (list[int]): Feature selection indices for upstream backbones.

    Example:
        >>> # Example with 3 backbones
        >>> fuse = CBFuse(idx=[1, 0])
        >>> xs = [
        ...     [torch.randn(2, 256, 32, 32), torch.randn(2, 512, 16, 16)],
        ...     [torch.randn(2, 512, 16, 16), torch.randn(2, 1024, 8, 8)],
        ...     [torch.randn(2, 1024, 8, 8)],  # Last backbone output should also be a list
        ... ]
        >>> fused = fuse(xs)
        >>> print(fused.shape)
        torch.Size([2, 1024, 8, 8])
    """

    def __init__(self, idx):
        """
        Initializes CBFuse module with layer index for selective feature fusion.

        Args:
            idx (list[int]): Indices specifying which feature maps to select.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """
        Forward pass through CBFuse layer.

        Selects and resizes feature maps from multiple backbones, then sums them to produce a fused output.

        Args:
            xs (list[list[torch.Tensor]] | list[torch.Tensor]):
                Feature maps from composite backbones.
                - `xs[:-1]`: List of lists containing feature maps from earlier backbones.
                - `xs[-1]`: The last backbone's output feature map (single tensor), used for target resolution.

        Returns:
            (torch.Tensor): The fused feature map with the same spatial resolution as `xs[-1]`.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """
    Faster Cross Stage Partial (CSP) Bottleneck with 3 convolutions for efficient feature processing.

    Implements an optimized CSP bottleneck layer with multiple bottleneck blocks. Enhances gradient flow
    while reducing computational complexity through partial feature aggregation.

    Args:
        c1 (int): Input channels from upstream layer
        c2 (int): Output channels for downstream layer
        n (int, optional): Number of bottleneck blocks. Default: 1
        shortcut (bool, optional): Enable residual shortcuts in bottlenecks. Default: False
        g (int, optional): Groups for grouped convolution in bottlenecks. Default: 1
        e (float, optional): Expansion ratio for hidden channels. Default: 0.5

    Attributes:
        cv1 (Conv): Initial 1x1 projection convolution
        cv2 (Conv): Secondary 1x1 projection convolution
        cv3 (Conv): Final 1x1 aggregation convolution
        m (nn.ModuleList): Stack of bottleneck blocks

    Example:
        >>> # Basic C3f block with 1 bottleneck
        >>> block = C3f(128, 256)
        >>> x = torch.randn(2, 128, 56, 56)
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([2, 256, 56, 56])

        >>> # Multi-bottleneck configuration with expansion
        >>> c3f = C3f(256, 512, n=3, e=0.75)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> print(c3f(x).shape)
        torch.Size([1, 512, 32, 32])

        >>> # With residual shortcuts and groups
        >>> bottleneck = C3f(64, 128, n=2, shortcut=True, g=2)
        >>> x = torch.randn(4, 64, 128, 128)
        >>> print(bottleneck(x).shape)
        torch.Size([4, 128, 128, 128])
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """
        Forward pass through C3f layer using split-transform-merge strategy.

        Args:
            x (Tensor): Input features of shape (B, c1, H, W)

        Returns:
            (Tensor): Processed features of shape (B, c2, H, W)
        """
        y = [self.cv2(x), self.cv1(x)]  # Initial split projections
        y.extend(m(y[-1]) for m in self.m)  # Sequential processing through bottlenecks
        return self.cv3(torch.cat(y, 1))  # Concatenate and fuse features


class C3k2(C2f):
    """
    Enhanced CSP Bottleneck with Dynamic Block Selection (C3k/Bottleneck).

    Implements a composite CSP bottleneck that dynamically selects between C3k blocks
    and standard Bottleneck blocks for feature processing. Inherits architecture
    fundamentals from C2f with enhanced block flexibility.

    Args:
        c1 (int): Input channels from upstream layer
        c2 (int): Output channels for downstream layer
        n (int, optional): Number of processing blocks. Default: 1
        c3k (bool, optional): Enable C3k blocks instead of Bottleneck. Default: False
        e (float, optional): Hidden channels expansion ratio. Default: 0.5
        g (int, optional): Groups for grouped convolution. Default: 1
        shortcut (bool, optional): Enable residual connections. Default: True

    Attributes:
        m (nn.ModuleList): Dynamic stack of processing blocks (C3k or Bottleneck)

    Example:
        >>> # Default configuration with Bottleneck blocks
        >>> block = C3k2(64, 128)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> print(block(x).shape)
        torch.Size([2, 128, 32, 32])

        >>> # C3k block activation with expansion
        >>> c3k2 = C3k2(256, 512, n=3, c3k=True, e=0.75)
        >>> print(c3k2(torch.randn(1, 256, 16, 16)).shape)
        torch.Size([1, 512, 16, 16])

        >>> # Mixed configuration with groups
        >>> module = C3k2(128, 256, n=2, g=2, shortcut=False)
        >>> x = torch.randn(4, 128, 64, 64)
        >>> print(module(x).shape)
        torch.Size([4, 256, 64, 64])
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module with dynamic block selection."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            (C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)) for _ in range(n)
        )


class C3k(C3):
    """
    Custom Kernel CSP Bottleneck (C3k) with Adaptive Receptive Field Control.

    Extends standard C3 architecture with configurable convolutional kernel sizes
    for enhanced feature extraction flexibility. Enables dynamic adjustment of
    receptive field characteristics.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int, optional): Number of bottleneck blocks. Default: 1
        shortcut (bool, optional): Enable residual connections. Default: True
        g (int, optional): Groups for grouped convolution. Default: 1
        e (float, optional): Hidden channels expansion ratio. Default: 0.5
        k (int, optional): Kernel size for bottleneck convolutions. Default: 3

    Attributes:
        m (nn.Sequential): Stack of Bottleneck layers with custom kernel sizes

    Example:
        >>> # Standard 3x3 kernel configuration
        >>> block = C3k(64, 128)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> print(block(x).shape)
        torch.Size([2, 128, 32, 32])

        >>> # Large kernel configuration
        >>> c3k_large = C3k(256, 512, k=5)
        >>> print(c3k_large(torch.randn(1, 256, 16, 16)).shape)
        torch.Size([1, 512, 16, 16])

        >>> # Asymmetric kernel configuration (requires Bottleneck modification)
        >>> # Note: Current implementation uses square kernels only
        >>> # Could be extended with separate width/height parameters
        >>> # c3k_asym = C3k(128, 256, k=(3,5))  # Hypothetical extension
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes C3k with configurable kernel size."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """
    RepVGG Depthwise Separable Block with structural reparameterization.

    Implements dual-branch depthwise convolution block for training, which can be fused
    into single 7x7 depthwise convolution during inference. Part of RepVGG architecture
    optimization strategy.

    Args:
        ed (int): Number of input/output channels (must equal groups for depthwise conv)

    Attributes:
        conv (Conv): 7x7 depthwise convolution branch
        conv1 (Conv): 3x3 depthwise convolution branch (padded to 7x7 during fusion)
        act (nn.SiLU): Activation function
        dim (int): Channel dimension preserved through operations

    Example:
        >>> # Training phase with dual branches
        >>> block = RepVGGDW(64).train()
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([1, 64, 32, 32])

        >>> # Inference phase after fusion
        >>> block.eval()
        >>> block.fuse()
        >>> out_fused = block(x)
        >>> torch.testing.assert_close(out, out_fused)  # Should match
    """

    def __init__(self, ed) -> None:
        """Initializes dual depthwise convolution branches."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)  # 7x7 DWConv
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)  # 3x3 DWConv
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Dual-branch forward pass during training.

        Args:
            x (Tensor): Input tensor of shape (B, ed, H, W)

        Returns:
            (Tensor): Activated sum of 7x7 and 3x3 branch outputs
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Single-branch forward pass for fused weights.

        Args:
            x (Tensor): Input tensor of shape (B, ed, H, W)

        Returns:
            (Tensor): Output using fused 7x7 convolution weights
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses 3x3 branch into 7x7 convolution weights.

        Mathematical Fusion:
            1. Pad 3x3 kernel to 7x7 with zeros
            2. Sum with original 7x7 kernel weights
            3. Update conv weights in-place
            4. Remove 3x3 branch reference
        """
        # Fuse Conv+BN for both branches
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        # Weight/Bias extraction
        conv_w, conv_b = conv.weight, conv.bias
        conv1_w, conv1_b = conv1.weight, conv1.bias

        # Zero-pad 3x3->7x7 kernel
        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])  # [left, right, top, bottom]

        # Combine weights
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        # Update parameters
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        # Replace layers
        self.conv = conv
        del self.conv1  # Remove reference to 3x3 branch


class CIB(nn.Module):
    """
    Compact Inverted Block (CIB) from YOLOv10.

    Implements an efficient inverted residual block using structural reparameterization technique from RepVGG.
    The "large kernel" refers to the 3x3 convolution enhanced with multi-branch training-time optimization.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        shortcut (bool, optional): Enable residual connection. Default: True
        e (float, optional): Hidden layer expansion ratio. Default: 0.5
        lk (bool, optional): Use RepVGG-style reparameterizable 3x3 conv (True) or standard 3x3 conv (False).
                            Implements training-time multi-branch -> inference-time single branch conversion. Default: False

    Attributes:
        cv1 (nn.Sequential): Core processing stages:
            1. 3x3 Depthwise Conv: Spatial feature mixing
            2. 1x1 Conv: Channel expansion (2×)
            3. Large Kernel Block: RepVGGDW (train-time multi-branch) or standard DWConv
            4. 1x1 Conv: Channel compression
            5. 3x3 Depthwise Conv: Final spatial processing
        add (bool): Residual connection enabled when input/output channels match

    Example:
        >>> # Training-phase RepVGG-style with multi-branch
        >>> cib = CIB(64, 64, lk=True).train()
        >>> x = torch.randn(1, 64, 32, 32)
        >>> print([m.shape for m in cib.cv1[2].body])  # Show RepVGG branches
        [torch.Size([128, 1, 3, 3]), torch.Size([128, 1, 1, 1]), torch.Size([128])]

        >>> # Inference-phase optimized single branch
        >>> cib.eval()
        >>> out = cib(x)
        >>> print(out.shape)
        torch.Size([1, 64, 32, 32])
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes CIB block with RepVGG-style optimization option."""
        super().__init__()
        c_ = int(c2 * e)  # Hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),  # Spatial mixing
            Conv(c1, 2 * c_, 1),  # Channel expansion
            (RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_)),  # Large kernel choice
            Conv(2 * c_, c2, 1),  # Channel compression
            Conv(c2, c2, 3, g=c2),  # Final processing
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Executes forward pass with optional residual connection.

        Args:
            x (Tensor): Input tensor (B, c1, H, W)

        Returns:
            (Tensor): Output tensor (B, c2, H, W) preserving spatial dimensions
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2f with RepVGG-enhanced Compact Inverted Blocks (CIB).

    YOLOv10-optimized variant using stack of CIB blocks.
    Combines channel splitting/merging with multi-branch->single branch optimization.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int, optional): Number of CIB blocks to stack. Default: 1
        shortcut (bool, optional): Enable cross-block residual connections. Default: False
        lk (bool, optional): Use RepVGG-style reparameterizable 3x3 conv in CIB blocks.
                           Enables multi-branch (3x3+1x1+identity) during training ->
                           converts to single 3x3 during inference. Default: False
        g (int, optional): Grouping factor for base convolutions. Default: 1
        e (float, optional): Expansion ratio for hidden channels in CIB blocks.
                           Hidden channels = c2 * e. Default: 0.5

    Attributes:
        m (nn.ModuleList): Stack of n CIB blocks with RepVGG optimization
        cv1 (Conv): Input channel splitting convolution (inherited from C2f)
        cv2 (Conv): Output channel merging convolution (inherited from C2f)

    Example:
        >>> # Basic usage with default parameters
        >>> model = C2fCIB(64, 128)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([2, 128, 32, 32])

        >>> # Advanced configuration with RepVGG and expansion
        >>> model = C2fCIB(128, 256, n=3, lk=True, e=0.75)
        >>> x = torch.randn(1, 128, 64, 64)
        >>> print(model(x).shape)
        torch.Size([1, 256, 64, 64])

        >>> # Inspect RepVGG branches during training
        >>> model.train()
        >>> print(model.m[0].cv1[2].body)  # Show RepVGG multi-branch layers
        [Conv2d(...), Conv2d(...), ...]
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes C2fCIB."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Standard Multi-Head Self-Attention module with positional encoding.

    Implements vanilla self-attention mechanism with convolutional projections and positional encoding.

    Args:
        dim (int): Input channel dimension
        num_heads (int, optional): Number of parallel attention heads. Default: 8
        attn_ratio (float, optional): Ratio of key dimension to head dimension. Default: 0.5

    Attributes:
        qkv (Conv): 1x1 convolution for generating query/key/value projections
        proj (Conv): 1x1 convolution for final output projection
        pe (Conv): Depthwise convolution for positional encoding
        scale (float): Normalization factor for attention scores

    Example:
        >>> # Basic self-attention operation
        >>> attn = Attention(dim=256)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> out = attn(x)
        >>> print(out.shape)
        torch.Size([2, 256, 32, 32])

        >>> # Different head configuration
        >>> model = Attention(dim=512, num_heads=16)
        >>> print(model(torch.randn(1, 512, 64, 64)).shape)
        torch.Size([1, 512, 64, 64])
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes attention components and projection layers."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.qkv = Conv(dim, h, 1, act=False)  # Query/Key/Value projection
        self.proj = Conv(dim, dim, 1, act=False)  # Output projection
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # Positional encoding

    def forward(self, x):
        """Performs standard self-attention computation.
        1. Project input to Q/K/V tensors
        2. Compute attention scores
        3. Apply softmax normalization
        4. Combine values with attention weights
        5. Add positional encoding
        6. Project to output space.

        Args:
            x (Tensor): Input tensor of shape (B, dim, H, W)

        Returns:
            (Tensor): Output tensor of same shape (B, dim, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # Generate Q/K/V projections
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        # Attention computation
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # Combine and project
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        x = x + self.pe(v.reshape(B, C, H, W))  # Positional encoding
        return self.proj(x)


class PSABlock(nn.Module):
    """
    Partial Self-Attention Block (PSABlock) with residual connections.

    Implements efficient partial self-attention mechanism from YOLOv10, combining multi-head attention and
    feed-forward operations in a transformer-style block optimized for vision tasks. Reduces head redundancy
    while maintaining spatial resolution.

    Attributes:
        attn (Attention): Partial self-attention module with optimized head configuration
        ffn (nn.Sequential): Feed-forward network with channel expansion/contraction (c → 2c → c)
        add (bool): Enables residual connections when True

    Args:
        c (int): Number of input/output channels
        attn_ratio (float, optional): Ratio of attention channels to total channels. Default: 0.5
        num_heads (int, optional): Number of parallel attention heads. Default: 4
        shortcut (bool, optional): Enables residual connections for stable training. Default: True

    Example:
        >>> # Basic usage with residual connections
        >>> block = PSABlock(c=128)
        >>> x = torch.randn(1, 128, 32, 32)
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([1, 128, 32, 32])

        >>> # Without residual connections
        >>> model = PSABlock(c=64, shortcut=False)
        >>> x = torch.randn(2, 64, 16, 16)
        >>> print(model(x).shape)
        torch.Size([2, 64, 16, 16])

        >>> # Verify attention transformation
        >>> block = PSABlock(256)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = block(x)
        >>> print(torch.allclose(x, out, atol=1e-3))  # Should be False
        False
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes partial self-attention and FFN layers with residual option."""
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),  # Channel expansion
            Conv(c * 2, c, 1, act=False),  # Channel compression
        )
        self.add = shortcut

    def forward(self, x):
        """
        Processes input through partial self-attention and FFN with optional residuals.

        1. Partial self-attention with residual: x' = x + attn(x)
        2. FFN with residual: out = x' + ffn(x')
        Residuals skipped when shortcut=False

        Args:
            x (Tensor): Input tensor of shape (B, c, H, W)

        Returns:
            (Tensor): Enhanced output of same shape (B, c, H, W)
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    Partial Self-Attention (PSA) module for efficient global feature modeling.

    Enhances standard self-attention by reducing head redundancy while maintaining spatial resolution. Based on
    YOLOv10's improvements showing 0.3% AP gain with 0.05ms latency reduction compared to standard transformer blocks.

    Attributes:
        attn (Attention): Partial self-attention module with optimized head configuration
        ffn (nn.Sequential): Feed-forward network with channel expansion/contraction

    Args:
        c1 (int): Input channels. Must equal c2.
        c2 (int): Output channels. Must equal c1.
        e (float, optional): Hidden dimension expansion ratio. Default: 0.5.

    Example:
        >>> # Basic usage with default parameters
        >>> psa = PSA(256, 256)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = psa(x)
        >>> print(out.shape)
        torch.Size([1, 256, 64, 64])

        >>> # With custom expansion ratio
        >>> model = PSA(128, 128, e=0.75)
        >>> print(model(torch.randn(2, 128, 32, 32)).shape)
        torch.Size([2, 128, 32, 32])
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes PSA with channel splitting, attention, and FFN components."""
        super().__init__()
        assert c1 == c2, f"PSA requires c1 == c2, got {c1} vs {c2}"
        self.c = int(c1 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, 2 * self.c, 1), Conv(2 * self.c, self.c, 1, act=False))

    def forward(self, x):
        """Processes input through partial self-attention and FFN with residuals."""
        a, b = self.cv1(x).split([self.c, self.c], dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat([a, b], 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Args:
        c1 (int): Number of input channels. Must equal c2.
        c2 (int): Number of output channels. Must equal c1.
        n (int, optional): Number of PSABlock modules to stack. Default is 1.
        e (float, optional): Hidden channels expansion ratio. Calculated as c1 * e. Default is 0.5.

    Attributes:
        c (int): Expanded hidden channels calculated as int(c1 * e).
        cv1 (Conv): 1x1 convolution that splits input into two hidden channel streams.
        cv2 (Conv): 1x1 convolution that merges processed features back to original channels.
        m (nn.Sequential): Sequence of PSABlock modules for feature processing.

    Examples:
        >>> # Basic usage with equal channels
        >>> c2psa = C2PSA(c1=256, c2=256, n=3)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = c2psa(x)
        >>> print(out.shape)
        torch.Size([1, 256, 64, 64])

        >>> # Using expansion ratio 0.25
        >>> model = C2PSA(c1=128, c2=128, e=0.25, n=2)
        >>> x = torch.randn(4, 128, 32, 32)
        >>> print(model(x).shape)
        torch.Size([4, 128, 32, 32])
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initializes the C2PSA module with specified parameters.

        Args:
            c1 (int): Input channels (must equal c2)
            c2 (int): Output channels (must equal c1)
            n (int, optional): Number of PSABlock modules. Default is 1.
            e (float, optional): Hidden channels expansion ratio. Default is 0.5.
        """
        super().__init__()
        assert c1 == c2, "C2PSA requires c1 == c2"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Split input into two streams
        self.cv2 = Conv(2 * self.c, c1, 1)  # Merge processed features

        self.m = nn.Sequential(*[PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)])

    def forward(self, x):
        """
        Forward pass through the C2PSA module.

        Splits input into two streams, processes one through PSABlocks, then merges back.

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)

        Returns:
            (torch.Tensor): Output tensor of shape (B, c2, H, W) with enhanced features

        Example:
            >>> # Verify input-output shape consistency
            >>> model = C2PSA(c1=64, c2=64)
            >>> x = torch.randn(3, 64, 128, 128)
            >>> out = model(x)
            >>> print(out.shape)
            torch.Size([3, 64, 128, 128])
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Args:
        c1 (int): Number of input channels. Must be equal to c2.
        c2 (int): Number of output channels. Must be equal to c1.
        n (int, optional): Number of PSA blocks. Default is 1.
        e (float, optional): Expansion ratio for hidden channels. Default is 0.5.

    Attributes:
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Examples:
        >>> # Basic usage with square input
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([1, 64, 128, 128])

        >>> # Edge case with minimal configuration
        >>> model = C2fPSA(c1=32, c2=32, n=1)
        >>> x = torch.randn(4, 32, 64, 64)
        >>> print(model(x).shape)
        torch.Size([4, 32, 64, 64])
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initializes the C2fPSA module with specified parameters.

        Args:
            c1 (int): Number of input channels. Must equal c2.
            c2 (int): Number of output channels. Must equal c1.
            n (int, optional): Number of PSA blocks. Default is 1.
            e (float, optional): Expansion ratio for hidden channels. Default is 0.5.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))

    def forward(self, x):
        """
        Forward pass through the C2fPSA module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, c2, H, W) maintaining spatial dimensions.
        """
        return super().forward(x)


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int): Kernel size for the depthwise convolution layer.
        s (int): Stride for the depthwise convolution layer.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces channels from c1 to c2.
        cv2 (Conv): Depthwise convolution layer with kernel size `k` and stride `s` for spatial downsampling.

    Examples:
        >>> # Basic usage with ResNet-like dimensions
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])

        >>> # Case with stride=1 (no spatial reduction)
        >>> model = SCDown(c1=32, c2=32, k=3, s=1)
        >>> x = torch.randn(2, 32, 64, 64)
        >>> print(model(x).shape)
        torch.Size([2, 32, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """
        Initializes the SCDown module with specified parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the depthwise convolution layer.
            s (int): Stride for the depthwise convolution layer.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Forward pass through the SCDown module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, c2, H//s, W//s) when using stride s > 1.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights,
    and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers.
            Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Example:
        >>> # Load a pretrained ResNet50 and get intermediate outputs
        >>> model = TorchVision("resnet50", weights="ResNet50_Weights.IMAGENET1K_V1", unwrap=True, split=True)
        >>> input_tensor = torch.rand(1, 3, 224, 224)
        >>> outputs = model(input_tensor)  # List of tensors from each layer

        >>> # Load EfficientNet without unwrapping
        >>> model = TorchVision("efficientnet_b0", unwrap=False)
        >>> output = model(input_tensor)  # Single output tensor
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output features. If `split` is True, returns a list of intermediate
                outputs from each layer; otherwise, returns the final output tensor.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y
