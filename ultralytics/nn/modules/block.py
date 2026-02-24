# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "SC_ELAN",
    "SC_ELAN_Fixed",
    "SC_ELAN_Efficient",
    "SC_ELAN_Dilated",
    "SC_ELAN_Slim",
    "SC_ELAN_LSKA",
    "SC_ELAN_LSKA_TSCG",
    "SC_ELAN_RepAdd",
    "SC_ELAN_RepExact",
    "TorchVision",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semseg: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semseg:
            semseg = self.semseg(feat)
            return (p, semseg)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det


class ContextAwareRepConv(nn.Module):
    """Context-Aware Reparameterizable Convolution.

    Integrates Pzconv's large kernel idea with RepVGG-style re-parameterization.
    Training: Multi-branch (1x1, 3x3, 5x5) to capture multi-scale context.
    Inference: Collapses into a single 3x3 convolution for speed.

    Attributes:
        deploy (bool): Whether the module is in deployment mode.
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        act (nn.Module): Activation function.
        rbr_reparam (nn.Conv2d): Reparameterized convolution (deploy mode only).
        rbr_identity (nn.BatchNorm2d): Identity branch with batch normalization.
        rbr_dense (nn.Sequential): Dense 3x3 convolution branch.
        rbr_context (nn.Sequential): Context branch with 5x5 depthwise convolution.
        rbr_1x1 (nn.Sequential): 1x1 convolution branch.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, g: int = 1, act: bool = True, deploy: bool = False):
        """Initialize ContextAwareRepConv module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size. Defaults to 3.
            s (int): Stride. Defaults to 1.
            p (int | None): Padding. Defaults to None (auto-pad).
            g (int): Groups for convolution. Defaults to 1.
            act (bool): Whether to use activation. Defaults to True.
            deploy (bool): Whether in deployment mode. Defaults to False.
        """
        super().__init__()
        self.deploy = deploy
        self.c1 = c1
        self.c2 = c2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(c1) if c2 == c1 and s == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            # Large kernel branch (Context Aware)
            self.rbr_context = nn.Sequential(
                nn.Conv2d(c1, c2, 5, s, autopad(5, p), groups=c1, bias=False),  # Depthwise 5x5
                nn.Conv2d(c2, c2, 1, 1, 0, bias=False),  # Pointwise 1x1
                nn.BatchNorm2d(c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, autopad(1, p), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through ContextAwareRepConv."""
        if self.deploy:
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(
            self.rbr_dense(inputs) + 
            self.rbr_1x1(inputs) + 
            self.rbr_context(inputs) + 
            id_out
        )


class SplitInteractionBlock(nn.Module):
    """Split Interaction Block for feature purification.

    Integrates FCM's interaction idea.
    Splits features and uses cross-branch attention to suppress background noise.

    Attributes:
        split_dim (int): The dimension for splitting features.
        spatial_att (nn.Sequential): Spatial attention generator.
        channel_att (nn.AdaptiveAvgPool2d): Channel attention pooling.
        fc_channel (nn.Sequential): Channel attention fully connected layers.
    """

    def __init__(self, dim: int):
        """Initialize SplitInteractionBlock.

        Args:
            dim (int): Input feature dimension.
        """
        super().__init__()
        self.split_dim = dim // 2
        
        # Spatial Attention Generator (for Branch 1)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(self.split_dim, 1, 7, padding=3),
            nn.Sigmoid()
        )
        # Channel Attention Generator (for Branch 2)
        self.channel_att = nn.AdaptiveAvgPool2d(1)
        self.fc_channel = nn.Sequential(
             nn.Conv2d(self.split_dim, self.split_dim, 1),
             nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SplitInteractionBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with interaction applied.
        """
        # 1. Split: Context vs Content
        x1, x2 = torch.split(x, self.split_dim, dim=1)
        
        # 2. Interaction
        # Use x2 (context) to spatially validate x1 (content)
        x1_out = x1 * self.spatial_att(x2)
        
        # Use x1 (content) to channel-wise validate x2 (context)
        x2_out = x2 * self.fc_channel(self.channel_att(x1))
        
        # 3. Merge
        return torch.cat([x1_out, x2_out], dim=1)


class SC_ELAN(nn.Module):
    """SC-ELAN: Spatial-Context Efficient Layer Aggregation Network.

    Combines ELAN backbone + Pzconv Context + FCM Interaction.
    Designed for enhanced small object detection through multi-scale context awareness
    and feature interaction mechanisms.

    Attributes:
        c (int): Half of the intermediate channels.
        cv1 (Conv): Initial 1x1 convolution for projection.
        cv2 (ContextAwareRepConv): First context-aware convolution.
        cv3 (ContextAwareRepConv): Second context-aware convolution.
        interaction (SplitInteractionBlock): Feature interaction block.
        cv4 (Conv): Final 1x1 convolution for aggregation.
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int, c5: int = 1):
        """Initialize SC_ELAN module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            c3 (int): Unused (kept for compatibility with C2f args).
            c4 (int): Unused (kept for compatibility).
            c5 (int): Unused (kept for compatibility). Defaults to 1.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, c2, 1, 1)
        
        # ELAN Backbone with ContextAware RepConvs
        self.cv2 = ContextAwareRepConv(c2 // 2, c2 // 2)
        self.cv3 = ContextAwareRepConv(c2 // 2, c2 // 2)
        
        # Interaction Block for cleanup
        self.interaction = SplitInteractionBlock(c2)
        
        # Final aggregation
        self.cv4 = Conv(c2 + (2 * (c2 // 2)), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with enhanced features.
        """
        # 1. Projection & Split
        y = list(self.cv1(x).chunk(2, 1))
        
        # 2. Context-Aware Processing Path
        # Process the second half through the chain
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        
        # 3. Concatenation (Gradient Highway)
        feat_cat = torch.cat(y, 1)
        
        # 4. Final Projection
        return self.cv4(feat_cat)


class DilatedRepConv(nn.Module):
    """Dilated Reparameterizable Convolution.

    Variant using Dilated Convolution instead of large dense kernels.
    Receptive field: 3x3 (local) + 3x3 dilated (global context).

    Attributes:
        deploy (bool): Whether the module is in deployment mode.
        act (nn.Module): Activation function.
        rbr_reparam (nn.Conv2d): Reparameterized convolution (deploy mode only).
        rbr_dense (nn.Sequential): Dense 3x3 convolution branch.
        rbr_dilated (nn.Sequential): Dilated 3x3 convolution branch.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, g: int = 1, act: bool = True, deploy: bool = False):
        """Initialize DilatedRepConv module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size. Defaults to 3.
            s (int): Stride. Defaults to 1.
            p (int | None): Padding. Defaults to None (auto-pad).
            g (int): Groups for convolution. Defaults to 1.
            act (bool): Whether to use activation. Defaults to True.
            deploy (bool): Whether in deployment mode. Defaults to False.
        """
        super().__init__()
        self.deploy = deploy
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            # Dilated Branch: Rate=2, behaves like 5x5 but far fewer params
            self.rbr_dilated = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, padding=2, dilation=2, groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through DilatedRepConv."""
        if self.deploy:
            return self.act(self.rbr_reparam(inputs))
        return self.act(self.rbr_dense(inputs) + self.rbr_dilated(inputs))


class SC_ELAN_Dilated(SC_ELAN):
    """SC-ELAN-Dilated: Variant focusing on receptive field expansion.

    Hypothesis: Small objects require a massive receptive field to be distinguished from background,
    but large dense kernels are heavy. Dilated convolutions offer a large view with zero extra parameters.

    This variant uses DilatedRepConv instead of ContextAwareRepConv for improved receptive field
    with minimal parameter overhead.
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int, c5: int = 1):
        """Initialize SC_ELAN_Dilated module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            c3 (int): Unused (kept for compatibility).
            c4 (int): Unused (kept for compatibility).
            c5 (int): Unused (kept for compatibility). Defaults to 1.
        """
        super().__init__(c1, c2, c3, c4, c5)
        # Override the conv layers with Dilated version
        self.cv2 = DilatedRepConv(c2 // 2, c2 // 2)
        self.cv3 = DilatedRepConv(c2 // 2, c2 // 2)


class SC_ELAN_Slim(nn.Module):
    """SC-ELAN-Slim: Lightweight variant focusing on speed/efficiency.

    Hypothesis: For edge devices, we need the "Context" but not the heavy "Split-Interaction" computation.
    This variant keeps the Pzconv context but simplifies the fusion.

    Attributes:
        c (int): Half of the intermediate channels.
        cv1 (Conv): Initial 1x1 convolution for projection.
        cv2 (ContextAwareRepConv): First context-aware convolution.
        cv3 (ContextAwareRepConv): Second context-aware convolution.
        cv4 (Conv): Final 1x1 convolution for aggregation.
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int, c5: int = 1):
        """Initialize SC_ELAN_Slim module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            c3 (int): Unused (kept for compatibility).
            c4 (int): Unused (kept for compatibility).
            c5 (int): Unused (kept for compatibility). Defaults to 1.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, c2, 1, 1)
        # Use simple Pzconv-style repconvs
        self.cv2 = ContextAwareRepConv(c2 // 2, c2 // 2)
        self.cv3 = ContextAwareRepConv(c2 // 2, c2 // 2)
        # Standard fusion without complex interaction
        self.cv4 = Conv(c2 + (2 * (c2 // 2)), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN_Slim module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with enhanced features.
        """
        y = list(self.cv1(x).chunk(2, 1))
        # Standard ELAN flow
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class LiteSplitInteraction(nn.Module):
    """Lightweight split interaction block for efficient cross-branch feature gating."""

    def __init__(self, dim: int, p: float = 0.5):
        """Initialize LiteSplitInteraction.

        Args:
            dim (int): Input/output channels.
            p (float): Ratio of channels for the first branch.
        """
        super().__init__()
        c1 = max(1, min(dim - 1, int(dim * p)))
        self.c1 = c1
        self.c2 = dim - c1

        self.spatial = nn.Sequential(
            DWConv(self.c2, self.c2, 3, 1),
            nn.Conv2d(self.c2, 1, 1),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c1, self.c2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight split interaction."""
        xa, xb = x.split((self.c1, self.c2), dim=1)
        xa = xa * self.spatial(xb)
        xb = xb * self.channel(xa)
        return torch.cat((xa, xb), dim=1)


class SC_ELAN_Efficient(nn.Module):
    """SC-ELAN-Efficient: SC-ELAN variant with elastic width and lightweight interaction.

    Keeps SC-ELAN principles (split -> context chain -> concat -> fusion + interaction) while
    reducing complexity through hidden-width scaling and lightweight split gating.
    """

    def __init__(self, c1: int, c2: int, e: float = 0.375, p: float = 0.5, c5: int = 1):
        """Initialize SC_ELAN_Efficient module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            e (float): Hidden width ratio relative to c2.
            p (float): Interaction split ratio.
            c5 (int): Unused (kept for YAML compatibility).
        """
        super().__init__()
        self.c = max(8, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Sequential(DWConv(self.c, self.c, 3, 1), Conv(self.c, self.c, 1, 1))
        self.cv3 = nn.Sequential(DWConv(self.c, self.c, 3, 1), Conv(self.c, self.c, 1, 1))
        self.cv4 = Conv(4 * self.c, c2, 1, 1)
        self.interaction = LiteSplitInteraction(c2, p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN_Efficient."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        y = self.cv4(torch.cat(y, 1))
        return self.interaction(y)


class SC_ELAN_Fixed(SC_ELAN):
    """SC-ELAN-Fixed: Corrected version of SC-ELAN with active feature interaction.

    This class addresses a bug in the original SC_ELAN where the SplitInteractionBlock 
    was initialized but never executed in the forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN_Fixed module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with enhanced features and interaction applied.
        """
        # 1. Projection & Split
        y = list(self.cv1(x).chunk(2, 1))
        
        # 2. Context-Aware Processing Path
        # Process the second half through the chain
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        
        # 3. Concatenation (Gradient Highway)
        feat_cat = torch.cat(y, 1)
        
        # 4. Final Projection followed by Interaction
        # Apply the interaction block to the output of the final projection (c2 channels)
        return self.interaction(self.cv4(feat_cat))


class LSKA(nn.Module):
    """LSKA (Large Separable Kernel Attention) module.

    A novel attention mechanism that decomposes large kernels into horizontal and vertical 
    1D convolutions to capture long-range dependencies efficiently.
    Based on: https://arxiv.org/abs/2309.01439

    Attributes:
        k_size (int): Large kernel size (e.g., 7, 11, ..., 53).
        conv0h (nn.Conv2d): Horizontal decomposition part 1.
        conv0v (nn.Conv2d): Vertical decomposition part 1.
        conv_spatial_h (nn.Conv2d): Horizontal decomposition part 2 (dilated).
        conv_spatial_v (nn.Conv2d): Vertical decomposition part 2 (dilated).
        conv1 (nn.Conv2d): Final 1x1 convolution.
    """

    def __init__(self, dim, k_size=7):
        """Initialize LSKA module."""
        super().__init__()
        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSKA module."""
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return x * attn


class SC_ELAN_LSKA(SC_ELAN):
    """SC-ELAN-LSKA: Variant replacing SplitInteractionBlock with LSKA.
    
    Uses Large Separable Kernel Attention (LSKA) for better long-range dependency modeling
    and efficient spatial attention.
    
    Attributes:
        interaction (LSKA): LSKA attention module replacing the original interaction block.
    """

    def __init__(self, c1, c2, c3, c4, c5=1, lsk_k: int = 7):
        """Initialize SC_ELAN_LSKA module.
        
        Args:
            c1, c2, c3, c4, c5: See SC_ELAN documentation.
            lsk_k (int): LSKA kernel size.
        """
        super().__init__(c1, c2, c3, c4, c5)
        # Replace interaction block with LSKA
        self.interaction = LSKA(c2, k_size=lsk_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN_LSKA module."""
        # 1. Projection & Split
        y = list(self.cv1(x).chunk(2, 1))
        
        # 2. Context-Aware Processing Path
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        
        # 3. Concatenation
        feat_cat = torch.cat(y, 1)
        
        # 4. Final Projection followed by LSKA Interaction
        return self.interaction(self.cv4(feat_cat))


class TinySelectiveContextGate(nn.Module):
    """Tiny-selective context gating.

    Generates a spatial gate from local detail cues and uses it to selectively blend
    context-enhanced features with identity features.
    """

    def __init__(self, c: int, reduction: int = 4):
        """Initialize TinySelectiveContextGate module.

        Args:
            c (int): Number of feature channels.
            reduction (int): Channel reduction ratio for gate predictor.
        """
        super().__init__()
        hidden = max(8, c // reduction)
        self.detail = nn.Sequential(
            DWConv(c, c, 3, 1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Blend identity and context features with a data-dependent gate."""
        gate = self.detail(x)
        return x + gate * (context - x)


class SC_ELAN_LSKA_TSCG(SC_ELAN_LSKA):
    """SC-ELAN-LSKA-TSCG: LSKA variant with Tiny-Selective Context Gate.

    Built on SC_ELAN_LSKA and adds a lightweight gate to selectively apply context
    enhancement on likely small-object-sensitive regions.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        c5: int = 1,
        lsk_k: int = 7,
        tscg_reduction: int = 4,
    ):
        """Initialize SC_ELAN_LSKA_TSCG module.

        Args:
            c1, c2, c3, c4, c5: See SC_ELAN documentation.
            lsk_k (int): LSKA kernel size.
            tscg_reduction (int): Reduction ratio for TinySelectiveContextGate.
        """
        super().__init__(c1, c2, c3, c4, c5, lsk_k=lsk_k)
        self.tscg = TinySelectiveContextGate(c2, reduction=tscg_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN_LSKA_TSCG module."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        feat = self.cv4(torch.cat(y, 1))
        context = self.interaction(feat)
        return self.tscg(feat, context)


class RepMultiKernelConv(nn.Module):
    """Re-parameterizable multi-kernel convolution.

    Training uses parallel branches (1x1, 3x3, 5x5, 7x7, optional identity BN).
    Deployment fuses all branches into one kxk conv (default k=7).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        s: int = 1,
        g: int = 1,
        act: bool = True,
        deploy: bool = False,
        max_k: int = 7,
    ):
        """Initialize RepMultiKernelConv.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            g (int): Convolution groups.
            act (bool): Use SiLU activation when True.
            deploy (bool): Build in deploy mode.
            max_k (int): Deploy kernel size.
        """
        super().__init__()
        self.deploy = deploy
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.g = g
        self.max_k = max_k
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, max_k, s, autopad(max_k), groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, autopad(1), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            self.rbr_3x3 = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, autopad(3), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            self.rbr_5x5 = nn.Sequential(
                nn.Conv2d(c1, c2, 5, s, autopad(5), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            self.rbr_7x7 = nn.Sequential(
                nn.Conv2d(c1, c2, 7, s, autopad(7), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse Conv2d + BatchNorm2d into equivalent kernel and bias."""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @staticmethod
    def _pad_kernel(kernel: torch.Tensor, target_k: int) -> torch.Tensor:
        """Pad kernel to target size (target_k x target_k)."""
        k = kernel.shape[-1]
        if k == target_k:
            return kernel
        p = (target_k - k) // 2
        return F.pad(kernel, [p, p, p, p])

    def _fuse_identity_bn(self, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert identity BN branch into an equivalent convolution kernel and bias."""
        input_dim = self.c1 // self.g
        kernel = torch.zeros((self.c2, input_dim, self.max_k, self.max_k), device=bn.weight.device, dtype=bn.weight.dtype)
        center = self.max_k // 2
        for i in range(self.c2):
            kernel[i, i % input_dim, center, center] = 1.0

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute equivalent fused kernel and bias in max_k space."""
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        k3, b3 = self._fuse_conv_bn(self.rbr_3x3[0], self.rbr_3x3[1])
        k5, b5 = self._fuse_conv_bn(self.rbr_5x5[0], self.rbr_5x5[1])
        k7, b7 = self._fuse_conv_bn(self.rbr_7x7[0], self.rbr_7x7[1])

        eq_k = self._pad_kernel(k1, self.max_k) + self._pad_kernel(k3, self.max_k) + self._pad_kernel(k5, self.max_k) + self._pad_kernel(k7, self.max_k)
        eq_b = b1 + b3 + b5 + b7

        if self.rbr_identity is not None:
            kid, bid = self._fuse_identity_bn(self.rbr_identity)
            eq_k = eq_k + kid
            eq_b = eq_b + bid

        return eq_k, eq_b

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuse training branches and convert module to deploy mode."""
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.c1, self.c2, self.max_k, self.s, autopad(self.max_k), groups=self.g, bias=True)
        self.rbr_reparam.weight.data.copy_(kernel)
        self.rbr_reparam.bias.data.copy_(bias)

        del self.rbr_1x1, self.rbr_3x3, self.rbr_5x5, self.rbr_7x7
        if self.rbr_identity is not None:
            del self.rbr_identity
        self.deploy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through re-parameterizable multi-kernel conv."""
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        out = self.rbr_1x1(x) + self.rbr_3x3(x) + self.rbr_5x5(x) + self.rbr_7x7(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)


class SC_ELAN_RepExact(nn.Module):
    """SC-ELAN-RepExact: exact-reparam single-path style SC-ELAN variant.

    Uses RepMultiKernelConv to emulate four receptive-field branches during training,
    and supports deployment as a single fused convolution path.
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int, c5: int = 1):
        """Initialize SC_ELAN_RepExact module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Unused (compatibility).
            c4 (int): Unused (compatibility).
            c5 (int): Unused (compatibility).
        """
        super().__init__()
        self.c = max(16, c2 // 2)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.rep = RepMultiKernelConv(self.c, self.c, s=1, g=self.c, deploy=False, max_k=7)
        self.cv2 = Conv(self.c, c2, 1, 1)

    @torch.no_grad()
    def switch_to_deploy(self):
        """Convert internal rep block to deploy mode."""
        self.rep.switch_to_deploy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN_RepExact."""
        return self.cv2(self.rep(self.cv1(x)))


class TinySelectiveContextGateV2(nn.Module):
    """Tiny-selective context gating with learnable stage bias.

    Extends TinySelectiveContextGate by adding a stage-adaptive bias before the
    sigmoid gate.  Shallow stages (negative bias) preserve raw detail; deep stages
    (positive bias) encourage stronger context injection.
    """

    def __init__(self, c: int, reduction: int = 4, stage_bias_init: float = 0.0):
        """Initialize TinySelectiveContextGateV2.

        Args:
            c (int): Number of feature channels.
            reduction (int): Channel reduction ratio for gate predictor.
            stage_bias_init (float): Initial value for the learnable stage bias.
        """
        super().__init__()
        hidden = max(8, c // reduction)
        self.gate_conv = nn.Sequential(
            DWConv(c, c, 3, 1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.stage_bias = nn.Parameter(torch.tensor(stage_bias_init))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Blend identity and context features with a stage-biased gate."""
        gate = torch.sigmoid(self.gate_conv(x) + self.stage_bias)
        return x + gate * (context - x)


class SC_ELAN_LSKA_TSCGv2(SC_ELAN_LSKA):
    """SC-ELAN-LSKA with stage-adaptive TinySelectiveContextGate v2.

    Identical topology to SC_ELAN_LSKA_TSCG but uses a learnable stage bias in the
    gate, allowing the network to automatically calibrate context injection strength
    per pyramid level.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        c5: int = 1,
        lsk_k: int = 7,
        tscg_reduction: int = 4,
        tscg_stage_bias: float = 0.0,
    ):
        """Initialize SC_ELAN_LSKA_TSCGv2.

        Args:
            c1, c2, c3, c4, c5: See SC_ELAN documentation.
            lsk_k (int): LSKA kernel size.
            tscg_reduction (int): Reduction ratio for the gate.
            tscg_stage_bias (float): Initial stage bias (e.g. -0.5 for P3, +0.5 for P5).
        """
        super().__init__(c1, c2, c3, c4, c5, lsk_k=lsk_k)
        self.tscg = TinySelectiveContextGateV2(c2, reduction=tscg_reduction, stage_bias_init=tscg_stage_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN_LSKA_TSCGv2."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        feat = self.cv4(torch.cat(y, 1))
        context = self.interaction(feat)
        return self.tscg(feat, context)


class P3_FRM(nn.Module):
    """P3 Feature Refinement Module for small object detection.

    Lightweight module inserted after the P3 neck output to enhance tiny-target
    features via local-contrast gating.  The contrast branch highlights regions
    where feature activations deviate from their local neighbourhood, providing
    an implicit saliency signal for small objects.
    """

    def __init__(self, c1: int, c2: int):
        """Initialize P3_FRM.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels (typically equal to c1).
        """
        super().__init__()
        c = c1
        self.refine = nn.Sequential(
            DWConv(c, c, 3, 1),
            Conv(c, c, 1, 1),
        )
        self.local_pool = nn.AvgPool2d(5, stride=1, padding=2)
        hidden = max(8, c // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c, 1),
            nn.Sigmoid(),
        )
        self.proj = Conv(c, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: DWConv refinement gated by local-contrast saliency."""
        feat = self.refine(x)
        contrast = x - self.local_pool(x)
        return self.proj(feat * self.gate(contrast) + x)


class SC_ELAN_RepAdd(nn.Module):
    """SC-ELAN-RepAdd: ELAN-style variant with additive fusion and rep branches.

    Keeps an ELAN-like progressive path but replaces concat-heavy fusion with weighted
    additive fusion to reduce sync/wait overhead on edge deployment targets.
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int, c5: int = 1):
        """Initialize SC_ELAN_RepAdd module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Unused (compatibility).
            c4 (int): Unused (compatibility).
            c5 (int): Unused (compatibility).
        """
        super().__init__()
        self.c = max(16, c2 // 4)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.rep1 = RepMultiKernelConv(self.c, self.c, s=1, g=self.c, deploy=False, max_k=7)
        self.rep2 = RepMultiKernelConv(self.c, self.c, s=1, g=self.c, deploy=False, max_k=7)
        self.rep3 = RepMultiKernelConv(self.c, self.c, s=1, g=self.c, deploy=False, max_k=7)
        self.alpha = nn.Parameter(torch.ones(3))
        self.cv2 = Conv(self.c, c2, 1, 1)

    @torch.no_grad()
    def switch_to_deploy(self):
        """Convert internal rep branches to deploy mode."""
        self.rep1.switch_to_deploy()
        self.rep2.switch_to_deploy()
        self.rep3.switch_to_deploy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SC_ELAN_RepAdd."""
        xa, xb = self.cv1(x).chunk(2, 1)
        y1 = self.rep1(xb)
        y2 = self.rep2(y1)
        y3 = self.rep3(y2)
        w = torch.softmax(self.alpha, dim=0)
        fused = xa + w[0] * y1 + w[1] * y2 + w[2] * y3
        return self.cv2(fused)
