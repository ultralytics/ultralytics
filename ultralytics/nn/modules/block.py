# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.ops import SqueezeExcitation

from .conv import (
    AsymmetricConv,
    AsymmetricDWConv,
    AsymmetricDWConvLightConv,
    CombConv,
    Conv,
    DWConv,
    GhostConv,
    LightConv,
    LightConvB,
    LightDSConv,
    QConv,
    RepConv,
    adderConv,
)
from .transformer import MSDATransformerBlock, TransformerBlock

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
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "FusedMBConv",
    "MBConv",
    "SABottleneck",
    "sa_layer",
    "C3SA",
    "LightC3x",
    "C3xTR",
    "C2HG",
    "C3xHG",
    "C2fx",
    "C2TR",
    "C3CTR",
    "C2DfConv",
    "DATransformerBlock",
    "C2fDA",
    "C3TR2",
    "HarDBlock",
    "MBC2f",
    "C2fTA",
    "C3xTA",
    "LightC2f",
    "LightBottleneck",
    "BLightC2f",
    "MSDAC3x",
    "QC2f",
    "LightDSConv",
    "LightDSConvC2f",
    "AsymmetricLightC2f",
    "AsymmetricLightBottleneckC2f",
    "C3xAsymmetricLightBottleneck",
    "adderBottleneck",
    "adderC2f",
    "ConvSelfAttention",
    "C2fOAttention",
)


class ConvSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8):
        super(ConvSelfAttention, self).__init__()
        assert out_channels % heads == 0, "num heads should be factorized by out channels"
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolutional layers for Q, K, V
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query_conv(x).view(B, self.heads, self.out_channels // self.heads, H * W)
        k = self.key_conv(x).view(B, self.heads, self.out_channels // self.heads, H * W)
        v = self.value_conv(x).view(B, self.heads, self.out_channels // self.heads, H * W)

        # Compute attention scores
        q = q.transpose(2, 3)  # (B, heads, H*W, C//heads)
        attn = torch.matmul(q, k) / (self.out_channels // self.heads) ** 0.5
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v.transpose(2, 3)).transpose(2, 3)  # (B, heads, C//heads, H*W)
        out = out.contiguous().view(B, -1, H, W)  # Merge heads

        # Apply final convolution
        out = self.output_conv(out)

        return out


class sa_layer(nn.Module):
    """
    Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class Conv3x3(Conv):
    # 3x3 convolution with padding
    def __init__(self, c1, c2, s=1, g=1, dilation=1):
        super(Conv3x3, self).__init__(c1, c2, k=3, s=s, g=g, act=True)
        self.conv.dilation = (dilation, dilation)
        self.conv.padding = (dilation, dilation)  # Update padding based on dilation


class Conv1x1(Conv):
    # 1x1 convolution
    def __init__(self, c1, c2, s=1):
        super(Conv1x1, self).__init__(c1, c2, k=1, s=s, act=True)


class SABottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(SABottleneck, self).__init__()
        self.expansion = 4
        c_ = int(c2 * e)  # hidden channels

        # Sesuaikan nama layer
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.bn1 = nn.BatchNorm2d(c_)
        self.cv2 = Conv(c_, c_, k[1], 1, g=g)
        self.bn2 = nn.BatchNorm2d(c_)
        self.cv3 = Conv(c_, c2 * self.expansion, 1, 1)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.sa = sa_layer(c2 * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.add = shortcut and c1 == c2 * self.expansion
        self.stride = 1  # atau sesuaikan dengan kebutuhan

    def forward(self, x):
        identity = x

        # Proses forward dengan penamaan yang disesuaikan
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.cv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.cv3(out)
        out = self.bn3(out)
        out = self.sa(out)

        if self.add:
            out += identity

        out = self.relu(out)
        return out


class FusedMBConv(nn.Module):
    """Fused MBConv, equivalent to EdgeResidual."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, se_ratio=0.25):
        super(FusedMBConv, self).__init__()
        self.has_skip = c1 == c2 and s == 1
        self.c2 = c2
        # Expansion Convolution
        self.conv_exp = Conv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act)

        self.triplet_attention = TripletAttention()

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(c2, int(c2 * se_ratio))

        # Point-wise Linear Projection
        self.conv_pwl = Conv(c2, c2, k=1, act=False)

    def forward(self, x):
        # print(self.c2)
        # print(f'x shape: {x.shape}')
        shortcut = x
        x = self.conv_exp(x)
        # print(f'before se shape: {x.shape}')
        x = self.triplet_attention(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        if self.has_skip:
            x = x + shortcut
        return x


class MBConv(nn.Module):
    """MBConv, equivalent to InvertedResidual."""

    def __init__(self, c1, c2, k=3, s=1, expand_ratio=1.0, p=None, g=1, d=1, act=True, se_ratio=0.25):
        super(MBConv, self).__init__()
        self.c2 = c2
        hidden_dim = int(round(c1 * expand_ratio))
        self.use_res_connect = s == 1 and c1 == c2

        layers = []
        if expand_ratio != 1:
            # Point-wise Expansion
            layers.append(Conv(c1, hidden_dim, k=1, act=act))

        layers.extend(
            [
                # Depth-wise Convolution
                Conv(hidden_dim, hidden_dim, k=k, s=s, p=p, g=hidden_dim, d=d, act=act),
                # Squeeze-and-Excitation
                SqueezeExcitation(hidden_dim, int(c2 * se_ratio)),
                # add Triplet Attention
                TripletAttention(),
                # Point-wise Linear Projection
                Conv(hidden_dim, c2, k=1, act=False),
            ]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # print(self.c2)
        # print(f'x shape: {x.shape}')
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
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
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ConvAttention(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.fx = nn.Conv2d(input_channel, 1, 1)
        self.gx = nn.Conv2d(input_channel, 1, 1)
        self.hx = nn.Conv2d(input_channel, 1, 1)

    def forward(self, x):
        fx = self.fx(x)
        gx = self.gx(x)
        hx = self.hx(x)

        fxgx = torch.matmul(fx, gx)

        fxgx = F.softmax(fxgx, dim=1)
        o = torch.matmul(hx, fxgx) * x
        return o


class C2fOAttention(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attention = ConvAttention(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out_c2f = self.cv2(torch.cat(y, 1))
        out_attention = self.attention(out_c2f)

        return out_attention


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
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class LightC3x(C3x):
    # (self, c1, c2, k=3, s=1, expand_ratio=1.0, p=None, g=1, d=1, act=True, se_ratio=0.25)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.Sequential(*(MBConv(self.c_, self.c_) for i in range(n)))


class C3SA(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(SABottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n))
        )


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3TR2(C3):
    """C3 module with DualTransformerBlock."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR2 module with DualTransformerBlock."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = DualTransformerBlock(c_, c_, 4, n)


class C3xTR(C3):
    """C3 module with cross-convolutions and transformer blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3xTR module with cross-convolutions and transformer blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # Menggantikan Bottleneck dengan TransformerBlock
        self.m = nn.Sequential(*(TransformerBlock(self.c_, self.c_, 4, 1) for _ in range(n)))


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class C2SA(nn.Module):
    """Faster Implementation of SABottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize SAbottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(SABottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2HG(nn.Module):
    def __init__(self, c1, c2, n=1, g=1, e=0.5, cm=128, k=3, n_hg=6, lightconv=False, shortcut_hg=False, act=nn.ReLU()):
        super().__init__()
        # Inisialisasi C2f
        self.c2f = C2f(c1, c2, n, shortcut=False, g=g, e=e)

        # Cari jumlah saluran output dari C2f
        c_out_c2f = c2

        # Inisialisasi HGBlock dengan jumlah saluran yang sesuai
        self.hgblock = HGBlock(
            c_out_c2f, cm, c_out_c2f, k=k, n=n_hg, lightconv=lightconv, shortcut=shortcut_hg, act=act
        )

    def forward(self, x):
        x = self.c2f(x)
        x = self.hgblock(x)
        return x


# Contoh inisialisasi model
model = C2HG(c1=64, c2=256, n=2, g=1, e=0.5, cm=128, k=3, n_hg=6, lightconv=False, shortcut_hg=False, act=nn.ReLU())


class C3xHG(nn.Module):
    def __init__(self, c1, c2, n=1, g=1, e=0.5, cm=128, k=3, n_hg=6, lightconv=False, shortcut_hg=False, act=nn.ReLU()):
        super().__init__()
        # Inisialisasi C3x
        self.c3x = C3x(c1, c2, n, shortcut=True, g=g, e=e)

        # Cari jumlah saluran output dari C3x
        c_out_c3x = c2

        # Inisialisasi HGBlock dengan jumlah saluran yang sesuai
        self.hgblock = HGBlock(
            c_out_c3x, cm, c_out_c3x, k=k, n=n_hg, lightconv=lightconv, shortcut=shortcut_hg, act=act
        )

    def forward(self, x):
        x = self.c3x(x)
        x = self.hgblock(x)
        return x


# Contoh inisialisasi model
model = C3xHG(c1=64, c2=256, n=2, g=1, e=0.5, cm=128, k=3, n_hg=6, lightconv=False, shortcut_hg=False, act=nn.ReLU())


class C2fx(nn.Module):
    """Faster Implementation of CSP Bottleneck with cross-convolutions (inspired by C3x)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # Use cross-convolution pattern in Bottleneck
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((1, 3), (3, 1)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through modified C2fx layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2TR(nn.Module):
    """Modified version of C2f using TransformerBlock instead of Bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions and TransformerBlock."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # Replace Bottleneck with TransformerBlock
        self.m = nn.ModuleList(TransformerBlock(self.c, self.c, 4, n) for _ in range(n))

    def forward(self, x):
        """Forward pass through modified C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3CTR(C3):
    """C3 module with modified TransformerBlock that uses Deformable Cross Attention."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, num_heads=4, n_levels=4, n_points=4):
        """Initialize C3Ghost module with TransformerBlock using Deformable Cross Attention."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = CrossDATransformerBlock(c_, c_, num_heads, n, n_levels=n_levels, n_points=n_points)


class C2DfConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = DfConv(c1, 2 * self.c, 1, 1)  # Replaced Conv with DfConv
        self.cv2 = DfConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2), Replaced Conv with DfConv
        # Here, Bottleneck should be compatible with DfConv if it uses convolution internally
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer using Deformable Convolution."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk() with Deformable Convolution."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fDA(nn.Module):
    """Modified C2f class with MSDeformAttn integration."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.ms_deform_attn = DATransformerBlock(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)

    def forward(self, x, refer_bbox, value_shapes):
        # Apply initial convolution
        y = list(self.cv1(x).chunk(2, 1))

        # Apply bottleneck layers
        for m in self.m:
            y.append(m(y[-1]))

        # Apply MSDeformAttn here (you may need to modify this based on your specific use case)
        # Note: You need to ensure that the dimensions of the input to MSDeformAttn match its expected input dimensions
        attn_output = self.ms_deform_attn(y[-1], refer_bbox, y[-1], value_shapes)

        # Continue with the rest of the C2f operations
        y.append(attn_output)
        return self.cv2(torch.cat(y, 1))


class HarDBlock(nn.Module):
    def get_link(self, layer, c1, gr, grmul):
        if layer == 0:
            return c1, 0, []
        c2 = gr
        link = []
        for i in range(10):
            dv = 2**i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    c2 *= grmul
        c2 = int(int(c2 + 1) / 2) * 2
        inch = 0
        for i in link:
            ch, _, _ = self.get_link(i, c1, gr, grmul)
            inch += ch
        return c2, inch, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, c1, gr, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            c2, inch, link = self.get_link(i + 1, c1, gr, grmul)
            self.links.append(link)

            if dwconv:
                layers_.append(CombConv(inch, c2))  # Asumsi DWConv memiliki parameter yang sesuai
            else:
                layers_.append(Conv(inch, c2))  # Asumsi Conv memiliki parameter yang sesuai

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += c2
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class MBC2f(nn.Module):
    def __init__(self, c1, c2, n=1, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        # Mengganti Bottleneck dengan MBConv tanpa shortcut
        self.m = nn.ModuleList(MBConv(self.c, self.c, g=g, k=((1, 3), (3, 1))) for _ in range(n))

    # ... (forward dan forward_split tetap sama)


##Triplet Attention
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=[(k - 1) // 2 for k in kernel_size], relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate(kernel_size=(1, 3))
        self.ChannelGateW = SpatialGate(kernel_size=(3, 1))
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(kernel_size=(3, 3))

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


##C2F with Triplet Attention
class C2fTA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

        # Menambahkan Triplet Attention
        self.triplet_attention = TripletAttention()

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))

        # Menerapkan Triplet Attention pada output dari setiap Bottleneck
        for m in self.m:
            bottleneck_output = m(y[-1])
            y.append(self.triplet_attention(bottleneck_output))

        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))

        # Menerapkan Triplet Attention pada output dari setiap Bottleneck
        for m in self.m:
            bottleneck_output = m(y[-1])
            y.append(self.triplet_attention(bottleneck_output))

        return self.cv2(torch.cat(y, 1))


# C3x with Triplet Attention
class C3xTA(C3):
    """C3 module with cross-convolutions and Triplet attention."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3x instance with Triplet Attention."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))

        # Tambahkan Triplet Attention
        self.triplet_attention = TripletAttention()

    def forward(self, x):
        # Menerapkan bottleneck dengan cross-convolutions
        y = self.m(x)

        # Menerapkan Triplet Attention
        y = self.triplet_attention(y)
        return y


class LightBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = LightConvB(c1, c_, k[0], 1)
        self.cv2 = LightConvB(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class LightC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = LightConvB(c1, 2 * self.c, 1, 1)
        self.cv2 = LightConvB((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class BLightC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = LightConvB(c1, 2 * self.c, 1, 1)
        self.cv2 = LightConvB((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class QC2f(nn.Module):
    """Modified C2f with Quantum Convolution."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, n_qubits=4, backend=None, shots=1024):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.qconv = QConv(n_qubits, backend, shots)  # Quantum convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.append(self.qconv(y[-1]))  # Apply quantum convolution here
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MSDAC3x(C3):
    """Modified C3 module with MSDATransformerBlock."""

    def __init__(self, c1, c2, num_heads, num_layers, n_levels=4, n_points=4, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(MSDATransformerBlock(self.c_, self.c_, num_heads, num_layers, n_levels, n_points) for _ in range(n))
        )

    def forward(self, x, refer_bbox, value_shapes, value_mask=None):
        x1 = self.cv1(x)
        x_transformed = self.m(x1, refer_bbox, value_shapes, value_mask) if self.m else x1
        return self.cv3(torch.cat((x_transformed, self.cv2(x)), 1))


class LightDSConvC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = LightDSConv(c1, 2 * self.c, 1, 1)
        self.cv2 = LightDSConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AsymmetricLightC2f(nn.Module):
    """Modified C2f with asymmetric bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize AsymmetricLightC2f with asymmetric bottleneck."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = AsymmetricConv(c1, 2 * self.c)  # Use Asymmetric Convolution
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        # Use AsymmetricBottleneck
        self.m = nn.ModuleList(AsymmetricBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through AsymmetricLightC2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AsymmetricBottleneck(nn.Module):
    """Bottleneck with Asymmetric Convolution."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with asymmetric convolution."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # Replace standard convolutions with asymmetric convolutions
        self.cv1 = AsymmetricConv(c1, c_, act=False)
        self.cv2 = AsymmetricConv(c_, c2, act=True)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass applies asymmetric convolutions to the input data."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class AsymmetricLightBottleneck(nn.Module):
    """Bottleneck with Asymmetric Depth-wise Convolution."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with asymmetric depth-wise convolution."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # Replace standard convolutions with asymmetric depth-wise convolutions
        self.cv1 = AsymmetricDWConv(c1, c_, act=False)
        self.cv2 = AsymmetricDWConv(c_, c2, act=True)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass applies asymmetric depth-wise convolutions to the input data."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class AsymmetricLightBottleneckC2f(nn.Module):
    """Modified C2f with LightConvB and AsymmetricBottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize ModifiedC2f with LightConvB and AsymmetricBottleneck."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # Replace the first convolution with LightConvB
        self.cv1 = AsymmetricDWConvLightConv(c1, 2 * self.c, act=True)

        # Replace the second convolution with standard Convolution
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Use AsymmetricBottleneck
        self.m = nn.ModuleList(AsymmetricLightBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through ModifiedC2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3xAsymmetricLightBottleneck(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(AsymmetricLightBottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n))
        )


class adderBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = adderConv(c1, c_, k[0], 1)  # Menggunakan kelas Conv yang telah dimodifikasi
        self.cv2 = adderConv(c_, c2, k[1], 1, g=g)  # Menggunakan kelas Conv yang telah dimodifikasi
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class adderC2f(nn.Module):
    # Implementasi C2f dengan adderConv
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([adderBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
