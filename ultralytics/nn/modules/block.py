# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torchvision.ops import SqueezeExcitation

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, DepthwiseSeparableConv, SqueezeExcite
from .transformer import TransformerBlock


__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3'
           ,'FusedMBConv','MBConv', 'SABottleneck', 'sa_layer', 'C3SA', 'LightC3x', 'C3xTR', 'C2HG', 'C3xHG', 'C2fx', 'C2TR', 'C3CTR', 'C2DfConv', 'DATransformerBlock', 'C2fDA', 'C3TR2', 'C2fHarDBlock')

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

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
        self.has_skip = (c1 == c2 and s == 1)
        self.c2 = c2
        # Expansion Convolution
        self.conv_exp = Conv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act)

        self.triplet_attention = TripletAttention()
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(c2, int(c2*se_ratio))
        
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
        
        layers.extend([
            # Depth-wise Convolution
            Conv(hidden_dim, hidden_dim, k=k, s=s, p=p, g=hidden_dim, d=d, act=act),
            # Squeeze-and-Excitation
            SqueezeExcitation(hidden_dim, int(c2*se_ratio)),
            # add Triplet Attention
            TripletAttention(),
            # Point-wise Linear Projection
            Conv(hidden_dim, c2, k=1, act=False)
        ])
        
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
    #(self, c1, c2, k=3, s=1, expand_ratio=1.0, p=None, g=1, d=1, act=True, se_ratio=0.25)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.Sequential(*(MBConv(self.c_, self.c_)for i in range(n)))

class C3SA(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(SABottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))

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
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

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

### Triplet attention
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
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
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
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

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
        self.hgblock = HGBlock(c_out_c2f, cm, c_out_c2f, k=k, n=n_hg, lightconv=lightconv, shortcut=shortcut_hg, act=act)

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
        self.hgblock = HGBlock(c_out_c3x, cm, c_out_c3x, k=k, n=n_hg, lightconv=lightconv, shortcut=shortcut_hg, act=act)

    def forward(self, x):
        x = self.c3x(x)
        x = self.hgblock(x)
        return x

# Contoh inisialisasi model
model = C3xHG(c1=64, c2=256, n=2, g=1, e=0.5, cm=128, k=3, n_hg=6, lightconv=False, shortcut_hg=False, act=nn.ReLU())


class C2fx(nn.Module):
    """Faster Implementation of CSP Bottleneck with cross-convolutions (inspired by C3x)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups, expansion."""
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
    

class C2fHarDBlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, growth_rate=32):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.growth_rate = growth_rate
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Define layers and links similar to HarDBlock
        layers_ = []
        self.links = []
        out_channels = 2 * self.c

        for i in range(n):
            link, layer_out_ch = self.get_link(i, self.c, growth_rate)
            self.links.append(link)
            layers_.append(Bottleneck(layer_out_ch, self.c, shortcut, g))  # Use your Bottleneck definition
            out_channels += self.c

        self.cv2 = Conv(out_channels, c2, 1)  # Adjust output channels
        self.m = nn.ModuleList(layers_)

    def get_link(self, layer_idx, base_ch, growth_rate):
        link = [max(0, layer_idx - 2), layer_idx - 1]
        layer_out_ch = base_ch + len(link) * growth_rate
        return link, layer_out_ch

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for idx, m in enumerate(self.m):
            link = self.links[idx]
            x = torch.cat([y[l] for l in link], 1)
            y.append(m(x))

        return self.cv2(torch.cat(y[1:], 1))  # Skip the first chunk
# Example usage
#c1, c2 = 256, 512  # Example input/output channels
#c2f_modified = C2fModified(c1, c2, n=4)

