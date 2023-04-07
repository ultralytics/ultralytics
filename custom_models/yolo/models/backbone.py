import torch
import torch.nn as nn

from typing import *

"""
Basic convolution layer structure in yolo
CBL = Convolution + Batch Norm + SiLU
"""

class CBL(nn.module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(CBL, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

        self.cb1 = nn.Sequential(
            conv,
            batch_norm,
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.cb1(x)


class Bottleneck(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
    """
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(Bottleneck, self).__init__()
        c_ = int(width_multiple*in_channels)
        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = CBL(c_, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x


class C3(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
        depth (int): it controls the number of times the bottleneck (residual block)
                        is repeated within the C3 block
        backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
                            it will be composed by bottlenecks 2 (check in the image linked below)
        https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png
    """
    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super(C3, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = CBL(in_channels,  c_, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    CBL(c_, c_, 1, 1, 0),
                    CBL(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = CBL(c_ * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels//2)

        self.c1 = CBL(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = CBL(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))


