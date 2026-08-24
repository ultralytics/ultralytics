# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Structured-pruned YOLOv8 blocks."""

import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv


class BottleneckPruned(nn.Module):
    """Bottleneck with channel widths reconstructed from pruning masks."""

    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, k[0], 1)
        self.cv2 = Conv(cv1out, cv2out, k[1], 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2fPruned(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, cv1in, cv1out, cv1_split_sections, inner_cv1outs, inner_cv2outs, cv2out, n=1, shortcut=False, g=1, e=0.5
    ):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = cv1_split_sections[1]  # hidden channels
        self.cv1_split_sections = cv1_split_sections
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        if shortcut:
            self.cv2 = Conv((2 + n) * self.c, cv2out, 1)  # optional act=FReLU(c2)
            # 如果是shortcut的情况下, 那么在Bottlenet内部的每一个cv2的输出必须和内部的cv1的输入通道数相同
            for i in range(n):
                assert inner_cv2outs[i] == self.c, "Shortcut channels must match"
            self.m = nn.ModuleList(
                BottleneckPruned(self.c, inner_cv1outs[i], self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                for i in range(n)
            )
        else:
            cv2_inchannels = cv1out + sum(inner_cv2outs)
            self.cv2 = Conv(cv2_inchannels, cv2out, 1)
            # 如果不是shortcut的情况下, 那么在Bottlenet内部的每一个cv2的输出和内部的cv1的输入通道数不一定相等
            self.m = nn.ModuleList()
            for i in range(n):
                self.m.append(
                    BottleneckPruned(self.c, inner_cv1outs[i], inner_cv2outs[i], shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                )
                self.c = inner_cv2outs[i]

    def forward(self, x):
        """在head部分的C2f层中, 由于没有shortcut残差结构, 因此C2f结构中的第一个cv1层是可以被剪枝的 但是剪完以后是不一定对称的, 因此要重新计算比例 例如,
        C2f结构中的第一个cv1层剪枝前输出通道数为256, chunck以后左右各式128, 但是剪枝后, cv1层输出通道数可能为120, 但是其中80落在左半区, 40落在右半区.
        """
        # y = list(self.cv1(x).chunk(2, 1))
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPFPruned(nn.Module):
    """SPPF block with channel widths reconstructed from pruning masks."""

    def __init__(self, cv1in, cv1out, cv2out, k=5):
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out * 4, cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
