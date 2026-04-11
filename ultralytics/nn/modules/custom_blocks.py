# ultralytics/nn/modules/custom_blocks.py

import torch
import torch.nn as nn

class MyConvBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
