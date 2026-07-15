# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math

import torch
import torch.nn as nn

import ultralytics.nn.modules as ult_modules
import ultralytics.nn.tasks as tasks


# dựa vào https://github.com/CVHub520/Convolution/blob/main/Ghost%20Convolution.py và
# https://github.com/sunsmarterjie/yolov12/blob/main/ultralytics/nn/modules/conv.py để truyền than số
# cho đúng dễ debug
class MultiScaleGhost(nn.Module):
    def __init__(self, c1, c2, k1=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        ratio = 2
        kernel_sizes = (3, 5, 7)  # để không thay đổi than số mặc định sử dụng tuple
        self.output = c2
        init_channels = math.ceil(c2 / ratio)
        new_channels = init_channels * (ratio - 1)
        activation = nn.ReLU(inplace=True) if act else nn.Identity()

        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, init_channels, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(init_channels),
            activation,
        )

        branch_channel = math.ceil(new_channels / len(kernel_sizes))
        self.branches = nn.ModuleList()  # ModuleList giúp chạy cả cpu/gpu để tránh gặp lỗi

        for kernel in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        init_channels,
                        init_channels,
                        kernel,
                        stride=1,
                        padding=kernel // 2,
                        groups=init_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(init_channels),
                    activation,
                    nn.Conv2d(init_channels, branch_channel, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                )
            )
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channel * len(kernel_sizes), new_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(new_channels),
            activation,
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x1))
        x2 = torch.cat(outputs, dim=1)
        x2 = self.fusion(x2)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.output, :, :]


# đăng ký vào module __init__ ultralytics
ult_modules.MultiScaleGhost = MultiScaleGhost
tasks.MultiScaleGhost = MultiScaleGhost

# rồi vào trực tiếp file tasks thêm vào
# from ultralytics.nn.modules import (
# MultiScaleGhost)
# base_module thêm MultiScaleGhost -> để tính output dựa trên input và ngược lại

if __name__ == "__main__":
    input = torch.randn(1, 3, 64, 64)
    model = MultiScaleGhost(c1=3, c2=64)
    print(model(input))
