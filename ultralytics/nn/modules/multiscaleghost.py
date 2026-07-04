import torch.nn as nn
import torch
import math
import ultralytics.nn.modules as ult_modules
import ultralytics.nn.tasks as tasks
# dựa vào https://github.com/CVHub520/Convolution/blob/main/Ghost%20Convolution.py và 
# https://github.com/sunsmarterjie/yolov12/blob/main/ultralytics/nn/modules/conv.py để truyền tham số 
# cho đúng dễ debug
class MultiScaleGhost(nn.Module):
    def __init__(self, c1, c2, k1=1, s=1, p=None, g=1, d=1, act=True):
        super(MultiScaleGhost, self).__init__()
        ratio=2
        kernel_sizes=(3, 5, 7) # để không thay đổi tham số mặc định sử dụng tuple
        self.output = c2
        init_channels = math.ceil(c2 / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, init_channels, kernel_size=k1, stride=s, padding=k1//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU() if act else nn.Identity()
        )

        branch_channel = math.ceil(new_channels / len(kernel_sizes))
        self.branches = nn.ModuleList() # ModuleList giúp chạy cả cpu/gpu để tránh gặp lỗi
        for kernel in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(init_channels, branch_channel, kernel, stride=1, padding=kernel//2, groups=math.gcd(init_channels, branch_channel), bias=False),
                    nn.BatchNorm2d(branch_channel),
                    nn.ReLU() if act else nn.Identity()
                )
            )
        
        self.fushion = nn.Sequential(
            nn.Conv2d(branch_channel*len(kernel_sizes), new_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU() if act else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x1))
        x2 = torch.cat(outputs, dim=1)
        x2 = self.fushion(x2)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.output, : , :]

# đăng ký vào module __init__ ultralytics  
ult_modules.MultiScaleGhost = MultiScaleGhost
tasks.MultiScaleGhost = MultiScaleGhost       

# rồi vào trực tiếp file tasks thêm vào 
# from ultralytics.nn.modules import (
    # MultiScaleGhost)
# base_module thêm MultiScaleGhost -> để tính output dựa trên input và ngược lại

# if __name__ == "__main__":
#     input = torch.randn(1, 3, 64, 64)
#     model = MultiScaleGhost(c1=3, c2=64)
#     print(model(input))