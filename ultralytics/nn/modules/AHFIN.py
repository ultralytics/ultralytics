import torch
import torch.nn as nn
import torch.nn.functional as F

class AHFIN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(AHFIN, self).__init__()
        self.num_inputs = len(in_channels_list)
        self.convs = nn.ModuleList()
        self.scales = nn.ParameterList()
        
        # 为每个输入特征图创建卷积层和尺度参数
        for i in range(self.num_inputs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_list[i], out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.scales.append(nn.Parameter(torch.ones(1, dtype=torch.float32)))
        
        # 用于生成权重图的卷积层
        self.weight_conv = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_list):
        # 对每个特征图进行卷积和尺度调整
        scaled_x_list = []
        for i in range(self.num_inputs):
            scaled_x = self.convs[i](x_list[i]) * self.scales[i]
            scaled_x_list.append(scaled_x)
        
        # 特征融合
        fused_x = sum(scaled_x_list)
        
        # 生成权重图
        weight = self.sigmoid(self.weight_conv(fused_x))
        
        # 对融合后的特征进行加权
        weighted_x = fused_x * weight
        
        return weighted_x

# 示例用法
if __name__ == "__main__":
    # 假设输入来自不同层次的特征图
    in_channels_list = [256, 512, 1024]
    out_channels = 256
    ahfin = AHFIN(in_channels_list, out_channels)
    
    # 创建示例输入张量
    x1 = torch.randn(1, 256, 64, 64)
    x2 = torch.randn(1, 512, 32, 32)
    x3 = torch.randn(1, 1024, 16, 16)
    x_list = [x1, x2, x3]
    
    # 前向传播
    output = ahfin(x_list)
    print(output.shape)