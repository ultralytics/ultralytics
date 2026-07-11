import torch
from torch import nn
import torch.nn.functional as F


def global_median_pooling(x):  #对输入特征图进行全局中值池化操作。

    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled #全局中值池化后的特征图，尺寸为 (batch_size, channels, 1, 1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        # 定义3×3 和 1x1 卷积层，用于减少和恢复特征维度
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1)) # 全局平均池化
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))# 全局最大池化
        median_pool = global_median_pooling(inputs)# 全局中值池化

        # 处理全局平均池化后的输出
        avg_out = self.fc1(avg_pool)# 通过第一个 1x1 卷积层减少特征维度
        avg_out = F.relu(avg_out, inplace=True) # 应用 ReLU 激活函数
        avg_out = self.fc2(avg_out)# 通过第二个 1x1 卷积层恢复特征维度
        avg_out = torch.sigmoid(avg_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 处理全局最大池化后的输出
        max_out = self.fc1(max_pool)# 通过第一个 1x1 卷积层减少特征维度
        max_out = F.relu(max_out, inplace=True) # 应用 ReLU 激活函数
        max_out = self.fc2(max_out) # 通过第二个 1x1 卷积层恢复特征维度
        max_out = torch.sigmoid(max_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 处理全局中值池化后的输出
        median_out = self.fc1(median_pool) # 通过第一个 1x1 卷积层减少特征维度
        median_out = F.relu(median_out, inplace=True) # 应用 ReLU 激活函数
        median_out = self.fc2(median_out) # 通过第二个 1x1 卷积层恢复特征维度
        median_out = torch.sigmoid(median_out) # 使用 Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内

        # 将三个池化结果的注意力图进行元素级相加
        out = avg_out + max_out + median_out
        return out


class MECS(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4):
        super(MECS , self).__init__()

        self.C = in_channels
        self.O = out_channels
        # 确保输入和输出通道数相同
        assert in_channels == out_channels, "Input and output channels must be the same"
        # 初始化通道注意力模块
        self.channel_attention = ChannelAttention(input_channels=in_channels,
                                                  internal_neurons=in_channels // channel_attention_reduce)

        # 定义 7x7 深度卷积层
        self.initial_depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)

        # 定义多个不同尺寸的深度卷积层
        self.depth_convs = nn.ModuleList([

            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels),
        ])
        # 定义 1x1 卷积层和激活函数
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        # 全局感知机
        inputs = self.pointwise_conv(inputs) #图中我画是先进入channel_attention，但是代码实际先进入pointwise_conv，不影响哈。
        inputs = self.act(inputs)

        # 通道注意力
        channel_att_vec = self.channel_attention(inputs)
        inputs = channel_att_vec * inputs

        # 先经过 5x5 深度卷积层
        initial_out = self.initial_depth_conv(inputs)

        # 空间注意力
        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        # 应用空间注意力
        spatial_att = self.pointwise_conv(spatial_out)
        out = spatial_att * inputs
        out = self.pointwise_conv(out)
        return out


if __name__ == '__main__':
    # 假设输入数据
    batch_size = 4
    channels = 16
    height = 64
    width = 64
    input_tensor = torch.randn(batch_size, channels, height, width).cuda()

    # 初始化 MECS 块
    cpca_block = MECS (in_channels=16, out_channels=16, channel_attention_reduce=4).cuda()

    # 通过 MECS 块处理输入
    output_tensor = cpca_block(input_tensor)

    # 打印输出张量的形状
    print(f"Output shape: {output_tensor.shape}")