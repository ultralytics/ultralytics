import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels

        # 调整输入通道数到统一的 out_channels
        self.conv_in = nn.ModuleList()
        for in_channels in in_channels_list:
            self.conv_in.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))

        # BiFPN 权重（可学习）
        self.weights = nn.ParameterList()
        for _ in range(num_layers):
            # 每个层有 5 个融合操作（P3-P7 的双向路径）
            self.weights.append(
                nn.Parameter(torch.ones(5, 3))  # 每个融合点有 2-3 个输入
            )

        # 卷积层用于融合后的特征处理
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                        )
                        for _ in range(5)  # P3-P7
                    ]
                )
            )

    def forward(self, inputs):
        # inputs: [P3, P4, P5] from backbone
        P3, P4, P5 = inputs

        # 调整通道数
        P3 = self.conv_in[0](P3)  # (B, out_channels, H/8, W/8)
        P4 = self.conv_in[1](P4)  # (B, out_channels, H/16, W/16)
        P5 = self.conv_in[2](P5)  # (B, out_channels, H/32, W/32)

        # BiFPN 层
        features = [P3, P4, P5]
        for layer in range(self.num_layers):
            # 自顶向下路径
            P5_td = features[2]  # P5
            P4_td = self.fuse_features(
                [features[1], F.interpolate(P5_td, scale_factor=2, mode="nearest")], self.weights[layer][0]
            )
            P4_td = self.conv_layers[layer][1](P4_td)
            P3_out = self.fuse_features(
                [features[0], F.interpolate(P4_td, scale_factor=2, mode="nearest")], self.weights[layer][1]
            )
            P3_out = self.conv_layers[layer][0](P3_out)

            # 自底向上路径
            P4_out = self.fuse_features(
                [features[1], P4_td, F.max_pool2d(P3_out, kernel_size=3, stride=2, padding=1)], self.weights[layer][2]
            )
            P4_out = self.conv_layers[layer][3](P4_out)
            P5_out = self.fuse_features(
                [features[2], P5_td, F.max_pool2d(P4_out, kernel_size=3, stride=2, padding=1)], self.weights[layer][4]
            )
            P5_out = self.conv_layers[layer][4](P5_out)

            # 更新特征
            features = [P3_out, P4_out, P5_out]

        return features

    def fuse_features(self, inputs, weights):
        # 归一化权重
        weights = F.softmax(weights, dim=1)
        # 加权融合
        out = 0
        for i, x in enumerate(inputs):
            out += weights[i] * x
        return out
