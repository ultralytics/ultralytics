
import math
import warnings

import torch
import torch.nn as nn

__all__ = ["BiFPN_Concat", "BiFPN_Concat2", "BiFPN_Concat3"]

from sympy.physics.paulialgebra import epsilon


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BiFPN_Concat(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
            w = self.w
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = [weight[0] * x[0], weight[1] * x[1]]
            return torch.cat(x, self.d)


# class BiFPN_Concat(nn.Module):
#     def __init__(self, c1, c2):
#         super(BiFPN_Concat, self).__init__()
#         self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 0.0001
#         self.conv = Conv(c1, c2, 1, 1, 0)
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         if len(x) == 2:
#             w = self.w1_weight
#             weight = w / (torch.sum(w, dim = 0) + self.epsilon)
#             x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
#         elif len(x) == 3:
#             w = self.w2_weight
#             weight = w / (torch.sum(w, dim = 0) + self.epsilon)
#             x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
#         else:
#             raise ValueError("BiFPN_Concat expects input of length 2 or 3")
#         return x

# class swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)
#
# class BiFPN_Concat(nn.Module):
#     def __init__(self, lenght):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(lenght, dtype=torch.float32), requires_grad=True)
#         self.swish = swish()
#         self.epsilon = 0.0001
#
#     def forward(self, x):
#         weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)
#         weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
#         stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
#         result = torch.sum(stacked_feature_maps, dim=0)
#         return result

class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] *x [1]]
        return torch.cat(x, self.d)


class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)