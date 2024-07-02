# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
<<<<<<< HEAD
=======
    "SqueezeExcite",
    "DepthwiseSeparableConv",
    "CombConv",
    "LightConvB",
    "LightChannelAttention",
    "LightSpatialAttention",
    "LightCBAM",
    "QConv",
    "LightDSConv",
    "AsymmetricDWConv",
    "AsymmetricDWConvLightConv",
    "AsymmetricConv",
    "adder",
    "adder2d",
    "adderConv",
>>>>>>> 2d87fb01604a79af96d1d3778626415fb4b54ac9
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation layer."""

    def __init__(self, c1, reduction_ratio=0.25, act=True):
        super(SqueezeExcite, self).__init__()
        c2 = max(1, int(c1 * reduction_ratio))
        # Using the Conv block with the format you provided
        self.conv_reduce = Conv(c1, c2, k=1, act=act)
        self.conv_expand = Conv(c2, c1, k=1, act=act)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        # Spatially average the input tensor along the dimensions HxW
        x_se = x.mean((2, 3), keepdim=True)
        print(f"after mean {x_se.shape}")
        x_se = self.conv_reduce(x_se)
        print(f"conv reduce {x_se.shape}")
        x_se = self.conv_expand(x_se)
        print(f"conv expand {x_se.shape}")

        return x * self.gate_activation(x_se)


class DepthwiseSeparableConv(nn.Module):
    """DepthwiseSeparable block."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super(DepthwiseSeparableConv, self).__init__()

        # Using the Conv block for depthwise and pointwise convolutions
        self.conv_dw = Conv(c1, c1, k=k, s=s, p=p, g=c1, d=d, act=act)
        self.conv_pw = Conv(c1, c2, k=1, act=act)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class AsymmetricConv(nn.Module):
    """Asymmetric Convolution."""

    def __init__(self, c1, c2, s=1, act=True):
        """Initialize Asymmetric Convolution with given parameters."""
        super().__init__()
        self.conv3x1 = Conv(c1, c2, (3, 1), s, act=False)
        self.conv1x3 = Conv(c2, c2, (1, 3), s, act=act)

    def forward(self, x):
        x = self.conv3x1(x)
        x = self.conv1x3(x)
        return x


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


import torch.nn as nn


class LightConvB(nn.Module):
    """Light convolution with Batch Normalization."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, act=True):
        """Initialize Conv layers with Batch Normalization and given arguments including activation."""
        super().__init__()
        self.conv1 = DWConv(c1, c2, 1, act=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = DWConv(c2, c2, k, act=None)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply 2 convolutions with Batch Normalization to input tensor."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act(x)


class AsymmetricDWConv(nn.Module):
    """Asymmetric Depth-wise Convolution."""

    def __init__(self, c1, c2, act=True):
        """Initialize asymmetric depth-wise convolution with given parameters."""
        super().__init__()
        # 3x1 followed by 1x3 convolutions
        self.conv3x1 = DWConv(c1, c2, (3, 1), act=False)
        self.conv1x3 = DWConv(c2, c2, (1, 3), act=act)

    def forward(self, x):
        x = self.conv3x1(x)
        x = self.conv1x3(x)
        return x


class AsymmetricDWConvLightConv(nn.Module):
    """Light convolution with Batch Normalization using asymmetric convolution."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, act=True):
        """Initialize asymmetric conv layers with Batch Normalization and given arguments including activation."""
        super().__init__()
        self.asymmetric_conv = AsymmetricDWConv(c1, c2, act=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply asymmetric convolution with Batch Normalization to input tensor."""
        x = self.asymmetric_conv(x)
        x = self.bn(x)
        return self.act(x)


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
<<<<<<< HEAD
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
=======
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
>>>>>>> 2d87fb01604a79af96d1d3778626415fb4b54ac9
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class CombConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bias=False, act=True):
        super().__init__()
        # Menggunakan Conv sebagai layer pertama
        self.layer1 = Conv(c1, c2, k, s, p, g, d, act)

        # Menggunakan DWConv sebagai layer kedua
        # Perhatikan bahwa dalam DWConv, parameter c2 dan k mungkin tidak digunakan sesuai dengan definisi DWConv yang diberikan
        self.layer2 = DWConv(c2, c2, k, s, d, act)

    def forward(self, x):
        # Mengaplikasikan layer1, kemudian layer2
        return self.layer2(self.layer1(x))


class LightChannelAttention(nn.Module):
    """Channel-attention module using LightConvB."""

    def __init__(self, channels: int):
        """Initializes the class with LightConvB layers."""
        super().__init__()
        self.cv1 = LightConvB(channels, channels, k=(1, 3), act=None)  # Cross Convolution with kernel (1,3)
        self.cv2 = LightConvB(channels, channels, k=(3, 1), act=None)  # Cross Convolution with kernel (3,1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using LightConvB layers."""
        x = self.cv1(self.pool(x))
        x = self.cv2(x)
        return x * self.act(x)


class LightSpatialAttention(nn.Module):
    """Spatial-attention module using LightConvB."""

    def __init__(self):
        """Initialize Spatial-attention module."""
        super().__init__()
        self.cv1 = LightConvB(2, 1, k=(1, 3), act=None)  # Cross Convolution with kernel (1,3)
        self.cv2 = LightConvB(1, 1, k=(3, 1), act=None)  # Cross Convolution with kernel (3,1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_conv1 = self.cv1(x_cat)
        x_conv2 = self.cv2(x_conv1)
        return x * self.act(x_conv2)


class LightCBAM(nn.Module):
    """Convolutional Block Attention Module with Cross Convolution."""

    def __init__(self, c1):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = LightChannelAttention(c1)
        self.spatial_attention = LightSpatialAttention()

    def forward(self, x):
        """Applies the forward pass through CBAM module."""
        return self.spatial_attention(self.channel_attention(x))


# Ensure the DWConv class is also defined if used in LightConvB


class QConv(nn.Module):
    """Quantum Convolutional Layer."""

    def __init__(self, n_qubits, backend, shots):
        super(QuantumConv, self).__init__()
        self.n_qubits = n_qubits
        self.feature_map = ZZFeatureMap(n_qubits)
        self.var_form = RealAmplitudes(n_qubits, reps=1)
        self.qcircuit = TorchConnector(self.feature_map.compose(self.var_form))
        self.backend = backend
        self.shots = shots

    def forward(self, x):
        # Quantum convolution logic
        return self.qcircuit(x, self.backend, self.shots)


class LightDSConv(nn.Module):
    """Light convolution with Batch Normalization."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, act=True):
        """Initialize Conv layers with Batch Normalization and given arguments including activation."""
        super().__init__()
        self.conv1 = DWConv(c1, c2, 1, act=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = DepthwiseSeparableConv(c2, c2, k, act=None)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply 2 convolutions with Batch Normalization to input tensor."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act(x)


def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(
        X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride
    ).view(n_x, -1, h_out * w_out)
    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    W_col = W.view(n_filters, -1)

    out = adder.apply(W_col, X_col)

    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()

    return out


class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col, X_col)
        output = -(W_col.unsqueeze(2) - X_col.unsqueeze(0)).abs().sum(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_col, X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0) - W_col.unsqueeze(2)) * grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col / grad_W_col.norm(p=2).clamp(min=1e-12) * math.sqrt(W_col.size(1) * W_col.size(0)) / 5
        grad_X_col = (-(X_col.unsqueeze(0) - W_col.unsqueeze(2)).clamp(-1, 1) * grad_output.unsqueeze(1)).sum(0)

        return grad_W_col, grad_X_col


class adder2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(
            nn.init.normal_(torch.randn(output_channel, input_channel, kernel_size, kernel_size))
        )
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output


class adderConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # Menggunakan adder2d
        self.conv = adder2d(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
