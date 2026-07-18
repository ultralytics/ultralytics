# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "ACConv",
    "Add",
    "DBBConv",
    "CBAM",
    "ChannelAttention",
    "Concat",
    "ScaledAdd",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "MobileOneConv",
    "MobileOneBlock",
    "ReparamLargeKernelConv",
    "FastViTDownsample",
    "RepConv",
    "RepGhostConv",
    "SpatialAttention",
    "WeightedFusion",
    "StripAttn",
    "GCAttn",
    "MogaGate",
    "GatedUpsample",
    "RepDWConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
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
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class ACConv(nn.Module):
    """Asymmetric Convolution Block with training and deploy modes.

    Uses 3x3 + 1x3 + 3x1 branches that fuse into a single 3x3 conv at inference.
    The asymmetric branches capture horizontal and vertical features more effectively than 1x1.

    Attributes:
        conv1 (Conv): 3x3 convolution (square branch).
        conv2 (Conv): 1x3 convolution (horizontal branch).
        conv3 (Conv): 3x1 convolution (vertical branch).
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.

    References:
        https://github.com/DingXiaoH/ACNet
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize ACConv with 3x3, 1x3, and 3x1 branches."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, 3, s, p=1, g=g, act=False)
        self.conv2 = Conv(c1, c2, (1, 3), s, p=(0, 1), g=g, act=False)
        self.conv3 = Conv(c1, c2, (3, 1), s, p=(1, 0), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + self.conv3(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Fuse all branches into equivalent 3x3 kernel and bias."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x3, bias1x3 = self._fuse_bn_tensor(self.conv2)
        kernel3x1, bias3x1 = self._fuse_bn_tensor(self.conv3)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return (
            kernel3x3 + F.pad(kernel1x3, [0, 0, 1, 1]) + F.pad(kernel3x1, [1, 1, 0, 0]) + kernelid,
            bias3x3 + bias1x3 + bias3x1 + biasid,
        )

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights."""
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
        """Fuse all branches into a single 3x3 conv for inference."""
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
        self.__delattr__("conv3")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class DBBConv(nn.Module):
    """Diverse Branch Block with training and deploy modes.

    Combines 3x3, 1x1→3x3 sequential, 1x1, and identity branches into a single 3x3 at inference.
    The diverse topology (sequential + parallel) provides richer gradient signal than uniform branches.

    References:
        https://github.com/DingXiaoH/DiverseBranchBlock
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize DBBConv with 3x3, 1x1→3x3, 1x1, and identity branches."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, 3, s, p=1, g=g, act=False)  # 3x3 branch
        self.conv2 = Conv(c1, c2, 1, s, p=0, g=g, act=False)  # 1x1 branch
        # Sequential 1x1 → 3x3 branch
        mid = max(c1 // 4 // g * g, g)
        self.seq1 = Conv(c1, mid, 1, 1, p=0, g=g, act=False)
        self.seq2 = Conv(mid, c2, 3, s, p=1, g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + self.seq2(self.seq1(x)) + id_out)

    def get_equivalent_kernel_bias(self):
        """Fuse all branches into equivalent 3x3 kernel and bias."""
        k3x3, b3x3 = self._fuse_bn_tensor(self.conv1)
        k1x1, b1x1 = self._fuse_bn_tensor(self.conv2)
        k_seq, b_seq = self._fuse_seq_branch()
        kid, bid = self._fuse_bn_tensor(self.bn)
        return k3x3 + F.pad(k1x1, [1, 1, 1, 1]) + k_seq + kid, b3x3 + b1x1 + b_seq + bid

    def _fuse_seq_branch(self):
        """Fuse sequential 1x1→3x3 branch into equivalent 3x3 kernel and bias."""
        k1, b1 = self._fuse_bn_tensor(self.seq1)  # [mid, c1//g, 1, 1]
        k3, b3 = self._fuse_bn_tensor(self.seq2)  # [c2, mid//g, 3, 3]
        # Fuse sequential convs per group
        g = self.g
        k1_groups = k1.chunk(g, dim=0)
        k3_groups = k3.chunk(g, dim=0)
        b1_groups = b1.chunk(g)
        b3_groups = b3.chunk(g)
        k_fused, b_fused = [], []
        for k1g, k3g, b1g, b3g in zip(k1_groups, k3_groups, b1_groups, b3_groups):
            # k1g: [mid/g, c1/g, 1, 1], k3g: [c2/g, mid/g, 3, 3]
            k_fused.append(torch.einsum("omhw,mi->oihw", k3g, k1g.squeeze(-1).squeeze(-1)))
            b_fused.append(k3g.sum(dim=(2, 3)) @ b1g + b3g)
        return torch.cat(k_fused), torch.cat(b_fused)

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights."""
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
        """Fuse all branches into a single 3x3 conv for inference."""
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
        self.__delattr__("seq1")
        self.__delattr__("seq2")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class RepGhostConv(nn.Module):
    """RepGhost module: Ghost convolution with reparameterizable DW branch.

    Training: 1x1 conv → primary features; DW 3x3 + identity shortcut → ghost features; cat both.
    Inference: identity shortcut fuses into DW 3x3, becoming a plain Ghost conv with no overhead.

    References:
        https://github.com/ChengpengChen/RepGhost
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepGhostConv."""
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        c_ = c2 // 2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.primary = Conv(c1, c_, 1, s, act=False)  # 1x1 primary
        self.cheap = Conv(c_, c_, 3, 1, g=c_, act=False)  # DW 3x3 cheap op
        self.shortcut = nn.BatchNorm2d(c_) if bn and s == 1 else None  # identity shortcut

    def forward_fuse(self, x):
        """Forward pass for deploy mode (shortcut fused into DW conv)."""
        y = self.act(self.primary(x))
        return torch.cat((y, self.act(self.cheap(y))), 1)

    def forward(self, x):
        """Forward pass for training mode."""
        y = self.act(self.primary(x))
        ghost = self.cheap(y)
        if self.shortcut is not None:
            ghost = ghost + self.shortcut(y)
        return torch.cat((y, self.act(ghost)), 1)

    def fuse_convs(self):
        """Fuse identity shortcut BN into DW conv."""
        if self.shortcut is None:
            return
        # Fuse DW conv + its BN
        dw_bn = self.cheap.bn
        std = (dw_bn.running_var + dw_bn.eps).sqrt()
        t = (dw_bn.weight / std).reshape(-1, 1, 1, 1)
        k_dw = self.cheap.conv.weight * t
        b_dw = dw_bn.bias - dw_bn.running_mean * dw_bn.weight / std

        # Fuse shortcut BN as identity DW kernel (1 at center per channel)
        sbn = self.shortcut
        std_s = (sbn.running_var + sbn.eps).sqrt()
        t_s = sbn.weight / std_s
        k_id = torch.zeros_like(k_dw)
        k_id[:, 0, 1, 1] = t_s
        b_id = sbn.bias - sbn.running_mean * sbn.weight / std_s

        # Replace cheap with fused plain conv (no BN)
        fused = nn.Conv2d(
            k_dw.shape[0], k_dw.shape[0], 3, 1, 1, groups=k_dw.shape[0], bias=True,
        ).requires_grad_(False)
        fused.weight.data = k_dw + k_id
        fused.bias.data = b_dw + b_id
        self.cheap = fused
        for para in self.cheap.parameters():
            para.detach_()
        self.__delattr__("shortcut")


class MobileOneConv(nn.Module):
    """MobileOne building block with reparameterizable branches.

    Uses k parallel 3x3 conv branches (each with BN) that fuse into a single 3x3 at inference.
    More parallel branches during training provide richer gradient signal.

    References:
        https://github.com/apple/ml-mobileone
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False, num_conv_branches=4):
        """Initialize MobileOneConv with multiple parallel 3x3 branches.

        Args:
            num_conv_branches (int): Number of parallel 3x3 conv branches. Default 4.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.num_conv_branches = num_conv_branches
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.convs = nn.ModuleList(Conv(c1, c2, 3, s, p=1, g=g, act=False) for _ in range(num_conv_branches))

    def forward_fuse(self, x):
        """Forward pass for deploy mode."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(sum(c(x) for c in self.convs) + id_out)

    def get_equivalent_kernel_bias(self):
        """Fuse all branches into equivalent 3x3 kernel and bias."""
        kernel, bias = 0, 0
        for conv in self.convs:
            k, b = self._fuse_bn_tensor(conv)
            kernel = kernel + k
            bias = bias + b
        kid, bid = self._fuse_bn_tensor(self.bn)
        return kernel + kid, bias + bid

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights."""
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
        """Fuse all branches into a single 3x3 conv for inference."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.convs[0].conv.in_channels,
            out_channels=self.convs[0].conv.out_channels,
            kernel_size=self.convs[0].conv.kernel_size,
            stride=self.convs[0].conv.stride,
            padding=self.convs[0].conv.padding,
            dilation=self.convs[0].conv.dilation,
            groups=self.convs[0].conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("convs")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class MobileOneBlock(nn.Module):
    """General MobileOne block with reparameterizable branches (kernel size and stride configurable).

    Unlike MobileOneConv (3x3 only), this supports arbitrary kernel size (e.g. 1x1 pointwise), stride-2
    downsampling, grouped/depthwise convs, an optional 1x1 scale branch (when k > 1) and an identity BN
    branch (when c1 == c2 and s == 1). Used by FastViTDownsample for the pointwise projection.

    References:
        https://github.com/apple/ml-fastvit (MobileOneBlock)
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, num_conv_branches=1, use_scale_branch=True):
        """Initialize MobileOneBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size of the main conv branches.
            s (int): Stride.
            p (int, optional): Padding. Autopadded when None.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
            num_conv_branches (int): Number of parallel k x k conv branches.
            use_scale_branch (bool): Add a parallel 1x1 scale branch when k > 1.
        """
        super().__init__()
        self.g = g
        self.s = s
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.num_conv_branches = num_conv_branches
        p = autopad(k, p)
        self.p = p
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None
        self.convs = nn.ModuleList(Conv(c1, c2, k, s, p=p, g=g, act=False) for _ in range(num_conv_branches))
        self.scale = Conv(c1, c2, 1, s, p=0, g=g, act=False) if use_scale_branch and k > 1 else None

    def forward_fuse(self, x):
        """Forward pass for deploy mode."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode."""
        out = sum(c(x) for c in self.convs)
        if self.scale is not None:
            out = out + self.scale(x)
        if self.bn is not None:
            out = out + self.bn(x)
        return self.act(out)

    def get_equivalent_kernel_bias(self):
        """Fuse all branches into an equivalent k x k kernel and bias."""
        kernel, bias = 0, 0
        for conv in self.convs:
            kk, bb = self._fuse_bn_tensor(conv)
            kernel = kernel + kk
            bias = bias + bb
        if self.scale is not None:
            sk, sb = self._fuse_bn_tensor(self.scale)
            pad = self.k // 2
            kernel = kernel + nn.functional.pad(sk, [pad, pad, pad, pad])
            bias = bias + sb
        kid, bid = self._fuse_bn_tensor(self.bn)
        return kernel + kid, bias + bid

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights."""
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
                kernel_value = np.zeros((self.c1, input_dim, self.k, self.k), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, self.k // 2, self.k // 2] = 1
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
        """Fuse all branches into a single k x k conv for inference."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.c1,
            out_channels=self.c2,
            kernel_size=self.k,
            stride=self.s,
            padding=self.p,
            groups=self.g,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("convs")
        if hasattr(self, "scale"):
            self.__delattr__("scale")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ReparamLargeKernelConv(nn.Module):
    """Reparameterizable large-kernel conv: large k x k branch + small kernel branch fused at inference.

    The downsampling conv of FastViT's PatchEmbed: a (grouped/depthwise) large kernel conv with stride,
    overparameterized with a parallel small-kernel branch during training, fused to a single large-kernel
    conv at deploy time.

    References:
        https://github.com/apple/ml-fastvit (ReparamLargeKernelConv) / https://arxiv.org/abs/2203.06717
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=7, s=2, g=None, small_kernel=3, act=True):
        """Initialize ReparamLargeKernelConv.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Large kernel size.
            s (int): Stride.
            g (int, optional): Groups. Defaults to gcd(c1, c2) for a depthwise-style conv.
            small_kernel (int | None): Small kernel size for the reparam branch. None disables it.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        assert small_kernel is None or small_kernel <= k, "small_kernel must be <= k"
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.g = g if g is not None else math.gcd(c1, c2)
        self.small_kernel = small_kernel
        self.p = k // 2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.lkb_origin = Conv(c1, c2, k, s, p=self.p, g=self.g, act=False)
        self.small_conv = Conv(c1, c2, small_kernel, s, p=small_kernel // 2, g=self.g, act=False) if small_kernel else None

    def forward_fuse(self, x):
        """Forward pass for deploy mode."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode."""
        out = self.lkb_origin(x)
        if self.small_conv is not None:
            out = out + self.small_conv(x)
        return self.act(out)

    @staticmethod
    def _fuse_bn_tensor(branch):
        """Fuse batch normalization with convolution weights for a Conv branch."""
        kernel = branch.conv.weight
        bn = branch.bn
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def get_equivalent_kernel_bias(self):
        """Fuse the large and small kernel branches into one large kernel and bias."""
        kernel, bias = self._fuse_bn_tensor(self.lkb_origin)
        if self.small_conv is not None:
            sk, sb = self._fuse_bn_tensor(self.small_conv)
            pad = (self.k - self.small_kernel) // 2
            kernel = kernel + nn.functional.pad(sk, [pad, pad, pad, pad])
            bias = bias + sb
        return kernel, bias

    def fuse_convs(self):
        """Fuse branches into a single large-kernel conv for inference."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.c1,
            out_channels=self.c2,
            kernel_size=self.k,
            stride=self.s,
            padding=self.p,
            groups=self.g,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")


class FastViTDownsample(nn.Module):
    """FastViT PatchEmbed downsampling block.

    Replaces a stride-2 Conv with FastViT's patch embedding: a reparameterizable large-kernel depthwise
    conv (stride 2) followed by a reparameterizable 1x1 pointwise MobileOne block. Both branches collapse
    to plain convs at inference via model.fuse().

    References:
        https://github.com/apple/ml-fastvit (PatchEmbed)
    """

    def __init__(self, c1, c2, k=7, small_kernel=3):
        """Initialize FastViTDownsample.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Large kernel (patch) size for the downsampling conv.
            small_kernel (int): Small reparam kernel size.
        """
        super().__init__()
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(c1, c2, k=k, s=2, g=math.gcd(c1, c2), small_kernel=small_kernel),
            MobileOneBlock(c2, c2, k=1, s=1, g=1),
        )

    def forward(self, x):
        """Forward pass."""
        return self.proj(x)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False, identity=True):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
            identity (bool): Use identity connection branch. Set to False for RepConvN (RepConv without identity),
                which avoids destroying the residual/concatenation gradient diversity per the YOLOv7 paper.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and identity and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
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
        """Fuse convolutions for inference by creating a single equivalent convolution."""
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
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Add(nn.Module):
    """Element-wise addition of a list of tensors."""

    def forward(self, x: list[torch.Tensor]):
        """Sum input tensors element-wise.

        Args:
            x (list[torch.Tensor]): List of input tensors with matching shapes.

        Returns:
            (torch.Tensor): Element-wise sum.
        """
        return sum(x)


class ScaledAdd(nn.Module):
    """Element-wise addition with a learnable scale on the residual input.

    Computes x[0] + alpha * x[1] where alpha is a learnable scalar initialized to 0,
    so training starts as pure x[0] and gradually learns the residual contribution.
    """

    def __init__(self):
        """Initialize ScaledAdd with alpha=0."""
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: list[torch.Tensor]):
        """Apply scaled addition.

        Args:
            x (list[torch.Tensor]): Two tensors [main, residual] with matching shapes.

        Returns:
            (torch.Tensor): main + alpha * residual.
        """
        return x[0] + self.alpha * x[1]


class WeightedFusion(nn.Module):
    """Learnable softmax-weighted fusion of same-shape features, followed by a 1x1 conv."""

    def __init__(self, c: int, num_inputs: int = 3):
        """Initialize WeightedFusion.

        Args:
            c (int): Number of channels of each input (and output).
            num_inputs (int): Number of input tensors to fuse.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs) / num_inputs)
        self.conv = nn.Conv2d(c, c, 1)

    def forward(self, x: list[torch.Tensor]):
        """Fuse input tensors with softmax-normalized learnable weights.

        Args:
            x (list[torch.Tensor]): Input tensors with matching shapes.

        Returns:
            (torch.Tensor): Fused tensor.
        """
        w = F.softmax(self.weights, dim=0)
        return self.conv(sum(wi * f for wi, f in zip(w, x)))


class RepDWConv(nn.Module):
    """Reparameterizable multi-dilation depthwise conv (ZipDepth-style parallel dilation).

    Trains with parallel DW 3x3 (dilation 1) + DW 3x3 (dilation 2) + BN identity branches,
    then fuses losslessly into a single DW 5x5 (a dilated 3x3 is a sparse 5x5). Doubles the
    receptive field of a plain DW 3x3 for the cost of a DW 5x5 at inference.
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c: int, act: bool | nn.Module = True, bn: bool = True):
        """Initialize RepDWConv.

        Args:
            c (int): Number of input/output channels.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization identity branch.
        """
        super().__init__()
        self.c = c
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.conv1 = Conv(c, c, 3, 1, g=c, act=False)
        self.conv2 = Conv(c, c, 3, 1, p=2, g=c, d=2, act=False)
        self.bn = nn.BatchNorm2d(c) if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through parallel branches (training mode)."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the fused DW 5x5 (deploy mode)."""
        return self.act(self.conv(x))

    @staticmethod
    def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """Fold a BN layer into preceding conv weights."""
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def get_equivalent_kernel_bias(self):
        """Merge all branches into a single equivalent DW 5x5 kernel and bias."""
        k1, b1 = self._fuse_bn(self.conv1.conv, self.conv1.bn)
        k2, b2 = self._fuse_bn(self.conv2.conv, self.conv2.bn)
        k = F.pad(k1, [1, 1, 1, 1])  # 3x3 d=1 -> 5x5 center
        kd = torch.zeros(self.c, 1, 5, 5, device=k.device, dtype=k.dtype)
        kd[:, :, ::2, ::2] = k2  # 3x3 d=2 -> sparse 5x5 taps
        k, b = k + kd, b1 + b2
        if self.bn is not None:
            std = (self.bn.running_var + self.bn.eps).sqrt()
            k[:, 0, 2, 2] += self.bn.weight / std
            b = b + self.bn.bias - self.bn.running_mean * self.bn.weight / std
        return k, b

    def fuse_convs(self):
        """Fuse branches into a single DW 5x5 convolution for inference."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(self.c, self.c, 5, 1, 2, groups=self.c, bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        self.__delattr__("bn")


class StripAttn(nn.Module):
    """Strip-pooling attention with a symmetric sigmoid gate (SPNet CVPR 2020, gate per ZipDepth).

    Pools horizontal and vertical strips for global axial context, mixes each strip with a
    depthwise 1D conv and a pointwise conv, then gates the input with 2*sigmoid so features
    can be both suppressed (<1) and amplified (>1). Cost is O(C^2 * (H + W)) — near-free.
    """

    def __init__(self, c: int, k: int = 3):
        """Initialize StripAttn.

        Args:
            c (int): Number of input/output channels.
            k (int): Kernel size of the depthwise 1D strip convs.
        """
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c, bias=False)
        self.conv_w = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False)
        self.fc_h = nn.Conv2d(c, c, 1)
        self.fc_w = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate input with broadcast sum of horizontal and vertical strip contexts."""
        gh = self.fc_h(self.conv_h(self.pool_h(x)))
        gw = self.fc_w(self.conv_w(self.pool_w(x)))
        return x * (2 * (gh + gw).sigmoid())


class MogaGate(nn.Module):
    """Gated multi-order aggregation (MogaNet, ICLR 2024), channel-preserving residual form.

    Feature decomposition amplifies per-channel deviations from the spatial mean, then a
    SiLU gate is multiplied with multi-order context from parallel DW 3x3 convs at
    dilations {1, 2, 3} (RF 3/5/7 — the mid-order band between conv locality and global
    pooling). All ops are DW or 1x1.
    """

    def __init__(self, c: int):
        """Initialize MogaGate.

        Args:
            c (int): Number of input/output channels.
        """
        super().__init__()
        self.gamma = nn.Parameter(1e-5 * torch.ones(c, 1, 1))
        self.gate = nn.Conv2d(c, c, 1)
        self.dw1 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw2 = nn.Conv2d(c, c, 3, 1, 2, dilation=2, groups=c, bias=False)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 3, dilation=3, groups=c, bias=False)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature decomposition, then gated multi-order context, residually."""
        y = x + self.gamma * (x - x.mean((2, 3), keepdim=True))  # feature decomposition
        ctx = self.dw1(y) + self.dw2(y) + self.dw3(y)  # fusable to a single DW 7x7
        return x + self.proj(F.silu(self.gate(y)) * F.silu(ctx))


class GCAttn(nn.Module):
    """Global Context block (GCNet, ICCVW 2019).

    Aggregates a single scene-level context vector via softmax-weighted spatial pooling,
    transforms it with a bottleneck channel MLP (norm-stabilized), and broadcast-adds
    it back. O(C*HW) pooling — near-free global context. Uses GroupNorm(1, ch) instead of
    the paper's LayerNorm([ch, 1, 1]) — identical normalization on a (b, ch, 1, 1) tensor,
    but with 1D affine params so optimizers that route >=2D weights to matrix-based updates
    (e.g. MuSGD/Muon) treat them as norm params.
    """

    def __init__(self, c: int, e: float = 0.25):
        """Initialize GCAttn.

        Args:
            c (int): Number of input/output channels.
            e (float): Bottleneck ratio for the channel transform.
        """
        super().__init__()
        ch = max(int(c * e), 16)
        self.mask = nn.Conv2d(c, 1, 1)
        self.transform = nn.Sequential(
            nn.Conv2d(c, ch, 1),
            nn.GroupNorm(1, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, c, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add transformed global context to every position."""
        b, c = x.shape[:2]
        attn = self.mask(x).flatten(2).softmax(-1)  # (b, 1, hw)
        ctx = (x.flatten(2) @ attn.transpose(-2, -1)).view(b, c, 1, 1)
        return x + self.transform(ctx)


class GatedUpsample(nn.Module):
    """Learned per-pixel blend of nearest and bilinear 2x upsampling (ZipDepth NPU path).

    A lightweight gate (1x1 reduce -> DW 5x5 -> 1x1 -> sigmoid) predicts alpha from the
    low-resolution input; output = alpha * nearest + (1 - alpha) * bilinear. Uses only
    standard Conv2d and interpolation ops, so it exports cleanly to NPU/mobile runtimes.
    """

    def __init__(self, c: int, scale: int = 2, e: float = 0.25):
        """Initialize GatedUpsample.

        Args:
            c (int): Number of input/output channels.
            scale (int): Upsampling factor.
            e (float): Bottleneck ratio for the gate.
        """
        super().__init__()
        ch = max(int(c * e), 16)
        self.scale = scale
        self.gate = nn.Sequential(Conv(c, ch, 1), Conv(ch, ch, 5, g=ch), nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample with a learned nearest/bilinear blend."""
        a = F.interpolate(self.gate(x), scale_factor=self.scale, mode="nearest")
        up_nn = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        up_bi = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return a * up_nn + (1 - a) * up_bi


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
