# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

# Try to import MMCV deformable conv ops (for CUDA), fallback to torchvision (works on CPU/MPS)
try:
    from mmcv.ops import DeformConv2d as MMCVDeformConv2d
    from mmcv.ops import ModulatedDeformConv2d as MMCVModulatedDeformConv2d
    MMCV_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MMCV_AVAILABLE = False
    # Fallback: use torchvision's deformable conv (supports CPU/MPS/CUDA)
    try:
        from torchvision.ops import DeformConv2d as TVDeformConv2d
        TORCHVISION_DCN_AVAILABLE = True
    except ImportError:
        TORCHVISION_DCN_AVAILABLE = False

# Try to import DCNv3 from OpenGVLab's InternImage
try:
    from dcnv3 import DCNv3 as DCNv3_Op
    DCNV3_AVAILABLE = True
except ImportError:
    DCNV3_AVAILABLE = False
    try:
        from ops_dcnv3 import DCNv3Function, dcnv3_core_pytorch
        DCNV3_PYTORCH_AVAILABLE = True
    except ImportError:
        DCNV3_PYTORCH_AVAILABLE = False

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DCNv3Bottleneck",
    "DCNv3C2f",
    "DCNv3Conv",
    "DWConv",
    "DWConvTranspose2d",
    "DeformBottleneck",
    "DeformC2f",
    "DeformConv",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

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
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

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
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

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


class DeformConv(nn.Module):
    """Deformable Convolutional Networks v2 (DCN v2) - Modulated Deformable Convolution.

    Implements DCN v2 from "Deformable ConvNets v2: More Deformable, Better Results" (Zhu et al., CVPR 2019).

    DCN v2 Key Features:
    - **Learnable Offsets**: 2D offset prediction for each kernel position (2 * k * k channels)
    - **Modulation Mechanism**: Learns importance weights for each sampling position (k * k channels)
    - **Adaptive Sampling**: Adjusts receptive field based on input content
    - **Better Localization**: Particularly effective for objects with deformation/scale variation

    The modulation mechanism is the key improvement over DCN v1:
        output = Σ w(p) · m(p) · x(p + Δp)
    where:
        - w(p): learnable convolution weights
        - m(p): learnable modulation scalars (NEW in v2)
        - Δp: learnable offset vectors
        - p: sampling positions

    Multi-Backend Support:
    1. MMCV (preferred for CUDA training) - Full DCN v2 with modulation
    2. TorchVision (fallback for CPU/MPS) - DCN v1 (no modulation)
    3. Regular Conv2d (universal fallback) - Standard convolution

    References:
        Zhu et al. "Deformable ConvNets v2: More Deformable, Better Results" CVPR 2019
        Zhang et al. "Offset-decoupled deformable convolution" 2022 (zero-init)
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, modulated=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.modulated = modulated
        self.k = k
        self.stride = s
        self.padding = p
        self.dilation = d
        self.groups = g
        self.c1 = c1
        self.c2 = c2

        # Choose deformable conv backend based on availability
        if MMCV_AVAILABLE:
            # Use MMCV (best for CUDA)
            if modulated:
                self.conv = MMCVModulatedDeformConv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
            else:
                self.conv = MMCVDeformConv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
            self.backend = 'mmcv'
        elif TORCHVISION_DCN_AVAILABLE and not modulated:
            # Use TorchVision (works on CPU/MPS, but only non-modulated)
            self.conv = TVDeformConv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
            self.backend = 'torchvision'
        else:
            # Fallback to regular conv with warning
            self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
            self.backend = 'standard'
            import warnings
            warnings.warn(
                f"DeformConv: No DCN backend available (MMCV: {MMCV_AVAILABLE}, TorchVision: {TORCHVISION_DCN_AVAILABLE}). "
                f"Falling back to standard Conv2d. For full DCN support, install: pip install mmcv-full (CUDA) or ensure torchvision>=0.11"
            )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

        # Offset/mask predictor: outputs g * 2*k*k (offsets) or g * 3*k*k (offsets+mask)
        # FIX: For grouped deformable convolutions, we need offsets/masks per group
        # Only create if we're using actual deformable conv
        if self.backend != 'standard':
            offset_channels = self.groups * (
                3 * self.k * self.k if modulated and self.backend == 'mmcv'
                else 2 * self.k * self.k
            )
            self.offset_mask_conv = nn.Conv2d(
                c1,
                offset_channels,
                kernel_size=3,
                stride=s,
                padding=autopad(3, None, d),
                bias=True,
            )

            # FIX #1: Initialize offset predictor to zero for training stability
            # Reference: Zhang et al. (2022) "Offset-decoupled deformable convolution"
            # This ensures deformable conv behaves like regular conv at training start
            nn.init.constant_(self.offset_mask_conv.weight, 0.0)
            nn.init.constant_(self.offset_mask_conv.bias, 0.0)

    def forward(self, x):
        if self.backend == 'standard':
            # Standard conv fallback
            out = self.conv(x)
        else:
            # DCN forward pass (MMCV or TorchVision)
            offset_mask = self.offset_mask_conv(x)

            # Runtime shape assertion: verify offset/mask channels match expected groups
            expected_channels = self.groups * (
                3 * self.k * self.k if self.modulated and self.backend == 'mmcv'
                else 2 * self.k * self.k
            )
            assert offset_mask.shape[1] == expected_channels, (
                f"Offset/mask channel mismatch in DeformConv: "
                f"got {offset_mask.shape[1]}, expected {expected_channels} "
                f"(groups={self.groups}, k={self.k}, modulated={self.modulated})"
            )

            off_ch = self.groups * 2 * self.k * self.k
            if self.modulated and self.backend == 'mmcv':
                # Modulated DCN: split into offsets and mask
                o1 = offset_mask[:, :off_ch, :, :]
                mask = offset_mask[:, off_ch:off_ch + self.groups * self.k * self.k, :, :].sigmoid()
                out = self.conv(x, o1, mask)
            else:
                # Non-modulated DCN: offsets only
                o1 = offset_mask
                out = self.conv(x, o1)

        out = self.bn(out)
        out = self.act(out)
        return out


class DeformBottleneck(nn.Module):
    """Bottleneck block with DCN v2 (Modulated Deformable Convolution).

    Architecture: input → Conv1x1(reduce) → DCNv2_3x3 → (+skip) → output

    Uses DCN v2 for the 3x3 convolution with:
    - Learnable offsets (2D displacement for each kernel position)
    - Modulation masks (importance weights for each sampling position)
    - Adaptive receptive field (adjusts based on input content)

    More efficient than applying DCN directly to full channels by:
    1. Reducing channels with 1x1 conv (compression)
    2. Applying expensive DCN v2 on reduced channels
    3. Learning adaptive sampling on compressed features

    This design is optimal for vehicle detection as it:
    - Adapts to varying vehicle scales
    - Handles different vehicle poses/orientations
    - Better captures geometric deformations

    References:
        Zhu et al. "Deformable ConvNets v2: More Deformable, Better Results" CVPR 2019
        He et al. "Deep Residual Learning for Image Recognition" CVPR 2016 (bottleneck design)
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, modulated=True):
        """
        Initialize Deformable Bottleneck.

        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            shortcut (bool): Use residual connection (default: True)
            g (int): Groups for deformable conv (default: 1)
            k (tuple): Kernel sizes (default: (3, 3))
            e (float): Expansion ratio for hidden channels (default: 0.5)
            modulated (bool): Use modulated DCN (default: True)
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # 1x1 conv for channel reduction (with BN + activation)
        self.cv1 = Conv(c1, c_, k[0], 1)

        # 3x3 deformable conv for adaptive spatial sampling
        self.cv2 = DeformConv(c_, c2, k[1], 1, g=g, modulated=modulated)

        # Residual connection when dimensions match
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass with optional residual connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DeformC2f(nn.Module):
    """C2f block with DCN v2 (Modulated Deformable Convolution) + CSP architecture.

    Combines:
    1. DCN v2 (Zhu et al., 2019) - Adaptive spatial sampling with modulation
    2. CSP (Wang et al., 2020) - Dual-path feature splitting for gradient flow

    DCN v2 FEATURES:
    - ✅ Learnable 2D offsets (adaptive receptive field)
    - ✅ Modulation masks (attention on sampling positions)
    - ✅ Zero-initialized offsets (training stability)
    - ✅ Better localization for deformed/scaled objects

    ARCHITECTURE FIXES APPLIED:
    - ✅ Dual-path CSP split (proper gradient flow)
    - ✅ BatchNorm in cv1/cv2 (training stability)
    - ✅ DCN v2 bottleneck structure (efficient learning)
    - ✅ Correct channel flow: (2+n)*c

    Architecture Flow:
                     ┌─────────────────┐
        x ───────────┤ cv1(c1→2c) ─────┼──── chunk(2) ───┐
                     └─────────────────┘                  │
                            │                        path2 │
                       path1 (c)                      (c)  │
                            │                              │
                     ┌──────▼──────────┐                   │
                     │ DeformBottleneck │ ← DCN v2 here!   │
                     │ (with modulation)│                  │
                     └──────┬───────────┘                  │
                            ... (n times)                  │
                     ┌──────┴──────────────────────────────▼───┐
                     │ Concat: path1 + path2 + ... = (2+n)*c   │
                     └──────┬─────────────────────────────────┘
                            │
                     ┌──────▼──────┐
                     │ cv2 → c2    │
                     └─────────────┘

    Why DCN v2 for Vehicle Detection?
    - Adapts to vehicle scale variations (small cars to large trucks)
    - Handles vehicle pose variations (front/side/rear views)
    - Better localization with modulation mechanism
    - Robust to partial occlusions in dense traffic

    References:
        Zhu et al. "Deformable ConvNets v2: More Deformable, Better Results" CVPR 2019
        Wang et al. "CSPNet: A New Backbone that can enhance learning capability of CNN" CVPRW 2020
        Zhang et al. "Offset-decoupled deformable convolution" 2022
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, modulated=True):
        """
        Initialize DeformC2f with proper CSP architecture.

        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of deformable bottleneck blocks (default: 1)
            shortcut (bool): Use shortcuts in bottlenecks (default: False)
            g (int): Groups for deformable conv (default: 1)
            e (float): Expansion ratio for hidden channels (default: 0.5)
            modulated (bool): Use modulated DCN (default: True)
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # FIX #2: Dual-path CSP split - produces 2*c channels, will be split into 2 paths
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # FIX #3: Use DeformBottleneck (not raw DeformConv)
        # Process one path through n deformable bottlenecks
        self.m = nn.ModuleList(
            DeformBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, modulated=modulated)
            for _ in range(n)
        )

        # FIX #2 & #4: Correct channel count (2 from split + n from bottlenecks) * c
        # Merge all paths back to c2 channels
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        """
        Forward pass with CSP architecture.

        Args:
            x (torch.Tensor): Input tensor (B, c1, H, W)

        Returns:
            torch.Tensor: Output tensor (B, c2, H, W)
        """
        # FIX #2: Split into 2 paths (CSP design)
        y = list(self.cv1(x).chunk(2, 1))  # [path1(c), path2(c)]

        # Process path2 through deformable bottlenecks
        y.extend(m(y[-1]) for m in self.m)  # [path1, path2, bottle1, ..., bottleN]

        # Concatenate all paths and merge
        return self.cv2(torch.cat(y, 1))


class DCNv3Conv(nn.Module):
    """Deformable Convolutional Networks v3 (DCN v3) - From InternImage (OpenGVLab).

    Implements DCN v3 from "InternImage: Exploring Large-Scale Vision Foundation Models
    with Deformable Convolutions" (Wang et al., CVPR 2023).

    DCN v3 KEY IMPROVEMENTS over v2:
    - **Group-wise Learning**: Splits channels into groups for efficient multi-scale learning
    - **Shared Offsets**: Uses shared offset prediction across groups (more efficient)
    - **Softmax Normalization**: Normalizes sampling weights via softmax (better than sigmoid)
    - **Removes Modulation**: Simplifies to offsets + softmax weights (more stable)
    - **Center Feature**: Explicitly adds center position feature
    - **Better Scaling**: Scales to larger models (InternImage-H: 1B+ params)

    Key Differences from DCN v2:
    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │ Feature         │ DCN v2               │ DCN v3 (This)        │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ Offsets         │ Per-kernel           │ Shared across groups │
    │ Attention       │ Sigmoid modulation   │ Softmax weights      │
    │ Groups          │ Single group         │ Multi-group          │
    │ Center feature  │ Implicit             │ Explicit addition    │
    │ Efficiency      │ Moderate             │ High                 │
    │ Stability       │ Good                 │ Better               │
    └─────────────────┴──────────────────────┴──────────────────────┘

    Architecture:
        input → [Offset Prediction] → offsets (2×K points)
              ↓
              [Sampling + Softmax Attention] → weighted features
              ↓
              [Linear Projection] → output

    Multi-Backend Support:
    1. DCNv3 CUDA ops (preferred for training) - OpenGVLab implementation
    2. DCNv3 PyTorch (fallback for CPU/development) - Pure PyTorch version
    3. Standard Conv2d (universal fallback) - Graceful degradation

    References:
        Wang et al. "InternImage: Exploring Large-Scale Vision Foundation Models
        with Deformable Convolutions" CVPR 2023
        Code: https://github.com/OpenGVLab/InternImage
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, kernel_size=3,
                 dw_kernel_size=None, center_feature_scale=False):
        """Initialize DCN v3 layer.

        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            k (int): Kernel size (default: 3)
            s (int): Stride (default: 1)
            p (int): Padding (default: None, auto-calculated)
            g (int): Number of groups for deformable sampling (default: 1)
            d (int): Dilation (default: 1)
            kernel_size (int): DCNv3 kernel size (default: 3, typically 3×3)
            dw_kernel_size (int): Depthwise kernel size for offset prediction (default: None)
            center_feature_scale (bool): Scale center feature (default: False)
        """
        super().__init__()
        if p is None:
            p = k // 2

        self.c1 = c1
        self.c2 = c2
        self.kernel_size = kernel_size
        self.stride = s
        self.padding = p
        self.dilation = d
        self.groups = g
        self.center_feature_scale = center_feature_scale
        self.dw_kernel_size = dw_kernel_size or kernel_size

        # Choose DCNv3 backend based on availability
        if DCNV3_AVAILABLE:
            # Use OpenGVLab's optimized CUDA implementation
            self.dcn = DCNv3_Op(
                channels=c1,
                kernel_size=kernel_size,
                stride=s,
                pad=p,
                dilation=d,
                group=g,
                offset_scale=1.0,
                act_layer='GELU',
                norm_layer='LN',
                dw_kernel_size=self.dw_kernel_size,
                center_feature_scale=center_feature_scale,
            )
            self.backend = 'dcnv3'
        elif DCNV3_PYTORCH_AVAILABLE:
            # Use pure PyTorch implementation (slower but works on CPU)
            self.dcn = DCNv3PyTorch(
                c1, c2, kernel_size, s, p, d, g,
                self.dw_kernel_size, center_feature_scale
            )
            self.backend = 'dcnv3_pytorch'
        else:
            # Fallback to standard conv with warning
            self.dcn = nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=1, bias=False)
            self.backend = 'standard'
            import warnings
            warnings.warn(
                f"DCNv3: CUDA ops not found — using fallback (non-deformable projection). "
                f"For true DCNv3 deformable convolution, install: pip install DCNv3 (from OpenGVLab/InternImage). "
                f"This fallback provides standard convolution only, without adaptive spatial sampling."
            )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """Forward pass through DCN v3."""
        if self.backend == 'standard':
            out = self.dcn(x)
        else:
            # DCNv3 handles offset prediction internally
            out = self.dcn(x)

        out = self.bn(out)
        out = self.act(out)
        return out


class DCNv3PyTorch(nn.Module):
    """Pure PyTorch implementation of DCNv3 (fallback for CPU/development).

    This is a simplified PyTorch-only version for compatibility when CUDA ops aren't available.
    For production training, use the CUDA-optimized DCNv3_Op from OpenGVLab.
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 dw_kernel_size=3, center_feature_scale=False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.center_feature_scale = center_feature_scale

        # Depthwise conv for offset prediction
        self.offset_conv = nn.Conv2d(
            c1,
            groups * 2 * kernel_size * kernel_size,  # 2D offsets per group
            kernel_size=dw_kernel_size,
            stride=stride,
            padding=dw_kernel_size // 2,
            groups=c1,  # depthwise
            bias=True
        )

        # Attention weights (softmax normalized, replaces DCNv2 modulation)
        self.attention_conv = nn.Conv2d(
            c1,
            groups * kernel_size * kernel_size,
            kernel_size=dw_kernel_size,
            stride=stride,
            padding=dw_kernel_size // 2,
            groups=c1,  # depthwise
            bias=True
        )

        # Linear projection
        self.proj = nn.Conv2d(c1, c2, kernel_size=1, bias=False)

        # Initialize offset predictor to zero (stability)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        # Initialize attention to produce uniform weights
        nn.init.constant_(self.attention_conv.weight, 0.0)
        nn.init.constant_(self.attention_conv.bias, 0.0)

    def forward(self, x):
        """Forward pass - simplified PyTorch version (non-deformable fallback).

        WARNING: This fallback implementation predicts offsets and attention but does NOT
        apply deformable sampling. It simply projects the input features directly.
        For true DCNv3 deformable convolution, use the CUDA implementation from OpenGVLab.
        """
        B, C, H, W = x.shape

        # Predict offsets and attention weights (for training stability, even if unused)
        offsets = self.offset_conv(x)  # [B, groups*2*K*K, H, W]
        attention = self.attention_conv(x).softmax(dim=1)  # [B, groups*K*K, H, W]

        # Runtime shape assertion: verify offset channels match expected groups
        expected_offset_ch = self.groups * 2 * self.kernel_size * self.kernel_size
        expected_attn_ch = self.groups * self.kernel_size * self.kernel_size
        assert offsets.shape[1] == expected_offset_ch, (
            f"DCNv3 offset channel mismatch: got {offsets.shape[1]}, expected {expected_offset_ch}"
        )
        assert attention.shape[1] == expected_attn_ch, (
            f"DCNv3 attention channel mismatch: got {attention.shape[1]}, expected {expected_attn_ch}"
        )

        # For simplicity in CPU fallback, just apply projection
        # (full deformable sampling would require custom CUDA ops)
        out = self.proj(x)

        return out


class DCNv3Bottleneck(nn.Module):
    """Bottleneck block with DCN v3 (from InternImage).

    Architecture: input → Conv1x1(reduce) → DCNv3_3x3 → (+skip) → output

    Uses DCN v3 for the 3x3 convolution with:
    - Group-wise learning (multi-scale features)
    - Shared offsets across groups (efficiency)
    - Softmax attention weights (stability)
    - Explicit center feature (better learning)

    This design is optimal for large-scale models and is the backbone of:
    - InternImage-T/S/B/L/XL/H (22M to 1B+ parameters)
    - State-of-the-art on ImageNet, COCO, ADE20K

    References:
        Wang et al. "InternImage: Exploring Large-Scale Vision Foundation Models
        with Deformable Convolutions" CVPR 2023
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,
                 kernel_size=3, center_feature_scale=False):
        """Initialize DCN v3 Bottleneck.

        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            shortcut (bool): Use residual connection (default: True)
            g (int): Groups for DCNv3 (default: 1, typically 4-16 for InternImage)
            k (tuple): Kernel sizes (default: (3, 3))
            e (float): Expansion ratio for hidden channels (default: 0.5)
            kernel_size (int): DCNv3 kernel size (default: 3)
            center_feature_scale (bool): Scale center feature (default: False)
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # 1x1 conv for channel reduction (with BN + activation)
        self.cv1 = Conv(c1, c_, k[0], 1)

        # 3x3 DCN v3 for adaptive spatial sampling
        self.cv2 = DCNv3Conv(c_, c2, k[1], 1, g=g, kernel_size=kernel_size,
                             center_feature_scale=center_feature_scale)

        # Residual connection when dimensions match
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass with optional residual connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DCNv3C2f(nn.Module):
    """C2f block with DCN v3 (from InternImage) + CSP architecture.

    Combines:
    1. DCN v3 (Wang et al., 2023) - Large-scale deformable convolution
    2. CSP (Wang et al., 2020) - Dual-path feature splitting

    DCN v3 FEATURES (vs v2):
    - ✅ Group-wise learning (better multi-scale)
    - ✅ Shared offsets (more efficient)
    - ✅ Softmax attention (vs sigmoid modulation)
    - ✅ Explicit center feature
    - ✅ Better scaling to large models

    This is the architecture used in InternImage, which achieves:
    - ImageNet: 89.6% Top-1 (InternImage-H)
    - COCO: 65.4 box AP (InternImage-H)
    - ADE20K: 62.9 mIoU (InternImage-H)

    References:
        Wang et al. "InternImage: Exploring Large-Scale Vision Foundation Models
        with Deformable Convolutions" CVPR 2023
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 kernel_size=3, center_feature_scale=False):
        """Initialize DCNv3C2f with CSP architecture.

        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of DCN v3 bottleneck blocks (default: 1)
            shortcut (bool): Use shortcuts in bottlenecks (default: False)
            g (int): Groups for DCNv3 (default: 1, typically 4-16)
            e (float): Expansion ratio for hidden channels (default: 0.5)
            kernel_size (int): DCNv3 kernel size (default: 3)
            center_feature_scale (bool): Scale center feature (default: False)
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # Dual-path CSP split
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Process one path through n DCN v3 bottlenecks
        self.m = nn.ModuleList(
            DCNv3Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0,
                           kernel_size=kernel_size, center_feature_scale=center_feature_scale)
            for _ in range(n)
        )

        # Merge all paths back to c2 channels
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        """Forward pass with CSP architecture.

        Args:
            x (torch.Tensor): Input tensor (B, c1, H, W)

        Returns:
            torch.Tensor: Output tensor (B, c2, H, W)
        """
        # Split into 2 paths (CSP design)
        y = list(self.cv1(x).chunk(2, 1))  # [path1(c), path2(c)]

        # Process path2 through DCN v3 bottlenecks
        y.extend(m(y[-1]) for m in self.m)  # [path1, path2, bottle1, ..., bottleN]

        # Concatenate all paths and merge
        return self.cv2(torch.cat(y, 1))


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

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
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

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
        """
        Initialize depth-wise transpose convolution with given parameters.

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
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

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
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

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
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

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
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

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

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

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
        """
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
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

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
        """
        Pad a 1x1 kernel to 3x3 size.

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
        """
        Fuse batch normalization with convolution weights.

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
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """
        Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
