# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

# Import MMCV deformable conv ops (CUDA-optimized)
try:
    from mmcv.ops import DeformConv2d as MMCVDeformConv2d
    from mmcv.ops import ModulatedDeformConv2d as MMCVModulatedDeformConv2d

    MMCV_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MMCV_AVAILABLE = False
    # Use torchvision's deformable conv (supports CPU/MPS/CUDA)
    try:
        from torchvision.ops import DeformConv2d as TVDeformConv2d

        TORCHVISION_DCN_AVAILABLE = True
    except ImportError:
        TORCHVISION_DCN_AVAILABLE = False

# Import DCNv3 from OpenGVLab's InternImage (CUDA-optimized)
try:
    from dcnv3 import DCNv3 as DCNv3_Op

    DCNV3_AVAILABLE = True
except ImportError:
    DCNV3_AVAILABLE = False

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

    Key Features:
    - Learnable 2D offsets: Predicts spatial displacement for each kernel position (2 * k * k channels)
    - Modulation mechanism: Learns importance weights for each sampling position (k * k channels)
    - Adaptive receptive field: Adjusts sampling locations based on input content
    - Enhanced localization: Effective for objects with geometric deformation and scale variation

    Formulation (key improvement over DCN v1):
        output = Σ w(p) · m(p) · x(p + Δp)
    where:
        w(p): learnable convolution weights
        m(p): learnable modulation scalars (NEW in v2)
        Δp: learnable offset vectors
        p: sampling grid positions

    References:
        Zhu et al. "Deformable ConvNets v2: More Deformable, Better Results" CVPR 2019
        Zhang et al. "Offset-decoupled deformable convolution" 2022
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

        # Select deformable conv backend based on availability
        if MMCV_AVAILABLE:
            # MMCV implementation (CUDA-optimized, supports both modulated and non-modulated)
            if modulated:
                self.conv = MMCVModulatedDeformConv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
            else:
                self.conv = MMCVDeformConv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
            self.backend = "mmcv"
        elif TORCHVISION_DCN_AVAILABLE and not modulated:
            # TorchVision implementation (CPU/MPS/CUDA support, non-modulated only)
            self.conv = TVDeformConv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
            self.backend = "torchvision"
        else:
            # Standard convolution fallback when no DCN backend is available
            self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
            self.backend = "standard"
            import warnings

            warnings.warn(
                f"DeformConv: No DCN backend available (MMCV: {MMCV_AVAILABLE}, TorchVision: {TORCHVISION_DCN_AVAILABLE}). "
                f"Using standard Conv2d. Install mmcv-full for CUDA support or torchvision>=0.11 for CPU/MPS support."
            )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

        # Offset/mask predictor for deformable convolution
        # Outputs: g * 2*k*k (offsets only) or g * 3*k*k (offsets + modulation masks)
        # Only created when using actual deformable conv backends
        # Reference: Zhu et al. (2019) Section 3.2 - "Modulated Deformable Convolution"
        if self.backend != "standard":
            offset_channels = self.groups * (
                3 * self.k * self.k if modulated and self.backend == "mmcv" else 2 * self.k * self.k
            )
            self.offset_mask_conv = nn.Conv2d(
                c1,
                offset_channels,
                kernel_size=3,
                stride=s,
                padding=autopad(3, None, d),
                bias=True,
            )

            # Zero-initialization for training stability
            # Ensures DCN behaves like regular convolution at initialization
            # Reference: Zhang et al. (2022) "Offset-decoupled deformable convolution" Section 3.3
            nn.init.constant_(self.offset_mask_conv.weight, 0.0)
            nn.init.constant_(self.offset_mask_conv.bias, 0.0)

    def forward(self, x):
        # Input validation for robustness
        assert x.ndim == 4, f"DeformConv expected 4D input (B,C,H,W), got {x.ndim}D with shape {x.shape}"
        assert x.shape[0] > 0, f"DeformConv: Empty batch not supported (batch_size={x.shape[0]})"
        assert x.shape[1] == self.c1, f"DeformConv: Channel mismatch - expected {self.c1}, got {x.shape[1]}"
        assert x.shape[2] >= self.k and x.shape[3] >= self.k, (
            f"DeformConv: Input spatial dims {x.shape[2:]} smaller than kernel size {self.k}"
        )

        if self.backend == "standard":
            # Standard convolution fallback
            out = self.conv(x)
        else:
            # Deformable convolution forward pass (MMCV or TorchVision)
            offset_mask = self.offset_mask_conv(x)

            # Verify offset/mask channels match expected configuration
            expected_channels = self.groups * (
                3 * self.k * self.k if self.modulated and self.backend == "mmcv" else 2 * self.k * self.k
            )
            assert offset_mask.shape[1] == expected_channels, (
                f"Offset/mask channel mismatch in DeformConv: "
                f"got {offset_mask.shape[1]}, expected {expected_channels} "
                f"(groups={self.groups}, k={self.k}, modulated={self.modulated})"
            )

            off_ch = self.groups * 2 * self.k * self.k
            if self.modulated and self.backend == "mmcv":
                # Modulated DCN: split into offsets and modulation masks
                # Reference: Zhu et al. (2019) Equation 4
                o1 = offset_mask[:, :off_ch, :, :]

                # Clip offsets to prevent extreme sampling locations
                # Prevents numerical instability during bilinear interpolation
                # Reference: Dai et al. (2017) "Deformable Convolutional Networks" Section 3.1
                max_offset = self.k * self.dilation * 2
                o1 = o1.clamp(-max_offset, max_offset)

                # Sigmoid activation for modulation masks (range [0,1])
                # Reference: Zhu et al. (2019) Section 3.2
                mask = offset_mask[:, off_ch : off_ch + self.groups * self.k * self.k, :, :].sigmoid()
                out = self.conv(x, o1, mask)
            else:
                # Non-modulated DCN: offsets only
                # Reference: Dai et al. (2017) "Deformable Convolutional Networks"
                o1 = offset_mask

                # Clip offsets to prevent extreme sampling locations
                max_offset = self.k * self.dilation * 2
                o1 = o1.clamp(-max_offset, max_offset)

                out = self.conv(x, o1)

        out = self.bn(out)
        out = self.act(out)
        return out


class DeformBottleneck(nn.Module):
    """
    Bottleneck block with DCN v2 (Modulated Deformable Convolution).

    Architecture: input → Conv1x1(reduce) → DCNv2_3x3 → (+skip) → output

    DCN v2 Features:
    - Learnable 2D offsets for each kernel position
    - Modulation masks (importance weights for sampling positions)
    - Content-adaptive receptive field

    Efficiency Strategy (based on ResNet bottleneck design):
    1. Channel reduction via 1x1 convolution (He et al. 2016)
    2. Apply DCN v2 on reduced feature channels (reduces computation)
    3. Learn adaptive sampling on compressed representation

    Benefits for Object Detection:
    - Handles scale variation (small to large objects) - Zhu et al. (2019) Section 4.2
    - Adapts to pose/orientation changes - demonstrated on COCO dataset
    - Robust to geometric deformations - Table 1 in Zhu et al. (2019)

    References:
        Zhu et al. "Deformable ConvNets v2: More Deformable, Better Results" CVPR 2019
        He et al. "Deep Residual Learning for Image Recognition" CVPR 2016
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
    """
    C2f block with DCN v2 (Modulated Deformable Convolution) + CSP architecture.

    Combines:
    1. DCN v2 (Zhu et al., 2019) - Adaptive spatial sampling with modulation
    2. CSP (Wang et al., 2020) - Dual-path feature splitting for gradient flow

    DCN v2 Features:
    - Learnable 2D offsets (content-adaptive receptive field) - Zhu et al. (2019) Eq. 4
    - Modulation masks (attention weights for sampling positions) - Zhu et al. (2019) Sec. 3.2
    - Zero-initialized offsets (training stability) - Zhang et al. (2022) Sec. 3.3
    - Enhanced localization for deformed/scaled objects - Zhu et al. (2019) Table 1

    Architecture Flow:
                     ┌─────────────────┐
        x ───────────┤ cv1(c1→2c) ─────┼──── chunk(2) ───┐
                     └─────────────────┘                  │
                            │                        path2 │
                       path1 (c)                      (c)  │
                            │                              │
                     ┌──────▼──────────┐                   │
                     │ DeformBottleneck │ ← DCN v2 here    │
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

    Benefits for Object Detection:
    - Adapts to scale variations (small to large objects) - Zhu et al. (2019) Sec. 4.2
    - Handles pose/orientation changes - validated on COCO detection benchmarks
    - Improved localization via modulation mechanism - Zhu et al. (2019) Fig. 5
    - Robust to partial occlusions - Wang et al. (2020) CSP gradient flow benefits

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

        # Dual-path CSP split - produces 2*c channels that will be split into 2 paths
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Process one path through n deformable bottlenecks
        self.m = nn.ModuleList(
            DeformBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, modulated=modulated) for _ in range(n)
        )

        # Merge all paths: (2 from split + n from bottlenecks) * c → c2 channels
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        """
        Forward pass with CSP architecture.

        Args:
            x (torch.Tensor): Input tensor (B, c1, H, W)

        Returns:
            torch.Tensor: Output tensor (B, c2, H, W)
        """
        # Split into 2 paths (CSP design)
        y = list(self.cv1(x).chunk(2, 1))  # [path1(c), path2(c)]

        # Process path2 through n deformable bottlenecks
        y.extend(m(y[-1]) for m in self.m)  # [path1, path2, bottle1, ..., bottleN]

        # Concatenate all paths and project to output channels
        return self.cv2(torch.cat(y, 1))


class DCNv3Conv(nn.Module):
    """Deformable Convolutional Networks v3 (DCN v3) - From InternImage (OpenGVLab).

    Implements DCN v3 from "InternImage: Exploring Large-Scale Vision Foundation Models
    with Deformable Convolutions" (Wang et al., CVPR 2023).

    Key Improvements over DCN v2:
    - Group-wise learning: Multi-scale feature learning via channel groups (Wang et al. 2023, Sec. 3.2)
    - Shared offsets: More efficient offset prediction across groups (Wang et al. 2023, Fig. 2)
    - Softmax normalization: Attention weights normalized via softmax (Wang et al. 2023, Eq. 3)
    - Simplified design: Offsets + softmax weights, removes separate modulation (Wang et al. 2023, Sec. 3.1)
    - Explicit center feature: Better gradient flow and learning (Wang et al. 2023, Sec. 3.2)
    - Better scalability: Supports large-scale models up to 1B+ params (Wang et al. 2023, Table 1)

    DCN v2 vs DCN v3 Comparison:
    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │ Feature         │ DCN v2               │ DCN v3               │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ Offsets         │ Per-kernel           │ Shared across groups │
    │ Attention       │ Sigmoid modulation   │ Softmax weights      │
    │ Groups          │ Single group         │ Multi-group          │
    │ Center feature  │ Implicit             │ Explicit addition    │
    │ Efficiency      │ Moderate             │ High                 │
    │ Stability       │ Good                 │ Better               │
    └─────────────────┴──────────────────────┴──────────────────────┘

    Architecture:
        input → [Offset Prediction] → offsets (2×K sampling points)
              ↓
              [Deformable Sampling + Softmax Attention] → weighted features
              ↓
              [Linear Projection] → output

    References:
        Wang et al. "InternImage: Exploring Large-Scale Vision Foundation Models
        with Deformable Convolutions" CVPR 2023
        Code: https://github.com/OpenGVLab/InternImage
    """

    def __init__(
        self, c1, c2, k=3, s=1, p=None, g=1, d=1, kernel_size=3, dw_kernel_size=None, center_feature_scale=False
    ):
        """
        Initialize DCN v3 layer.

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

        # Select DCNv3 backend based on availability
        if DCNV3_AVAILABLE:
            # OpenGVLab's CUDA-optimized implementation
            self.dcn = DCNv3_Op(
                channels=c1,
                kernel_size=kernel_size,
                stride=s,
                pad=p,
                dilation=d,
                group=g,
                offset_scale=1.0,
                act_layer="GELU",
                norm_layer="LN",
                dw_kernel_size=self.dw_kernel_size,
                center_feature_scale=center_feature_scale,
            )
            self.backend = "dcnv3"
        else:
            # Standard convolution fallback when DCNv3 is not available
            self.dcn = nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=1, bias=False)
            self.backend = "standard"
            import warnings

            warnings.warn(
                "DCNv3: CUDA implementation not found. Using standard convolution. "
                "Install DCNv3 for deformable convolution: pip install DCNv3 (from OpenGVLab/InternImage)"
            )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """Forward pass through DCN v3."""
        # Input validation for robustness
        assert x.ndim == 4, f"DCNv3Conv expected 4D input (B,C,H,W), got {x.ndim}D with shape {x.shape}"
        assert x.shape[0] > 0, f"DCNv3Conv: Empty batch not supported (batch_size={x.shape[0]})"
        assert x.shape[1] == self.c1, f"DCNv3Conv: Channel mismatch - expected {self.c1}, got {x.shape[1]}"
        assert x.shape[2] >= self.kernel_size and x.shape[3] >= self.kernel_size, (
            f"DCNv3Conv: Input spatial dims {x.shape[2:]} smaller than kernel size {self.kernel_size}"
        )

        if self.backend == "standard":
            out = self.dcn(x)
        else:
            # DCNv3 handles offset prediction and sampling internally
            # Offset clipping and normalization handled inside CUDA ops
            # Reference: Wang et al. (2023) Section 3.2 - "Implementation Details"
            out = self.dcn(x)

        out = self.bn(out)
        out = self.act(out)
        return out


class DCNv3Bottleneck(nn.Module):
    """
    Bottleneck block with DCN v3 (from InternImage).

    Architecture: input → Conv1x1(reduce) → DCNv3_3x3 → (+skip) → output

    DCN v3 Features:
    - Group-wise learning for multi-scale features (Wang et al. 2023, Sec. 3.2)
    - Shared offsets across groups (Wang et al. 2023, Fig. 2 - reduces parameters)
    - Softmax attention weights (Wang et al. 2023, Eq. 3 - improved stability)
    - Explicit center feature (Wang et al. 2023, Sec. 3.2 - better gradient flow)

    Used in InternImage models:
    - InternImage-T/S/B/L/XL/H (22M to 1B+ parameters)
    - State-of-the-art performance: ImageNet (89.6%), COCO (65.4 AP), ADE20K (62.9 mIoU)

    References:
        Wang et al. "InternImage: Exploring Large-Scale Vision Foundation Models
        with Deformable Convolutions" CVPR 2023
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, kernel_size=3, center_feature_scale=False):
        """
        Initialize DCN v3 Bottleneck.

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
        self.cv2 = DCNv3Conv(c_, c2, k[1], 1, g=g, kernel_size=kernel_size, center_feature_scale=center_feature_scale)

        # Residual connection when dimensions match
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass with optional residual connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DCNv3C2f(nn.Module):
    """
    C2f block with DCN v3 (from InternImage) + CSP architecture.

    Combines:
    1. DCN v3 (Wang et al., 2023) - Large-scale deformable convolution
    2. CSP (Wang et al., 2020) - Dual-path feature splitting

    DCN v3 Features (vs v2):
    - Group-wise learning (improved multi-scale representation) - Wang et al. (2023) Sec. 3.2
    - Shared offsets (more efficient, fewer parameters) - Wang et al. (2023) Fig. 2
    - Softmax attention (vs sigmoid modulation) - Wang et al. (2023) Eq. 3
    - Explicit center feature (better learning) - Wang et al. (2023) Sec. 3.2
    - Better scalability to large models - Wang et al. (2023) Table 1

    InternImage Performance (Wang et al. 2023, Tables 2-4):
    - ImageNet: 89.6% Top-1 (InternImage-H)
    - COCO: 65.4 box AP (InternImage-H)
    - ADE20K: 62.9 mIoU (InternImage-H)

    References:
        Wang et al. "InternImage: Exploring Large-Scale Vision Foundation Models
        with Deformable Convolutions" CVPR 2023
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_size=3, center_feature_scale=False):
        """
        Initialize DCNv3C2f with CSP architecture.

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
            DCNv3Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=(3, 3),
                e=1.0,
                kernel_size=kernel_size,
                center_feature_scale=center_feature_scale,
            )
            for _ in range(n)
        )

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
