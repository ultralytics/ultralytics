import inspect
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import autopad

try:  # torchvision is required for deformable convolutions
    from torchvision.ops import DeformConv2d
except ImportError:  # pragma: no cover - handled lazily at runtime
    DeformConv2d = None


class ChannelAttention(nn.Module):
    """Lightweight squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16, activation: Optional[nn.Module] = None) -> None:
        super().__init__()
        reduction = max(reduction, 1)
        hidden = max(channels // reduction, 1)
        act = activation if activation is not None else nn.SiLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = act
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.avg_pool(x)
        attn = self.fc1(attn)
        attn = self.act(attn)
        attn = self.fc2(attn)
        return x * self.sigmoid(attn)


class ConvAttnDeform(nn.Module):
    """Deformable convolution block with channel attention and activation."""

    default_act = nn.SiLU()  # default activation

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Union[Sequence[int], int] = 3,
        s: Union[Sequence[int], int] = 1,
        p=None,
        g: int = 1,
        d: Union[Sequence[int], int] = 1,
        act=True,
        deform_groups: int = 1,
        attn_reduction: Optional[int] = 16,
        zero_init_offset: bool = True,
    ) -> None:
        """Construct a deformable convolution followed by BatchNorm, attention, and activation."""
        super().__init__()
        if DeformConv2d is None:  # pragma: no cover
            raise ImportError(
                "ConvAttnDeform requires torchvision>=0.12 for torchvision.ops.DeformConv2d. "
                "Install torchvision or replace ConvAttnDeform with a standard Conv block."
            )
        if c1 % g:
            raise ValueError(f"ConvAttnDeform received c1={c1} incompatible with groups g={g}.")
        if deform_groups < 1:
            raise ValueError(f"deform_groups must be >= 1 (received {deform_groups}).")

        kernel = [k, k] if isinstance(k, int) else list(k)
        if len(kernel) != 2:
            raise ValueError(f"Kernel size must have one or two integers, received {k}.")
        kernel_size = tuple(kernel)

        stride = (s, s) if isinstance(s, int) else tuple(s)
        if isinstance(d, int):
            dilation = (d, d)
            d_for_pad = d
        else:
            dilation = tuple(d)
            d_for_pad = list(d)
        padding = autopad(kernel, p, d_for_pad)
        if isinstance(padding, list):
            padding = tuple(padding)

        deform_kwargs = dict(
            in_channels=c1,
            out_channels=c2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        deform_sig = inspect.signature(DeformConv2d).parameters
        if "groups" in deform_sig:
            deform_kwargs["groups"] = g
        if "deformable_groups" in deform_sig:
            deform_kwargs["deformable_groups"] = deform_groups
        self.deform_conv = DeformConv2d(**deform_kwargs)
        self.deform_groups = getattr(self.deform_conv, "deformable_groups", deform_kwargs.get("deformable_groups", 1))
        if deform_groups != self.deform_groups:
            if deform_groups != 1:
                raise ValueError(
                    f"ConvAttnDeform requested deform_groups={deform_groups} but torchvision only supports "
                    f"{self.deform_groups} in this build."
                )

        # Offset predictor: 2 offsets per kernel coordinate per deformable group.
        offset_channels = self.deform_groups * 2 * kernel_size[0] * kernel_size[1]
        self.offset_conv = nn.Conv2d(
            c1,
            offset_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        if zero_init_offset:
            nn.init.constant_(self.offset_conv.weight, 0.0)
            nn.init.constant_(self.offset_conv.bias, 0.0)

        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.attn = ChannelAttention(c2, reduction=attn_reduction) if attn_reduction else None
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deformable convolution followed by BN, optional attention, and activation."""
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.bn(x)
        if self.attn is not None:
            x = self.attn(x)
        return self.act(x)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Fused forward (BatchNorm folding is not supported; falls back to standard forward)."""
        return self.forward(x)
