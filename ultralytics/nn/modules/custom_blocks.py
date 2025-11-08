from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import autopad


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


class ConvAttnLite(nn.Module):
    """CPU-friendly convolutional block with depthwise filtering and channel attention."""

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
        expand_ratio: float = 1.5,
        attn_reduction: Optional[int] = 16,
    ) -> None:
        """Construct an inverted bottleneck-style conv block with attention."""
        super().__init__()
        if g != 1:
            raise ValueError("ConvAttnLite does not support grouped convolutions.")

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

        hidden = max(int(c1 * expand_ratio), 1)
        self.expand = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, 1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Conv2d(hidden, c2, 1, 1, bias=False)

        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.attn = ChannelAttention(c2, reduction=attn_reduction) if attn_reduction else None
        self.act = nn.SiLU(inplace=True) if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.use_shortcut = c1 == c2 and stride == (1, 1) and dilation == (1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run lightweight conv-attention block."""
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        x = self.bn(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.use_shortcut:
            x = x + residual
        return self.act(x)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Fused forward (BatchNorm folding is not supported; falls back to standard forward)."""
        return self.forward(x)
