# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""OSNet backbone for person re-identification.

Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019.
arXiv:1905.00953 — reference implementation: https://github.com/KaiyangZhou/deep-person-reid

Bundled here as a single ``OSNetBackbone`` module that the YAML parser can
instantiate in one line, e.g.::

    backbone:
      - [-1, 1, OSNetBackbone, ["x1_0"]]

Output channel count matches the third-stage width of the selected variant
(512 for ``x1_0``, 256 for ``x0_5``). ~2.2M params for x1_0.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ("OSNetBackbone", "OSBlock")


class _ConvBNReLU(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int = 1, p: int = 0, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Conv1x1(_ConvBNReLU):
    def __init__(self, c1: int, c2: int):
        super().__init__(c1, c2, 1, 1, 0)


class _Conv1x1Linear(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))


class _LightConv3x3(nn.Module):
    """1x1 pointwise + 3x3 depthwise (OSNet's factorized conv)."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv2(self.conv1(x))))


class _LightConvStream(nn.Module):
    """A chain of ``depth`` LightConv3x3 blocks — each stream in OSBlock uses a different depth."""

    def __init__(self, c1: int, c2: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = [_LightConv3x3(c1, c2)]
        for _ in range(depth - 1):
            layers.append(_LightConv3x3(c2, c2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _ChannelGate(nn.Module):
    """Unified Aggregation Gate — per-channel dynamic attention over multi-scale streams."""

    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        hidden = max(c // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, c, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w


class OSBlock(nn.Module):
    """Omni-Scale bottleneck: 4 parallel LightConv streams of increasing depth,
    aggregated through a shared Channel Gate, then residual.
    """

    def __init__(self, c1: int, c2: int, reduction: int = 4):
        super().__init__()
        mid = c2 // reduction
        self.conv1 = _Conv1x1(c1, mid)
        self.stream_a = _LightConvStream(mid, mid, 1)
        self.stream_b = _LightConvStream(mid, mid, 2)
        self.stream_c = _LightConvStream(mid, mid, 3)
        self.stream_d = _LightConvStream(mid, mid, 4)
        self.gate = _ChannelGate(mid)
        self.conv3 = _Conv1x1Linear(mid, c2)
        self.shortcut = None if c1 == c2 else _Conv1x1Linear(c1, c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        y = self.conv1(x)
        y = self.gate(self.stream_a(y)) + self.gate(self.stream_b(y)) + self.gate(self.stream_c(y)) + self.gate(self.stream_d(y))
        y = self.conv3(y)
        return self.act(y + identity)


class OSNetBackbone(nn.Module):
    """OSNet backbone (stem + 3 OSBlock stages + 2 downsamplings between stages).

    Output shape is ``(B, stage_channels[-1], H/16, W/16)`` after the stem's 4×
    downsample plus two further 2× downsamples between stages.
    """

    _VARIANTS: dict[str, tuple[int, tuple[int, int, int], tuple[int, int, int]]] = {
        #     stem_ch    stage channels       blocks per stage
        "x1_0": (64, (256, 384, 512), (2, 2, 2)),
        "x0_75": (48, (192, 288, 384), (2, 2, 2)),
        "x0_5": (32, (128, 192, 256), (2, 2, 2)),
        "x0_25": (16, (64, 96, 128), (2, 2, 2)),
    }

    def __init__(self, variant: str = "x1_0"):
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"unknown OSNet variant {variant!r}; choose from {list(self._VARIANTS)}")
        stem_ch, stage_ch, blocks = self._VARIANTS[variant]

        self.stem = nn.Sequential(
            _ConvBNReLU(3, stem_ch, k=7, s=2, p=3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        stages: list[nn.Module] = []
        c = stem_ch
        for i, (out_c, n) in enumerate(zip(stage_ch, blocks)):
            for j in range(n):
                stages.append(OSBlock(c if j == 0 else out_c, out_c))
            c = out_c
            if i < len(stage_ch) - 1:
                # transition: 1×1 conv to keep channels, then average-pool 2× down
                stages.append(_Conv1x1(c, c))
                stages.append(nn.AvgPool2d(2, 2))

        self.stages = nn.Sequential(*stages)
        self.out_channels = stage_ch[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))
