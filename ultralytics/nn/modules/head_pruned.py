# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Structured-pruned YOLOv8 detection head compatible with official Ultralytics KD."""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import DFL
from .conv import Conv
from .head import Detect


class DetectPruned(Detect):
    """YOLOv8 Detect head whose internal branch widths come from pruning masks.

    Subclassing the official :class:`Detect` is intentional: the official
    ``DistillationModel`` locates the three neck features through the Detect
    head's ``f`` indices. The forward/loss output contract therefore remains
    the official Ultralytics contract; only the branch construction differs.
    """

    legacy = True

    def __init__(
        self,
        cv2x0_outs,
        cv2x1_outs,
        cv3x0_outs,
        cv3x1_outs,
        nc: int = 80,
        ch: tuple = (),
    ):
        """Initialize a structured-pruned detection head."""
        nn.Module.__init__(self)
        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.end2end = False

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, cv2x0_out, 3),
                Conv(cv2x0_out, cv2x1_out, 3),
                nn.Conv2d(cv2x1_out, 4 * self.reg_max, 1),
            )
            for x, cv2x0_out, cv2x1_out in zip(ch, cv2x0_outs, cv2x1_outs)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, cv3x0_out, 3),
                Conv(cv3x0_out, cv3x1_out, 3),
                nn.Conv2d(cv3x1_out, self.nc, 1),
            )
            for x, cv3x0_out, cv3x1_out in zip(ch, cv3x0_outs, cv3x1_outs)
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
