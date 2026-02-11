from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


AUX_SPECS = {"lr_distance": 1, "dimensions": 3, "orientation": 2, "depth": 1}


def _branch(in_ch: int, out_ch: int, hidden: int = 256) -> nn.Sequential:
    """Simple conv branch for dense per-location prediction."""
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, 1, 1),
        nn.BatchNorm2d(hidden),
        nn.SiLU(inplace=True),
        nn.Conv2d(hidden, out_ch, 1, 1, 0),
    )


def _deep_branch(in_ch: int, out_ch: int, hidden: int = 64) -> nn.Sequential:
    """Deeper conv branch for depth-critical predictions (lr_distance, depth)."""
    return nn.Sequential(
        Conv(in_ch, hidden, 3),
        Conv(hidden, hidden, 3),
        Conv(hidden, hidden, 3),
        nn.Conv2d(hidden, out_ch, 1),
    )


class Stereo3DDetHeadYOLO11(Detect):
    """Multi-scale stereo 3D detection head (Pose-pattern).

    Receives P3/P4/P5 feature maps from FPN+PAN neck.  Per-scale aux branches
    predict stereo/3D quantities; outputs are flattened to [B, C, HW_total].

    Args:
        nc: Number of classes.
        ch: Tuple of per-scale input channels, e.g. (256, 512, 1024).
    """

    def __init__(self, nc: int = 3, reg_max: int = 16, ch: tuple = ()):
        if isinstance(reg_max, (list, tuple)):  # YAML [nc] — ch landed in reg_max slot
            ch, reg_max = reg_max, 16
        super().__init__(nc=nc, ch=ch)  # multi-scale Detect (builds with reg_max=16)

        # Override reg_max if requested (e.g. reg_max=1 disables DFL)
        if reg_max != self.reg_max:
            self.reg_max = reg_max
            self.no = nc + reg_max * 4
            c2 = max(16, ch[0] // 4, reg_max * 4)
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
            )
            from ultralytics.nn.modules.block import DFL
            self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        # Hidden size scales with model width (same pattern as Pose.cv4)
        hidden = max(ch[0] // 4, max(AUX_SPECS.values()))
        depth_hidden = max(ch[0] // 2, 64)  # wider hidden for depth-critical branches

        # Per-scale aux branches (like Pose.cv4)
        self.aux = nn.ModuleDict()
        for name, out_c in AUX_SPECS.items():
            if name in ("lr_distance", "depth"):
                self.aux[name] = nn.ModuleList(_deep_branch(x, out_c, depth_hidden) for x in ch)
            else:
                self.aux[name] = nn.ModuleList(_branch(x, out_c, hidden) for x in ch)

    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor] | tuple:
        """Forward pass: process aux BEFORE Detect.forward (Detect modifies x in-place)."""
        bs = x[0].shape[0]

        # Aux branches — collect before Detect mutates x
        aux_out: Dict[str, torch.Tensor] = {}
        for name, branches in self.aux.items():
            out_c = AUX_SPECS[name]
            aux_out[name] = torch.cat(
                [branches[i](x[i]).view(bs, out_c, -1) for i in range(self.nl)], -1
            )  # [B, C, HW_total]

        det = Detect.forward(self, x)  # standard multi-scale detect

        if self.training:
            return det, aux_out  # det is list of feat maps, aux_out is dict of [B,C,HW]
        return {"det": det, **aux_out}
