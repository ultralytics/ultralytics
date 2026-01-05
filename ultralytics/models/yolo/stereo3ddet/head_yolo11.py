from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect


def _branch(in_ch: int, out_ch: int, hidden: int = 256) -> nn.Sequential:
    """Simple conv branch for dense per-location prediction."""
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, 1, 1),
        nn.BatchNorm2d(hidden),
        nn.SiLU(inplace=True),
        nn.Conv2d(hidden, out_ch, 1, 1, 0),
    )


class Stereo3DDetHeadYOLO11(nn.Module):
    """P3-only stereo3ddet head that reuses YOLO11 Detect assignment for positives.

    Outputs:
      - det: YOLO Detect head outputs (list with 1 feature map) for bbox/cls (DFL format).
      - aux branches: dense maps aligned with P3 grid, trained only on assigner positives.
    """

    def __init__(self, nc: int, ch: int = 256):
        super().__init__()
        self.nc = nc

        # YOLO11-style bbox+cls head (P3-only = single detection layer).
        self.detect = Detect(nc=nc, ch=(ch,))

        # Auxiliary stereo/3D branches (kept dense, masked by fg_mask in loss).
        self.aux = nn.ModuleDict(
            {
                "lr_distance": _branch(ch, 1),
                "right_width": _branch(ch, 1),
                "dimensions": _branch(ch, 3),
                "orientation": _branch(ch, 8),
                "vertices": _branch(ch, 8),
                "vertex_offset": _branch(ch, 8),
                "vertex_dist": _branch(ch, 4),
            }
        )

    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a list of feature maps from the neck; for P3-only we expect length==1.
        if not isinstance(x, list):
            x = [x]
        if len(x) != 1:
            # Keep it strict for P3-only first iteration to avoid silent shape bugs.
            raise ValueError(f"Stereo3DDetHeadYOLO11 expects 1 feature map (P3-only), got {len(x)}")

        feat = x[0]
        out: Dict[str, torch.Tensor] = {"det": self.detect([feat])}
        for k, m in self.aux.items():
            out[k] = m(feat)
        return out



