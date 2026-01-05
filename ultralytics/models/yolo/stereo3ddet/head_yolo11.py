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


class Stereo3DDetHeadYOLO11(Detect):
    """P3-only stereo3ddet head implemented as a Detect subclass (like Segment/Pose).

    This keeps YOLO's standard Detect behavior (stride/bias init, export behavior) while returning a dict:
      - det: Detect outputs (training: list[Tensor], inference: (y, x))
      - aux branches: dense maps aligned with P3 grid
    """

    def __init__(self, nc: int, ch: int = 256):
        super().__init__(nc=nc, ch=(ch,))  # P3-only Detect

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
        if not isinstance(x, list):
            x = [x]
        if len(x) != 1:
            raise ValueError(f"Stereo3DDetHeadYOLO11 expects 1 feature map (P3-only), got {len(x)}")

        feat = x[0]
        det_out = super().forward([feat])  # keep Detect semantics (training vs inference)

        out: Dict[str, torch.Tensor] = {"det": det_out}
        for k, m in self.aux.items():
            out[k] = m(feat)
        return out



