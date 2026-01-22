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

        # Standard aux branches (use fused or left features)
        self.aux = nn.ModuleDict(
            {
                "dimensions": _branch(ch, 3),
                "orientation": _branch(ch, 8),
                "vertices": _branch(ch, 8),
                "vertex_offset": _branch(ch, 8),
                "vertex_dist": _branch(ch, 4),
            }
        )
        
        # Stereo association branches - take CONCATENATED left+right (2*ch channels)
        # These need to see both views separately to learn correspondence
        self.stereo_aux = nn.ModuleDict(
            {
                "lr_distance": _branch(ch * 2, 1),  # Input is 2*ch
                "right_width": _branch(ch * 2, 1),  # Input is 2*ch
            }
        )

    def forward(self, x: List[torch.Tensor], x_stereo: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: List containing fused/left feature map [B, C, H, W]
            x_stereo: Concatenated [left, right] features [B, 2C, H, W] for stereo branches.
                     If None (e.g., during initialization), duplicates x[0] to create [B, 2C, H, W]
        """
        if not isinstance(x, list):
            x = [x]
        if len(x) != 1:
            raise ValueError(f"Stereo3DDetHeadYOLO11 expects 1 feature map (P3-only), got {len(x)}")

        feat = x[0]
        det_out = super().forward([feat])  # keep Detect semantics (training vs inference)

        out: Dict[str, torch.Tensor] = {"det": det_out}
        
        # 3D branches use fused/left features
        for k, m in self.aux.items():
            out[k] = m(feat)
        
        # Stereo branches use concatenated left+right
        # If x_stereo not provided (initialization), duplicate features
        if x_stereo is None:
            x_stereo = torch.cat([feat, feat], dim=1)  # [B, 2C, H, W]
        for k, m in self.stereo_aux.items():
            out[k] = m(x_stereo)
        
        return out



