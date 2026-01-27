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
    
    Args:
        nc: Number of classes.
        ch: P3 feature channels (for detection and 3D aux branches).
    """

    def __init__(self, nc: int, ch: int = 256):
        super().__init__(nc=nc, ch=(ch,))  # P3-only Detect

        # Standard aux branches (use fused P3 features)
        self.aux = nn.ModuleDict(
            {
                "dimensions": _branch(ch, 3),
                "orientation": _branch(ch, 8),
                "vertices": _branch(ch, 8),
                "vertex_offset": _branch(ch, 8),
                "vertex_dist": _branch(ch, 4),
            }
        )
        
        # Stereo branches are lazily initialized based on actual input channels
        self.stereo_aux = None
        self._stereo_ch = None

    def _get_stereo_aux(self, stereo_ch: int, device) -> nn.ModuleDict:
        """Lazily create stereo branches based on actual input channels."""
        if self.stereo_aux is None or self._stereo_ch != stereo_ch:
            self._stereo_ch = stereo_ch
            self.stereo_aux = nn.ModuleDict(
                {
                    "lr_distance": _branch(stereo_ch, 1),
                    "right_width": _branch(stereo_ch, 1),
                }
            ).to(device)
        return self.stereo_aux

    def forward(self, x: List[torch.Tensor], x_stereo: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: List containing fused/left feature map [B, C, H, W]
            x_stereo: Concatenated [left, right] features [B, stereo_ch, H, W] for stereo branches.
                     If None (e.g., during initialization), stereo outputs are zeros.
        """
        if not isinstance(x, list):
            x = [x]
        if len(x) != 1:
            raise ValueError(f"Stereo3DDetHeadYOLO11 expects 1 feature map (P3-only), got {len(x)}")

        feat = x[0]
        det_out = super().forward([feat])  # keep Detect semantics (training vs inference)

        out: Dict[str, torch.Tensor] = {"det": det_out}
        
        # 3D branches use fused P3 features
        for k, m in self.aux.items():
            out[k] = m(feat)
        
        # Stereo branches - lazily initialized based on actual input channels
        if x_stereo is not None:
            stereo_aux = self._get_stereo_aux(x_stereo.shape[1], x_stereo.device)
            for k, m in stereo_aux.items():
                out[k] = m(x_stereo)
        else:
            # Initialization pass - return zeros
            B, _, H, W = feat.shape
            out["lr_distance"] = torch.zeros(B, 1, H, W, device=feat.device, dtype=feat.dtype)
            out["right_width"] = torch.zeros(B, 1, H, W, device=feat.device, dtype=feat.dtype)
        
        return out



