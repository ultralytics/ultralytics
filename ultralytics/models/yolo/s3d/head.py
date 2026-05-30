from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

DEPTH_BINS = 16
DEPTH_MIN = 2.0
DEPTH_MAX = 80.0

AUX_SPECS = {"lr_distance": 1, "dimensions": 3, "orientation": 2, "depth": DEPTH_BINS}


class DepthDFL(nn.Module):
    """DFL-style decode for depth bins: softmax → weighted sum → scale to log-depth range."""

    def __init__(self, n_bins: int = DEPTH_BINS, d_min: float = DEPTH_MIN, d_max: float = DEPTH_MAX):
        super().__init__()
        self.n_bins = n_bins
        log_min, log_max = math.log(d_min), math.log(d_max)
        self.register_buffer("bin_values", torch.linspace(log_min, log_max, n_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode bin logits [B, n_bins, HW] → log-depth [B, 1, HW]."""
        weights = x.softmax(dim=1)  # [B, n_bins, HW]
        return (weights * self.bin_values.view(1, -1, 1)).sum(dim=1, keepdim=True)  # [B, 1, HW]


def get_aux_specs(depth_mode: str = "both") -> dict[str, int]:
    """Return filtered AUX_SPECS based on depth_mode.

    Args:
        depth_mode: "both" (default), "lr_only", or "depth_only".
    """
    specs = dict(AUX_SPECS)
    if depth_mode == "lr_only":
        specs.pop("depth", None)
    elif depth_mode == "depth_only":
        specs.pop("lr_distance", None)
    elif depth_mode != "both":
        raise ValueError(f"Unknown depth_mode: {depth_mode!r}. Expected 'both', 'lr_only', or 'depth_only'.")
    return specs


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


class Stereo3DDetHead(Detect):
    """Multi-scale stereo 3D detection head (Pose-pattern).

    Receives P3/P4/P5 feature maps from FPN+PAN neck, plus optional cost volume
    features that are fed ONLY to depth branches (lr_distance, depth) at P3 scale.
    This keeps P3/P4/P5 clean for 2D detection, avoiding 2D-3D task conflict.

    Args:
        nc: Number of classes.
        reg_max: DFL channels (forced to 1).
        end2end: End-to-end mode (forced to False).
        ch: Tuple of per-scale input channels, e.g. (256, 512, 1024) or
            (256, 512, 1024, 64) where the 4th element is cost volume channels.
    """

    def __init__(self, nc: int = 3, reg_max: int = 1, end2end: bool = False, ch: tuple = ()):
        if isinstance(reg_max, (list, tuple)):  # YAML [nc] — ch landed in reg_max slot
            ch, reg_max = reg_max, 1

        # Detect cost volume channels (4th element beyond P3/P4/P5)
        ch = list(ch)
        self.cv_ch = ch.pop() if len(ch) > 3 else 0
        ch = tuple(ch)

        super().__init__(nc=nc, reg_max=1, end2end=False, ch=ch)  # Force reg_max=1, end2end=False

        # Force reg_max=1 (no DFL) — stereo 3D detection doesn't benefit from DFL
        self.reg_max = 1
        self.no = nc + 4  # 4 direct bbox offsets, no distribution
        c2 = max(16, ch[0] // 4, 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1)) for x in ch
        )
        self.dfl = nn.Identity()

        self.aux_specs = dict(AUX_SPECS)  # mutable copy
        self.depth_dfl = DepthDFL(DEPTH_BINS, DEPTH_MIN, DEPTH_MAX)

        # Hidden size scales with model width (same pattern as Pose.cv4)
        hidden = max(ch[0] // 4, max(self.aux_specs.values()))
        depth_hidden = max(ch[0] // 2, 64)  # wider hidden for depth-critical branches

        # Per-scale aux branches (like Pose.cv4)
        # Depth branches at P3 (scale 0) get cost volume concat → wider input
        self.aux = nn.ModuleDict()
        for name, out_c in self.aux_specs.items():
            if name in ("lr_distance", "depth"):
                branches = []
                for i, x in enumerate(ch):
                    in_ch = x + self.cv_ch if i == 0 else x  # P3 gets cost vol
                    branches.append(_deep_branch(in_ch, out_c, depth_hidden))
                self.aux[name] = nn.ModuleList(branches)
            else:
                self.aux[name] = nn.ModuleList(_branch(x, out_c, hidden) for x in ch)

    def set_depth_mode(self, mode: str) -> None:
        """Prune aux branches to match depth_mode ('both', 'lr_only', 'depth_only')."""
        self.aux_specs = get_aux_specs(mode)
        for name in list(self.aux.keys()):
            if name not in self.aux_specs:
                del self.aux[name]

    @property
    def one2many(self):
        """Returns the one-to-many head components including aux branches."""
        return dict(box_head=self.cv2, cls_head=self.cv3, aux_branches=self.aux)

    def forward_head(
        self, x: list[torch.Tensor], box_head=None, cls_head=None, aux_branches=None
    ) -> dict[str, torch.Tensor]:
        """Forward pass: compute detection + aux predictions.

        If cost volume is present (4th element in x), it is separated and
        concatenated with P3 ONLY for depth branches (lr_distance, depth).
        2D detection (box/cls) uses clean P3/P4/P5 features.
        """
        # Separate cost volume from feature maps
        cost_vol = None
        if self.cv_ch > 0 and len(x) > self.nl:
            cost_vol = x[self.nl]
            x = list(x[: self.nl])

        # 2D detection on clean features
        preds = super().forward_head(x, box_head, cls_head)  # {boxes, scores, feats}

        if aux_branches is not None:
            bs = x[0].shape[0]
            for name, branches in aux_branches.items():
                out_c = self.aux_specs[name]
                feats = []
                for i in range(self.nl):
                    feat = x[i]
                    # Concat cost volume with P3 only for depth branches
                    if cost_vol is not None and i == 0 and name in ("lr_distance", "depth"):
                        feat = torch.cat([feat, cost_vol], dim=1)
                    feats.append(branches[i](feat).view(bs, out_c, -1))
                preds[name] = torch.cat(feats, -1)  # [B, C, HW_total]

        # Decode depth bins → scalar log-depth (keep raw logits for loss/export)
        if "depth" in preds:
            depth_logits = preds["depth"]  # raw logits [B, 16, HW]
            preds["depth"] = self.depth_dfl(depth_logits)  # decoded [B, 1, HW]
            if self.training or self.export:
                preds["depth_bins"] = depth_logits  # raw logits for DFLoss / ONNX export

        return preds

    def forward(self, x):
        """Forward with aux output concatenation in export mode."""
        preds = self.forward_head(x, **self.one2many)
        if self.training:
            return preds
        y = self._inference(preds)  # [B, nc+4, anchors]

        if self.export:
            # Concat aux outputs: lr_distance, dimensions, orientation, depth_bins
            aux_tensors = []
            for name in ("lr_distance", "dimensions", "orientation"):
                if name in preds:
                    aux_tensors.append(preds[name])  # [B, C, anchors]
            if "depth_bins" in preds:
                aux_tensors.append(preds["depth_bins"])  # [B, 16, anchors] raw logits
            if aux_tensors:
                y = torch.cat([y, *aux_tensors], dim=1)  # [B, 7+22, anchors]
            return y

        return (y, preds)
