from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

DEPTH_BINS = 16
DEPTH_MIN = 2.0
DEPTH_MAX = 80.0

# Cost-volume disparity constants — MUST mirror the StereoCostVolume layer args in the YAML
# (single source of truth for SoftArgmaxDisparity bin centers).
COST_MAX_DISP = 48
COST_NUM_BINS = 24

# 'cost_disp' is decoded from the raw cost-volume bins (not a conv branch), so it is NOT in
# AUX_SPECS (which drives self.aux conv branches). It is appended to preds in forward_head.
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


class SoftArgmaxDisparity(nn.Module):
    """Differentiable expected disparity over fixed correlation-bin centers.

    Given the raw [B, D, *] correlation maps from StereoCostVolume, applies a softmax
    over the D disparity bins and returns the weighted sum with the bin centers, i.e.
    the expected feature-pixel disparity at the cost-volume stride.
    """

    def __init__(self, disparities: list[int]):
        super().__init__()
        self.register_buffer("bin_centers", torch.tensor(disparities, dtype=torch.float32))

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        """Decode correlation bins [B, D, *] → expected disparity [B, 1, *]."""
        weights = bins.softmax(dim=1)  # [B, D, *]
        shape = [1, -1] + [1] * (bins.dim() - 2)
        return (weights * self.bin_centers.view(*shape)).sum(dim=1, keepdim=True)  # [B, 1, *]


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

        # Soft-argmax disparity decoder over the cost-volume's raw correlation bins.
        # Bin centers mirror StereoCostVolume.disparities = linspace(0, max_disp, num_bins).round().
        cost_disparities = torch.linspace(0, COST_MAX_DISP, COST_NUM_BINS).round().int().tolist()
        self.cost_disp_decoder = SoftArgmaxDisparity(cost_disparities)

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
        cost_bins = None
        if self.cv_ch > 0 and len(x) > self.nl:
            cost_vol = x[self.nl]
            # With return_bins enabled the cost volume is a (refined, raw_bins) tuple.
            if isinstance(cost_vol, (tuple, list)):
                cost_vol, cost_bins = cost_vol
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

            # Cost-volume soft-argmax disparity → aux 'cost_disp' map [B, 1, HW_total].
            # Aligns 1:1 with the P3 portion of HW_total. Stereo disparity is decoded only at the
            # P3 grid; P4/P5 anchors fall back to the per-anchor lr_distance regression so larger,
            # nearby objects detected at coarse scales are not regressed to a single mean value.
            if cost_bins is not None and "lr_distance" in preds:
                p3_h, p3_w = x[0].shape[2], x[0].shape[3]
                hw_total = preds["lr_distance"].shape[2]
                input_w = p3_w * 8  # P3 stride is 8
                # Interpolate raw bins to the P3 grid, then soft-argmax → feature-px disparity.
                bins_p3 = F.interpolate(cost_bins, size=(p3_h, p3_w), mode="bilinear", align_corners=False)
                disp_featpx = self.cost_disp_decoder(bins_p3.flatten(2))  # [B, 1, p3_h*p3_w]
                # Convert feature-pixel disparity (stride 8) to the SAME log-normalized space as
                # the lr_distance GT: disparity_pixels = disp_featpx * 8; normalized = / input_w.
                disp_log = torch.log((disp_featpx * 8.0 / input_w).clamp(min=1e-6))  # [B, 1, p3_n]
                p3_n = p3_h * p3_w
                cost_disp = disp_log.new_zeros(bs, 1, hw_total)
                cost_disp[:, :, :p3_n] = disp_log
                if hw_total > p3_n:
                    # Non-P3 anchors fall back to the per-anchor lr_distance prediction (no
                    # regression vs. the prior behaviour where lr_distance covered all scales).
                    cost_disp[:, :, p3_n:] = preds["lr_distance"][:, :, p3_n:]
                preds["cost_disp"] = cost_disp

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
            # Concat aux outputs: lr_distance, cost_disp, dimensions, orientation, depth_bins
            aux_tensors = []
            for name in ("lr_distance", "cost_disp", "dimensions", "orientation"):
                if name in preds:
                    aux_tensors.append(preds[name])  # [B, C, anchors]
            if "depth_bins" in preds:
                aux_tensors.append(preds["depth_bins"])  # [B, 16, anchors] raw logits
            if aux_tensors:
                y = torch.cat([y, *aux_tensors], dim=1)  # [B, 7+22, anchors]
            return y

        return (y, preds)
