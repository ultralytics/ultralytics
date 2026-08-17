# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math

import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

from .orientation import ORIENT_CHANNELS

# 64 log-spaced depth bins is the measured optimum of an inverted-U curve, not a guess. On the
# drive-disjoint Chen split: 64 beats 16 by 1.82 Car AP3D@0.7 Mod, beats it at every GT range band, and cuts
# median depth error ~24% on the screening subset, 128 ties, and 192/256 are WORSE than 16 because very fine
# bins spread the DFL target too thin to optimise. The gain is a genuinely better depth branch (+28.3% with
# the cue fusion bypassed), not the fusion re-weighting a smaller bin spread also causes. Override per model
# with `training: depth_bins:` in a model YAML.
DEPTH_BINS = 64
DEPTH_MIN = 2.0
DEPTH_MAX = 80.0

# Orientation is a MultiBin head (ORIENT_CHANNELS = N_bins*(1 conf + 2 residual)); see orientation.py.
# lr_distance is 2ch (value + log-variance for the heteroscedastic depth NLL); proj_offset is the
# 2ch (Δu, Δv) projected-3D-center offset. Both are always on (the promoted Tier-1 localization path).
AUX_SPECS = {"lr_distance": 2, "proj_offset": 2, "dimensions": 3, "orientation": ORIENT_CHANNELS, "depth": DEPTH_BINS}


class DepthDFL(nn.Module):
    """DFL-style decode for depth bins: softmax → weighted sum → scale to log-depth range."""

    def __init__(self, n_bins: int = DEPTH_BINS, d_min: float = DEPTH_MIN, d_max: float = DEPTH_MAX):
        """Set up the depth bin grid spanning [d_min, d_max] in log space."""
        super().__init__()
        self.n_bins = n_bins
        log_min, log_max = math.log(d_min), math.log(d_max)
        self.register_buffer("bin_values", torch.linspace(log_min, log_max, n_bins))

    def _set_range(self, d_min: float, d_max: float) -> None:
        """Set the decodable depth range in meters."""
        if not 0 < d_min < d_max:
            raise ValueError(f"Depth range must satisfy 0 < depth_min < depth_max, got ({d_min}, {d_max})")
        self.bin_values = torch.linspace(
            math.log(d_min), math.log(d_max), self.n_bins, device=self.bin_values.device, dtype=self.bin_values.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode bin logits [B, n_bins, HW] → log-depth [B, 1, HW].

        Args:
            x: Bin logits [B, n_bins, HW].
        """
        weights = x.softmax(dim=1)  # [B, n_bins, HW]
        return (weights * self.bin_values.view(1, -1, 1)).sum(dim=1, keepdim=True)  # [B, 1, HW]


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

    Receives P3/P4/P5 feature maps from FPN+PAN neck, plus optional cost volume features that are fed to the depth
    branches (lr_distance, depth) at every scale, pooled to each scale's grid. This keeps P3/P4/P5 clean for 2D
    detection, avoiding 2D-3D task conflict, while keeping the right image reachable from every depth prediction.

    Args:
        nc: Number of classes.
        reg_max: DFL channels (forced to 1).
        end2end: End-to-end mode (forced to False).
        ch: Tuple of per-scale input channels, e.g. (256, 512, 1024) or (256, 512, 1024, 64) where the 4th element is
            cost volume channels.
    """

    def __init__(self, nc: int = 3, reg_max: int = 1, end2end: bool = False, ch: tuple = ()):
        """Build the 2D detection head plus the depth, dimension, orientation and disparity branches."""
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
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1)) for x in ch)
        self.dfl = nn.Identity()

        self.aux_specs = dict(AUX_SPECS)  # mutable copy
        self.depth_dfl = DepthDFL(DEPTH_BINS, DEPTH_MIN, DEPTH_MAX)
        # The decode grid owns the branch width — one logit per bin — so deriving it here keeps AUX_SPECS
        # from becoming a second, disagreeing source.
        self.aux_specs["depth"] = self.depth_dfl.n_bins

        # Hidden size scales with model width (same pattern as Pose.cv4)
        hidden = max(ch[0] // 4, max(self.aux_specs.values()))
        depth_hidden = max(ch[0] // 2, 64)  # wider hidden for depth-critical branches

        # Per-scale aux branches (like Pose.cv4)
        # Depth branches get the cost volume concatenated at EVERY scale → wider input
        self.aux = nn.ModuleDict()
        for name, out_c in self.aux_specs.items():
            if name in ("lr_distance", "depth"):
                self.aux[name] = nn.ModuleList(_deep_branch(x + self.cv_ch, out_c, depth_hidden) for x in ch)
            else:
                self.aux[name] = nn.ModuleList(_branch(x, out_c, hidden) for x in ch)

    def configure_depth(self, n_bins: int | None = None) -> None:
        """Retarget the depth path's bin count, resizing the branch to match.

        Called at model construction from the YAML `training:` block (see `Stereo3DDetModel`), before any
        weights matter, so only the branches' final 1x1 conv is rebuilt. Keeping this on the head means the
        bin count never has to be changed by mutating the module-level `AUX_SPECS`, which is shared state.

        Args:
            n_bins: New depth bin count. None keeps the current one.
        """
        d = self.depth_dfl
        n_bins = d.n_bins if n_bins is None else int(n_bins)
        if n_bins < 2:
            raise ValueError(f"depth_bins must be at least 2, got {n_bins}")
        log_min, log_max = float(d.bin_values[0]), float(d.bin_values[-1])
        self.depth_dfl = DepthDFL(n_bins, math.exp(log_min), math.exp(log_max))
        self.depth_dfl.to(d.bin_values.device)

        if "depth" not in self.aux_specs:  # pruned by depth_mode='lr_only'; nothing to resize
            return
        self.aux_specs["depth"] = n_bins
        for branch in self.aux["depth"]:
            last = branch[-1]
            branch[-1] = nn.Conv2d(last.in_channels, n_bins, 1).to(last.weight.device)

    def set_depth_mode(self, mode: str) -> None:
        """Prune aux branches to match depth_mode ('both', 'lr_only', 'depth_only').

        Filters this head's own `aux_specs` rather than rebuilding them from the module-level `AUX_SPECS`:
        the branches in `self.aux` were sized from the instance, so re-reading the global would silently
        replace a customized channel count (e.g. a non-default `depth_dfl.n_bins`) with the default and
        make `forward_head`'s `view(bs, out_c, -1)` reshape against the wrong width.
        """
        drop = {"lr_only": "depth", "depth_only": "lr_distance"}
        if mode not in {*drop, "both"}:
            raise ValueError(f"Unknown depth_mode: {mode!r}. Expected 'both', 'lr_only', or 'depth_only'.")
        self.aux_specs.pop(drop.get(mode), None)
        for name in list(self.aux.keys()):
            if name not in self.aux_specs:
                del self.aux[name]

    @property
    def one2many(self):
        """Returns the one-to-many head components including aux branches."""
        return {"box_head": self.cv2, "cls_head": self.cv3, "aux_branches": self.aux}

    def forward_head(
        self, x: list[torch.Tensor], box_head=None, cls_head=None, aux_branches=None
    ) -> dict[str, torch.Tensor]:
        """Forward pass: compute detection + aux predictions.

        If cost volume is present (4th element in x), it is separated and concatenated into the depth
        branches (lr_distance, depth) at EVERY scale, pooled to each scale's grid. It is the only
        right-image path into those branches, so a scale that misses it predicts depth monocularly no
        matter how it is trained (guarded by test_stereo_reaches_every_scale). 2D detection (box/cls)
        uses clean P3/P4/P5 features.
        """
        # Separate cost volume from feature maps, then pool it onto each scale's grid. The channel axis
        # encodes disparity in input pixels, which is grid-independent, so spatial pooling is meaningful.
        cost_vols, cv_disparity = None, None
        if self.cv_ch > 0 and len(x) > self.nl:
            cost_vol = x[self.nl]
            x = list(x[: self.nl])
            cv_disparity = cost_vol[:, :1]  # StereoCostVolume soft-argmin readout, [0,1] over its grid
            cost_vols = [
                cost_vol
                if f.shape[-2:] == cost_vol.shape[-2:]
                else nn.functional.adaptive_avg_pool2d(cost_vol, f.shape[-2:])
                for f in x
            ]

        # 2D detection on clean features
        preds = super().forward_head(x, box_head, cls_head)  # {boxes, scores, feats}
        if cv_disparity is not None:
            preds["cv_disparity"] = cv_disparity  # [B, 1, H/8, W/8], for direct disparity supervision

        if aux_branches is not None:
            bs = x[0].shape[0]
            for name, branches in aux_branches.items():
                out_c = self.aux_specs[name]
                feats = []
                for i in range(self.nl):
                    feat = x[i]
                    if cost_vols is not None and name in ("lr_distance", "depth"):
                        feat = torch.cat([feat, cost_vols[i]], dim=1)
                    feats.append(branches[i](feat).view(bs, out_c, -1))
                preds[name] = torch.cat(feats, -1)  # [B, C, HW_total]

        if "lr_distance" in preds:
            lr = preds["lr_distance"]
            preds["lr_distance"] = lr[:, :1]  # value
            preds["lr_logvar"] = lr[:, 1:2]  # log-variance (heteroscedastic depth NLL)

        # Decode depth bins → scalar log-depth (keep raw logits for loss/export)
        if "depth" in preds:
            depth_logits = preds["depth"]  # branch output [B, n_bins, HW] — one logit per decode bin
            preds["depth"] = self.depth_dfl(depth_logits)  # decoded [B, 1, HW]
            preds["depth_bins"] = depth_logits  # raw logits: DFLoss / ONNX export / eval-time DFL variance
            # Ship the grid the logits are defined on. _set_range() retargets it per dataset, so a decoder
            # that rebuilds it from DEPTH_MIN/DEPTH_MAX silently reads the bins on the wrong axis.
            preds["depth_bin_values"] = self.depth_dfl.bin_values

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
                aux_tensors.append(preds["depth_bins"])  # [B, n_bins, anchors] raw logits
            if aux_tensors:
                y = torch.cat([y, *aux_tensors], dim=1)  # [B, nc+4+74, anchors] (74 = 1+3+6+64)
            return y

        return (y, preds)
