# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: bbox-mask renderer, heatmap bias fusion, optional SegBranch.

Soft-hint fusion: a 1-channel mask is turned into a bounded per-pixel bias added
(broadcast over channels) to PAN features before the Detect head. PAN feature
addition keeps the Detect head unmodified, lets reg and cls both see the bias
(empirical question — see spec §2), and is bounded vs the previous multiplicative
amplifier that forced detections.

See docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = (
    "BboxMaskRenderer",
    "HeatmapBiasFusion",
    "HeatmapFiLMFusion",
    "SegBranch",
    "binary_seg_loss",
    "BackboneMemoryBank",
)


class BboxMaskRenderer(nn.Module):
    """Render normalized YOLO-format bboxes into a 1xHxW mask.

    Two modes:
      - "rect":  hard rectangle (inside bbox = 1, outside = 0)
      - "gauss": per-bbox 2D Gaussian centered at bbox center,
                 sigma_x = w * sigma_factor, sigma_y = h * sigma_factor;
                 multiple bboxes in the same image are combined with max.

    Output spatial size is fixed at construction (default 80 to match P3).
    """

    def __init__(self, mask_size: int = 80, mode: str = "rect", sigma_factor: float = 0.25):
        super().__init__()
        assert mode in ("rect", "gauss"), f"mode must be 'rect' or 'gauss', got {mode!r}"
        self.mask_size = int(mask_size)
        self.mode = mode
        self.sigma_factor = float(sigma_factor)
        ys, xs = torch.meshgrid(
            torch.arange(self.mask_size, dtype=torch.float32),
            torch.arange(self.mask_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", xs + 0.5, persistent=False)
        self.register_buffer("grid_y", ys + 0.5, persistent=False)

    def forward(self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        H = self.mask_size
        device = self.grid_x.device
        dtype = self.grid_x.dtype
        mask = torch.zeros(batch_size, 1, H, H, device=device, dtype=dtype)
        if bboxes.numel() == 0:
            return mask

        bboxes = bboxes.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)

        cx = bboxes[:, 0] * H
        cy = bboxes[:, 1] * H
        w = bboxes[:, 2] * H
        h = bboxes[:, 3] * H

        if self.mode == "rect":
            x1 = (cx - w / 2)[:, None, None]
            x2 = (cx + w / 2)[:, None, None]
            y1 = (cy - h / 2)[:, None, None]
            y2 = (cy + h / 2)[:, None, None]
            inside = (
                (self.grid_x[None] >= x1)
                & (self.grid_x[None] < x2)
                & (self.grid_y[None] >= y1)
                & (self.grid_y[None] < y2)
            ).to(dtype)
        else:  # gauss
            sigma_x = (w * self.sigma_factor).clamp(min=0.5)
            sigma_y = (h * self.sigma_factor).clamp(min=0.5)
            dx = self.grid_x[None] - cx[:, None, None]
            dy = self.grid_y[None] - cy[:, None, None]
            inside = torch.exp(
                -(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2))
            )

        for b in range(batch_size):
            sel = batch_idx == b
            if sel.any():
                mask[b, 0] = inside[sel].max(dim=0).values
        return mask

    def extra_repr(self) -> str:
        return f"mask_size={self.mask_size}, mode={self.mode!r}, sigma_factor={self.sigma_factor}"


class HeatmapBiasFusion(nn.Module):
    """Soft-hint fusion: 1-ch mask -> bounded per-pixel bias broadcast onto PAN features.

    Output shape ``(B, 1, H, W)`` — the caller broadcasts (adds) it to a PAN feature
    of shape ``(B, C, H, W)``. The conv stack is SHARED across PAN scales; the caller
    is responsible for resizing the mask to each scale before calling forward.

    Per-scale magnitude is controlled by ``beta[i]``, initialized to zero so training
    starts as pure passthrough (vanilla YOLO). Without a hard cap, beta can in
    principle grow large; that is intentional — the detection loss decides how much
    to lean on the heatmap.

    Output per pixel is in ``[-beta_i, +beta_i]`` via tanh.
    """

    def __init__(self, num_scales: int = 3, c_mid: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, c_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_mid, 1, 3, padding=1),
        )
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def forward(self, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return bias (B, 1, H, W) for the given PAN scale.

        Args:
            mask: (B, 1, H, W) already resized to the target PAN scale.
            scale_idx: index into ``self.beta``.

        Returns:
            Bias tensor (B, 1, H, W) in ``[-beta_i, +beta_i]``.
        """
        return self.beta[scale_idx] * torch.tanh(self.conv(mask))


class HeatmapFiLMFusion(nn.Module):
    """Residual grouped-FiLM fusion: the prior modulates a projected copy of each PAN feature.

    Richer than ``HeatmapBiasFusion`` (1-channel additive bias broadcast over all channels). Each
    PAN feature is projected into a modulation space, split into ``num_groups`` channel groups, and
    each group is scaled by a prior-derived spatial map (grouped FiLM: ``V * (1 + gamma)``, or the
    bounded gate ``V * (1 + tanh(gamma))`` in ``(0, 2)`` when ``gamma_bound``). The modulated feature
    is projected back and added as a LayerScale-gated residual, so the main detection path is
    preserved and the branch starts as a near-identity (``alpha`` init ~0).

    The prior conv (mask -> per-group ``gamma``) is shared across PAN scales since its input is
    always the 1-channel mask; ``proj_in`` / ``proj_out`` are per-scale because PAN channel counts
    differ. ``forward`` returns the residual increment ``alpha * dP`` of shape ``(B, C, H, W)``;
    the caller adds it to the PAN feature and applies mask-dropout, mirroring the bias path.
    """

    def __init__(
        self,
        pan_channels: list[int],
        num_groups: int = 16,
        group_dim: int = 16,
        prior_mid: int = 32,
        alpha_init: float = 1e-4,
        gamma_bound: bool = False,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.group_dim = int(group_dim)
        self.c_mod = self.num_groups * self.group_dim
        self.gamma_bound = bool(gamma_bound)
        # Prior -> per-group spatial scale gamma. Shared across scales (input is the 1-ch mask).
        self.prior_conv = nn.Sequential(
            nn.Conv2d(1, prior_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(prior_mid, self.num_groups, 3, padding=1),
        )
        # Per-scale projections into / out of the modulation space (PAN channels differ per scale).
        self.proj_in = nn.ModuleList(Conv(c, self.c_mod, k=1) for c in pan_channels)
        self.proj_out = nn.ModuleList(nn.Conv2d(self.c_mod, c, 1) for c in pan_channels)
        # LayerScale: per-scale, per-channel residual gate, init ~0 so the branch starts inert.
        self.alpha = nn.ParameterList(
            nn.Parameter(torch.full((c,), float(alpha_init))) for c in pan_channels
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return the residual increment ``(B, C, H, W)`` to add to ``feat``.

        Args:
            feat: PAN feature ``(B, C, H, W)`` at scale ``scale_idx``.
            mask: prior ``(B, 1, H, W)`` already resized to ``feat``'s spatial size.
            scale_idx: index into the per-scale projections / LayerScale.

        Returns:
            Residual increment ``(B, C, H, W)``; ~0 at init via the LayerScale gate.
        """
        b, _, h, w = feat.shape
        gamma = self.prior_conv(mask)  # (B, num_groups, H, W)
        # Bounded gate 1+tanh(gamma) in (0, 2) (multiplicative suppress/amplify, training-stable)
        # vs raw 1+gamma (unbounded). gamma_bound selects between them.
        scale = (1.0 + torch.tanh(gamma)) if self.gamma_bound else (1.0 + gamma)
        v = self.proj_in[scale_idx](feat)  # (B, c_mod, H, W)
        v = v.view(b, self.num_groups, self.group_dim, h, w)
        v = v * scale.unsqueeze(2)  # grouped FiLM scale
        v = v.reshape(b, self.c_mod, h, w)
        d = self.proj_out[scale_idx](v)  # (B, C, H, W)
        return self.alpha[scale_idx].view(1, -1, 1, 1) * d


class SegBranch(nn.Module):
    """Lightweight semantic-segmentation head that predicts a 1-channel anomaly heatmap.

    Consumes the P3 and P4 PAN features and emits per-pixel logits at P3 resolution
    (e.g. 80x80 for 640 input). A P4 auxiliary head provides deep supervision during
    training.
    """

    def __init__(self, ch: tuple, nc: int = 1, c_mid: int | None = None):
        super().__init__()
        self.nc = nc
        c_mid = ch[0] if c_mid is None else c_mid
        self.classifier = nn.Sequential(Conv(ch[0], c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None

    def forward(self, x: list[torch.Tensor]):
        logits = self.classifier(x[0])
        if self.training and self.aux_head is not None:
            return logits, self.aux_head(x[1])
        return logits


def binary_seg_loss(
    logits: torch.Tensor, target: torch.Tensor, aux_logits: torch.Tensor | None = None, aux_weight: float = 0.4
) -> torch.Tensor:
    """BCE + soft-Dice loss for a single-channel anomaly heatmap."""
    if target.shape[2:] != logits.shape[2:]:
        target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
    target = target.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    inter = (prob * target).sum(dim=(1, 2, 3))
    card = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1.0) / (card + 1.0)).mean()
    loss = bce + dice
    if aux_logits is not None:
        aux_t = target
        if aux_t.shape[2:] != aux_logits.shape[2:]:
            aux_t = F.interpolate(target, size=aux_logits.shape[2:], mode="nearest")
        loss = loss + aux_weight * F.binary_cross_entropy_with_logits(aux_logits, aux_t)
    return loss


class BackboneMemoryBank(nn.Module):
    """Memory-bank anomaly heatmap from backbone features (v1 ADMBHead inference logic).

    Stores L2-normalised normal-image backbone features. During inference, scores
    each spatial position via Noisy-OR cosine similarity against the bank, producing a
    (B, 1, H, W) heatmap that feeds into HeatmapBiasFusion as a prior.

    Two modes controlled by ``update``:
      - ``True`` (build): ``forward()`` returns zeros — bank not yet frozen.
      - ``False`` (inference): ``forward()`` returns anomaly scores in [0, 1].

    Calibration modes (``calibrate`` parameter):

      - ``"auto"`` (default): sample from the full accumulated bank and calibrate β
        so that the 90th-percentile normal feature scores ``calibration_target_score``.
        Coreset is skipped in this mode.
      - ``"compactness"``: first subsample via greedy k-centre coreset, then measure
        local neighbour density (compactness) on the coreset to calibrate β.
        Scores naturally map to [0, 1] because compactness reflects the true normal
        manifold tightness.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        K: int = 5,
        accumulate_thresh: float = 0.4,
        score_filter_kernel: int = 1,
        max_bank_size: int | None = None,
        auto_temperature: bool = True,
        calibration_target_score: float = 0.2,
        calibrate: str = "auto",
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.K = int(K)
        self.accumulate_thresh = float(accumulate_thresh)
        self.score_filter_kernel = int(score_filter_kernel)
        self.max_bank_size = max_bank_size
        self.auto_temperature = bool(auto_temperature)
        self.calibration_target_score = float(calibration_target_score)
        self.calibrate = calibrate
        self._calibrated = False
        self.register_buffer("memory_bank", torch.empty(0, 0), persistent=True)
        self.feature_dim: int | None = None
        self.update = True
        self._bb_layer_indices: list[int] = []

    @property
    def built(self) -> bool:
        return not self.update

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_bank(self, features: torch.Tensor) -> None:
        """Direct-set the memory bank from pre-extracted L2-normalised features [M, C]."""
        if features.numel() == 0:
            return
        self.feature_dim = features.shape[1]
        self.memory_bank = F.normalize(features.to(self.memory_bank.device), p=2, dim=1)

    def freeze_memory_bank(self) -> None:
        """Compress, optionally calibrate, and freeze the bank for inference.

        Two calibration strategies (set via ``self.calibrate``):

        * ``"auto"`` — sample from the full bank and calibrate β (skips coreset).
        * ``"compactness"`` — greedy k-centre coreset first, then measure local
          neighbour density on the coreset to calibrate β.  Because the coreset
          represents the normal manifold compactness, scores naturally map to [0, 1].
        """
        mem = self._effective_bank()
        if self.calibrate == "compactness":
            # Coreset first, then calibrate from its spatial compactness.
            if self.max_bank_size is not None and mem.shape[0] > self.max_bank_size:
                self.memory_bank = self._coreset_subsample(mem, self.max_bank_size)
            mem = self._effective_bank()
            if mem.shape[0] > 0:
                self._calibrate_compactness(mem)
        elif self.auto_temperature and mem.shape[0] > 0:
            self._calibrate_temperature(mem)
        if self.max_bank_size is not None and mem.shape[0] > self.max_bank_size:
            if not self.auto_temperature and self.calibrate != "compactness":
                self.memory_bank = self._coreset_subsample(mem, self.max_bank_size)
        self.update = False

    def reset_memory_bank(self) -> None:
        """Clear the bank and return to build mode."""
        self.memory_bank = torch.empty(0, 0, device=self.memory_bank.device)
        self.feature_dim = None
        self._calibrated = False
        self.update = True

    def accumulate_features(self, feat_dict: dict[int, torch.Tensor]) -> None:
        """Extract and accumulate backbone features into the memory bank (build phase).

        Fused backbone features are L2-normalised per spatial position and concatenated
        onto the existing bank. Call ``freeze_memory_bank()`` to compress and freeze.
        """
        if not feat_dict:
            return
        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        C, H, W = fused.shape[1], fused.shape[2], fused.shape[3]
        if self.feature_dim is None:
            self.feature_dim = C
        flat = fused.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        normed = F.normalize(flat, p=2, dim=1)
        cur = self._effective_bank()
        if cur.shape[0] > 0:
            self.memory_bank = torch.cat([cur, normed.to(cur.device)], dim=0)
        else:
            self.memory_bank = normed

    def forward(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Produce (B, 1, H, W) anomaly heatmap from backbone features.

        During build mode or when the bank is empty, returns zeros.
        """
        b, device = self._resolve_batch_size(feat_dict)
        h = feat_dict[list(feat_dict.keys())[0]].shape[2] if feat_dict else 80
        w = feat_dict[list(feat_dict.keys())[0]].shape[3] if feat_dict else 80
        if self.update:
            return torch.zeros(b, 1, h, w, device=device)
        mem = self._effective_bank()
        if mem.shape[0] == 0:
            return torch.zeros(b, 1, h, w, device=device)
        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        if fused.shape[1] != self.feature_dim:
            return torch.zeros(b, 1, fused.shape[2], fused.shape[3], device=device)
        bh, bw = fused.shape[2], fused.shape[3]
        flat = fused.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        scores = self._anomaly_scores(flat, mem)  # [B*H*W]
        return scores.view(b, 1, bh, bw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_batch_size(feat_dict: dict[int, torch.Tensor]) -> tuple[int, torch.device]:
        for v in feat_dict.values():
            return v.shape[0], v.device
        return 1, torch.device("cpu")

    def _effective_bank(self) -> torch.Tensor:
        """Return real memory-bank entries excluding zero-padding placeholders."""
        mem = self.memory_bank
        if self.feature_dim is None or mem.numel() == 0 or mem.shape[0] == 0:
            return mem[:0]
        valid = mem.norm(dim=1) > 0
        return mem[valid]

    def _build_fused_feature(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Gather backbone features at configured layer indices, upsample to P3 scale, concat."""
        indices = self._bb_layer_indices
        if not indices:
            return feat_dict[list(feat_dict.keys())[0]]
        feats = [feat_dict[i] for i in indices if i in feat_dict]
        if not feats:
            return feat_dict[list(feat_dict.keys())[0]]
        target_size = feats[0].shape[-2:]
        aligned = [
            F.interpolate(f, size=target_size, mode="nearest") if f.shape[-2:] != target_size else f
            for f in feats
        ]
        return torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]

    def estimate_temperature(self) -> float:
        """Estimate current auto-calibrated β from the bank WITHOUT modifying state."""
        import math

        mem = self._effective_bank()
        if mem.shape[0] < 2:
            return self.temperature
        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            n_sample = min(512, mem.shape[0])
            idx = torch.randperm(mem.shape[0], device=mem.device)[:n_sample]
            sample = mem[idx]
            sim = sample @ mem.t()
            topk_sim = sim.topk(k=k, dim=1).values
            mean_topk = topk_sim.mean(dim=1)
            s_max = mean_topk.max().clamp(0.0, 1.0 - 1e-4).item()
            beta = -math.log(1.0 - self.calibration_target_score) / max(1.0 - s_max, 1e-6)
            beta = max(0.1, min(20.0, beta))
        return beta

    def _calibrate_temperature(self, mem: torch.Tensor) -> None:
        """Auto-calibrate β from the current memory bank.

        Solves for β such that the maximum mean top-K cosine similarity against
        the bank scores ``calibration_target_score``.

        Formula: β = −ln(1 − target) / (1 − s_max)
        """
        old_temp = self.temperature
        beta = self.estimate_temperature()
        self.temperature = beta
        self._calibrated = True
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "BackboneMemoryBank: auto-calibrated temperature %.3f → %.3f  (target=%.2f)",
            old_temp, beta, self.calibration_target_score,
        )

    def _calibrate_compactness(self, mem: torch.Tensor) -> None:
        """Calibrate β from coreset spatial compactness.

        Measures local neighbour density on the coreset: for each centre, compute
        mean cosine similarity to its top-K neighbours within the coreset.
        Global compactness = mean of these local densities.
        β = −ln(1 − target) / (1 − compactness).

        Because compactness reflects the true normal-manifold tightness (coreset
        removes redundant features), the resulting β maps anomaly scores naturally
        to [0, 1].
        """
        import math

        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            n_sample = min(512, mem.shape[0])
            idx = torch.randperm(mem.shape[0], device=mem.device)[:n_sample]
            sample = mem[idx]
            # cosine similarity of each sampled centre vs all coreset centres
            sim = sample @ mem.t()  # [n_sample, M]
            # exclude self-match by masking diagonal
            diag_mask = torch.eye(n_sample, mem.shape[0], device=mem.device, dtype=torch.bool)
            sim.masked_fill_(diag_mask, -1.0)
            topk_sim = sim.topk(k=k, dim=1).values  # [n_sample, k]
            local_density = topk_sim.mean(dim=1)     # [n_sample] per-centre compactness
            compactness = local_density.mean().clamp(0.0, 1.0 - 1e-4).item()
        old_temp = self.temperature
        beta = -math.log(1.0 - self.calibration_target_score) / max(1.0 - compactness, 1e-6)
        beta = max(0.1, min(20.0, beta))
        self.temperature = beta
        self._calibrated = True
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "BackboneMemoryBank: compactness-calibrated temp %.3f → %.3f  "
            "(compactness=%.4f, target=%.2f)",
            old_temp, beta, compactness, self.calibration_target_score,
        )

    def _anomaly_scores(self, features: torch.Tensor, mem: torch.Tensor | None = None) -> torch.Tensor:
        """Noisy-OR cosine anomaly scores ∈ [0, 1] for every spatial position.

        Args:
            features: [N, C] query features.
            mem: [M, C] memory bank. If None, uses ``self._effective_bank()``.

        Returns:
            [N] tensor of anomaly scores.
        """
        if mem is None:
            mem = self._effective_bank()
        if mem.numel() == 0 or mem.shape[0] == 0:
            return torch.full((features.shape[0],), 0.5, device=features.device)
        q = F.normalize(features.view(-1, self.feature_dim), p=2, dim=1)
        k = min(self.K, mem.shape[0])
        sim = q @ mem.t()  # [N, M]
        sim = torch.exp(-self.temperature * (1 - sim))  # psi(x) = exp(-beta*(1-cos))
        topk_sim = sim.topk(k=k, dim=1).values  # [N, k]
        log_prob = torch.log((1 - topk_sim).clamp(min=1e-8)).mean(dim=1)
        return (torch.exp(log_prob)*1.0) .clamp(0, 1)

    @staticmethod
    def _coreset_subsample(mem: torch.Tensor, max_size: int) -> torch.Tensor:
        """Greedy k-center coreset on L2-normalised features using cosine distance.

        Complexity: O(max_size × M). Features must be L2-normalised.
        """
        from ultralytics.utils import TQDM

        M = mem.shape[0]
        if M <= max_size:
            return mem
        device = mem.device
        dist = torch.full((M,), float("inf"), device=device, dtype=torch.float32)
        selected: list[int] = []
        mean = mem.mean(dim=0)
        mean = mean / mean.norm().clamp(min=1e-8)
        seed = int((mem @ mean).argmax().item())
        selected.append(seed)
        for _ in TQDM(range(max_size - 1), desc="Coreset subsample", leave=False):
            centre = mem[selected[-1]].unsqueeze(0)
            cos_sim = (mem @ centre.t()).squeeze(1)
            new_dist = (1.0 - cos_sim).clamp(min=0.0)
            dist = torch.minimum(dist, new_dist)
            selected.append(int(dist.argmax().item()))
        return mem[torch.tensor(selected, device=device)]


if __name__ == "__main__":
    # Smoke test: render, fusion init=0 passthrough, fusion with learned beta bounded,
    # broadcast onto a C-channel PAN feature.
    B, H = 4, 80
    renderer = BboxMaskRenderer(mask_size=H, mode="rect")
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.75, 0.75, 0.30, 0.10],
            [0.50, 0.50, 0.10, 0.10],
        ]
    )
    batch_idx = torch.tensor([0, 0, 2], dtype=torch.long)
    mask = renderer(bboxes, batch_idx, B)
    assert mask.shape == (B, 1, H, H)
    assert mask[1].sum().item() == 0.0  # image with no bboxes

    fusion = HeatmapBiasFusion(num_scales=3)
    # beta init=0 -> bias is exactly zero for every scale
    for s in range(3):
        bias = fusion(mask, s)
        assert bias.shape == (B, 1, H, H)
        assert bias.abs().max().item() == 0.0, f"scale {s} bias not zero at init"
    print("HeatmapBiasFusion init=0 passthrough OK.")

    # With beta set non-zero, output is bounded in [-beta, +beta]
    with torch.no_grad():
        fusion.beta.fill_(1.5)
    bias = fusion(mask, 0)
    assert bias.abs().max().item() <= 1.5 + 1e-6, "bias exceeded beta after tanh"
    print(f"HeatmapBiasFusion beta=1.5 bounded OK (max abs = {bias.abs().max().item():.4f}).")

    # Broadcast onto a C-channel PAN feature: (B, 1, H, W) + (B, C, H, W) -> (B, C, H, W).
    p = torch.randn(B, 256, H, H)
    p_fused = p + fusion(mask, 0)
    assert p_fused.shape == p.shape
    print(f"Broadcast OK: P (B,256,H,W) + bias (B,1,H,W) -> {tuple(p_fused.shape)}.")

    # Resize-per-scale smoke
    mask_p4 = F.interpolate(mask, size=(40, 40), mode="bilinear", align_corners=False)
    mask_p5 = F.interpolate(mask, size=(20, 20), mode="bilinear", align_corners=False)
    assert fusion(mask_p4, 1).shape == (B, 1, 40, 40)
    assert fusion(mask_p5, 2).shape == (B, 1, 20, 20)
    print("HeatmapBiasFusion multi-scale OK.")

    print("\n--- HeatmapFiLMFusion ---")
    pan_ch = [256, 512, 512]  # yolo26m PAN channel counts (differ per scale)
    film = HeatmapFiLMFusion(pan_channels=pan_ch, num_groups=16, group_dim=16)
    feats = [torch.randn(B, c, s, s) for c, s in zip(pan_ch, (80, 40, 20))]
    masks = [
        mask,
        F.interpolate(mask, size=(40, 40), mode="bilinear", align_corners=False),
        F.interpolate(mask, size=(20, 20), mode="bilinear", align_corners=False),
    ]
    # alpha init ~0 -> increment is negligible (near-vanilla start).
    for i, (f, msk) in enumerate(zip(feats, masks)):
        delta = film(f, msk, i)
        assert delta.shape == f.shape, f"scale {i}: {tuple(delta.shape)} != {tuple(f.shape)}"
        assert delta.abs().max().item() < 1e-2, f"scale {i} increment not ~0 at init"
    print("HeatmapFiLMFusion init near-identity + per-scale shapes OK.")

    # With alpha grown, the increment is non-trivial and still shaped like the PAN feature.
    with torch.no_grad():
        for a in film.alpha:
            a.fill_(1.0)
    delta = film(feats[0], masks[0], 0)
    assert delta.shape == feats[0].shape and delta.abs().max().item() > 0.0
    print(f"HeatmapFiLMFusion alpha=1 active OK (max abs = {delta.abs().max().item():.4f}).")

    # gamma_bound: bounded 1+tanh(gamma) gate; still near-identity at init via alpha.
    film_b = HeatmapFiLMFusion(pan_channels=pan_ch, gamma_bound=True)
    db = film_b(feats[0], masks[0], 0)
    assert db.shape == feats[0].shape and db.abs().max().item() < 1e-2, "gamma_bound init not ~0"
    print("HeatmapFiLMFusion gamma_bound init near-identity OK.")

    print("\n--- BackboneMemoryBank ---")
    mb = BackboneMemoryBank(temperature=3.0, K=5)
    assert not mb.built, "fresh bank should not be built (update=True)"
    feat_dict = {4: torch.randn(2, 512, 80, 80), 6: torch.randn(2, 512, 40, 40), 10: torch.randn(2, 1024, 20, 20)}
    mb._bb_layer_indices = [4, 6, 10]
    # Build mode: returns zeros
    out = mb(feat_dict)
    assert out.shape == (2, 1, 80, 80), f"expected (2,1,80,80) got {tuple(out.shape)}"
    assert out.abs().max().item() == 0.0, "build mode should return zeros"
    # Load bank and freeze
    bank_features = torch.randn(100, 512)
    mb.load_bank(bank_features)
    mb.freeze_memory_bank()
    assert mb.built, "should be built after freeze"
    # Inference mode: non-zero heatmap
    out = mb(feat_dict)
    assert out.shape == (2, 1, 80, 80)
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0, "scores should be in [0,1]"
    # Reset
    mb.reset_memory_bank()
    assert not mb.built
    # Coreset smoke
    big_bank = torch.randn(200, 512)
    mb.load_bank(big_bank)
    mb.freeze_memory_bank()  # max_bank_size=None → no compression
    mb2 = BackboneMemoryBank(max_bank_size=50)
    mb2._bb_layer_indices = [4]
    mb2.load_bank(big_bank)
    mb2.freeze_memory_bank()
    assert mb2.memory_bank.shape[0] == 50, f"coreset should keep 50, got {mb2.memory_bank.shape[0]}"
    print("BackboneMemoryBank smoke OK.")

    print("\nAll smoke tests passed.")
