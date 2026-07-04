# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: heatmap fusion and memory-bank scorer.

Soft-hint fusion: a 1-channel memory-bank heatmap is turned into a bounded per-pixel bias
added (broadcast over channels) to PAN features before the Detect head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "BboxMaskRenderer",
    "HeatmapBiasFusion",
    "BackboneMemoryBank",
    "HeatmapProcessor",
)


class BboxMaskRenderer(nn.Module):
    """Render normalized YOLO-format bboxes into a 1xHxW mask.

    Two modes:
      - "rect":  hard rectangle (inside bbox = 1, outside = 0)
      - "gauss": per-bbox 2D Gaussian centered at bbox center,
                 sigma_x = w * sigma_factor, sigma_y = h * sigma_factor;
                 multiple bboxes in the same image are combined with max.

    ``sigma_factor`` may be a scalar (fixed width) or a ``[lo, hi]`` range. With a range, each
    bbox draws its own factor ~ U(lo, hi) while training (prior-width augmentation toward the
    diffuse inference heatmap) and uses the midpoint deterministically at eval.

    Output spatial size is fixed at construction (default 80 to match P3).
    """

    def __init__(self, mask_size: int = 80, mode: str = "rect", sigma_factor: float | list = 0.25):
        super().__init__()
        assert mode in ("rect", "gauss"), f"mode must be 'rect' or 'gauss', got {mode!r}"
        self.mask_size = int(mask_size)
        self.mode = mode
        if isinstance(sigma_factor, (list, tuple)):
            self.sigma_lo, self.sigma_hi = float(sigma_factor[0]), float(sigma_factor[1])
        else:
            self.sigma_lo = self.sigma_hi = float(sigma_factor)
        self.sigma_factor = self.sigma_lo  # back-compat scalar handle (== lo)
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
            # Per-bbox sigma factor: random in [lo, hi] while training, midpoint at eval.
            if self.training and self.sigma_hi > self.sigma_lo:
                sf = torch.empty_like(w).uniform_(self.sigma_lo, self.sigma_hi)
            else:
                sf = 0.5 * (self.sigma_lo + self.sigma_hi)
            sigma_x = (w * sf).clamp(min=0.5)
            sigma_y = (h * sf).clamp(min=0.5)
            dx = self.grid_x[None] - cx[:, None, None]
            dy = self.grid_y[None] - cy[:, None, None]
            inside = torch.exp(-(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2)))

        for b in range(batch_size):
            sel = batch_idx == b
            if sel.any():
                mask[b, 0] = inside[sel].max(dim=0).values
        return mask

    def render_per_instance(
        self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int, sigma_factor: float | None = None
    ) -> list[torch.Tensor]:
        """Render each bbox as its own gauss mask, grouped per image (NO per-image max merge).

        Training-only target for query-to-instance matching. Unlike ``forward``, the per-bbox
        masks are kept separate so each can be Hungarian-matched to a query attention map.

        Args:
            bboxes: (N, 4) normalized YOLO ``[cx, cy, w, h]``.
            batch_idx: (N,) image index per bbox.
            batch_size: number of images B.
            sigma_factor: fixed gauss width factor; defaults to ``self.sigma_lo`` (independent of
                the fusion-prior sigma so the query GT stays sharp).

        Returns:
            list of length B, each ``(N_b, H, H)`` in [0, 1] (empty ``(0, H, H)`` for images with
            no bboxes).
        """
        H = self.mask_size
        device = self.grid_x.device
        dtype = self.grid_x.dtype
        empty = torch.zeros(0, H, H, device=device, dtype=dtype)
        if bboxes.numel() == 0:
            return [empty for _ in range(batch_size)]
        bboxes = bboxes.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)
        cx = bboxes[:, 0] * H
        cy = bboxes[:, 1] * H
        w = bboxes[:, 2] * H
        h = bboxes[:, 3] * H
        sf = float(self.sigma_lo if sigma_factor is None else sigma_factor)
        sigma_x = (w * sf).clamp(min=0.5)
        sigma_y = (h * sf).clamp(min=0.5)
        dx = self.grid_x[None] - cx[:, None, None]
        dy = self.grid_y[None] - cy[:, None, None]
        inst = torch.exp(
            -(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2))
        )  # (N, H, H)
        return [inst[batch_idx == b] for b in range(batch_size)]

    def extra_repr(self) -> str:
        return f"mask_size={self.mask_size}, mode={self.mode!r}, sigma_factor=[{self.sigma_lo}, {self.sigma_hi}]"


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
        dtype = next(self.conv.parameters()).dtype
        return self.beta[scale_idx] * torch.tanh(self.conv(mask.to(dtype)))


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
        temperature: float = 5.0,
        K: int = 5,
        max_bank_size: int | None = 10000,
        calibration_target_score: float = 0.4,
        calibration_target_quantile: float = 0.95,
        hmap_stretch_strength: float = 0.0,
        holdout_max: int = 5000,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.K = int(K)
        self.max_bank_size = max_bank_size
        self.calibration_target_score = float(calibration_target_score)
        self.calibration_target_quantile = float(calibration_target_quantile)
        self.hmap_stretch_strength = float(hmap_stretch_strength)
        self.holdout_max = int(holdout_max)
        self.register_buffer("memory_bank", torch.empty(0, 0), persistent=True)
        self.feature_dim: int | None = None
        self.update = True
        self._bb_layer_indices: list[int] = []
        self._bank_chunks: list[torch.Tensor] = []
        self._compactness: float | None = None  # normal-manifold tightness from coreset
        self._threshold: float | None = None  # sigmoid threshold in cosine space
        self.score_chunk_elems = 1 << 27  # max elements per similarity slice in _anomaly_scores

    def load_bank(self, features: torch.Tensor) -> None:
        """Direct-set the memory bank from pre-extracted L2-normalised features [M, C]."""
        if features.numel() == 0:
            return
        if not torch.isfinite(features).all():
            raise ValueError(f"BackboneMemoryBank.load_bank: features contain NaN/Inf ({features.shape})")
        self.feature_dim = features.shape[1]
        self.memory_bank = F.normalize(features.to(self.memory_bank.device), p=2, dim=1).float()
        self.update = False
        self._calibrate_compactness(self._effective_bank())

    def freeze_memory_bank(self) -> None:
        """Coreset-compress, calibrate via compactness + holdout, then freeze the bank."""
        if self._bank_chunks:
            self.memory_bank = torch.cat(self._bank_chunks, dim=0)
            self._bank_chunks.clear()
        mem = self._effective_bank()

        # Coreset subsample, collect holdout (features not selected into coreset)
        holdout = None
        if self.max_bank_size is not None and mem.shape[0] > self.max_bank_size:
            self.memory_bank, coreset_idx = self._coreset_subsample(mem, self.max_bank_size, return_indices=True)
            holdout_mask = torch.ones(mem.shape[0], dtype=torch.bool, device=mem.device)
            holdout_mask[coreset_idx] = False
            holdout = mem[holdout_mask]
        mem = self._effective_bank()

        if mem.shape[0] > 0:
            self._calibrate_compactness(mem)
            if holdout is not None and holdout.shape[0] > 0:
                if holdout.shape[0] > self.holdout_max:
                    idx = torch.randperm(holdout.shape[0], device=holdout.device)[: self.holdout_max]
                    holdout = holdout[idx]
                self._calibrate_threshold_from_holdout(holdout, mem)

        self.update = False

    def reset_memory_bank(self) -> None:
        """Clear the bank and return to build mode."""
        self.memory_bank = torch.empty(0, 0, device=self.memory_bank.device)
        self.feature_dim = None
        self._compactness = None
        self._threshold = None
        self.update = True
        self._bank_chunks: list[torch.Tensor] = []  # defer cat until freeze

    def accumulate_features(self, feat_dict: dict[int, torch.Tensor]) -> None:
        """Extract and accumulate backbone features into the memory bank (build phase).

        Fused backbone features are L2-normalised per spatial position and appended
        to a chunk list; the full bank is materialised once in ``freeze_memory_bank``
        to avoid O(N²) reallocation from repeated ``torch.cat``.
        """
        if not feat_dict:
            return
        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        C = fused.shape[1]
        if self.feature_dim is None:
            self.feature_dim = C
        flat = fused.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        normed = F.normalize(flat, p=2, dim=1).float()
        self._bank_chunks.append(normed)

    def forward(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Produce (B, 1, H, W) anomaly heatmap from backbone features."""
        first = next(iter(feat_dict.values()))
        b, device, h, w = first.shape[0], first.device, first.shape[2], first.shape[3]
        mem = self._effective_bank()
        if self.update or mem.shape[0] == 0:
            return torch.zeros(b, 1, h, w, device=device)

        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        flat = fused.permute(0, 2, 3, 1).reshape(-1, fused.shape[1])
        scores = self._anomaly_scores(flat, mem)  # [B*H*W]
        hmap = scores.view(b, 1, fused.shape[2], fused.shape[3])
        s = self.hmap_stretch_strength
        if s:
            hmap = (hmap + s * hmap * hmap).clamp(0, 1)
        return hmap

    def _effective_bank(self) -> torch.Tensor:
        """Return real memory-bank entries excluding zero-padding placeholders."""
        if self.update and self._bank_chunks:
            return torch.cat(self._bank_chunks, dim=0)
        mem = self.memory_bank
        if self.feature_dim is None or mem.shape[0] == 0:
            return mem[:0]
        return mem[mem.norm(dim=1) > 0]

    def _build_fused_feature(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Gather backbone features at configured layer indices and concat."""
        indices = self._bb_layer_indices
        if not indices:
            return feat_dict[list(feat_dict.keys())[0]]
        feats = [feat_dict[i] for i in indices if i in feat_dict]
        if not feats:
            return feat_dict[list(feat_dict.keys())[0]]
        target_size = feats[0].shape[-2:]
        aligned = [
            F.interpolate(f, size=target_size, mode="nearest") if f.shape[-2:] != target_size else f for f in feats
        ]
        return torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]

    def _measure_compactness(self, mem: torch.Tensor) -> float:
        """Compute compactness from the bank without touching ``self.temperature``.

        Used for lazy restore after state_dict load and by ``_calibrate_compactness``.
        """
        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            n_sample = min(512, mem.shape[0])
            idx = torch.randperm(mem.shape[0], device=mem.device)[:n_sample]
            sample = mem[idx]
            sim = sample @ mem.t()  # [n_sample, M]
            sim[torch.arange(n_sample, device=mem.device), idx] = -1.0
            topk_sim = sim.topk(k=k, dim=1).values
            local_density = topk_sim.mean(dim=1)
            return local_density.mean().clamp(0.0, 1.0 - 1e-4).item()

    def _calibrate_compactness(self, mem: torch.Tensor) -> None:
        """Measure compactness and calibrate sigmoid threshold in cosine space.

        Compactness = mean local cosine density on the coreset.  The sigmoid
        operates directly on cosine similarity so that the dynamic range is
        never compressed by the bank spread:

            psi = sigmoid(β × (cos − threshold_cos))

        threshold_cos = compactness − logit(1−target)/β.

        A normal query has cos ≈ compactness → psi ≈ 1−target → normal
        anomaly score ≈ target.  β is the user-controlled ``temperature``.
        """
        import math

        compactness = self._measure_compactness(mem)
        self._compactness = compactness
        beta = self.temperature
        t = self.calibration_target_score
        logit = math.log(max((1.0 - t) / max(t, 1e-6), 1e-6))
        self._threshold = compactness - logit / max(beta, 0.1)

    def _score_with_threshold(
        self, features: torch.Tensor, mem: torch.Tensor, threshold_cos: float, beta: float | None = None
    ) -> torch.Tensor:
        """Compute anomaly scores with specific threshold & β (stateless, for calibration)."""
        if beta is None:
            beta = self.temperature
        q = F.normalize(features.view(-1, self.feature_dim), p=2, dim=1).to(mem.dtype)
        k = min(self.K, mem.shape[0])
        n, m = q.shape[0], mem.shape[0]
        chunk = max(1, int(getattr(self, "score_chunk_elems", 1 << 27)) // max(m, 1))
        out = []
        for i in range(0, n, chunk):
            cos = q[i : i + chunk] @ mem.t()
            psi = torch.sigmoid(beta * (cos - threshold_cos))
            topk = psi.topk(k=k, dim=1).values
            log_prob = torch.log((1.0 - topk).clamp(min=1e-8)).mean(dim=1)
            out.append(torch.exp(log_prob))
        return torch.cat(out).clamp(0, 1).to(features.dtype)

    def _calibrate_threshold_from_holdout(self, holdout: torch.Tensor, mem: torch.Tensor) -> None:
        """Calibrate β & threshold so p95 of hold-out normal scores ≈ target.

        The user β is a sensitivity floor — it is never lowered.  For each
        candidate β, binary-searches ``threshold_cos`` to put p95 at
        ``calibration_target_score``, then picks the β that gives the tightest
        normal-score distribution (smallest p95−p5).
        """
        import math
        import logging

        logger = logging.getLogger(__name__)

        n_holdout = holdout.shape[0]
        target = self.calibration_target_score
        target_q = self.calibration_target_quantile  # e.g. 0.95 → 95% of normal scores ≤ target
        c = self._compactness
        beta0 = self.temperature

        # β candidates: user β as floor, log-spaced upward
        beta_candidates = [beta0]
        n_extra = 8
        for i in range(1, n_extra + 1):
            b = beta0 * (10 ** (i / n_extra))
            if b > 100:
                break
            beta_candidates.append(round(b, 4))

        # Pre-compute cosine matrix and top-K once (sigmoid is monotonic,
        # so the top-K indices of cos are identical to top-K of psi).
        q_feats = F.normalize(holdout.view(-1, self.feature_dim), p=2, dim=1)
        cos_mat = q_feats @ mem.t()  # [N_holdout, M]
        k = min(self.K, mem.shape[0])
        topk_cos = cos_mat.topk(k=k, dim=1).values  # [N_holdout, k] — only these matter

        # z-score for the target quantile (√2 · erf⁻¹(2q−1))
        z_q = math.sqrt(2) * torch.erfinv(torch.tensor(2.0 * target_q - 1.0)).item()

        def _scores_for(beta, thresh):
            """Compute anomaly scores from pre-computed top-K cos values."""
            psi = torch.sigmoid(beta * (topk_cos - thresh))
            log_prob = torch.log((1.0 - psi).clamp(min=1e-8)).mean(dim=1)
            return torch.exp(log_prob).clamp(0, 1)

        def _tail_stat(scores):
            """Gaussian tail: μ + z·σ."""
            return scores.mean().item() + z_q * scores.std().item()

        def _spread(scores):
            """Score spread: standard deviation (smaller = tighter normal distribution)."""
            return scores.std().item()

        def _find_thresh(beta):
            """Binary-search threshold so tail-stat ≈ target."""
            half_range = 3.0 / max(beta, 0.1)
            lo, hi = c - half_range, c + half_range
            s_lo = _tail_stat(_scores_for(beta, lo))
            s_hi = _tail_stat(_scores_for(beta, hi))
            for _ in range(5):
                if s_lo > target and lo > c - 5.0:
                    lo -= half_range
                    s_lo = _tail_stat(_scores_for(beta, lo))
                if s_hi < target and hi < c + 5.0:
                    hi += half_range
                    s_hi = _tail_stat(_scores_for(beta, hi))
            if not (s_lo <= target <= s_hi):
                return None, float("inf"), float("inf")
            for _ in range(20):
                mid = (lo + hi) / 2
                sm = _tail_stat(_scores_for(beta, mid))
                if sm > target:
                    hi = mid
                else:
                    lo = mid
            thresh = (lo + hi) / 2
            scores = _scores_for(beta, thresh)
            achieved = _tail_stat(scores)
            spread = _spread(scores)
            return thresh, achieved, spread

        best_beta, best_thresh, best_achieved, best_spread = beta0, c, float("inf"), float("inf")
        for beta in beta_candidates:
            thresh, achieved, spread = _find_thresh(beta)
            if thresh is not None and spread < best_spread:
                best_spread, best_achieved, best_beta, best_thresh = spread, achieved, beta, thresh

        old_beta = self.temperature
        self.temperature = best_beta
        self._threshold = best_thresh
        logger.debug(
            "BackboneMemoryBank: gauss calibration  n=%d  q=%.2f  z=%.4f  "
            "β %.3f→%.3f  thresh(formula)=%.4f→(holdout)=%.4f  "
            "target=%.2f→achieved=%.4f  spread=%.4f",
            n_holdout,
            target_q,
            z_q,
            old_beta,
            best_beta,
            c - math.log(max((1.0 - target) / max(target, 1e-6), 1e-6)) / max(old_beta, 0.1),
            best_thresh,
            target,
            best_achieved,
            best_spread,
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
        if mem.shape[0] == 0 or self._threshold is None:
            return torch.full((features.shape[0],), 0.5, device=features.device)
        q = F.normalize(features.view(-1, self.feature_dim), p=2, dim=1).to(mem.dtype)
        k = min(self.K, mem.shape[0])
        n, m = q.shape[0], mem.shape[0]
        chunk = max(1, int(self.score_chunk_elems) // max(m, 1))
        beta, thresh = self.temperature, self._threshold
        out = []
        for i in range(0, n, chunk):
            cos = q[i : i + chunk] @ mem.t()  # [chunk, M] cosine similarities
            if self.training:
                cos = cos.masked_fill(cos > 0.999, float("-inf"))
            psi = torch.sigmoid(beta * (cos - thresh))  # cos-space sigmoid
            topk = psi.topk(k=k, dim=1).values  # [chunk, k]
            log_prob = torch.log((1.0 - topk).clamp(min=1e-8)).mean(dim=1)
            out.append(torch.exp(log_prob))
        return torch.cat(out).clamp(0, 1).to(features.dtype)

    @staticmethod
    def _coreset_subsample(mem: torch.Tensor, max_size: int, return_indices: bool = False):
        """Greedy k-center coreset on L2-normalised features using cosine distance.

        Complexity: O(max_size × M). Features must be L2-normalised.

        Batched greedy: selects ``batch_size`` farthest points per iteration and
        computes distances against all of them in one GEMM call, amortizing the
        large bank read over multiple centres and trading a small amount of
        greediness for a large wall-clock speedup on memory-bandwidth-limited
        devices (MPS / CPU).

        Args:
            mem: [M, C] L2-normalised feature bank.
            max_size: Target coreset size.
            return_indices: If True, also return the indices of selected rows.

        Returns:
            Coreset tensor, or (coreset, indices) if ``return_indices``.
        """
        from ultralytics.utils import TQDM

        M = mem.shape[0]
        if M <= max_size:
            idx = torch.arange(M, device=mem.device)
            return (mem, idx) if return_indices else mem
        device = mem.device
        BATCH = 64  # centres per GEMM call — amortizes bank reads
        dist = torch.full((M,), float("inf"), device=device, dtype=torch.float32)
        selected: list[int] = []
        mean = mem.mean(dim=0)
        mean = mean / mean.norm().clamp(min=1e-8)
        seed = int((mem @ mean).argmax().item())
        selected.append(seed)
        # seed distance so the first topk isn't random (all-inf)
        centre = mem[seed].unsqueeze(0)
        seed_cos = (mem @ centre.t()).squeeze(1)
        dist = (1.0 - seed_cos).clamp(min=0.0)
        n_needed = max_size - len(selected)
        pbar = TQDM(total=n_needed, desc="Coreset subsample", leave=False)
        while n_needed > 0:
            k = min(BATCH, max_size - len(selected))
            _, top_idx = dist.topk(k)
            selected.extend(top_idx.tolist())
            centres = mem[top_idx]  # [k, C]
            cos_sim = mem @ centres.t()  # [M, k]
            new_dist = (1.0 - cos_sim).clamp(min=0.0).min(dim=1).values  # [M]
            dist = torch.minimum(dist, new_dist)
            n_needed = max_size - len(selected)
            pbar.update(k)
        pbar.close()
        sel = torch.tensor(selected, device=device)
        return (mem[sel], sel) if return_indices else mem[sel]


class HeatmapProcessor(nn.Module):
    """Post-process a memory-bank heatmap prior before it is fused into PAN features.

    Encapsulates the inference-time transforms that were previously scattered inside
    ``YOLOAnomalyV2Model``: edge-suppression window, min-max stretch, and gaussian/mean
    blur. The processing knobs are owned here as baked-in defaults and are not configurable.
    """

    def __init__(
        self,
        mask_size: int = 80,
        norm: str = "none",
        smooth_kernel: int = 5,
        edge_weight: bool = True,
        edge_p: float = 4.0,
        edge_m: float = 4.4,
        edge_sigma: float = 1.0,
    ):
        super().__init__()
        self.mask_size = int(mask_size)
        self.norm = str(norm)
        self.smooth_kernel = int(smooth_kernel)
        self.edge_weight = bool(edge_weight)
        self.edge_p = float(edge_p)
        self.edge_m = float(edge_m)
        self.edge_sigma = float(edge_sigma)
        self._edge_weight_cache: tuple | None = None

    def forward(self, hmap: torch.Tensor) -> torch.Tensor:
        """Apply edge-weight, normalization, and smoothing to ``hmap``.

        Args:
            hmap: (B, 1, H, W) raw memory-bank heatmap.

        Returns:
            Processed heatmap of the same shape.
        """
        if hmap is None or hmap.numel() == 0:
            return hmap

        # Edge-suppression window: down-weight borders where peripheral patches score
        # high from boundary effects rather than real defects.
        if self.edge_weight:
            hmap = hmap * self._edge_weight(
                hmap,
                p=self.edge_p,
                m=self.edge_m,
                sigma=self.edge_sigma,
            )

        # Per-image min-max normalization: stretch each sample's prior to [0, 1].
        if self.norm == "minmax":
            b = hmap.shape[0]
            flat = hmap.reshape(b, -1)
            lo = flat.min(dim=1, keepdim=True).values
            hi = flat.max(dim=1, keepdim=True).values
            hmap = ((flat - lo) / (hi - lo).clamp_min(1e-6)).reshape_as(hmap)
        elif self.norm in ("gaussian", "mean"):
            hmap = self._smooth_prior(hmap, self.norm, self.smooth_kernel)

        return hmap

    @staticmethod
    def _smooth_prior(mask: torch.Tensor, mode: str, kernel: int) -> torch.Tensor:
        """Blur the prior heatmap while preserving its [0, 1] scale and blob structure."""
        k = max(1, int(kernel)) | 1  # force odd so padding keeps H, W
        if k < 3:
            return mask
        if mode == "mean":
            ker = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype) / float(k * k)
        else:  # gaussian
            ax = torch.arange(k, device=mask.device, dtype=mask.dtype) - (k - 1) / 2.0
            g = torch.exp(-(ax**2) / (2.0 * (k / 6.0) ** 2))
            g = g / g.sum()
            ker = (g[:, None] * g[None, :])[None, None]
        return F.conv2d(mask, ker, padding=k // 2)

    def _edge_weight(self, mask: torch.Tensor, p: float, m: float, sigma: float) -> torch.Tensor:
        """Fixed squircle-Gaussian center-weight window matching ``mask``'s HxW (cached)."""
        h, w = mask.shape[-2], mask.shape[-1]
        key = (h, w, p, m, sigma, mask.device, mask.dtype)
        cache = self._edge_weight_cache
        if cache is None or cache[0] != key:
            yc = (torch.arange(h, device=mask.device, dtype=torch.float32) - (h - 1) / 2.0).abs() / max(
                (h - 1) / 2.0, 1e-6
            )
            xc = (torch.arange(w, device=mask.device, dtype=torch.float32) - (w - 1) / 2.0).abs() / max(
                (w - 1) / 2.0, 1e-6
            )
            dist = (xc[None, :] ** p + yc[:, None] ** p) ** (1.0 / p)
            wmap = torch.exp(-(dist**m) / (2.0 * sigma**m)).to(mask.dtype)
            cache = (key, wmap[None, None])
            self._edge_weight_cache = cache
        return cache[1]
