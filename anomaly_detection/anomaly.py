"""Training-free patch-NN anomaly detection with swappable backbone.

Architecture
============

::

    AnomalyBase                            # Pipeline contract (constructor + 4 standard modules below).
     ├── AnomalyDINO                       # DINOv2 ViT, single layer (no fusion).
     └── _MultiLayerAnomaly                # Adds: hooks → self._cached dict, multi-layer fuse.
          ├── AnomalyYOLO                  # YOLO backbone (Ultralytics ckpt), taps {bb, neck, cv3}.
          └── AnomalyPatchCore             # torchvision ResNet/WideResNet, layer1..4.

The four standard modules every subclass plugs into (see ``AnomalyBase`` docstring for full
contract). Override what you need:

1. **Backbone**          — ``_build_model`` / ``_build_preprocess`` / ``_extract``.
2. **Bank construction** — ``load_support_set`` is the for-loop; subclasses can override
   ``_filter_tokens`` to drop irrelevant patches (e.g. PCA foreground mask) before banking.
3. **Compaction**        — ``_compact``; default = PatchCore-style greedy FPS coreset (with optional
   random pre-sample). Override for KMeans / random / no-op.
4. **Scoring**           — ``_score_patches`` (cosine 1-NN or L2 1-NN) + ``_aggregate_image_score``
   (top-k% mean) + ``_upsample_pixel_map``.

Usage
=====

::

    python tools/anomaly_dino.py --backbone dino_vits14
    python tools/anomaly_dino.py --backbone yolo_p4_bb --weight yoloe-26l-seg.pt --coreset 10000 --coreset_presample 100000
    python tools/anomaly_dino.py --backbone wrn50_l23 --fuse patchcore --patchsize 3

Caveats
=======

* MPS adaptive_avg_pool1d on non-divisible (in, out) sizes is unimplemented → we run pool on CPU.
* MPS allocator can silently corrupt large/high-dim banks across categories → ``empty_cache`` between cats.
* MPS argmin on large/duplicate-heavy banks gives wrong results → FPS runs on CPU.
"""
import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from mvtec_yolo import CATEGORIES as MVTEC_CATEGORIES, get_mvtec_yolo_data


# ---------------------------------------------------------------------------
# Pixel-AUROC GT mask helper (copied from YOLOA fork's AnomalyValidator so this
# script does not depend on the fork's ultralytics package).
# ---------------------------------------------------------------------------


class AnomalyValidator:
    """Minimal stand-in exposing only the GT-mask rasterizer used by the eval driver."""

    @staticmethod
    def _label_path(im_file: str) -> Path | None:
        """Resolve YOLO label .txt for an image. Tries ``labels/`` mirror first, then sibling."""
        p = Path(im_file)
        parts = p.parts
        if "images" in parts:
            idx = parts.index("images")
            mirrored = Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")
            if mirrored.exists():
                return mirrored
        sibling = p.with_suffix(".txt")
        return sibling if sibling.exists() else None

    @classmethod
    def _gt_mask_from_polygons(cls, im_file: str, mask_size: int) -> "np.ndarray | None":
        """Rasterize all instance polygons from the YOLO label file into a binary mask.

        Returns ``None`` if no label file or no polygons are found.
        Output shape: (mask_size, mask_size), uint8, values in {0, 1}.
        """
        import cv2

        label_path = cls._label_path(im_file)
        if label_path is None:
            return None
        try:
            text = label_path.read_text().strip()
        except OSError:
            return None
        if not text:
            return None

        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        any_drawn = False
        for line in text.splitlines():
            tokens = line.strip().split()
            # YOLO seg polygon line: cls x1 y1 x2 y2 ... xn yn  (normalized, pairs of points)
            if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
                continue
            try:
                coords = np.array(tokens[1:], dtype=np.float32)
            except ValueError:
                continue
            pts = (coords.reshape(-1, 2) * mask_size).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
            any_drawn = True
        return mask if any_drawn else None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DINOV2_HUB = {  # short → (torch.hub entry, feat_dim)
    "vits14": ("dinov2_vits14_reg", 384),
    "vitb14": ("dinov2_vitb14_reg", 768),
    "vitl14": ("dinov2_vitl14_reg", 1024),
}
_DINO_PATCH = 14
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_CONVNEXT_CONFIGS = {
    # name → (depths, dims)
    "tiny":  ([3, 3, 9, 3],  [96, 192, 384, 768]),
    "small": ([3, 3, 27, 3], [96, 192, 384, 768]),
    "base":  ([3, 3, 27, 3], [128, 256, 512, 1024]),
    "large": ([3, 3, 27, 3], [192, 384, 768, 1536]),
}

_CONVNEXT_WEIGHTS = Path(__file__).parent / "dinov3_weights" / "convnext"

_YOLO_LAYER_IDX = {"P3": 0, "P4": 1, "P5": 2}
_YOLO_TAPS = ("bb", "neck", "cv3")  # bb=pre-FPN backbone stage, neck=PAN-FPN output (= Detect input), cv3=cls-branch
_YOLO_BB_LAYER = {"P3": 4, "P4": 6, "P5": 10}

_RESNET_LAYERS = ("l1", "l2", "l3", "l4")
_RESNET_LAYER_ATTR = {"l1": "layer1", "l2": "layer2", "l3": "layer3", "l4": "layer4"}

_FUSE_CHOICES = ("concat", "sum", "avg", "patchcore", "concat_pool")
_SCORE_CHOICES = ("cosine", "l2")
_BANK_MODES = ("flat", "obma")  # flat = dump everything (optional FPS), obma = OBMA accept/reject

# ---------------------------------------------------------------------------
# OBMA + Noisy-OR helpers
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _noisy_or_score(features: torch.Tensor, bank: torch.Tensor,
                    temperature: float, K: int) -> torch.Tensor:
    """Per-position Noisy-OR anomaly probability ∈ [0, 1].

    Args:
        features: [N, D] L2-normalised query features.
        bank: [M, D] L2-normalised memory bank.
        temperature: β — Noisy-OR sharpening factor (set by auto-calibration).
        K: top-K nearest bank entries used per query position.

    Returns:
        [N] tensor, higher = more anomalous. 0.5 fallback if bank is empty.

    Formula:
        ψ(x) = exp(-β·(1-x))                       # cos-sim → "match probability"
        score = geom_mean(1 - ψ(top-K cos-sim))    # ≈ 1 - P(any top-K matches)
    """
    if bank.numel() == 0 or bank.shape[0] == 0:
        return torch.full((features.shape[0],), 0.5,
                          device=features.device, dtype=features.dtype)
    sim = features @ bank.t()                                       # [N, M] cos sim
    sim_t = torch.exp(-temperature * (1.0 - sim))                   # ψ(x) ∈ [0, 1]
    k = min(K, bank.shape[0])
    topk = sim_t.topk(k=k, dim=1).values                            # [N, k]
    log_complement = torch.log((1.0 - topk).clamp(min=1e-8))        # [N, k]
    return torch.exp(log_complement.mean(dim=1)).clamp(0.0, 1.0)    # [N]


@torch.inference_mode()
def _calibrate_obma_temperature(sample: torch.Tensor, bank: torch.Tensor,
                                K: int, target_score: float) -> float:
    """Solve for β so a 'typical normal' position scores at ``target_score``.

    Derivation (single-entry approximation, see AnomalyDINO/PatchCore literature):
        score ≈ 1 - exp(-β·(1 - s_typical))
        ⇒  β = -ln(1 - target_score) / (1 - s_typical)

    where s_typical = 90th percentile of mean-top-K cosine similarity for normal
    features queried against the bank. Capped to [0.1, 20.0] for safety.

    Args:
        sample: [N, D] L2-normalised normal features (subset of support set).
        bank: [M, D] L2-normalised memory bank.
        K: top-K nearest bank entries.
        target_score: desired score for typical-normal (default 0.2).
    """
    k = min(K, bank.shape[0])
    sim = sample @ bank.t()                                  # [N, M]
    topk_sim = sim.topk(k=k, dim=1).values                   # [N, k]
    mean_topk = topk_sim.mean(dim=1)                         # [N]
    s_typical = float(mean_topk.quantile(0.90).clamp(0.0, 1.0 - 1e-4).item())
    beta = -math.log(1.0 - target_score) / (1.0 - s_typical)
    return max(0.1, min(20.0, beta))


@torch.inference_mode()
def _obma_accept_one_image(bank: torch.Tensor, cand_feats: torch.Tensor,
                           accumulate_thresh: float, temperature: float,
                           K: int) -> torch.Tensor:
    """Sequentially append novel features from one image's candidates to the bank.

    Greedy dedup within the image: candidates are sorted by initial novelty (highest
    Noisy-OR score first); the first is always accepted, then subsequent candidates
    are re-scored against the *growing* bank before accept/reject. This guarantees
    no two features contributed by the same image are mutually redundant.

    Optimisation: a single [N, cur+N] sim matrix is built once and a new column is
    appended per accept (instead of recomputing the full matmul each step).

    Args:
        bank: [M, D] current bank (on device, L2-normalised).
        cand_feats: [N, D] candidate features sorted descending by initial novelty.
        accumulate_thresh: minimum Noisy-OR score to accept.
        temperature: β for Noisy-OR scoring.
        K: top-K nearest bank entries.

    Returns:
        [M+k, D] updated bank with the k accepted candidates appended.
    """
    M0 = bank.shape[0]
    N = cand_feats.shape[0]
    if N == 0:
        return bank
    D = cand_feats.shape[1]
    device = cand_feats.device
    dtype = bank.dtype if M0 > 0 else cand_feats.dtype

    buf = torch.empty((M0 + N, D), device=device, dtype=dtype)
    if M0 > 0:
        buf[:M0] = bank.to(device=device, dtype=dtype)
    cur = M0

    # Sim cache: cols 0..cur = sim against existing bank; appended cols = sim against accepts.
    S = torch.empty((N, M0 + N), device=device, dtype=dtype)
    if M0 > 0:
        S[:, :M0] = cand_feats @ bank.t()
    M_view = M0

    # Always accept the most novel candidate first.
    buf[cur] = cand_feats[0]
    S[:, M_view] = cand_feats @ cand_feats[0]
    M_view += 1
    cur += 1
    pos = 1

    while pos < N:
        # Inline Noisy-OR over the already-cached sim sub-matrix.
        sim_sub = S[pos:, :M_view]                                # [N-pos, M_view]
        sim_t = torch.exp(-temperature * (1.0 - sim_sub))
        k = min(K, M_view)
        topk = sim_t.topk(k=k, dim=1).values
        log_comp = torch.log((1.0 - topk).clamp(min=1e-8))
        scores = torch.exp(log_comp.mean(dim=1)).clamp(0.0, 1.0)  # [N-pos]
        passing = (scores > accumulate_thresh).nonzero(as_tuple=True)[0]
        if passing.numel() == 0:
            break
        j = pos + int(passing[0].item())
        buf[cur] = cand_feats[j]
        S[:, M_view] = cand_feats @ cand_feats[j]
        M_view += 1
        cur += 1
        pos = j + 1

    return buf[:cur].clone()


# ---------------------------------------------------------------------------
# Base class — defines the contract for the 4 standard modules
# ---------------------------------------------------------------------------


class AnomalyBase:
    """Training-free patch-NN anomaly detector. Subclass to plug in a backbone.

    Standard modules (override as needed):

    * **Backbone** (``_build_model``, ``_build_preprocess``, ``_extract``):
      ``_extract(x: [B, 3, H, W]) → [B, N, D]`` returns L2-normalize-ready patch tokens.

    * **Bank construction** (``load_support_set`` for-loop, ``_filter_tokens`` hook):
      iterate over support images in batches, run ``_extract``, flatten to ``[N, D]``,
      optionally drop tokens via ``_filter_tokens``, concat all, L2-normalize, then compact.

    * **Compaction** (``_compact``): default = PatchCore greedy FPS coreset (with optional
      random pre-sample). Triggered iff ``self.coreset_size`` is set and smaller than bank.

    * **Scoring** (``_score_patches``, ``_aggregate_image_score``, ``_upsample_pixel_map``):
      1-NN against bank under the chosen metric, top-k% mean as image score, bilinear
      upsample of patch grid as pixel map.

    Common state set by subclass ``__init__``:
        ``self.model``, ``self.tf`` (PIL → tensor), ``self.grid``, ``self.feat_dim``,
        ``self.device``, ``self.backbone`` (descriptive tag).
    """

    def __init__(
        self,
        imgsz: int,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
        score_metric: str = "cosine",
        coreset_size: int | None = None,
        coreset_presample: int | None = None,
        bank_mode: str = "flat",
        obma_thresh: float = 0.4,
        obma_K: int = 15,
        obma_target_score: float = 0.2,
        obma_temperature: float | None = None,
    ) -> None:
        if score_metric not in _SCORE_CHOICES:
            raise ValueError(f"score_metric must be in {_SCORE_CHOICES}, got {score_metric!r}")
        if bank_mode not in _BANK_MODES:
            raise ValueError(f"bank_mode must be in {_BANK_MODES}, got {bank_mode!r}")
        self.imgsz = imgsz
        self.top_pct = top_pct
        self.sim_chunk = sim_chunk
        self.score_metric = score_metric
        self.coreset_size = coreset_size
        self.coreset_presample = coreset_presample
        # OBMA + Noisy-OR config.
        self.bank_mode = bank_mode
        self.obma_thresh = obma_thresh
        self.obma_K = obma_K
        self.obma_target_score = obma_target_score
        # User-set temperature; auto-calibrated to ``self._obma_temperature`` if None.
        self.obma_temperature = obma_temperature
        self._obma_temperature: float | None = obma_temperature  # filled by calibration
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        # Subclass must set: model, tf, grid (H, W), feat_dim, backbone (tag).
        self.bank: torch.Tensor | None = None

    # ── 1. Backbone ─────────────────────────────────────────────────────
    def _build_model(self) -> None:  # subclass: assign self.model, register hooks
        raise NotImplementedError

    def _build_preprocess(self) -> None:  # subclass: assign self.tf
        raise NotImplementedError

    @torch.inference_mode()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:  # → [B, N, D]
        raise NotImplementedError

    def _load(self, path: str) -> torch.Tensor:
        return self.tf(Image.open(path).convert("RGB"))

    # ── 2. Bank construction ────────────────────────────────────────────
    def load_support_set(self, image_paths: list[str], batch: int = 8) -> None:
        """Build memory bank. Dispatches on ``self.bank_mode``."""
        if self.bank_mode == "obma":
            return self._load_support_obma(image_paths)
        # Default = flat: dump all tokens, optional FPS coreset.
        feats: list[torch.Tensor] = []
        for i in range(0, len(image_paths), batch):
            xs = torch.stack([self._load(p) for p in image_paths[i:i + batch]], dim=0)
            tokens = self._extract(xs).reshape(-1, self.feat_dim).cpu()  # keep big intermediate on CPU
            tokens = self._filter_tokens(tokens)
            feats.append(tokens)
        bank = F.normalize(torch.cat(feats, dim=0), dim=-1)
        bank = self._compact(bank)
        self.bank = bank.to(self.device)

    @torch.inference_mode()
    def _load_support_obma(self, image_paths: list[str], initial_temp: float = 3.0,
                           calib_sample_per_img: int = 200) -> None:
        """OBMA bank construction: per-image accept/reject + sequential dedup.

        Iterates images one-by-one. Image 0 seeds the bank with all its (filtered)
        features. From image 1 onwards, only positions whose Noisy-OR score against
        the current bank exceeds ``self.obma_thresh`` are eligible; eligible candidates
        are then deduped sequentially against the *growing* bank within that image.

        After the full sweep, auto-calibrate β so a typical-normal position scores at
        ``self.obma_target_score``. Skipped if ``self.obma_temperature`` was set explicitly.

        **MPS note:** the sequential dedup loop is run on CPU. Per-iteration ``.item()``
        synchronisations are ~1000× cheaper on CPU than MPS, and tensor sizes are
        small (N ≲ 400 candidates × bank ≲ 100k rows) so CPU matmul wins overall.
        """
        bank_cpu: torch.Tensor | None = None    # OBMA build runs on CPU (see docstring)
        calib_sample: list[torch.Tensor] = []   # for post-build β calibration

        t_start = time.perf_counter()
        log_every = max(1, len(image_paths) // 10)
        for idx, path in enumerate(image_paths):
            x = self._load(path).unsqueeze(0)
            # Extract on device, immediately move to CPU for the OBMA loop.
            tokens = self._extract(x).reshape(-1, self.feat_dim).cpu()
            tokens = self._filter_tokens(tokens)
            normed = F.normalize(tokens, p=2, dim=-1).contiguous()      # [N, D] on CPU

            # Sub-sample for calibration (cap memory ~ N_images × cap × D).
            n_keep = min(calib_sample_per_img, normed.shape[0])
            calib_sample.append(normed[:n_keep])

            # Bootstrap: image 0 dumps everything.
            if bank_cpu is None or bank_cpu.shape[0] == 0:
                bank_cpu = normed
                print(f"  obma: img {idx + 1}/{len(image_paths)} bank={bank_cpu.shape[0]} "
                      f"(seed) elapsed={time.perf_counter() - t_start:.1f}s", flush=True)
                continue

            # Score all positions, take candidates above threshold.
            scores = _noisy_or_score(normed, bank_cpu, initial_temp, self.obma_K)
            cand_mask = scores > self.obma_thresh
            if not cand_mask.any():
                if (idx % log_every) == 0 or idx == len(image_paths) - 1:
                    print(f"  obma: img {idx + 1}/{len(image_paths)} bank={bank_cpu.shape[0]} "
                          f"added=0 elapsed={time.perf_counter() - t_start:.1f}s", flush=True)
                continue
            cand_idx = cand_mask.nonzero(as_tuple=True)[0]
            cand_idx = cand_idx[scores[cand_idx].argsort(descending=True)]
            cand_feats = normed[cand_idx]

            new_bank = _obma_accept_one_image(bank_cpu, cand_feats, self.obma_thresh,
                                              initial_temp, self.obma_K)
            added = new_bank.shape[0] - bank_cpu.shape[0]
            bank_cpu = new_bank
            if (idx % log_every) == 0 or idx == len(image_paths) - 1:
                print(f"  obma: img {idx + 1}/{len(image_paths)} bank={bank_cpu.shape[0]} "
                      f"added={added} elapsed={time.perf_counter() - t_start:.1f}s", flush=True)

        assert bank_cpu is not None, "OBMA: empty support set"

        # Calibrate β on CPU using the support feature sample, then move bank to device.
        if self.obma_temperature is None:
            sample = torch.cat(calib_sample, dim=0)
            self._obma_temperature = _calibrate_obma_temperature(
                sample, bank_cpu, self.obma_K, self.obma_target_score,
            )
            print(f"  obma: auto-calibrated β={self._obma_temperature:.3f} "
                  f"(target_score={self.obma_target_score}, |sample|={sample.shape[0]})",
                  flush=True)
        else:
            self._obma_temperature = self.obma_temperature
            print(f"  obma: using user-specified β={self._obma_temperature:.3f}", flush=True)

        # Final bank moves to device for fast Noisy-OR scoring at inference time.
        self.bank = bank_cpu.contiguous().to(self.device)

    def _filter_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hook: drop / reweight patch tokens before banking. Default = identity.

        Override for tricks like AnomalyDINO's PCA foreground masking on object classes,
        or to skip border patches with low gradient.
        """
        return tokens

    # ── 3. Compaction ───────────────────────────────────────────────────
    def _compact(self, bank: torch.Tensor) -> torch.Tensor:
        """Default = greedy FPS coreset with optional random pre-sample (PatchCore)."""
        if not (self.coreset_size and self.coreset_size < bank.shape[0]):
            return bank
        if self.coreset_presample and self.coreset_presample < bank.shape[0]:
            g = torch.Generator(device="cpu").manual_seed(0)
            pre_idx = torch.randperm(bank.shape[0], generator=g)[: self.coreset_presample]
            print(f"  coreset: pre-sample {bank.shape[0]} → {self.coreset_presample} before FPS")
            bank = bank[pre_idx].contiguous()
        return _greedy_fps(bank, self.coreset_size, sim_chunk=self.sim_chunk)

    # ── 4. Scoring ──────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict_one(self, image_path: str, eval_size: int = 256) -> tuple[float, np.ndarray]:
        """Score one test image. Return ``(image_score, pixel_map[eval_size, eval_size])``.

        Dispatches by ``self.bank_mode``:
          * ``flat`` — per-patch 1-NN distance (cosine or L2) → top-k% mean.
          * ``obma`` — per-patch Noisy-OR probability ∈ [0,1] → max over positions.
        """
        if self.bank is None:
            raise RuntimeError("call load_support_set() before predict_one()")
        x = self._load(image_path).unsqueeze(0)
        feats = F.normalize(self._extract(x).squeeze(0), dim=-1)   # [N, D]

        if self.bank_mode == "obma":
            assert self._obma_temperature is not None
            probs = _noisy_or_score(feats, self.bank,
                                    self._obma_temperature, self.obma_K)  # [N] ∈ [0,1]
            image_score = float(probs.max().item())
            amap = self._upsample_pixel_map(probs, eval_size)
            return image_score, amap

        dists = self._score_patches(feats)                         # [N]
        image_score = self._aggregate_image_score(dists)
        amap = self._upsample_pixel_map(dists, eval_size)
        return image_score, amap

    def _score_patches(self, feats: torch.Tensor) -> torch.Tensor:
        """1-NN distance per patch against the bank. Chunked over bank dim to bound memory."""
        if self.score_metric == "cosine":
            # For L2-normalized bank+query, cosine sim == dot product. Distance = 1 - sim.
            max_sim = torch.full((feats.shape[0],), -1.0, device=feats.device, dtype=feats.dtype)
            for s in range(0, self.bank.shape[0], self.sim_chunk):
                sims = feats @ self.bank[s:s + self.sim_chunk].T  # [N, chunk]
                max_sim = torch.maximum(max_sim, sims.max(dim=-1).values)
            return 1.0 - max_sim
        # L2 distance (squared, since monotonic ordering is what matters for top-k and AUROC).
        min_d2 = torch.full((feats.shape[0],), float("inf"), device=feats.device, dtype=feats.dtype)
        for s in range(0, self.bank.shape[0], self.sim_chunk):
            d2 = torch.cdist(feats, self.bank[s:s + self.sim_chunk]).pow(2)
            min_d2 = torch.minimum(min_d2, d2.min(dim=-1).values)
        return min_d2

    def _aggregate_image_score(self, dists: torch.Tensor) -> float:
        """Top-``top_pct`` mean of per-patch distances. ``top_pct=0.01`` ≈ PatchCore/AnomalyDINO default."""
        n = dists.numel()
        k = max(1, int(round(n * self.top_pct)))
        return float(dists.topk(k).values.mean())

    def _upsample_pixel_map(self, dists: torch.Tensor, eval_size: int) -> np.ndarray:
        H, W = self.grid
        amap = dists.reshape(1, 1, H, W).float()
        amap = F.interpolate(amap, size=(eval_size, eval_size), mode="bilinear", align_corners=False)
        return amap.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Coreset helper (module-level so subclasses can swap _compact entirely without inheriting it)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _greedy_fps(bank: torch.Tensor, n_keep: int, sim_chunk: int = 4096, seed: int = 0) -> torch.Tensor:
    """Greedy farthest-point sampling over L2-normalized features (cosine metric).

    Runs on CPU: MPS argmin gives wrong results on large/duplicate-heavy banks (silently
    corrupts grid-like patterns), and MPS sync overhead per-iter dominates the matmul cost
    anyway. Pass a CUDA bank explicitly to get GPU speedup.
    """
    orig_device = bank.device
    if orig_device.type == "mps":
        bank = bank.cpu()
    M, _ = bank.shape
    device = bank.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    start = int(torch.randint(M, (1,), generator=g).item())

    max_sim = torch.full((M,), -1.0, device=device, dtype=bank.dtype)
    selected = torch.empty(n_keep, dtype=torch.long, device=device)
    selected[0] = start
    cur = bank[start]
    t0 = time.perf_counter()
    for i in range(1, n_keep):
        sims = bank @ cur
        max_sim = torch.maximum(max_sim, sims)
        nxt = torch.argmin(max_sim)
        selected[i] = nxt
        cur = bank[nxt]
        if i % 2000 == 0:
            print(f"  coreset: {i}/{n_keep}  elapsed={time.perf_counter() - t0:.1f}s")
    return bank[selected].contiguous().to(orig_device)


# ---------------------------------------------------------------------------
# DINOv2 — single layer, no fusion
# ---------------------------------------------------------------------------


class AnomalyDINO(AnomalyBase):
    """DINOv2 ViT + memory-bank anomaly detector (Damm et al., WACV 2025).

    Single layer (last block patch tokens), no multi-layer fusion. PCA foreground mask
    omitted for minimal baseline — override ``_filter_tokens`` to add it back.
    """

    def __init__(
        self,
        backbone: str = "vits14",
        imgsz: int = 448,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
        score_metric: str = "cosine",
        coreset_size: int | None = None,
        coreset_presample: int | None = None,
        bank_mode: str = "flat",
        obma_thresh: float = 0.4,
        obma_K: int = 15,
        obma_target_score: float = 0.2,
        obma_temperature: float | None = None,
    ) -> None:
        if backbone not in _DINOV2_HUB:
            raise ValueError(f"backbone must be in {list(_DINOV2_HUB)}, got {backbone!r}")
        if imgsz % _DINO_PATCH != 0:
            raise ValueError(f"imgsz must be divisible by {_DINO_PATCH}, got {imgsz}")
        super().__init__(imgsz=imgsz, device=device, top_pct=top_pct, sim_chunk=sim_chunk,
                         score_metric=score_metric, coreset_size=coreset_size,
                         coreset_presample=coreset_presample,
                         bank_mode=bank_mode, obma_thresh=obma_thresh, obma_K=obma_K,
                         obma_target_score=obma_target_score, obma_temperature=obma_temperature)
        self.dino_short = backbone
        self.backbone = f"dino_{backbone}"
        self._build_model()
        self._build_preprocess()
        n = imgsz // _DINO_PATCH
        self.grid = (n, n)
        # feat_dim set in _build_model from the hub registry.

    def _build_model(self) -> None:
        hub_name, self.feat_dim = _DINOV2_HUB[self.dino_short]
        self.model = torch.hub.load("facebookresearch/dinov2", hub_name).to(self.device).eval()

    def _build_preprocess(self) -> None:
        self.tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    @torch.inference_mode()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x.to(self.device))
        return out["x_norm_patchtokens"]  # [B, N, D]


# ---------------------------------------------------------------------------
# DINOv3 ConvNeXt (distilled, single-layer from stage 3)
# ---------------------------------------------------------------------------

class ConvNeXtLayerNorm(nn.Module):
    """LayerNorm with channels_first / channels_last support."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """DINOv3 ConvNeXt block: dwconv → LN → pwconv1 → GELU → pwconv2 → gamma + residual."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ConvNeXtLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()  # inference only

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        return residual + self.drop_path(x)


class DINOv3ConvNeXt(nn.Module):
    """Official DINOv3 ConvNeXt backbone (matches checkpoint keys exactly)."""

    def __init__(
        self,
        depths: list[int],
        dims: list[int],
        patch_size: int | None = None,
    ):
        super().__init__()
        # Stem + 3 downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            ConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                ConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))
        # Stages
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])
            ]))
        # Norms: Identity for stages 0-2, LayerNorm for stage 3 (matches checkpoint)
        self.norms = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.norms.append(self.norm)
        # Housekeeping
        self.patch_size = patch_size
        self.n_storage_tokens = 0
        self.embed_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward features returning same dict format as DINOv2 hub model."""
        h, w = x.shape[-2:]
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x_pool = x.mean([-2, -1])                               # [B, C]
        x = torch.flatten(x, 2).transpose(1, 2)                 # [B, HW, C]
        x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))  # [B, 1+HW, C]
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1:],
            "x_prenorm": x,
        }

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Alias matching DINOv2 hub API."""
        return self.forward(x)


class AnomalyConvNeXt(AnomalyBase):
    """DINOv3-distilled ConvNeXt + memory-bank anomaly detector.

    Single layer (stage 3, stride 32), same scoring as AnomalyDINO.
    """

    def __init__(
        self,
        backbone: str = "small",
        imgsz: int = 448,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
        score_metric: str = "cosine",
        coreset_size: int | None = None,
        coreset_presample: int | None = None,
        bank_mode: str = "flat",
        obma_thresh: float = 0.4,
        obma_K: int = 15,
        obma_target_score: float = 0.2,
        obma_temperature: float | None = None,
        weight: str | Path | None = None,
    ) -> None:
        if backbone not in _CONVNEXT_CONFIGS:
            raise ValueError(f"backbone must be in {list(_CONVNEXT_CONFIGS)}, got {backbone!r}")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be divisible by 32, got {imgsz}")
        super().__init__(imgsz=imgsz, device=device, top_pct=top_pct, sim_chunk=sim_chunk,
                         score_metric=score_metric, coreset_size=coreset_size,
                         coreset_presample=coreset_presample,
                         bank_mode=bank_mode, obma_thresh=obma_thresh, obma_K=obma_K,
                         obma_target_score=obma_target_score, obma_temperature=obma_temperature)
        self.convnext_name = backbone
        self.backbone = f"convnext_{backbone}"
        self._weight = weight
        self._build_model()
        self._build_preprocess()
        n = imgsz // 32
        self.grid = (n, n)

    def _build_model(self) -> None:
        depths, dims = _CONVNEXT_CONFIGS[self.convnext_name]
        self.feat_dim = dims[3]
        self.model = DINOv3ConvNeXt(depths, dims).to(self.device).eval()
        if self._weight:
            ckpt_path = Path(self._weight)
        else:
            matches = sorted(_CONVNEXT_WEIGHTS.glob(
                f"dinov3_convnext_{self.convnext_name}_pretrain_lvd1689m-*.pth"))
            if not matches:
                raise FileNotFoundError(f"No checkpoint for {self.convnext_name} in {_CONVNEXT_WEIGHTS}")
            ckpt_path = matches[0]
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd, strict=True)

    def _build_preprocess(self) -> None:
        self.tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    @torch.inference_mode()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x.to(self.device))
        return out["x_norm_patchtokens"]  # [B, N, D]


class _MultiLayerAnomaly(AnomalyBase):
    """Adds hooks → ``self._cached: dict[lname → BCHW]`` + multi-layer fuse to ``_extract``.

    Subclasses provide ``_register_hooks(layers)`` and set ``self.layers`` (highest-res first).
    """

    fuse: str
    layers: list[str]
    patchsize: int
    pretrain_dim: int
    target_dim: int
    _cached: dict[str, torch.Tensor]

    def __init__(
        self,
        imgsz: int,
        layers: list[str],
        fuse: str = "concat",
        patchsize: int = 1,
        pretrain_dim: int = 1024,
        target_dim: int = 1024,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
        score_metric: str = "cosine",
        coreset_size: int | None = None,
        coreset_presample: int | None = None,
        bank_mode: str = "flat",
        obma_thresh: float = 0.4,
        obma_K: int = 15,
        obma_target_score: float = 0.2,
        obma_temperature: float | None = None,
    ) -> None:
        if fuse not in _FUSE_CHOICES:
            raise ValueError(f"fuse must be in {_FUSE_CHOICES}, got {fuse!r}")
        if patchsize % 2 == 0:
            raise ValueError(f"patchsize must be odd, got {patchsize}")
        super().__init__(imgsz=imgsz, device=device, top_pct=top_pct, sim_chunk=sim_chunk,
                         score_metric=score_metric, coreset_size=coreset_size,
                         coreset_presample=coreset_presample,
                         bank_mode=bank_mode, obma_thresh=obma_thresh, obma_K=obma_K,
                         obma_target_score=obma_target_score, obma_temperature=obma_temperature)
        self.layers = layers
        self.fuse = fuse
        self.patchsize = patchsize
        self.pretrain_dim = pretrain_dim
        self.target_dim = target_dim
        self._cached = {}

    def _probe_and_set_dims(self) -> None:
        """Run a dummy forward to discover ``self.grid`` and ``self.feat_dim`` from cached features."""
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device))
        missing = [l for l in self.layers if l not in self._cached]
        assert not missing, f"hooks didn't fire for {missing}"
        ref = self._cached[self.layers[0]]
        self.grid = (ref.shape[2], ref.shape[3])
        dims = [self._cached[l].shape[1] for l in self.layers]
        if self.fuse == "concat":
            self.feat_dim = sum(dims)
        elif self.fuse in ("patchcore", "concat_pool"):
            self.feat_dim = self.target_dim
        else:  # sum / avg need matching dims
            if len(set(dims)) > 1:
                raise ValueError(f"fuse={self.fuse!r} needs equal channel dims, got {dict(zip(self.layers, dims))}")
            self.feat_dim = dims[0]

    @torch.inference_mode()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        self._cached = {}
        self.model(x.to(self.device))
        H, W = self.grid
        B = x.shape[0]
        if self.fuse == "patchcore":
            return self._fuse_patchcore(B, H, W)
        if self.fuse == "concat_pool":
            return self._fuse_concat_pool(B, H, W)
        return self._fuse_simple(B, H, W)

    def _fuse_simple(self, B: int, H: int, W: int) -> torch.Tensor:
        """concat / sum / avg over upsampled channel-aligned feature maps."""
        feats = []
        for lname in self.layers:
            f = self._cached[lname]
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            feats.append(f)
        if len(feats) == 1:
            fused = feats[0]
        elif self.fuse == "concat":
            fused = torch.cat(feats, dim=1)
        elif self.fuse == "sum":
            fused = torch.stack(feats, dim=0).sum(dim=0)
        else:  # avg
            fused = torch.stack(feats, dim=0).mean(dim=0)
        return fused.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, D]

    def _fuse_patchcore(self, B: int, H: int, W: int) -> torch.Tensor:
        """3x3 unfold + per-layer adaptive_avg_pool → concat → adaptive_avg_pool to target_dim."""
        ps, pad = self.patchsize, self.patchsize // 2
        pool_dev = "cpu" if self.device == "mps" else self.device
        per_layer = []
        for lname in self.layers:
            f = self._cached[lname]
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            patches = F.unfold(f, kernel_size=ps, stride=1, padding=pad)        # [B, C*ps^2, H*W]
            patches = patches.transpose(1, 2).reshape(B * H * W, -1)            # [B*N, C*ps^2]
            pooled = F.adaptive_avg_pool1d(patches.to(pool_dev).unsqueeze(1), self.pretrain_dim).squeeze(1)
            per_layer.append(pooled)
        merged = torch.cat(per_layer, dim=-1) if len(per_layer) > 1 else per_layer[0]
        final = F.adaptive_avg_pool1d(merged.unsqueeze(1), self.target_dim).squeeze(1)
        return final.to(self.device).reshape(B, H * W, self.target_dim)

    def _fuse_concat_pool(self, B: int, H: int, W: int) -> torch.Tensor:
        """unfold + concat across layers + single adaptive_avg_pool to target_dim (no per-layer pool)."""
        ps, pad = self.patchsize, self.patchsize // 2
        pool_dev = "cpu" if self.device == "mps" else self.device
        feats_list = []
        for lname in self.layers:
            f = self._cached[lname]
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            patches = F.unfold(f, kernel_size=ps, stride=1, padding=pad)
            patches = patches.transpose(1, 2).reshape(B * H * W, -1)
            feats_list.append(patches)
        merged = torch.cat(feats_list, dim=-1) if len(feats_list) > 1 else feats_list[0]
        final = F.adaptive_avg_pool1d(merged.to(pool_dev).unsqueeze(1), self.target_dim).squeeze(1)
        return final.to(self.device).reshape(B, H * W, self.target_dim)


# ---------------------------------------------------------------------------
# YOLO backbone
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ConvNeXt multi-layer — hooks on stage 2+3, PatchCore fusion
# ---------------------------------------------------------------------------

class AnomalyConvNeXtMulti(_MultiLayerAnomaly):
    """DINOv3 ConvNeXt with multi-layer fusion (PatchCore-style).

    Hooks stage 2 (stride 16) and stage 3 (stride 32). Layer names are "s2", "s3".
    """

    def __init__(
        self,
        backbone: str = "small",
        imgsz: int = 320,
        layers: str | list[str] = "s2",
        fuse: str = "patchcore",
        patchsize: int = 3,
        pretrain_dim: int = 1024,
        target_dim: int = 1024,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
        score_metric: str = "cosine",
        coreset_size: int | None = None,
        coreset_presample: int | None = None,
        bank_mode: str = "flat",
        obma_thresh: float = 0.4,
        obma_K: int = 15,
        obma_target_score: float = 0.2,
        obma_temperature: float | None = None,
        weight: str | Path | None = None,
    ) -> None:
        if backbone not in _CONVNEXT_CONFIGS:
            raise ValueError(f"backbone must be in {list(_CONVNEXT_CONFIGS)}, got {backbone!r}")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be divisible by 32, got {imgsz}")
        if isinstance(layers, str):
            layers = [layers]
        valid = {"s2", "s3"}
        if not all(l in valid for l in layers):
            raise ValueError(f"layers must be subset of {valid}, got {layers}")
        layers = sorted(set(layers))
        super().__init__(imgsz=imgsz, layers=layers, fuse=fuse, patchsize=patchsize,
                         pretrain_dim=pretrain_dim, target_dim=target_dim, device=device,
                         top_pct=top_pct, sim_chunk=sim_chunk, score_metric=score_metric,
                         coreset_size=coreset_size, coreset_presample=coreset_presample,
                         bank_mode=bank_mode, obma_thresh=obma_thresh, obma_K=obma_K,
                         obma_target_score=obma_target_score, obma_temperature=obma_temperature)
        self.convnext_name = backbone
        self.backbone = self._make_tag()
        self._weight = weight
        self._build_model()
        self._build_preprocess()
        self._probe_and_set_dims()

    def _make_tag(self) -> str:
        digits = "".join(l[-1] for l in self.layers)
        if self.fuse == "patchcore":
            suf = f"_pc{self.patchsize}_e{self.pretrain_dim}_t{self.target_dim}"
        else:
            suf = f"_{self.fuse}" if len(self.layers) > 1 else ""
        return f"convnext_{self.convnext_name}_s{digits}{suf}"

    def _build_model(self) -> None:
        depths, dims = _CONVNEXT_CONFIGS[self.convnext_name]
        self.model = DINOv3ConvNeXt(depths, dims).to(self.device).eval()
        if self._weight:
            ckpt_path = Path(self._weight)
        else:
            matches = sorted(_CONVNEXT_WEIGHTS.glob(
                f"dinov3_convnext_{self.convnext_name}_pretrain_lvd1689m-*.pth"))
            if not matches:
                raise FileNotFoundError(f"No checkpoint for {self.convnext_name} in {_CONVNEXT_WEIGHTS}")
            ckpt_path = matches[0]
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd, strict=True)
        _STAGE_IDX = {"s2": 2, "s3": 3}
        for lname in self.layers:
            def _make(k=lname):
                def h(_m, _i, output):
                    self._cached[k] = output
                return h
            self.model.stages[_STAGE_IDX[lname]].register_forward_hook(_make())

    def _build_preprocess(self) -> None:
        self.tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


class AnomalyYOLO(_MultiLayerAnomaly):
    """YOLO/YOLOE backbone, hooked at one of three taps (see ``_YOLO_TAPS``)."""

    def __init__(
        self,
        weight: str = "yolo26l.pt",
        imgsz: int = 640,
        layers: str | list[str] = "P3",
        tap: str = "neck",
        fuse: str = "concat",
        patchsize: int = 1,
        pretrain_dim: int = 1024,
        target_dim: int = 1024,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
        score_metric: str = "cosine",
        coreset_size: int | None = None,
        coreset_presample: int | None = None,
        bank_mode: str = "flat",
        obma_thresh: float = 0.4,
        obma_K: int = 15,
        obma_target_score: float = 0.2,
        obma_temperature: float | None = None,
    ) -> None:
        if isinstance(layers, str):
            layers = [layers]
        if not all(l in _YOLO_LAYER_IDX for l in layers):
            raise ValueError(f"each layer must be in {list(_YOLO_LAYER_IDX)}, got {layers}")
        if tap not in _YOLO_TAPS:
            raise ValueError(f"tap must be in {_YOLO_TAPS}, got {tap!r}")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be divisible by 32 (YOLO stride), got {imgsz}")
        layers = sorted(set(layers), key=lambda l: _YOLO_LAYER_IDX[l])
        super().__init__(imgsz=imgsz, layers=layers, fuse=fuse, patchsize=patchsize,
                         pretrain_dim=pretrain_dim, target_dim=target_dim, device=device,
                         top_pct=top_pct, sim_chunk=sim_chunk, score_metric=score_metric,
                         coreset_size=coreset_size, coreset_presample=coreset_presample,
                         bank_mode=bank_mode, obma_thresh=obma_thresh, obma_K=obma_K,
                         obma_target_score=obma_target_score, obma_temperature=obma_temperature)
        self.weight = weight
        self.tap = tap
        self.backbone = self._make_tag()
        self._build_model()
        self._build_preprocess()
        self._probe_and_set_dims()

    def _make_tag(self) -> str:
        digits = "".join(l[-1] for l in self.layers)
        if self.fuse == "patchcore":
            suf = f"_pc{self.patchsize}_e{self.pretrain_dim}_t{self.target_dim}"
        elif self.fuse == "concat_pool":
            suf = f"_cp{self.patchsize}_t{self.target_dim}"
        else:
            suf = f"_{self.fuse}" if (len(self.layers) > 1 and self.fuse != "concat") else ""
        return f"yolo_p{digits}_{self.tap}" + suf

    def _build_model(self) -> None:
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import Detect

        self.model = YOLO(self.weight).model.to(self.device).eval()
        det = next((m for m in self.model.modules() if isinstance(m, Detect)), None)
        if self.tap in ("neck", "cv3") and det is None:
            raise ValueError(f"tap={self.tap!r} requires a Detect head, but {self.weight} has none "
                             "(e.g. semantic-seg or classify model). Use tap='bb' instead.")

        for lname in self.layers:
            idx = _YOLO_LAYER_IDX[lname]
            if self.tap == "neck":
                # Detect's forward input = list[P3, P4, P5(, text)].
                def _make(i, k=lname):
                    def h(_m, inputs):
                        self._cached[k] = inputs[0][i]
                    return h
                det.register_forward_pre_hook(_make(idx))
            elif self.tap == "bb":
                # Pre-FPN backbone stage. Indices below assume yolo26/v8/v11 detection topology.
                stage = self.model.model[_YOLO_BB_LAYER[lname]]
                def _make_bb(k=lname):
                    def h(_m, _i, output):
                        self._cached[k] = output
                    return h
                stage.register_forward_hook(_make_bb())
            else:  # "cv3"
                def _make_cv3(k=lname, i=idx):
                    def h(_m, _i, output):
                        self._cached[k] = output
                    return h
                det.cv3[idx][1].register_forward_hook(_make_cv3())

    def _build_preprocess(self) -> None:
        # YOLO: simple resize + /255, no ImageNet norm.
        self.tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])


# ---------------------------------------------------------------------------
# torchvision ResNet (PatchCore default = WideResNet50 + layer2+3)
# ---------------------------------------------------------------------------


class AnomalyPatchCore(_MultiLayerAnomaly):
    """torchvision ResNet/WideResNet, ImageNet-pretrained, hooks on layer1..4 stages.

    Paper PatchCore config: ``arch=wide_resnet50_2, layers=['l2','l3'], fuse='patchcore',
    patchsize=3, pretrain_dim=1024, target_dim=1024, score_metric='l2'``.
    """

    def __init__(
        self,
        arch: str = "wide_resnet50_2",
        imgsz: int = 224,
        layers: str | list[str] = "l2",
        fuse: str = "concat",
        patchsize: int = 1,
        pretrain_dim: int = 1024,
        target_dim: int = 1024,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
        score_metric: str = "cosine",
        coreset_size: int | None = None,
        coreset_presample: int | None = None,
        bank_mode: str = "flat",
        obma_thresh: float = 0.4,
        obma_K: int = 15,
        obma_target_score: float = 0.2,
        obma_temperature: float | None = None,
    ) -> None:
        if isinstance(layers, str):
            layers = [layers]
        if not all(l in _RESNET_LAYER_ATTR for l in layers):
            raise ValueError(f"each layer must be in {_RESNET_LAYERS}, got {layers}")
        layers = sorted(set(layers), key=lambda l: _RESNET_LAYERS.index(l))
        super().__init__(imgsz=imgsz, layers=layers, fuse=fuse, patchsize=patchsize,
                         pretrain_dim=pretrain_dim, target_dim=target_dim, device=device,
                         top_pct=top_pct, sim_chunk=sim_chunk, score_metric=score_metric,
                         coreset_size=coreset_size, coreset_presample=coreset_presample,
                         bank_mode=bank_mode, obma_thresh=obma_thresh, obma_K=obma_K,
                         obma_target_score=obma_target_score, obma_temperature=obma_temperature)
        self.arch = arch
        self.backbone = self._make_tag()
        self._build_model()
        self._build_preprocess()
        self._probe_and_set_dims()

    def _make_tag(self) -> str:
        digits = "".join(l[-1] for l in self.layers)
        if self.fuse == "patchcore":
            suf = f"_pc{self.patchsize}_e{self.pretrain_dim}_t{self.target_dim}"
        elif self.fuse == "concat_pool":
            suf = f"_cp{self.patchsize}_t{self.target_dim}"
        else:
            suf = f"_{self.fuse}" if (len(self.layers) > 1 and self.fuse != "concat") else ""
        return f"{self.arch}_l{digits}{suf}"

    def _build_model(self) -> None:
        from torchvision import models
        ctor = getattr(models, self.arch)
        self.model = ctor(weights="DEFAULT").to(self.device).eval()
        for lname in self.layers:
            stage = getattr(self.model, _RESNET_LAYER_ATTR[lname])
            def _make(k=lname):
                def h(_m, _i, output):
                    self._cached[k] = output
                return h
            stage.register_forward_hook(_make())

    def _build_preprocess(self) -> None:
        self.tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_visualization(image_path: str, amap: np.ndarray, score: float, save_path: Path,
                       eval_size: int = 256, gt_mask: np.ndarray | None = None) -> None:
    """Save side-by-side viz: [original | heatmap | overlay (+ GT contour if given)]."""
    import cv2

    orig = cv2.imread(str(image_path))
    if orig is None:
        return
    orig = cv2.resize(orig, (eval_size, eval_size))

    # Normalize heatmap to [0, 255] uint8 then colorize (JET: blue=low, red=high).
    amap_n = amap - amap.min()
    if amap_n.max() > 0:
        amap_n = amap_n / amap_n.max()
    heatmap = cv2.applyColorMap((amap_n * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay: original × 0.5 + heatmap × 0.5.
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)

    # If GT mask provided, draw its outline on the overlay in green.
    if gt_mask is not None:
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    # Score label (white text with black outline for visibility on any background).
    txt = f"score={score:.4f}"
    cv2.putText(overlay, txt, (5, eval_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, txt, (5, eval_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    combined = cv2.hconcat([orig, heatmap, overlay])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), combined)


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------


def val_one_category(category: str, ad: AnomalyBase, eval_size: int = 256,
                     save_vis_dir: Path | None = None,
                     save_vis_samples_dir: Path | None = None,
                     vis_thresh: float = 0.4) -> dict:
    """Build bank from train/good, score every test image, compute image/pixel AUROC.

    If ``save_vis_dir`` is given, also dump per-image [orig|heatmap|overlay] triptychs to
    ``<save_vis_dir>/<category>/<good|anom>/<stem>.jpg``.

    If ``save_vis_samples_dir`` is given, dump exactly ONE normal + ONE anomaly viz per
    category to ``<save_vis_samples_dir>/<category>/{normal,anomaly}.jpg``:
      * normal  = the *good* test image with the lowest score (cleanest example)
      * anomaly = the *anomalous* test image with the highest score, only if > ``vis_thresh``
        (skipped if no anomaly crosses the threshold)
    """
    from sklearn.metrics import roc_auc_score

    # MPS allocator state from previous category can silently corrupt downstream ops.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    data = get_mvtec_yolo_data(category)
    ad.bank = None
    t0 = time.perf_counter()
    ad.load_support_set(data["train_im_list"])
    build_s = time.perf_counter() - t0
    print(f"[{category}] bank built in {build_s:.1f}s  "
          f"(support={len(data['train_im_list'])}, bank_size={ad.bank.shape[0]})", flush=True)

    t1 = time.perf_counter()
    img_s, img_y, pix_s, pix_y = [], [], [], []
    # Track best normal (lowest-score good) and best anomaly (highest-score anom > thresh)
    # for the sample-export mode.
    best_normal: tuple[float, str, np.ndarray] | None = None        # (score, im, hm)
    best_anomaly: tuple[float, str, np.ndarray, np.ndarray | None] | None = None  # (score, im, hm, gt)
    for im in data["test_im_list"]:
        score, hm = ad.predict_one(im, eval_size=eval_size)
        is_anom = im not in data["test_good_im_list"]
        img_s.append(score)
        img_y.append(int(is_anom))
        gt = None
        if is_anom:
            gt = AnomalyValidator._gt_mask_from_polygons(im, eval_size)
            if gt is not None:
                pix_s.append(hm.flatten())
                pix_y.append(gt.flatten())
        if save_vis_dir is not None:
            sub = "anom" if is_anom else "good"
            out = save_vis_dir / category / sub / f"{Path(im).stem}.jpg"
            save_visualization(im, hm, score, out, eval_size=eval_size, gt_mask=gt)
        # Track best representatives for the sample export.
        if save_vis_samples_dir is not None:
            if is_anom:
                if score > vis_thresh and (best_anomaly is None or score > best_anomaly[0]):
                    best_anomaly = (score, im, hm, gt)
            else:
                if best_normal is None or score < best_normal[0]:
                    best_normal = (score, im, hm)
    val_s = time.perf_counter() - t1

    # Dump the chosen normal + anomaly exemplars after the full pass.
    if save_vis_samples_dir is not None:
        cat_dir = save_vis_samples_dir / category
        if best_normal is not None:
            s, im, hm = best_normal
            save_visualization(im, hm, s, cat_dir / "normal.jpg", eval_size=eval_size, gt_mask=None)
            print(f"  vis-samples: normal  score={s:.4f}  {Path(im).name}", flush=True)
        if best_anomaly is not None:
            s, im, hm, gt = best_anomaly
            save_visualization(im, hm, s, cat_dir / "anomaly.jpg", eval_size=eval_size, gt_mask=gt)
            print(f"  vis-samples: anomaly score={s:.4f}  {Path(im).name}", flush=True)
        elif save_vis_samples_dir is not None:
            print(f"  vis-samples: anomaly SKIPPED — no test image scored > {vis_thresh}", flush=True)

    image_auroc = float(roc_auc_score(img_y, img_s)) if len(set(img_y)) > 1 else float("nan")
    pixel_auroc = float("nan")
    if pix_s:
        ps, pl = np.concatenate(pix_s), np.concatenate(pix_y)
        if pl.any() and not pl.all():
            pixel_auroc = float(roc_auc_score(pl, ps))

    return {
        "category": category,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "n_support": len(data["train_im_list"]),
        "n_test": len(data["test_im_list"]),
        "n_anom_with_gt": len(pix_s),
        "build_s": round(build_s, 1),
        "val_s":   round(val_s, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_RESNET_ARCH_MAP = {
    "wrn50": "wide_resnet50_2", "rn50": "resnet50", "rn18": "resnet18",
    "rn34": "resnet34", "rn101": "resnet101", "rn152": "resnet152",
}


def _build(args: argparse.Namespace) -> tuple[AnomalyBase, Path]:
    """Construct the requested model + return its default CSV path."""
    bb = args.backbone
    kw = dict(
        score_metric=args.score_metric,
        coreset_size=args.coreset, coreset_presample=args.coreset_presample,
        bank_mode=args.bank_mode,
        obma_thresh=args.obma_thresh, obma_K=args.obma_K,
        obma_target_score=args.obma_target_score, obma_temperature=args.obma_temp,
    )
    if bb.startswith("dino_"):
        short = bb.removeprefix("dino_")
        ad = AnomalyDINO(backbone=short, imgsz=args.imgsz or 448, **kw)
    elif bb.startswith("convnext_"):
        rest = bb.removeprefix("convnext_")
        if "_s" in rest:
            name, _, stages_str = rest.partition("_s")
            if not stages_str or not all(d in "234" for d in stages_str):
                raise ValueError(f"convnext stage digits must be ⊆ {{2,3,4}}, got {stages_str!r}")
            ad = AnomalyConvNeXtMulti(
                backbone=name, imgsz=args.imgsz or 320,
                layers=[f"s{d}" for d in stages_str], fuse=args.fuse,
                patchsize=args.patchsize, pretrain_dim=args.pretrain_dim, target_dim=args.target_dim,
                **kw,
            )
        else:
            ad = AnomalyConvNeXt(backbone=rest, imgsz=args.imgsz or 448, **kw)
    elif bb.startswith("yolo_p"):
        rest = bb.removeprefix("yolo_p")
        digits, _, tap = rest.partition("_")
        if not digits or not all(d in "345" for d in digits):
            raise ValueError(f"yolo backbone digits must be ⊆ {{3,4,5}}, got {digits!r}")
        ad = AnomalyYOLO(
            weight=args.weight, imgsz=args.imgsz or 640,
            layers=[f"P{d}" for d in digits], tap=tap or "neck", fuse=args.fuse,
            patchsize=args.patchsize, pretrain_dim=args.pretrain_dim, target_dim=args.target_dim,
            **kw,
        )
    elif "_l" in bb and bb.partition("_l")[0] in _RESNET_ARCH_MAP:
        arch_key, _, digits = bb.partition("_l")
        if not digits or not all(d in "1234" for d in digits):
            raise ValueError(f"resnet backbone digits must be ⊆ {{1,2,3,4}}, got {digits!r}")
        ad = AnomalyPatchCore(
            arch=_RESNET_ARCH_MAP[arch_key], imgsz=args.imgsz or 224,
            layers=[f"l{d}" for d in digits], fuse=args.fuse,
            patchsize=args.patchsize, pretrain_dim=args.pretrain_dim, target_dim=args.target_dim,
            **kw,
        )
    else:
        raise ValueError(f"unknown --backbone {bb!r}")

    tag = ad.backbone
    if isinstance(ad, AnomalyYOLO):
        tag = f"{tag}_{Path(args.weight).stem}_imgsz{ad.imgsz}"
    elif isinstance(ad, AnomalyPatchCore):
        tag = f"{tag}_imgsz{ad.imgsz}"
    elif isinstance(ad, (AnomalyConvNeXt, AnomalyConvNeXtMulti)):
        tag = f"{tag}_imgsz{ad.imgsz}"
    else:  # DINO
        tag = f"{tag}_imgsz{ad.imgsz}"

    if args.coreset:
        tag += f"_cs{args.coreset}" + (f"_pre{args.coreset_presample}" if args.coreset_presample else "")
    if args.score_metric != "cosine":
        tag += f"_{args.score_metric}"
    if args.bank_mode != "flat":
        tag += f"_{args.bank_mode}"
        if args.bank_mode == "obma":
            tag += f"_t{args.obma_thresh:g}_K{args.obma_K}_ts{args.obma_target_score:g}"
            if args.obma_temp is not None:
                tag += f"_T{args.obma_temp:g}"
    return ad, Path(f"./runs/temp/anomaly_{tag}_mvtec_metrics.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _yolo_digits = ("3", "4", "5", "34", "35", "45", "345")
    yolo_choices = [f"yolo_p{d}_{t}" for d in _yolo_digits for t in _YOLO_TAPS]
    _resnet_digits = ("1", "2", "3", "4", "12", "13", "14", "23", "24", "34", "123", "234", "1234")
    resnet_choices = [f"{a}_l{d}" for a in _RESNET_ARCH_MAP for d in _resnet_digits]

    _cn_sizes = list(_CONVNEXT_CONFIGS)
    _cn_stages = ("2", "3", "23")
    convnext_choices = [f"convnext_{k}" for k in _cn_sizes] + \
                       [f"convnext_{k}_s{s}" for k in _cn_sizes for s in _cn_stages]

    p.add_argument("--backbone", default="dino_vits14",
                   choices=[f"dino_{k}" for k in _DINOV2_HUB] + convnext_choices + yolo_choices + resnet_choices,
                   help="Feature extractor. dino_*: DINOv2. convnext_*[_s<digits>]: DINOv3-distilled ConvNeXt "
                        "(single or multi-stage). yolo_p<digits>[_<tap>]: YOLO. "
                        "{wrn50,rn18,rn34,rn50,rn101,rn152}_l<digits>: torchvision ResNet.")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Input size. Defaults: 448 (dino) / 640 (yolo) / 224 (resnet).")
    p.add_argument("--weight",
                   default="/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa/26m_mergedata_v3/weights/best.pt",
                   help="YOLO checkpoint (only used for yolo_* backbones).")
    p.add_argument("--category", default=None,
                   help=f"Comma-separated subset of {MVTEC_CATEGORIES}. Default: all.")
    p.add_argument("--fuse", choices=_FUSE_CHOICES, default="concat",
                   help="Multi-layer fusion. concat=channel-cat (dim grows); sum/avg=elementwise "
                        "(equal-dim layers); patchcore=2-stage adaptive pool (paper); "
                        "concat_pool=1-stage adaptive pool.")
    p.add_argument("--patchsize", type=int, default=3,
                   help="Spatial unfold (PatchCore paper default = 3). Only used with patchcore/concat_pool.")
    p.add_argument("--pretrain_dim", type=int, default=1024,
                   help="Per-layer adaptive_avg_pool target (patchcore only). Paper default = 1024.")
    p.add_argument("--target_dim", type=int, default=1024,
                   help="Final per-patch dim after cross-layer adaptive_avg_pool. Paper default = 1024.")
    p.add_argument("--score_metric", choices=_SCORE_CHOICES, default="cosine",
                   help="1-NN metric: cosine (default) or l2 (PatchCore paper).")
    p.add_argument("--coreset", type=int, default=None,
                   help="Compact bank to N points via greedy FPS (PatchCore-style coreset).")
    p.add_argument("--coreset_presample", type=int, default=None,
                   help="Random pre-sample to this size BEFORE FPS. Only with --coreset.")
    p.add_argument("--bank-mode", choices=_BANK_MODES, default="flat",
                   help="Bank construction & scoring: 'flat' = dump-all + 1-NN cosine top-k%% mean "
                        "(default). 'obma' = OBMA accept/reject + Noisy-OR max scoring "
                        "(scores ∈ [0,1], auto-calibrated so typical-normal ≈ target_score).")
    p.add_argument("--obma-thresh", type=float, default=0.4,
                   help="OBMA accumulation threshold: features scored above this against the "
                        "current bank are accepted (default 0.4). Only used with --bank-mode obma.")
    p.add_argument("--obma-K", type=int, default=15,
                   help="Top-K nearest bank entries used for Noisy-OR scoring (default 15).")
    p.add_argument("--obma-target-score", type=float, default=0.2,
                   help="Auto-calibration target: solve β so a typical-normal position scores at this "
                        "value (default 0.2). Must be < --obma-thresh.")
    p.add_argument("--obma-temp", type=float, default=None,
                   help="Manually set Noisy-OR temperature β (skips auto-calibration). Useful for "
                        "reproducibility or tuning. Reasonable range: 1–10.")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path.")
    p.add_argument("--save-vis", type=Path, default=None,
                   help="If set, save per-image [orig|heatmap|overlay] triptych to this dir "
                        "(grouped by <category>/<good|anom>/). GT polygon outline drawn in green on overlay.")
    p.add_argument("--save-vis-samples", type=Path, default=None,
                   help="If set, save exactly ONE normal + ONE anomaly viz per category to "
                        "<dir>/<category>/{normal,anomaly}.jpg. Normal = good with lowest score; "
                        "anomaly = anomalous with highest score > --vis-thresh (skipped if none).")
    p.add_argument("--vis-thresh", type=float, default=0.4,
                   help="Minimum score for an anomaly to be eligible as the per-category exemplar "
                        "(default 0.4 — matches OBMA's accumulate_thresh default).")
    args = p.parse_args()

    if args.category:
        cats = [c.strip() for c in args.category.split(",") if c.strip()]
        unknown = [c for c in cats if c not in MVTEC_CATEGORIES]
        if unknown:
            raise SystemExit(f"unknown category: {unknown}. Valid: {MVTEC_CATEGORIES}")
    else:
        cats = list(MVTEC_CATEGORIES)

    ad, out_csv = _build(args)
    if args.out is not None:
        out_csv = args.out
    elif len(cats) < len(MVTEC_CATEGORIES):
        out_csv = out_csv.with_name(out_csv.stem + f"_{'_'.join(cats)}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"ready: backbone={ad.backbone} imgsz={ad.imgsz} device={ad.device} "
          f"feat_dim={ad.feat_dim} grid={ad.grid} score={ad.score_metric}"
          + (f" coreset={args.coreset}" if args.coreset else ""))

    rows: list[dict] = []
    for cat in cats:
        print(f"\n{'=' * 60}\n[{cat}] starting\n{'=' * 60}", flush=True)
        try:
            row = val_one_category(cat, ad,
                                   save_vis_dir=args.save_vis,
                                   save_vis_samples_dir=args.save_vis_samples,
                                   vis_thresh=args.vis_thresh)
        except Exception as e:
            print(f"[{cat}] FAILED: {e!r}", flush=True)
            row = {"category": cat, "image_auroc": float("nan"), "pixel_auroc": float("nan"),
                   "n_support": 0, "n_test": 0, "n_anom_with_gt": 0,
                   "build_s": 0.0, "val_s": 0.0}
        rows.append(row)
        print(f"[{cat}] image={row['image_auroc']:.4f} pixel={row['pixel_auroc']:.4f} "
              f"build={row['build_s']:.1f}s val={row['val_s']:.1f}s "
              f"(support={row['n_support']}, test={row['n_test']}, anom_w_gt={row['n_anom_with_gt']})",
              flush=True)

    def _avg(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r[key], float) and r[key] == r[key]]
        return statistics.fmean(vals) if vals else float("nan")

    avg_row = {
        "category": "AVERAGE",
        "image_auroc": _avg("image_auroc"),
        "pixel_auroc": _avg("pixel_auroc"),
        "n_support": sum(r["n_support"] for r in rows),
        "n_test": sum(r["n_test"] for r in rows),
        "n_anom_with_gt": sum(r["n_anom_with_gt"] for r in rows),
        "build_s": round(sum(r["build_s"] for r in rows), 1),
        "val_s":   round(sum(r["val_s"]   for r in rows), 1),
    }

    fields = ["category", "image_auroc", "pixel_auroc", "build_s", "val_s",
              "n_support", "n_test", "n_anom_with_gt"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows + [avg_row]:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"\nSaved {len(rows) + 1} rows ({len(rows)} categories + 1 AVERAGE) → {out_csv}")
    print("\n=== summary ===")
    for r in rows + [avg_row]:
        print(f"  {r['category']:>12s}  img={r['image_auroc']:.4f}  pix={r['pixel_auroc']:.4f}  "
              f"build={r['build_s']:>6.1f}s  val={r['val_s']:>6.1f}s")


if __name__ == "__main__":
    main()
