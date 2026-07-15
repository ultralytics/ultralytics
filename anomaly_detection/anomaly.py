"""Training-free patch-NN anomaly detection with swappable backbone.

Architecture
============

::

    AnomalyBase                 # Full pipeline: feature extraction + fusion, bank, calibration, scoring.
     ├── AnomalyDINO            # DINOv2 ViT tokens.
     ├── AnomalyConvNeXt        # DINOv3-distilled ConvNeXt tokens (single stage).
     ├── AnomalyConvNeXtMulti   # ConvNeXt multi-stage hooks (s2/s3) + fusion.
     ├── AnomalyYOLO            # YOLO/YOLOE backbone, taps {bb, neck, cv3}.
     └── AnomalyPatchCore       # torchvision ResNet/WideResNet, layer1..4.

Every backbone inherits ``AnomalyBase`` directly and overrides just the section-1 hooks
(see ``AnomalyBase`` docstring). The rest is shared:

1. **Backbone + extraction** — ``_build_backbone`` (build model + register hooks); token
   backbones also override ``_forward_features`` (reshape ViT tokens to BCHW). Layer choice,
   multi-layer fusion (``fuse=``), grid/dim probing, and ``_build_preprocess`` are shared.
2. **Bank construction + calibration** — ``load_support_set``: dump every patch, k-center
   coreset (dropped rows = normal hold-out), then compactness + hold-out calibration of the
   cos-space sigmoid Noisy-OR scorer. Ported from YOLOA BackboneMemoryBank.
3. **Scoring**           — ``_score_bank`` (cos-space sigmoid Noisy-OR, ∈ [0,1], max over
   positions as image score) + ``_upsample_pixel_map`` (bilinear grid → pixel map).

Usage
=====

::

    python tools/anomaly_dino.py --backbone dino_vits14
    python tools/anomaly_dino.py --backbone yolo_p4_bb --weight yoloe-26l-seg.pt --coreset 10000
    python tools/anomaly_dino.py --backbone wrn50_l23 --fuse patchcore --patchsize 3

Caveats
=======

* MPS adaptive_avg_pool1d on non-divisible (in, out) sizes is unimplemented → we run pool on CPU.
* MPS allocator can silently corrupt large/high-dim banks across categories → ``empty_cache`` between cats.
* MPS argmin on large/duplicate-heavy banks gives wrong results → FPS runs on CPU.
"""
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from block import DINOv3ConvNeXt, load_convnext_sd
from mvtec_yolo import CATEGORIES as MVTEC_CATEGORIES, get_mvtec_yolo_data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DINOV2_HUB = {  # short → (torch.hub entry, feat_dim)
    "dinov2_vits14": ("dinov2_vits14_reg", 384),
    "dinov2_vitb14": ("dinov2_vitb14_reg", 768),
    "dinov2_vitl14": ("dinov2_vitl14_reg", 1024),
}
_DINO_PATCH = 14
_DINOV3_PATCH = 16
_DINOV3_HUB = {  # short → (hub entry, feat_dim, checksum)
    "vits16":      ("dinov3_vits16",      384,  "08c60483"),
    "vits16plus":  ("dinov3_vits16plus",  384,  "4057cbaa"),
    "vitb16":      ("dinov3_vitb16",      768,  "73cec8be"),
    "vitl16":      ("dinov3_vitl16",      1024, "8aa4cbdd"),
    "vith16plus":  ("dinov3_vith16plus",  1280, "7c1da9a5"),
}
_DINOV3_REPO = str(Path(__file__).parent / "dinov3")
_DINOV3_WEIGHTS_DIR = Path(__file__).parent / "dinov3_weights"
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
_SCORE_CHUNK_ELEMS = 1 << 27  # max [query, bank] elements per similarity slice (scoring memory knob)

# ---------------------------------------------------------------------------
# Memory-bank helpers — k-center coreset + pluggable scoring/calibration.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _coreset_kcenter(mem: torch.Tensor, max_size: int, return_indices: bool = False):
    """Batched greedy k-center coreset on L2-normalised features (cosine distance).

    Selects ``BATCH`` farthest points per iteration and updates min-distances against
    all of them in one GEMM, amortising the bank read — a small greediness/speed
    trade that wins on memory-bandwidth-limited devices. Returns the coreset, or
    ``(coreset, indices)`` when ``return_indices``.
    """
    M = mem.shape[0]
    if M <= max_size:
        idx = torch.arange(M, device=mem.device)
        return (mem, idx) if return_indices else mem
    device = mem.device
    BATCH = 64  # centres per GEMM call
    selected: list[int] = []
    mean = mem.mean(dim=0)
    mean = mean / mean.norm().clamp(min=1e-8)
    seed = int((mem @ mean).argmax().item())
    selected.append(seed)
    dist = (1.0 - (mem @ mem[seed])).clamp(min=0.0)  # seed distance so first top-k isn't all-inf
    t0 = time.perf_counter()
    while len(selected) < max_size:
        k = min(BATCH, max_size - len(selected))
        _, top_idx = dist.topk(k)
        selected.extend(top_idx.tolist())
        new_dist = (1.0 - mem @ mem[top_idx].t()).clamp(min=0.0).min(dim=1).values
        dist = torch.minimum(dist, new_dist)
        if len(selected) % 2000 < BATCH:
            print(f"  coreset: {len(selected)}/{max_size}  elapsed={time.perf_counter() - t0:.1f}s")
    sel = torch.tensor(selected, device=device)
    return (mem[sel], sel) if return_indices else mem[sel]


# ── Scoring: Noisy-OR ──────────────────────────────────────────────────

@torch.inference_mode()
def _score_noisy_or(features: torch.Tensor, bank: torch.Tensor, K: int,
                    state: dict) -> torch.Tensor:
    """Cos-space sigmoid Noisy-OR anomaly scores ∈ [0, 1] per query position."""
    if bank.numel() == 0 or bank.shape[0] == 0:
        return torch.full((features.shape[0],), 0.5, device=features.device, dtype=features.dtype)
    beta, threshold = state["beta"], state["threshold"]
    k = min(K, bank.shape[0])
    n, m = features.shape[0], bank.shape[0]
    chunk = max(1, _SCORE_CHUNK_ELEMS // max(m, 1))
    out: list[torch.Tensor] = []
    for i in range(0, n, chunk):
        cos = features[i:i + chunk] @ bank.t()
        psi = torch.sigmoid(beta * (cos - threshold))
        topk = psi.topk(k=k, dim=1).values
        log_prob = torch.log((1.0 - topk).clamp(min=1e-8)).mean(dim=1)
        out.append(torch.exp(log_prob))
    return torch.cat(out).clamp(0.0, 1.0).to(features.dtype)


def _calibrate_noisy_or(bank: torch.Tensor, holdout: torch.Tensor | None,
                        K: int, temperature: float, target_score: float,
                        target_quantile: float, device: str) -> dict:
    """Compactness + hold-out calibration → ``{beta, threshold}``."""
    # Compactness
    k = min(K, bank.shape[0])
    n_sample = min(512, bank.shape[0])
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(bank.shape[0], generator=g)[:n_sample].to(bank.device)
    sim = bank[idx] @ bank.t()
    sim[torch.arange(n_sample, device=bank.device), idx] = -1.0
    compactness = sim.topk(k=k, dim=1).values.mean().clamp(0.0, 1.0 - 1e-4).item()

    beta = float(temperature)
    logit = math.log(max((1.0 - target_score) / max(target_score, 1e-6), 1e-6))
    threshold = compactness - logit / max(beta, 0.1)

    if holdout is not None and holdout.shape[0] > 0:
        holdout = holdout[:5000].to(device)
        beta, threshold = _refine_noisy_or(holdout, bank, k, compactness,
                                           beta, target_score, target_quantile)
    return {"beta": beta, "threshold": threshold}


@torch.inference_mode()
def _refine_noisy_or(holdout: torch.Tensor, bank: torch.Tensor, K: int,
                     compactness: float, beta0: float, target: float,
                     target_q: float) -> tuple[float, float]:
    """Gaussian-tail hold-out refinement → ``(best_beta, best_threshold)``."""
    k = min(K, bank.shape[0])
    topk_cos = (F.normalize(holdout, p=2, dim=1) @ bank.t()).topk(k=k, dim=1).values
    z_q = math.sqrt(2) * torch.erfinv(torch.tensor(2.0 * target_q - 1.0)).item()

    def scores_for(beta: float, thresh: float) -> torch.Tensor:
        psi = torch.sigmoid(beta * (topk_cos - thresh))
        return torch.exp(torch.log((1.0 - psi).clamp(min=1e-8)).mean(dim=1)).clamp(0.0, 1.0)

    def tail(s: torch.Tensor) -> float:
        return s.mean().item() + z_q * s.std().item()

    beta_candidates = [beta0]
    for i in range(1, 9):
        b = beta0 * (10 ** (i / 8))
        if b > 100:
            break
        beta_candidates.append(round(b, 4))

    def find_thresh(beta: float):
        hr = 3.0 / max(beta, 0.1)
        lo, hi = compactness - hr, compactness + hr
        s_lo, s_hi = tail(scores_for(beta, lo)), tail(scores_for(beta, hi))
        for _ in range(5):
            if s_lo > target and lo > compactness - 5.0:
                lo -= hr
                s_lo = tail(scores_for(beta, lo))
            if s_hi < target and hi < compactness + 5.0:
                hi += hr
                s_hi = tail(scores_for(beta, hi))
        if not (s_lo <= target <= s_hi):
            return None, float("inf")
        for _ in range(20):
            mid = (lo + hi) / 2.0
            if tail(scores_for(beta, mid)) > target:
                hi = mid
            else:
                lo = mid
        th = (lo + hi) / 2.0
        return th, scores_for(beta, th).std().item()

    best_beta, best_thresh, best_spread = beta0, compactness, float("inf")
    for beta in beta_candidates:
        th, spread = find_thresh(beta)
        if th is not None and spread < best_spread:
            best_spread, best_beta, best_thresh = spread, beta, th
    return best_beta, best_thresh


# ── Scoring: 1-NN ──────────────────────────────────────────────────────

@torch.inference_mode()
def _score_1nn(features: torch.Tensor, bank: torch.Tensor, K: int,
               state: dict) -> torch.Tensor:
    """1-NN anomaly scores: 1 − max cosine similarity to the bank."""
    if bank.numel() == 0 or bank.shape[0] == 0:
        return torch.full((features.shape[0],), 0.5, device=features.device, dtype=features.dtype)
    cos = features @ bank.t()
    return (1.0 - cos.max(dim=1).values).clamp(0.0, 1.0).to(features.dtype)


def _calibrate_1nn(bank: torch.Tensor, holdout: torch.Tensor | None,
                   K: int, temperature: float, target_score: float,
                   target_quantile: float, device: str) -> dict:
    """1-NN is calibration-free."""
    return {}


_SCORERS = {
    "noisy_or": (_score_noisy_or, _calibrate_noisy_or),
    "1nn":      (_score_1nn,      _calibrate_1nn),
}


def _tokens_to_bchw(tok: torch.Tensor) -> torch.Tensor:
    """Square-grid patch tokens ``[B, N, D]`` → feature map ``[B, D, h, w]`` (h=w=√N)."""
    B, N, D = tok.shape
    h = int(round(N ** 0.5))
    if h * h != N:
        raise ValueError(f"non-square token grid: N={N}")
    return tok.transpose(1, 2).reshape(B, D, h, h)


# ---------------------------------------------------------------------------
# Base class — backbone-agnostic pipeline: extraction+fusion + bank + scoring
# ---------------------------------------------------------------------------


class AnomalyBase:
    """Training-free patch-NN anomaly detector. Subclass to plug in a backbone.

    A backbone is plugged in by overriding just the two hooks in section 1; everything
    else (layer selection, multi-layer fusion, grid/dim probing, preprocessing, bank
    construction, calibration, scoring) is shared here.

    * **Backbone + feature extraction**:
      - ``_build_backbone`` (required): build ``self.model``, register any hooks into
        ``self._cached``, set ``self.backbone`` tag.
      - ``_forward_features`` (optional): run the model → ``{name: [B, C, H, W]}``. Default
        returns the hook-cached maps; token backbones (ViT) override it to reshape their
        ``[B, N, D]`` tokens to BCHW via ``_tokens_to_bchw``.
      - ``_build_preprocess`` (optional): default = ImageNet norm; override for e.g. YOLO.
      Layer choice / single-vs-multi / fusion is config: ``layers=``, ``fuse=`` (concat /
      sum / avg / patchcore / concat_pool), ``patchsize/pretrain_dim/target_dim``.

    * **Bank construction + calibration** (``load_support_set``, ``_filter_tokens`` hook):
      dump every support patch, k-center coreset to ``coreset_size`` (keeping the dropped
      features as a normal hold-out), then calibrate the cos-space sigmoid Noisy-OR scorer
      (compactness threshold, refined on the hold-out). Ported from YOLOA BackboneMemoryBank.

    * **Scoring** (``_score_bank``, ``_upsample_pixel_map``): per-patch cos-space sigmoid
      Noisy-OR probability ∈ [0, 1], max over positions as the image score, bilinear
      upsample of the patch grid as the pixel map.
    """

    def __init__(
        self,
        imgsz: int,
        device: str | None = None,
        *,
        layers: str | list[str] | None = None,
        fuse: str = "concat",
        patchsize: int = 1,
        pretrain_dim: int = 1024,
        target_dim: int = 1024,
        coreset_size: int | None = 10000,
        K: int = 5,
        scoring: str = "noisy_or",
        temperature: float = 3.0,
        target_score: float = 0.2,
        target_quantile: float = 0.95,
    ) -> None:
        if fuse not in _FUSE_CHOICES:
            raise ValueError(f"fuse must be in {_FUSE_CHOICES}, got {fuse!r}")
        if patchsize % 2 == 0:
            raise ValueError(f"patchsize must be odd, got {patchsize}")
        if scoring not in _SCORERS:
            raise ValueError(f"scoring must be in {list(_SCORERS)}, got {scoring!r}")
        self.imgsz = imgsz
        # Feature-extraction config (consumed by the shared _extract / fusion below).
        self.layers = None if layers is None else ([layers] if isinstance(layers, str) else list(layers))
        self.fuse = fuse                                # concat / sum / avg / patchcore / concat_pool
        self.patchsize = patchsize                      # spatial unfold for patchcore / concat_pool
        self.pretrain_dim = pretrain_dim
        self.target_dim = target_dim
        self._cached: dict[str, torch.Tensor] = {}      # hooks write BCHW maps here
        # Bank / scoring config.
        self.coreset_size = coreset_size                # k-center bank cap (None = keep all patches)
        self.K = K                                      # top-K neighbours per query
        self.scoring = scoring                          # "noisy_or" | "1nn"
        self.temperature = float(temperature)           # β floor / starting point (noisy_or only)
        self.target_score = float(target_score)         # a typical-normal query scores here
        self.target_quantile = float(target_quantile)   # hold-out quantile placed at target_score
        self._score_state: dict = {}                    # opaque state set by calibration
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.bank: torch.Tensor | None = None
        # Setup is orchestrated here; subclasses only fill the backbone hooks below.
        self._build_backbone()      # build self.model (+ hooks), set self.backbone
        self._build_preprocess()    # set self.tf
        self._probe_and_set_dims()  # discover self.grid / self.feat_dim (and default self.layers)

    # ── 1. Backbone + feature extraction ────────────────────────────────
    # Subclass MUST implement _build_backbone; MAY override _forward_features (token backbones)
    # and _build_preprocess. Layer selection, multi-layer fusion, grid/dim probing are all shared.
    def _build_backbone(self) -> None:
        """Subclass: construct ``self.model``, register any hooks into ``self._cached``, set ``self.backbone``."""
        raise NotImplementedError

    def _forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the backbone → tapped feature maps ``{name: [B, C, H, W]}``.

        Default = hook-based: run the model and return whatever the registered hooks cached.
        Token backbones (ViT) override this to reshape their ``[B, N, D]`` tokens to BCHW.
        """
        self._cached = {}
        self.model(x.to(self.device))
        return self._cached

    def _build_preprocess(self) -> None:
        """Default: resize + ImageNet normalisation. Override for e.g. YOLO (/255, no norm)."""
        self.tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def _load(self, path: str) -> torch.Tensor:
        return self.tf(Image.open(path).convert("RGB"))

    def _probe_and_set_dims(self) -> None:
        """Dummy forward to discover ``self.grid``, ``self.feat_dim`` (and default ``self.layers``)."""
        with torch.inference_mode():
            maps = self._forward_features(torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device))
        if self.layers is None:
            self.layers = list(maps.keys())
        missing = [l for l in self.layers if l not in maps]
        assert not missing, f"feature maps missing for {missing} (got {list(maps.keys())})"
        ref = maps[self.layers[0]]
        self.grid = (ref.shape[2], ref.shape[3])
        dims = [maps[l].shape[1] for l in self.layers]
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
        """Backbone features → fused patch tokens ``[B, H*W, D]`` (shared by every backbone)."""
        maps = self._forward_features(x.to(self.device))
        H, W = self.grid
        B = x.shape[0]
        if self.fuse == "patchcore":
            return self._fuse_patchcore(maps, B, H, W)
        if self.fuse == "concat_pool":
            return self._fuse_concat_pool(maps, B, H, W)
        return self._fuse_simple(maps, B, H, W)

    def _fuse_simple(self, maps: dict, B: int, H: int, W: int) -> torch.Tensor:
        """concat / sum / avg over upsampled, channel-aligned feature maps."""
        feats = []
        for lname in self.layers:
            f = maps[lname]
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

    def _fuse_patchcore(self, maps: dict, B: int, H: int, W: int) -> torch.Tensor:
        """unfold + per-layer adaptive_avg_pool → concat → adaptive_avg_pool to target_dim (PatchCore)."""
        ps, pad = self.patchsize, self.patchsize // 2
        pool_dev = "cpu" if self.device == "mps" else self.device
        per_layer = []
        for lname in self.layers:
            f = maps[lname]
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            patches = F.unfold(f, kernel_size=ps, stride=1, padding=pad)        # [B, C*ps^2, H*W]
            patches = patches.transpose(1, 2).reshape(B * H * W, -1)            # [B*N, C*ps^2]
            pooled = F.adaptive_avg_pool1d(patches.to(pool_dev).unsqueeze(1), self.pretrain_dim).squeeze(1)
            per_layer.append(pooled)
        merged = torch.cat(per_layer, dim=-1) if len(per_layer) > 1 else per_layer[0]
        final = F.adaptive_avg_pool1d(merged.unsqueeze(1), self.target_dim).squeeze(1)
        return final.to(self.device).reshape(B, H * W, self.target_dim)

    def _fuse_concat_pool(self, maps: dict, B: int, H: int, W: int) -> torch.Tensor:
        """unfold + concat across layers + single adaptive_avg_pool to target_dim (no per-layer pool)."""
        ps, pad = self.patchsize, self.patchsize // 2
        pool_dev = "cpu" if self.device == "mps" else self.device
        feats_list = []
        for lname in self.layers:
            f = maps[lname]
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            patches = F.unfold(f, kernel_size=ps, stride=1, padding=pad)
            patches = patches.transpose(1, 2).reshape(B * H * W, -1)
            feats_list.append(patches)
        merged = torch.cat(feats_list, dim=-1) if len(feats_list) > 1 else feats_list[0]
        final = F.adaptive_avg_pool1d(merged.to(pool_dev).unsqueeze(1), self.target_dim).squeeze(1)
        return final.to(self.device).reshape(B, H * W, self.target_dim)

    # ── 2. Bank construction + calibration ──────────────────────────────
    @torch.inference_mode()
    def load_support_set(self, image_paths: list[str], batch: int = 8) -> None:
        """Build the memory bank from normal images, then calibrate the scorer."""
        feats: list[torch.Tensor] = []
        for i in range(0, len(image_paths), batch):
            xs = torch.stack([self._load(p) for p in image_paths[i:i + batch]], dim=0)
            tokens = self._extract(xs).reshape(-1, self.feat_dim).cpu()  # keep big intermediate on CPU
            tokens = self._filter_tokens(tokens)
            feats.append(tokens)
        mem = F.normalize(torch.cat(feats, dim=0), dim=-1)  # [M, D] on CPU

        # Coreset on CPU (MPS argmin is unreliable on large banks); keep dropped rows as hold-out.
        holdout: torch.Tensor | None = None
        if self.coreset_size and self.coreset_size < mem.shape[0]:
            bank, sel = _coreset_kcenter(mem, self.coreset_size, return_indices=True)
            mask = torch.ones(mem.shape[0], dtype=torch.bool)
            mask[sel] = False
            holdout = mem[mask]
            mem = bank
        self.bank = mem.to(self.device)

        _, calibrate = _SCORERS[self.scoring]
        self._score_state = calibrate(
            self.bank, holdout, self.K,
            self.temperature, self.target_score, self.target_quantile, self.device,
        )
        print(f"  bank={self.bank.shape[0]} scoring={self.scoring} "
              f"state={ {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in self._score_state.items()} }",
              flush=True)

    def _filter_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hook: drop / reweight patch tokens before banking. Default = identity.

        Override for tricks like AnomalyDINO's PCA foreground masking on object classes,
        or to skip border patches with low gradient.
        """
        return tokens

    # ── 3. Scoring ──────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict_one(self, image_path: str, eval_size: int | tuple[int, int] = 256) -> tuple[float, np.ndarray]:
        """Score one test image → ``(image_score, pixel_map)``.

        If ``eval_size`` is an int the pixel map is square; a ``(H, W)`` tuple
        gives the exact output dimensions (used by bbox evaluation).
        """
        if self.bank is None:
            raise RuntimeError("call load_support_set() before predict_one()")
        score_fn, _ = _SCORERS[self.scoring]
        x = self._load(image_path).unsqueeze(0)
        feats = F.normalize(self._extract(x).squeeze(0), dim=-1)   # [N, D]
        probs = score_fn(feats, self.bank, self.K, self._score_state)  # [N] ∈ [0,1]
        image_score = float(probs.max().item())
        amap = self._upsample_pixel_map(probs, eval_size)
        return image_score, amap

    def _upsample_pixel_map(self, scores: torch.Tensor, eval_size: int | tuple[int, int]) -> np.ndarray:
        H, W = self.grid
        amap = scores.reshape(1, 1, H, W).float()
        size = (eval_size, eval_size) if isinstance(eval_size, int) else eval_size
        amap = F.interpolate(amap, size=size, mode="bilinear", align_corners=False)
        return amap.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# DINOv2 — single layer, no fusion
# ---------------------------------------------------------------------------


class AnomalyDINO(AnomalyBase):
    """DINOv2 ViT + memory-bank anomaly detector (Damm et al., WACV 2025).

    Single layer (last block patch tokens), no multi-layer fusion. PCA foreground mask
    omitted for minimal baseline — override ``_filter_tokens`` to add it back.
    """

    def __init__(self, backbone: str = "dinov2_vits14", imgsz: int = 448, **kw) -> None:
        if backbone not in _DINOV2_HUB:
            raise ValueError(f"backbone must be in {list(_DINOV2_HUB)}, got {backbone!r}")
        if imgsz % _DINO_PATCH != 0:
            raise ValueError(f"imgsz must be divisible by {_DINO_PATCH}, got {imgsz}")
        self.dino_short = backbone
        super().__init__(imgsz=imgsz, **kw)

    def _build_backbone(self) -> None:
        hub_name, _ = _DINOV2_HUB[self.dino_short]
        self.model = torch.hub.load("facebookresearch/dinov2", hub_name).to(self.device).eval()
        self.backbone = self.dino_short

    def _forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tok = self.model.forward_features(x.to(self.device))["x_norm_patchtokens"]  # [B, N, D]
        return {"tok": _tokens_to_bchw(tok)}


class AnomalyConvNeXt(AnomalyBase):
    """DINOv3-distilled ConvNeXt + memory-bank anomaly detector.

    Single layer (stage 3, stride 32), same scoring as AnomalyDINO.
    """

    def __init__(self, backbone: str = "small", imgsz: int = 448,
                 weight: str | Path | None = None, **kw) -> None:
        if backbone not in _CONVNEXT_CONFIGS:
            raise ValueError(f"backbone must be in {list(_CONVNEXT_CONFIGS)}, got {backbone!r}")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be divisible by 32, got {imgsz}")
        self.convnext_name = backbone
        self._weight = weight
        super().__init__(imgsz=imgsz, **kw)

    def _build_backbone(self) -> None:
        depths, dims = _CONVNEXT_CONFIGS[self.convnext_name]
        self.model = DINOv3ConvNeXt(depths, dims).to(self.device).eval()
        self.model.load_state_dict(load_convnext_sd(self.convnext_name, self._weight, _CONVNEXT_WEIGHTS), strict=True)
        self.backbone = f"convnext_{self.convnext_name}"

    def _forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tok = self.model.forward_features(x.to(self.device))["x_norm_patchtokens"]  # [B, N, D]
        return {"tok": _tokens_to_bchw(tok)}


# ---------------------------------------------------------------------------
# DINOv3 ViT — torch.hub.load from local dinov3 repo
# ---------------------------------------------------------------------------


class AnomalyDINOv3(AnomalyBase):
    """DINOv3 ViT + memory-bank anomaly detector.

    Single layer (last block patch tokens), same scoring as AnomalyDINO.
    Supports vits16 / vits16plus / vitb16 / vitl16 / vith16plus.
    """

    def __init__(self, backbone: str = "vits16", imgsz: int = 448, **kw) -> None:
        if backbone not in _DINOV3_HUB:
            raise ValueError(f"backbone must be in {list(_DINOV3_HUB)}, got {backbone!r}")
        if imgsz % _DINOV3_PATCH != 0:
            raise ValueError(f"imgsz must be divisible by {_DINOV3_PATCH}, got {imgsz}")
        self.dinov3_short = backbone
        super().__init__(imgsz=imgsz, **kw)

    def _build_backbone(self) -> None:
        hub_name, _, checksum = _DINOV3_HUB[self.dinov3_short]
        weight_path = _DINOV3_WEIGHTS_DIR / f"dinov3_{self.dinov3_short}_pretrain_lvd1689m-{checksum}.pth"
        self.model = torch.hub.load(_DINOV3_REPO, hub_name, source="local",
                                    pretrained=False).to(self.device).eval()
        sd = torch.load(str(weight_path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd, strict=True)
        self.backbone = f"dinov3_{self.dinov3_short}"

    def _forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tok = self.model.forward_features(x.to(self.device))["x_norm_patchtokens"]  # [B, N, D]
        return {"tok": _tokens_to_bchw(tok)}


# ---------------------------------------------------------------------------
# ConvNeXt multi-layer — hooks on stage 2+3, PatchCore fusion
# ---------------------------------------------------------------------------

class AnomalyConvNeXtMulti(AnomalyBase):
    """DINOv3 ConvNeXt with multi-layer fusion (PatchCore-style).

    Hooks stage 2 (stride 16) and stage 3 (stride 32). Layer names are "s2", "s3".
    """

    def __init__(self, backbone: str = "small", imgsz: int = 320, layers: str | list[str] = "s2",
                 fuse: str = "patchcore", patchsize: int = 3,
                 weight: str | Path | None = None, **kw) -> None:
        if backbone not in _CONVNEXT_CONFIGS:
            raise ValueError(f"backbone must be in {list(_CONVNEXT_CONFIGS)}, got {backbone!r}")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be divisible by 32, got {imgsz}")
        layers = [layers] if isinstance(layers, str) else list(layers)
        if not all(l in {"s2", "s3"} for l in layers):
            raise ValueError(f"layers must be subset of {{'s2','s3'}}, got {layers}")
        self.convnext_name = backbone
        self._weight = weight
        super().__init__(imgsz=imgsz, layers=sorted(set(layers)), fuse=fuse, patchsize=patchsize, **kw)

    def _make_tag(self) -> str:
        digits = "".join(l[-1] for l in self.layers)
        if self.fuse == "patchcore":
            suf = f"_pc{self.patchsize}_e{self.pretrain_dim}_t{self.target_dim}"
        else:
            suf = f"_{self.fuse}" if len(self.layers) > 1 else ""
        return f"convnext_{self.convnext_name}_s{digits}{suf}"

    def _build_backbone(self) -> None:
        depths, dims = _CONVNEXT_CONFIGS[self.convnext_name]
        self.model = DINOv3ConvNeXt(depths, dims).to(self.device).eval()
        self.model.load_state_dict(load_convnext_sd(self.convnext_name, self._weight, _CONVNEXT_WEIGHTS), strict=True)
        _STAGE_IDX = {"s2": 2, "s3": 3}
        for lname in self.layers:
            def _make(k=lname):
                def h(_m, _i, output):
                    self._cached[k] = output
                return h
            self.model.stages[_STAGE_IDX[lname]].register_forward_hook(_make())
        self.backbone = self._make_tag()


class AnomalyYOLO(AnomalyBase):
    """YOLO/YOLOE backbone, hooked at one of three taps (see ``_YOLO_TAPS``)."""

    def __init__(self, weight: str = "yolo26l.pt", imgsz: int = 640,
                 layers: str | list[str] = "P3", tap: str = "neck", **kw) -> None:
        layers = [layers] if isinstance(layers, str) else list(layers)
        if not all(l in _YOLO_LAYER_IDX for l in layers):
            raise ValueError(f"each layer must be in {list(_YOLO_LAYER_IDX)}, got {layers}")
        if tap not in _YOLO_TAPS:
            raise ValueError(f"tap must be in {_YOLO_TAPS}, got {tap!r}")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be divisible by 32 (YOLO stride), got {imgsz}")
        self.weight = weight
        self.tap = tap
        super().__init__(imgsz=imgsz, layers=sorted(set(layers), key=lambda l: _YOLO_LAYER_IDX[l]), **kw)

    def _make_tag(self) -> str:
        digits = "".join(l[-1] for l in self.layers)
        if self.fuse == "patchcore":
            suf = f"_pc{self.patchsize}_e{self.pretrain_dim}_t{self.target_dim}"
        elif self.fuse == "concat_pool":
            suf = f"_cp{self.patchsize}_t{self.target_dim}"
        else:
            suf = f"_{self.fuse}" if (len(self.layers) > 1 and self.fuse != "concat") else ""
        return f"yolo_p{digits}_{self.tap}" + suf

    def _build_backbone(self) -> None:
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
        self.backbone = self._make_tag()

    def _build_preprocess(self) -> None:
        # YOLO: simple resize + /255, no ImageNet norm.
        self.tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])


# ---------------------------------------------------------------------------
# torchvision ResNet (PatchCore default = WideResNet50 + layer2+3)
# ---------------------------------------------------------------------------


class AnomalyPatchCore(AnomalyBase):
    """torchvision ResNet/WideResNet, ImageNet-pretrained, hooks on layer1..4 stages.

    Paper-style PatchCore config: ``arch=wide_resnet50_2, layers=['l2','l3'], fuse='patchcore',
    patchsize=3, pretrain_dim=1024, target_dim=1024`` (scoring here is cos-space Noisy-OR, not 1-NN L2).
    """

    def __init__(self, arch: str = "wide_resnet50_2", imgsz: int = 224,
                 layers: str | list[str] = "l2", **kw) -> None:
        layers = [layers] if isinstance(layers, str) else list(layers)
        if not all(l in _RESNET_LAYER_ATTR for l in layers):
            raise ValueError(f"each layer must be in {_RESNET_LAYERS}, got {layers}")
        self.arch = arch
        super().__init__(imgsz=imgsz, layers=sorted(set(layers), key=lambda l: _RESNET_LAYERS.index(l)), **kw)

    def _make_tag(self) -> str:
        digits = "".join(l[-1] for l in self.layers)
        if self.fuse == "patchcore":
            suf = f"_pc{self.patchsize}_e{self.pretrain_dim}_t{self.target_dim}"
        elif self.fuse == "concat_pool":
            suf = f"_cp{self.patchsize}_t{self.target_dim}"
        else:
            suf = f"_{self.fuse}" if (len(self.layers) > 1 and self.fuse != "concat") else ""
        return f"{self.arch}_l{digits}{suf}"

    def _build_backbone(self) -> None:
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
        self.backbone = self._make_tag()


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

class AnomalyValidator:
    """MVTec-style evaluator for an :class:`AnomalyBase` model.

    Owns the whole validation path: build the bank from ``train/good``, score every test
    image, rasterize GT masks, compute image/pixel metrics via anomalib, optionally save
    visualizations, aggregate across categories, and write the CSV + summary.

    Metrics per category (all via anomalib; AP = AUPR, F1Max = F1-at-best-threshold):
      image  — AUROC, AP, F1Max
      pixel  — AUROC, AP, F1Max, AUPRO (region localization)
    Pixel metrics use anomaly images only (same population as the legacy pixel-AUROC).
    """

    METRIC_KEYS = ["image_auroc", "image_ap", "image_f1max", "image_f1max_thresh",
                   "pixel_auroc", "pixel_ap", "pixel_f1max", "pixel_f1max_thresh", "pixel_aupro"]

    def __init__(self, eval_size: int = 256, save_vis_dir: Path | None = None,
                 save_vis_samples_dir: Path | None = None, vis_thresh: float = 0.4) -> None:
        self.eval_size = eval_size
        self.save_vis_dir = save_vis_dir
        self.save_vis_samples_dir = save_vis_samples_dir
        self.vis_thresh = vis_thresh

    # ── GT mask rasterization ───────────────────────────────────────────
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

    # ── Metrics (anomalib) ──────────────────────────────────────────────
    @staticmethod
    def _anomalib_metric(cls, fields: list[str], batch) -> "tuple[float, float | None]":
        """Instantiate an anomalib metric, feed one batch, compute → ``(value, threshold_or_None)``."""
        m = cls(fields=fields)
        m.update(batch)
        val = float(m.compute())
        thresh = float(m.threshold) if hasattr(m, "threshold") else None
        return val, thresh

    @classmethod
    def _compute_metrics(cls, img_s: list[float], img_y: list[int],
                         pro_maps: list[np.ndarray], pro_gts: list[np.ndarray]) -> dict:
        """Image- and pixel-level metrics via anomalib (see class docstring)."""
        from types import SimpleNamespace

        from anomalib.metrics import AUPR, AUPRO, AUROC, F1Max

        out = {k: float("nan") for k in cls.METRIC_KEYS}
        if len(set(img_y)) > 1:
            b = SimpleNamespace(pred_score=torch.tensor(img_s, dtype=torch.float32),
                                gt_label=torch.tensor(img_y, dtype=torch.int32))
            out["image_auroc"], _ = cls._anomalib_metric(AUROC, ["pred_score", "gt_label"], b)
            out["image_ap"], _ = cls._anomalib_metric(AUPR, ["pred_score", "gt_label"], b)
            out["image_f1max"], out["image_f1max_thresh"] = cls._anomalib_metric(F1Max, ["pred_score", "gt_label"], b)
        if pro_maps:
            amap = torch.from_numpy(np.stack(pro_maps)).float()               # [N, H, W]
            gtm = torch.from_numpy(np.stack(pro_gts).astype(np.int32))        # [N, H, W]
            if bool(gtm.any()) and not bool(gtm.all()):
                b = SimpleNamespace(anomaly_map=amap, gt_mask=gtm)
                out["pixel_auroc"], _ = cls._anomalib_metric(AUROC, ["anomaly_map", "gt_mask"], b)
                out["pixel_ap"], _ = cls._anomalib_metric(AUPR, ["anomaly_map", "gt_mask"], b)
                out["pixel_f1max"], out["pixel_f1max_thresh"] = cls._anomalib_metric(F1Max, ["anomaly_map", "gt_mask"], b)
                out["pixel_aupro"], _ = cls._anomalib_metric(AUPRO, ["anomaly_map", "gt_mask"], b)
        return out

    # ── Visualization ───────────────────────────────────────────────────
    @staticmethod
    def _save_visualization(image_path: str, amap: np.ndarray, score: float, save_path: Path,
                            eval_size: int = 256, gt_mask: np.ndarray | None = None) -> None:
        """Save side-by-side viz: [original | heatmap | overlay (+ GT contour if given)]."""
        import cv2

        orig = cv2.imread(str(image_path))
        if orig is None:
            return
        orig = cv2.resize(orig, (eval_size, eval_size))

        # Colorize with absolute score values (0→blue, 1→red), no per-image normalization.
        amap_n = np.clip(amap, 0.0, 1.0)
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
        # Also save standalone heatmap with min/max labels.
        heatmap_path = save_path.with_stem(save_path.stem + "_heatmap")
        hm_min, hm_max = float(amap.min()), float(amap.max())
        heatmap_labeled = heatmap.copy()
        for i, (text, pos) in enumerate(
            [(f"min={hm_min:.4f}", (4, 14)), (f"max={hm_max:.4f}", (4, 30))]
        ):
            cv2.putText(heatmap_labeled, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap_labeled, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(heatmap_path), heatmap_labeled)

    # ── Per-category validation ─────────────────────────────────────────
    def val_category(self, category: str, ad: "AnomalyBase") -> dict:
        """Build bank from train/good, score every test image, return the metrics row.

        Honors ``save_vis_dir`` (per-image triptychs) and ``save_vis_samples_dir`` (one
        normal + one anomaly exemplar per category, anomaly only if score > ``vis_thresh``).
        """
        eval_size = self.eval_size
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
        img_s, img_y, pro_maps, pro_gts = [], [], [], []  # image scores/labels + anomaly-image 2D maps/masks
        # Best normal (lowest-score good) / best anomaly (highest-score anom > thresh) for sample export.
        best_normal: tuple[float, str, np.ndarray] | None = None
        best_anomaly: tuple[float, str, np.ndarray, np.ndarray | None] | None = None
        for im in data["test_im_list"]:
            score, hm = ad.predict_one(im, eval_size=eval_size)
            is_anom = im not in data["test_good_im_list"]
            img_s.append(score)
            img_y.append(int(is_anom))
            gt = None
            if is_anom:
                gt = self._gt_mask_from_polygons(im, eval_size)
                if gt is not None:
                    pro_maps.append(hm)   # 2D [H, W] anomaly map
                    pro_gts.append(gt)    # 2D [H, W] binary GT
            if self.save_vis_dir is not None:
                sub = "anom" if is_anom else "good"
                out = self.save_vis_dir / category / sub / f"{Path(im).stem}.jpg"
                self._save_visualization(im, hm, score, out, eval_size=eval_size, gt_mask=gt)
            if self.save_vis_samples_dir is not None:
                if is_anom:
                    if score > self.vis_thresh and (best_anomaly is None or score > best_anomaly[0]):
                        best_anomaly = (score, im, hm, gt)
                elif best_normal is None or score < best_normal[0]:
                    best_normal = (score, im, hm)
        val_s = time.perf_counter() - t1

        # Dump the chosen normal + anomaly exemplars after the full pass.
        if self.save_vis_samples_dir is not None:
            cat_dir = self.save_vis_samples_dir / category
            if best_normal is not None:
                s, im, hm = best_normal
                self._save_visualization(im, hm, s, cat_dir / "normal.jpg", eval_size=eval_size, gt_mask=None)
                print(f"  vis-samples: normal  score={s:.4f}  {Path(im).name}", flush=True)
            if best_anomaly is not None:
                s, im, hm, gt = best_anomaly
                self._save_visualization(im, hm, s, cat_dir / "anomaly.jpg", eval_size=eval_size, gt_mask=gt)
                print(f"  vis-samples: anomaly score={s:.4f}  {Path(im).name}", flush=True)
            else:
                print(f"  vis-samples: anomaly SKIPPED — no test image scored > {self.vis_thresh}", flush=True)

        metrics = self._compute_metrics(img_s, img_y, pro_maps, pro_gts)
        return {
            "category": category,
            **metrics,
            "n_support": len(data["train_im_list"]),
            "n_test": len(data["test_im_list"]),
            "n_anom_with_gt": len(pro_maps),
            "build_s": round(build_s, 1),
            "val_s": round(val_s, 1),
        }

    # ── Full sweep across categories ────────────────────────────────────
    def run(self, ad: "AnomalyBase", categories: list[str], out_csv: Path) -> list[dict]:
        """Validate every category, aggregate an AVERAGE row, write the CSV, print a summary."""
        rows: list[dict] = []
        for cat in categories:
            print(f"\n{'=' * 60}\n[{cat}] starting\n{'=' * 60}", flush=True)
            try:
                row = self.val_category(cat, ad)
            except Exception as e:
                print(f"[{cat}] FAILED: {e!r}", flush=True)
                row = {"category": cat, **{k: float("nan") for k in self.METRIC_KEYS},
                       "n_support": 0, "n_test": 0, "n_anom_with_gt": 0, "build_s": 0.0, "val_s": 0.0}
            rows.append(row)
            print(f"[{cat}] img AUROC={row['image_auroc']:.4f} AP={row['image_ap']:.4f} "
                  f"F1={row['image_f1max']:.4f}@{row['image_f1max_thresh']:.4f} | "
                  f"pix AUROC={row['pixel_auroc']:.4f} AP={row['pixel_ap']:.4f} "
                  f"F1={row['pixel_f1max']:.4f}@{row['pixel_f1max_thresh']:.4f} AUPRO={row['pixel_aupro']:.4f} "
                  f"| build={row['build_s']:.1f}s val={row['val_s']:.1f}s "
                  f"(test={row['n_test']}, anom_w_gt={row['n_anom_with_gt']})", flush=True)

        def _avg(key: str) -> float:
            vals = [r[key] for r in rows if isinstance(r[key], float) and r[key] == r[key]]
            return statistics.fmean(vals) if vals else float("nan")

        avg_row = {
            "category": "AVERAGE",
            **{k: _avg(k) for k in self.METRIC_KEYS},
            "n_support": sum(r["n_support"] for r in rows),
            "n_test": sum(r["n_test"] for r in rows),
            "n_anom_with_gt": sum(r["n_anom_with_gt"] for r in rows),
            "build_s": round(sum(r["build_s"] for r in rows), 1),
            "val_s": round(sum(r["val_s"] for r in rows), 1),
        }

        fields = ["category", *self.METRIC_KEYS, "build_s", "val_s", "n_support", "n_test", "n_anom_with_gt"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows + [avg_row]:
                w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})
        print(f"\nSaved {len(rows) + 1} rows ({len(rows)} categories + 1 AVERAGE) → {out_csv}")
        print("\n=== summary (i=image p=pixel · AUROC/AP/F1max@thresh, pixel adds AUPRO) ===")
        for r in rows + [avg_row]:
            print(f"  {r['category']:>12s}  "
                  f"i:{r['image_auroc']:.4f}/{r['image_ap']:.4f}/{r['image_f1max']:.4f}@{r['image_f1max_thresh']:.4f}  "
                  f"p:{r['pixel_auroc']:.4f}/{r['pixel_ap']:.4f}/{r['pixel_f1max']:.4f}@{r['pixel_f1max_thresh']:.4f}/{r['pixel_aupro']:.4f}")
        return rows


# ---------------------------------------------------------------------------
# Bbox mAP evaluator — heatmap → connected components → COCO eval
# ---------------------------------------------------------------------------


_BBOX_THRESHOLDS = np.arange(0.05, 1.0, 0.05)  # 19 thresholds
_BBOX_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


class AnomalyBboxValidator:
    """Heatmap → bbox mAP evaluator via COCO JSON + :mod:`faster_coco_eval`.

    Heatmaps are computed once per image at original resolution.  Connected-
    components is re-run per threshold (cheap) and each threshold's predictions
    are saved as a flat COCO-results JSON.  GT is derived from YOLO-seg polygon
    labels (min/max of polygon vertices → bounding box) and written once as a
    standard COCO annotation file.

    Parameters
    ----------
    min_area : int
        Minimum connected-component area (pixels) for a blob to produce a bbox.
    save_dir : Path or None
        Directory for per-category ``gt.json`` / ``pred_t0.XX.json`` files.
    """

    def __init__(self, min_area: int = 5, save_dir: Path | None = None):
        self.min_area = min_area
        self.save_dir = save_dir

    # ── Label file resolution ──────────────────────────────────────────

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

    # ── GT: polygon → bbox ─────────────────────────────────────────────

    @staticmethod
    def _polygons_to_gt_annotations(
        label_path: Path, ori_w: int, ori_h: int
    ) -> list[dict]:
        """Parse YOLO seg label file → COCO annotation dicts (xywh pixel coords)."""
        try:
            text = label_path.read_text().strip()
        except OSError:
            return []
        if not text:
            return []
        anns = []
        for line in text.splitlines():
            tokens = line.strip().split()
            if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
                continue
            try:
                coords = np.array(tokens[1:], dtype=np.float32).reshape(-1, 2)
            except ValueError:
                continue
            x1 = max(float(coords[:, 0].min()) * ori_w, 0)
            y1 = max(float(coords[:, 1].min()) * ori_h, 0)
            x2 = min(float(coords[:, 0].max()) * ori_w, ori_w)
            y2 = min(float(coords[:, 1].max()) * ori_h, ori_h)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            anns.append({"bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2)})
        return anns

    # ── Pred: heatmap → connected components → bboxes ──────────────────

    @staticmethod
    def _heatmap_to_boxes(
        heatmap: np.ndarray, threshold: float, min_area: int
    ) -> np.ndarray:
        """Connected-components on thresholded heatmap → ``(N, 5)`` [x1, y1, x2, y2, score].

        Returns empty ``(0, 5)`` array when no components survive filtering.
        """
        import cv2

        mask = (heatmap >= threshold).astype(np.uint8)
        if mask.sum() == 0:
            return np.empty((0, 5), dtype=np.float32)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes: list[list[float]] = []
        for lbl in range(1, stats.shape[0]):
            x, y, w, h, area = stats[lbl]
            if area < min_area:
                continue
            score = float(heatmap[labels == lbl].mean())
            boxes.append([float(x), float(y), float(x + w), float(y + h), score])
        if not boxes:
            return np.empty((0, 5), dtype=np.float32)
        boxes_arr = np.array(boxes, dtype=np.float32)
        boxes_arr = boxes_arr[boxes_arr[:, 4].argsort()[::-1]]
        return boxes_arr

    # ── COCO JSON builders ─────────────────────────────────────────────

    @staticmethod
    def _build_gt_coco(
        image_entries: list[dict],
    ) -> dict:
        """Build a standard COCO GT dict from pre-collected image metadata.

        *image_entries* is a list of dicts with keys ``image_id`` (int),
        ``file_name``, ``width``, ``height``, and ``annotations`` (list of
        ``{"bbox": [x,y,w,h], "area": float}``).
        """
        images = []
        annotations = []
        ann_id = 0
        for entry in image_entries:
            images.append({
                "id": entry["image_id"],
                "file_name": entry["file_name"],
                "width": entry["width"],
                "height": entry["height"],
            })
            for ann in entry["annotations"]:
                annotations.append({
                    "id": ann_id,
                    "image_id": entry["image_id"],
                    "category_id": 0,
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0,
                })
                ann_id += 1
        return {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "anomaly"}],
        }

    @staticmethod
    def _build_pred_json(
        image_ids: list[int],
        per_image_boxes: list[np.ndarray],
    ) -> list[dict]:
        """Build flat COCO-results list from per-image bbox arrays.

        Each bbox array is ``(N, 5)`` [x1, y1, x2, y2, score] in pixel coords.
        Output bbox is ``[x, y, w, h]`` (xywh).
        """
        preds: list[dict] = []
        for img_id, boxes in zip(image_ids, per_image_boxes):
            for b in boxes:
                x1, y1, x2, y2, score = b
                w = max(x2 - x1, 0)
                h = max(y2 - y1, 0)
                if w <= 0 or h <= 0:
                    continue
                preds.append({
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(w), 2), round(float(h), 2)],
                    "score": round(float(score), 5),
                })
        return preds

    # ── COCO eval via faster-coco-eval ─────────────────────────────────

    @staticmethod
    def _eval_coco(gt_path: str, pred_path: str) -> dict[str, float]:
        """Run :class:`COCOeval_faster` on the given GT and prediction JSON files.

        Returns dict with keys ``map50, map, map75``.
        """
        from faster_coco_eval import COCO, COCOeval_faster

        anno = COCO(gt_path)
        pred = anno.loadRes(pred_path)
        val = COCOeval_faster(anno, pred, iouType="bbox")
        val.evaluate()
        val.accumulate()
        val.summarize()
        stats = val.stats_as_dict
        return {
            "map50": round(float(stats.get("AP_50", 0.0)), 4),
            "map": round(float(stats.get("AP_all", 0.0)), 4),
            "map75": round(float(stats.get("AP_75", 0.0)), 4),
        }

    # ── Per-category evaluation ────────────────────────────────────────

    def val_category(self, category: str, ad: "AnomalyBase") -> list[dict]:
        """Build bank, score all test images, compute mAP across 19 thresholds.

        Returns 19 rows — one per threshold in ``_BBOX_THRESHOLDS``.
        """
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        data = get_mvtec_yolo_data(category)
        test_good_set = set(data["test_good_im_list"])
        test_imgs = data["test_im_list"]

        # ── Build bank ─────────────────────────────────────────────────
        ad.bank = None
        t0 = time.perf_counter()
        ad.load_support_set(data["train_im_list"])
        build_s = time.perf_counter() - t0
        print(f"[{category}] bank built in {build_s:.1f}s  "
              f"(support={len(data['train_im_list'])}, bank_size={ad.bank.shape[0]})", flush=True)

        # ── Score all test images (once) + collect metadata ─────────────
        t1 = time.perf_counter()
        image_entries: list[dict] = []
        heatmaps: list[np.ndarray] = []
        image_ids: list[int] = []

        for img_id, im in enumerate(test_imgs):
            # original image size
            with Image.open(im) as pil:
                ori_w, ori_h = pil.size
            _, hm = ad.predict_one(im, eval_size=(ori_h, ori_w))

            label_path = self._label_path(im)
            is_good = im in test_good_set
            annotations: list[dict] = []
            if label_path is not None and not is_good:
                annotations = self._polygons_to_gt_annotations(label_path, ori_w, ori_h)

            image_entries.append({
                "image_id": img_id,
                "file_name": Path(im).name,
                "width": ori_w,
                "height": ori_h,
                "annotations": annotations,
            })
            heatmaps.append(hm)
            image_ids.append(img_id)

        val_s = time.perf_counter() - t1

        # ── GT COCO JSON (once) ─────────────────────────────────────────
        gt_json = self._build_gt_coco(image_entries)
        save_dir = self.save_dir / category
        save_dir.mkdir(parents=True, exist_ok=True)
        gt_path = str(save_dir / "gt.json")
        with open(gt_path, "w") as f:
            json.dump(gt_json, f, indent=2)

        # ── Per-threshold: CC → pred JSON → COCO eval ──────────────────
        rows: list[dict] = []
        for threshold in _BBOX_THRESHOLDS:
            threshold = round(float(threshold), 2)
            per_image_boxes = [
                self._heatmap_to_boxes(hm, threshold, self.min_area)
                for hm in heatmaps
            ]
            pred_list = self._build_pred_json(image_ids, per_image_boxes)
            n_pred = len(pred_list)

            pred_path = str(save_dir / f"pred_t{threshold:.2f}.json")
            with open(pred_path, "w") as f:
                json.dump(pred_list, f, indent=2)

            try:
                coco_metrics = self._eval_coco(gt_path, pred_path)
            except Exception:
                coco_metrics = {"map50": float("nan"), "map": float("nan"), "map75": float("nan")}

            n_gt = sum(len(e["annotations"]) for e in image_entries)
            rows.append({
                "category": category,
                "threshold": threshold,
                **coco_metrics,
                "n_pred": n_pred,
                "n_gt": n_gt,
            })

        print(f"[{category}] {val_s:.1f}s val  "
              f"(test={len(test_imgs)}, anom_w_gt="
              f"{sum(1 for e in image_entries if e['annotations'])}  "
              f"t={_BBOX_THRESHOLDS[0]:.2f}..{_BBOX_THRESHOLDS[-1]:.2f})", flush=True)
        return rows

    # ── Full sweep ─────────────────────────────────────────────────────

    def run(self, ad: "AnomalyBase", categories: list[str], out_csv: Path) -> list[dict]:
        """Validate every category, compute AVERAGE per threshold, write CSV."""
        rows: list[dict] = []
        for cat in categories:
            print(f"\n{'=' * 60}\n[{cat}] starting\n{'=' * 60}", flush=True)
            try:
                cat_rows = self.val_category(cat, ad)
            except Exception as e:
                print(f"[{cat}] FAILED: {e!r}", flush=True)
                cat_rows = [{
                    "category": cat, "threshold": round(float(t), 2),
                    "map50": float("nan"), "map": float("nan"), "map75": float("nan"),
                    "n_pred": 0, "n_gt": 0,
                } for t in _BBOX_THRESHOLDS]
            rows.extend(cat_rows)
            # Print best threshold for this category
            valid = [r for r in cat_rows if r["map50"] == r["map50"]]
            if valid:
                best = max(valid, key=lambda r: r["map50"])
                print(f"[{cat}] best mAP50={best['map50']:.4f} @ t={best['threshold']:.2f}", flush=True)

        # AVERAGE per threshold
        fields = ["category", "threshold", "map50", "map", "map75", "n_pred", "n_gt"]
        avg_rows: list[dict] = []
        for t in _BBOX_THRESHOLDS:
            t = round(float(t), 2)
            t_rows = [r for r in rows if r["threshold"] == t]
            avg = {"category": "AVERAGE", "threshold": t}
            for k in ["map50", "map", "map75"]:
                vals = [r[k] for r in t_rows if isinstance(r[k], float) and r[k] == r[k]]
                avg[k] = round(statistics.fmean(vals), 4) if vals else float("nan")
            avg["n_pred"] = sum(r["n_pred"] for r in t_rows)
            avg["n_gt"] = sum(r["n_gt"] for r in t_rows)
            avg_rows.append(avg)

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows + avg_rows:
                w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                            for k in fields})
        print(f"\nSaved {len(rows) + len(avg_rows)} rows → {out_csv}")

        # Print summary: best threshold per category
        print("\n=== bbox mAP summary (best mAP50 per category) ===")
        for cat in categories:
            cat_rows = [r for r in rows if r["category"] == cat]
            valid = [r for r in cat_rows if r["map50"] == r["map50"]]
            if valid:
                best = max(valid, key=lambda r: r["map50"])
                print(f"  {cat:>12s}  mAP50={best['map50']:.4f}  mAP={best['map']:.4f}  "
                      f"mAP75={best['map75']:.4f}  @ t={best['threshold']:.2f}")
        return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_RESNET_ARCH_MAP = {
    "wrn50": "wide_resnet50_2", "rn50": "resnet50", "rn18": "resnet18",
    "rn34": "resnet34", "rn101": "resnet101", "rn152": "resnet152",
}


def _build(args: argparse.Namespace) -> tuple[AnomalyBase, Path]:
    """Construct the requested model + return its default run directory."""
    bb = args.backbone
    kw = dict(
        coreset_size=args.coreset, K=args.K, scoring=args.scoring,
        temperature=args.temperature, target_score=args.target_score,
        target_quantile=args.target_quantile, device=args.device,
    )
    if bb.startswith("dinov2_"):
        ad = AnomalyDINO(backbone=bb, imgsz=args.imgsz or 448, **kw)
    elif bb.startswith("dinov3_"):
        short = bb.removeprefix("dinov3_")
        ad = AnomalyDINOv3(backbone=short, imgsz=args.imgsz or 448, **kw)
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
        wp = Path(args.weight)
        tag = f"{tag}_{wp.parent.name}_{wp.stem}_imgsz{ad.imgsz}"
    elif isinstance(ad, AnomalyPatchCore):
        tag = f"{tag}_imgsz{ad.imgsz}"
    elif isinstance(ad, (AnomalyConvNeXt, AnomalyConvNeXtMulti)):
        tag = f"{tag}_imgsz{ad.imgsz}"
    else:  # DINO
        tag = f"{tag}_imgsz{ad.imgsz}"

    if args.coreset:
        tag += f"_cs{args.coreset}"
    tag += f"_sc{args.scoring}_K{args.K}_T{args.temperature:g}_ts{args.target_score:g}_q{args.target_quantile:g}"
    return ad, Path(f"./runs/{tag}")


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

    dinov2_choices = list(_DINOV2_HUB)
    dinov3_choices = [f"dinov3_{k}" for k in _DINOV3_HUB]

    p.add_argument("--backbone", default="dinov2_vits14",
                   choices=dinov2_choices + dinov3_choices + convnext_choices + yolo_choices + resnet_choices,
                   help="Feature extractor. dinov2_*: DINOv2. dinov3_*: DINOv3 ViT. "
                        "convnext_*[_s<digits>]: DINOv3-distilled ConvNeXt "
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
    p.add_argument("--coreset", type=int, default=10000,
                   help="k-center coreset bank cap (default 10000; 0 = keep all patches). "
                        "Dropped features become the calibration hold-out.")
    p.add_argument("--K", type=int, default=5,
                   help="Top-K nearest bank entries per query (default 5; ignored by 1nn).")
    p.add_argument("--scoring", default="noisy_or", choices=list(_SCORERS),
                   help="Scoring method: noisy_or (sigmoid Noisy-OR) or 1nn (1 − max cos).")
    p.add_argument("--temperature", type=float, default=5.0,
                   help="β floor / starting point (default 3.0); hold-out calibration only raises it.")
    p.add_argument("--target-score", type=float, default=0.4,
                   help="Calibration target: a typical-normal query scores at this value (default 0.2).")
    p.add_argument("--target-quantile", type=float, default=0.95,
                   help="Hold-out quantile placed at --target-score (default 0.95).")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path.")
    p.add_argument("--save-vis", action="store_true",
                   help="If set, save per-image [orig|heatmap|overlay] triptych to "
                        "runs/<tag>/vis/ (grouped by <category>/<good|anom>/).")
    p.add_argument("--save-vis-samples", action="store_true",
                   help="If set, save exactly ONE normal + ONE anomaly viz per category to "
                        "runs/<tag>/vis/samples/<category>/{normal,anomaly}.jpg.")
    p.add_argument("--vis-thresh", type=float, default=0.4,
                   help="Minimum score for an anomaly to be eligible as the per-category exemplar "
                        "(default 0.4 — matches OBMA's accumulate_thresh default).")
    p.add_argument("--device", default=None,
                   help="Device override (e.g. 'mps', 'cuda', 'cpu'). Default: auto-detect.")
    p.add_argument("--bbox", action="store_true",
                   help="Also run bbox mAP evaluation (heatmap -> connected components -> COCO mAP).")
    p.add_argument("--bbox-min-area", type=int, default=5,
                   help="Min connected-component area for heatmap->bbox (default 5).")
    args = p.parse_args()

    if args.category:
        cats = [c.strip() for c in args.category.split(",") if c.strip()]
        unknown = [c for c in cats if c not in MVTEC_CATEGORIES]
        if unknown:
            raise SystemExit(f"unknown category: {unknown}. Valid: {MVTEC_CATEGORIES}")
    else:
        cats = list(MVTEC_CATEGORIES)

    ad, run_dir = _build(args)
    if args.out is not None:
        out_csv = args.out
    else:
        stem = "metrics"
        if len(cats) < len(MVTEC_CATEGORIES):
            stem += f"_{'_'.join(cats)}"
        out_csv = run_dir / f"{stem}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    save_vis_dir = run_dir / "vis" if args.save_vis else None
    save_vis_samples_dir = run_dir / "vis" / "samples" if args.save_vis_samples else None

    print(f"run_dir={run_dir}  backbone={ad.backbone} imgsz={ad.imgsz} device={ad.device} "
          f"feat_dim={ad.feat_dim} grid={ad.grid} K={ad.K} target_score={ad.target_score}"
          + (f" coreset={args.coreset}" if args.coreset else ""))

    validator = AnomalyValidator(
        eval_size=256,
        save_vis_dir=save_vis_dir,
        save_vis_samples_dir=save_vis_samples_dir,
        vis_thresh=args.vis_thresh,
    )
    validator.run(ad, cats, out_csv)

    if args.bbox:
        bbox_out = run_dir / f"bbox_{out_csv.stem}.csv"
        bbox_validator = AnomalyBboxValidator(
            min_area=args.bbox_min_area,
            save_dir=run_dir / "bbox_eval",
        )
        bbox_validator.run(ad, cats, bbox_out)


if __name__ == "__main__":
    main()
