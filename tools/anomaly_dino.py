"""Training-free anomaly detection: memory bank of normal patch tokens, swappable backbone.

Pipeline (identical regardless of backbone — this is the whole point of the file):
  - Extract patch tokens from each training image, L2-normalize, concat → bank [M, D].
  - For each test image: per-patch cosine similarity vs bank, 1-NN → distance per patch.
  - Image score = mean of top-1% patch distances.
  - Pixel map = patch dist grid bilinear-upsampled to eval_size.
  - GT mask via AnomalyValidator._gt_mask_from_polygons (same eval path as YOLOAnomaly).

Backbones (selected via ``--backbone``):
  - ``dino_vits14`` / ``dino_vitb14`` / ``dino_vitl14``: DINOv2 patch tokens
    (Damm et al., WACV 2025). Default imgsz=448 (must be a multiple of 14).
  - ``yolo_p3`` / ``yolo_p4`` / ``yolo_p5``: YOLO backbone features at the chosen
    pyramid level. Default imgsz=640 (must be a multiple of 32). Weight via ``--weight``.

Use ``--backbone yolo_p3`` to isolate the backbone variable: if numbers stay near
YOLOAnomaly's, the gap is the YOLO backbone itself; if they jump toward DINO's, the gap
is in YOLOAnomaly's bank-construction (EM/coreset) or score aggregation.

Intentionally omits two paper tricks to keep the baseline minimal:
  * PCA foreground masking (objects)  — would lift object-class numbers a few points
  * rotation augmentation (objects)   — same
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultra_ext.yoloa import MVTEC_CATEGORIES, get_mvtec_yolo_data
from ultralytics.models.yolo.model import AnomalyValidator


_DINOV2_HUB = {  # short name -> (torch.hub entry, feat_dim)
    "vits14": ("dinov2_vits14_reg", 384),
    "vitb14": ("dinov2_vitb14_reg", 768),
    "vitl14": ("dinov2_vitl14_reg", 1024),
}
_DINO_PATCH = 14
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_YOLO_LAYER_IDX = {"P3": 0, "P4": 1, "P5": 2}
# tap = where we tap the network for patch tokens:
#   "bb"  → backbone stage output (pre-FPN); layer indices below are for yolo26 / v8 / v11 detection.
#   "pre" → Detect module input (post-neck/FPN, pre-head); current default.
#   "cv3" → cls-branch c3-dim feat right before the 1x1 nc-projection.
_YOLO_TAPS = ("bb", "pre", "cv3")
_YOLO_BB_LAYER = {"P3": 4, "P4": 6, "P5": 10}


class AnomalyDINO:
    """DINOv2 + memory-bank anomaly detector (Damm et al., 2024)."""

    def __init__(
        self,
        backbone: str = "vits14",
        imgsz: int = 448,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
    ) -> None:
        if backbone not in _DINOV2_HUB:
            raise ValueError(f"backbone must be one of {list(_DINOV2_HUB)}")
        if imgsz % _DINO_PATCH != 0:
            raise ValueError(f"imgsz must be divisible by {_DINO_PATCH}, got {imgsz}")
        self.backbone = backbone
        self.imgsz = imgsz
        self.top_pct = top_pct
        self.sim_chunk = sim_chunk
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")

        hub_name, self.feat_dim = _DINOV2_HUB[backbone]
        self.model = torch.hub.load("facebookresearch/dinov2", hub_name).to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        n = imgsz // _DINO_PATCH
        self.grid = (n, n)
        self.bank: torch.Tensor | None = None    # [M, D], L2-normalized
        self.coreset_size: int | None = None     # if set, subsample bank to this many points via greedy FPS
        self.coreset_presample: int | None = None  # if set, randomly subsample to this size before FPS

    def _load(self, path: str) -> torch.Tensor:
        return self.tf(Image.open(path).convert("RGB"))

    @torch.inference_mode()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return DINOv2 patch tokens [B, N, D] for input [B, 3, H, W]."""
        out = self.model.forward_features(x.to(self.device))
        return out["x_norm_patchtokens"]

    def load_support_set(self, image_paths: list[str], batch: int = 8) -> None:
        """Build (and L2-normalize) the memory bank from normal images.

        Optionally compact via greedy FPS (PatchCore-style coreset) when ``self.coreset_size`` is set.
        Big intermediate banks live on CPU to avoid blowing MPS' 30 GiB watermark; only the
        final (post-coreset) bank moves to ``self.device``.
        """
        feats: list[torch.Tensor] = []
        for i in range(0, len(image_paths), batch):
            xs = torch.stack([self._load(p) for p in image_paths[i:i + batch]], dim=0)
            tokens = self._extract(xs)  # [B, N, D]
            feats.append(tokens.reshape(-1, tokens.shape[-1]).cpu())
        bank = F.normalize(torch.cat(feats, dim=0), dim=-1)
        if self.coreset_size and self.coreset_size < bank.shape[0]:
            if self.coreset_presample and self.coreset_presample < bank.shape[0]:
                # PatchCore-style speed trick: random pre-sample before FPS.
                g = torch.Generator(device="cpu").manual_seed(0)
                pre_idx = torch.randperm(bank.shape[0], generator=g)[: self.coreset_presample]
                print(f"  coreset: pre-sample {bank.shape[0]} → {self.coreset_presample} before FPS")
                bank = bank[pre_idx].contiguous()
            bank = _greedy_fps(bank, self.coreset_size, sim_chunk=self.sim_chunk)
        self.bank = bank.to(self.device)

    @torch.inference_mode()
    def predict_one(self, image_path: str, eval_size: int = 256) -> tuple[float, np.ndarray]:
        """Return (image_score, pixel_map[eval_size, eval_size])."""
        if self.bank is None:
            raise RuntimeError("call load_support_set() before predict_one()")
        x = self._load(image_path).unsqueeze(0)
        feats = F.normalize(self._extract(x).squeeze(0), dim=-1)  # [N, D]

        # 1-NN cosine distance against the bank, chunked over the bank dim to bound memory.
        max_sim = torch.full((feats.shape[0],), -1.0, device=feats.device, dtype=feats.dtype)
        for s in range(0, self.bank.shape[0], self.sim_chunk):
            sims = feats @ self.bank[s:s + self.sim_chunk].T  # [N, chunk]
            max_sim = torch.maximum(max_sim, sims.max(dim=-1).values)
        dists = 1.0 - max_sim  # [N], min cosine distance

        n = dists.numel()
        k = max(1, int(round(n * self.top_pct)))
        image_score = float(dists.topk(k).values.mean())

        H, W = self.grid
        amap = dists.reshape(1, 1, H, W).float()
        amap = F.interpolate(amap, size=(eval_size, eval_size), mode="bilinear", align_corners=False)
        return image_score, amap.squeeze().cpu().numpy()


@torch.inference_mode()
def _greedy_fps(bank: torch.Tensor, n_keep: int, sim_chunk: int = 4096, seed: int = 0) -> torch.Tensor:
    """Greedy farthest-point sampling over L2-normalized features (cosine metric).

    Returns the subsampled bank as a contiguous tensor. O(n_keep * M * D) — runs on CPU
    because MPS produces wrong argmin results for large/high-dim banks (silently corrupts
    grid-like patterns where many points have near-identical features). The matmul cost
    is dwarfed by the per-iter sync overhead anyway, so CPU is competitive with GPU here.
    Use CUDA explicitly via device='cuda' on the bank to get GPU speedup.
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
        sims = bank @ cur                       # [M]
        max_sim = torch.maximum(max_sim, sims)
        nxt = torch.argmin(max_sim)
        selected[i] = nxt
        cur = bank[nxt]
        if i % 2000 == 0:
            print(f"  coreset: {i}/{n_keep}  elapsed={time.perf_counter() - t0:.1f}s")
    return bank[selected].contiguous().to(orig_device)


class AnomalyYOLO(AnomalyDINO):
    """Same pipeline as AnomalyDINO, but patch tokens come from a YOLO pyramid level."""

    def __init__(
        self,
        weight: str = "yolo26l.pt",
        imgsz: int = 640,
        layers: str | list[str] = "P3",
        tap: str = "pre",
        fuse: str = "concat",
        patchsize: int = 1,
        pretrain_dim: int = 1024,
        target_dim: int = 1024,
        device: str | None = None,
        top_pct: float = 0.01,
        sim_chunk: int = 4096,
    ) -> None:
        if isinstance(layers, str):
            layers = [layers]
        if not all(l in _YOLO_LAYER_IDX for l in layers):
            raise ValueError(f"each layer must be one of {list(_YOLO_LAYER_IDX)}, got {layers}")
        if tap not in _YOLO_TAPS:
            raise ValueError(f"tap must be one of {_YOLO_TAPS}")
        if fuse not in ("concat", "sum", "avg", "patchcore", "concat_pool"):
            raise ValueError(f"fuse must be one of (concat, sum, avg, patchcore, concat_pool), got {fuse!r}")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be divisible by 32 (YOLO stride), got {imgsz}")
        if patchsize % 2 == 0:
            raise ValueError(f"patchsize must be odd (for symmetric padding), got {patchsize}")
        # Order by ascending P-level so the highest-resolution feature comes first (fusion target).
        layers = sorted(set(layers), key=lambda l: _YOLO_LAYER_IDX[l])
        self.layers = layers
        self.tap = tap
        self.fuse = fuse
        self.patchsize = patchsize
        self.pretrain_dim = pretrain_dim
        self.target_dim = target_dim
        digits = "".join(l[-1] for l in layers)
        if fuse == "patchcore":
            fuse_suffix = f"_pc{patchsize}_e{pretrain_dim}_t{target_dim}"
        elif fuse == "concat_pool":
            fuse_suffix = f"_cp{patchsize}_t{target_dim}"
        else:
            fuse_suffix = f"_{fuse}" if (len(layers) > 1 and fuse != "concat") else ""
        self.backbone = f"yolo_p{digits}" + (f"_{tap}" if tap != "pre" else "") + fuse_suffix
        self.imgsz = imgsz
        self.top_pct = top_pct
        self.sim_chunk = sim_chunk
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")

        from ultralytics import YOLO
        from ultralytics.nn.modules.head import Detect

        self.model = YOLO(weight).model.to(self.device).eval()
        self._cached: dict[str, torch.Tensor] = {}
        det = next(m for m in self.model.modules() if isinstance(m, Detect))

        for lname in layers:
            idx = _YOLO_LAYER_IDX[lname]
            if tap == "pre":
                # Detect's forward input = list[P3, P4, P5(, text)]. One pre-hook per Detect is enough,
                # but registering once and indexing all levels is simpler than tracking state.
                def _make_pre(i, k=lname):
                    def h(_m, inputs):
                        self._cached[k] = inputs[0][i]
                    return h
                det.register_forward_pre_hook(_make_pre(idx))
            elif tap == "bb":
                # Pre-FPN backbone stage. Layer indices are yolo26/v8/v11-specific.
                bb_layer = self.model.model[_YOLO_BB_LAYER[lname]]
                def _make_bb(k=lname):
                    def h(_m, _inputs, output):
                        self._cached[k] = output
                    return h
                bb_layer.register_forward_hook(_make_bb())
            else:  # "cv3": output of cls-branch's penultimate block (c3-dim, before 1x1 nc-projection).
                def _make_cv3(k=lname):
                    def h(_m, _inputs, output):
                        self._cached[k] = output
                    return h
                det.cv3[idx][1].register_forward_hook(_make_cv3())

        # YOLO preprocessing: resize (square) + ToTensor (/255). No ImageNet norm.
        self.tf = transforms.Compose([
            transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        # Probe feat_dim and grid via a dummy forward.
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, imgsz, imgsz, device=self.device))
        missing = [l for l in layers if l not in self._cached]
        assert not missing, f"hooks didn't fire for {missing}"
        ref = self._cached[layers[0]]  # highest-res level → fusion target grid
        self.grid = (ref.shape[2], ref.shape[3])
        dims = [self._cached[l].shape[1] for l in layers]
        if fuse == "concat":
            self.feat_dim = sum(dims)
        elif fuse in ("patchcore", "concat_pool"):
            self.feat_dim = target_dim  # adaptive pool produces target_dim regardless of inputs
        else:  # sum / avg require matching channel dims
            if len(set(dims)) > 1:
                raise ValueError(f"fuse={fuse!r} needs equal channel dims across layers, got {dict(zip(layers, dims))}")
            self.feat_dim = dims[0]
        self.bank: torch.Tensor | None = None
        self.coreset_size: int | None = None
        self.coreset_presample: int | None = None

    @torch.inference_mode()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, N, D] patch tokens.

        Multi-layer flow (concat/sum/avg): upsample each layer to highest-res grid → fuse channel-wise.
        PatchCore flow: per-layer 3x3 unfold → per-layer adaptive_avg_pool to ``pretrain_dim`` →
        layer-wise concat → adaptive_avg_pool to ``target_dim`` (paper's exact pipeline).
        """
        self._cached = {}
        self.model(x.to(self.device))
        H, W = self.grid
        B = x.shape[0]

        if self.fuse == "patchcore":
            return self._extract_patchcore(B, H, W)
        if self.fuse == "concat_pool":
            return self._extract_concat_pool(B, H, W)

        feats = []
        for lname in self.layers:
            f = self._cached[lname]  # [B, C, h, w]
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            feats.append(f)
        if len(feats) == 1:
            fused = feats[0]
        elif self.fuse == "concat":
            fused = torch.cat(feats, dim=1)
        elif self.fuse == "sum":
            fused = torch.stack(feats, dim=0).sum(dim=0)
        else:  # "avg"
            fused = torch.stack(feats, dim=0).mean(dim=0)
        return fused.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, D]

    def _extract_patchcore(self, B: int, H: int, W: int) -> torch.Tensor:
        """PatchCore-style extraction: 3x3 unfold + two-stage adaptive avg-pool.

        Simplification vs upstream: we upsample feature maps to the reference grid BEFORE the
        3x3 unfold (upstream unfolds first then upsamples the patch grid). The two orderings
        produce slightly different boundary behavior but the same overall shape and semantics.

        MPS workaround: adaptive_avg_pool1d on non-divisible (input, output) sizes is unimplemented
        on MPS as of 2026, so we run the pool step on CPU and move the result back.
        """
        ps = self.patchsize
        pad = ps // 2
        pool_dev = "cpu" if self.device == "mps" else self.device
        per_layer = []
        for lname in self.layers:
            f = self._cached[lname]  # [B, C, h, w]
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            # Unfold ps×ps neighborhood per spatial position: each token becomes a C*ps^2 vector.
            patches = F.unfold(f, kernel_size=ps, stride=1, padding=pad)  # [B, C*ps^2, H*W]
            patches = patches.transpose(1, 2).reshape(B * H * W, -1)     # [B*N, C*ps^2]
            pooled = F.adaptive_avg_pool1d(patches.to(pool_dev).unsqueeze(1), self.pretrain_dim).squeeze(1)
            per_layer.append(pooled)                                      # [B*N, pretrain_dim]
        merged = torch.cat(per_layer, dim=-1) if len(per_layer) > 1 else per_layer[0]
        final = F.adaptive_avg_pool1d(merged.unsqueeze(1), self.target_dim).squeeze(1)
        return final.to(self.device).reshape(B, H * W, self.target_dim)

    def _extract_concat_pool(self, B: int, H: int, W: int) -> torch.Tensor:
        """Single-stage variant of PatchCore: concat raw (optionally unfolded) features then one pool."""
        ps = self.patchsize
        pad = ps // 2
        pool_dev = "cpu" if self.device == "mps" else self.device
        feats_list = []
        for lname in self.layers:
            f = self._cached[lname]  # [B, C, h, w]
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            patches = F.unfold(f, kernel_size=ps, stride=1, padding=pad)  # [B, C*ps^2, H*W]
            patches = patches.transpose(1, 2).reshape(B * H * W, -1)     # [B*N, C*ps^2]
            feats_list.append(patches)
        merged = torch.cat(feats_list, dim=-1) if len(feats_list) > 1 else feats_list[0]
        final = F.adaptive_avg_pool1d(merged.to(pool_dev).unsqueeze(1), self.target_dim).squeeze(1)
        return final.to(self.device).reshape(B, H * W, self.target_dim)


def val_one_category(category: str, ad: AnomalyDINO, eval_size: int = 256) -> dict:
    """Build bank from train/good, score every test image, compute image/pixel AUROC."""
    from sklearn.metrics import roc_auc_score

    # MPS allocator state from previous category can silently corrupt downstream ops
    # (esp. 1024-dim banks with coreset → AUROC collapses to 0.5). Clear cache to be safe.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    data = get_mvtec_yolo_data(category)
    ad.bank = None
    t0 = time.perf_counter()
    ad.load_support_set(data["train_im_list"])
    build_s = time.perf_counter() - t0
    print(f"[{category}] bank built in {build_s:.1f}s  "
          f"(support={len(data['train_im_list'])}, bank_size={ad.bank.shape[0]})")

    t1 = time.perf_counter()
    img_s, img_y, pix_s, pix_y = [], [], [], []
    for im in data["test_im_list"]:
        score, hm = ad.predict_one(im, eval_size=eval_size)
        is_anom = im not in data["test_good_im_list"]
        img_s.append(score)
        img_y.append(int(is_anom))
        if is_anom:
            gt = AnomalyValidator._gt_mask_from_polygons(im, eval_size)
            if gt is not None:
                pix_s.append(hm.flatten())
                pix_y.append(gt.flatten())
    val_s = time.perf_counter() - t1

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


def _build(args: argparse.Namespace) -> tuple[AnomalyDINO, str, Path]:
    """Return (model, tag, default_csv_path) for the requested backbone."""
    bb = args.backbone
    if bb.startswith("dino_"):
        short = bb.removeprefix("dino_")  # vits14 / vitb14 / vitl14
        imgsz = args.imgsz or 448
        ad = AnomalyDINO(backbone=short, imgsz=imgsz)
        tag = f"dino_{short}_imgsz{imgsz}"
    elif bb.startswith("yolo_p"):
        rest = bb.removeprefix("yolo_p")          # "3", "34_bb", "345_cv3"
        digits, _, tap = rest.partition("_")
        if not digits or not all(d in "345" for d in digits):
            raise ValueError(f"yolo backbone digits must be from {{3,4,5}}, got {digits!r}")
        layers = [f"P{d}" for d in digits]
        tap = tap or "pre"
        imgsz = args.imgsz or 640
        ad = AnomalyYOLO(
            weight=args.weight, imgsz=imgsz, layers=layers, tap=tap, fuse=args.fuse,
            patchsize=args.patchsize, pretrain_dim=args.pretrain_dim, target_dim=args.target_dim,
        )
        tap_suffix = f"_{tap}" if tap != "pre" else ""
        if args.fuse == "patchcore":
            fuse_suffix = f"_pc{args.patchsize}_e{args.pretrain_dim}_t{args.target_dim}"
        elif args.fuse == "concat_pool":
            fuse_suffix = f"_cp{args.patchsize}_t{args.target_dim}"
        else:
            fuse_suffix = f"_{args.fuse}" if (len(layers) > 1 and args.fuse != "concat") else ""
        tag = f"yolo_p{''.join(sorted(digits))}{tap_suffix}{fuse_suffix}_{Path(args.weight).stem}_imgsz{imgsz}"
    else:
        raise ValueError(f"unknown --backbone {bb!r}")
    out_csv = Path(f"./runs/temp/anomaly_{tag}_mvtec_metrics.csv")
    return ad, tag, out_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # All non-empty subsets of {3,4,5}, kept in ascending order so the highest-res level is first.
    _yolo_digits = ("3", "4", "5", "34", "35", "45", "345")
    yolo_choices = [
        f"yolo_p{d}" + (f"_{t}" if t != "pre" else "")
        for d in _yolo_digits for t in _YOLO_TAPS
    ]
    p.add_argument("--backbone", default="dino_vits14",
                   choices=[f"dino_{k}" for k in _DINOV2_HUB] + yolo_choices,
                   help="Feature extractor. yolo_p<digits> with digits ⊆ {3,4,5}: "
                        "single = one level, multi = bilinear-upsample lower-res to highest-res grid + channel-concat. "
                        "Suffix _bb = pre-FPN backbone stage; no suffix = pre-Detect (post-neck); "
                        "_cv3 = cls-branch c3-dim feat before nc-projection. Pipeline downstream is identical.")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Input size. Defaults: 448 (dino) / 640 (yolo).")
    p.add_argument(
        "--weight",
        default="/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa/26m_mergedata_v3/weights/best.pt",
        help="YOLO checkpoint (only used for yolo_* backbones).",
    )
    p.add_argument("--category", default=None,
                   help=f"Comma-separated subset of {MVTEC_CATEGORIES}. Default: all.")
    p.add_argument("--fuse", choices=("concat", "sum", "avg", "patchcore", "concat_pool"), default="concat",
                   help="Multi-layer fusion. concat = channel-concat (dim grows); "
                        "sum/avg = element-wise (dim preserved, equal-dim layers only); "
                        "patchcore = unfold + two-stage adaptive_avg_pool (paper's exact pipeline); "
                        "concat_pool = unfold + concat across layers + single adaptive_avg_pool to target_dim "
                        "(no per-layer pool; needs --patchsize and --target_dim).")
    p.add_argument("--patchsize", type=int, default=3,
                   help="Spatial neighborhood for PatchCore unfold (paper default = 3). Only used with --fuse patchcore.")
    p.add_argument("--pretrain_dim", type=int, default=1024,
                   help="Per-layer target dim after first adaptive_avg_pool. PatchCore paper default = 1024.")
    p.add_argument("--target_dim", type=int, default=1024,
                   help="Final per-patch feature dim after cross-layer adaptive_avg_pool. PatchCore paper default = 1024.")
    p.add_argument("--coreset", type=int, default=None,
                   help="If set, compact the bank to this many points via greedy FPS (PatchCore-style). "
                        "Skipped when bank is already smaller.")
    p.add_argument("--coreset_presample", type=int, default=None,
                   help="If set, randomly subsample bank to this size BEFORE running FPS. "
                        "PatchCore-style speed trick — much faster, near-identical quality. "
                        "Only takes effect with --coreset.")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path.")
    args = p.parse_args()

    if args.category:
        cats = [c.strip() for c in args.category.split(",") if c.strip()]
        unknown = [c for c in cats if c not in MVTEC_CATEGORIES]
        if unknown:
            raise SystemExit(f"unknown category: {unknown}. Valid: {MVTEC_CATEGORIES}")
    else:
        cats = list(MVTEC_CATEGORIES)

    ad, tag, out_csv = _build(args)
    if args.coreset:
        ad.coreset_size = args.coreset
        ad.coreset_presample = args.coreset_presample
        cs_tag = f"_cs{args.coreset}" + (f"_pre{args.coreset_presample}" if args.coreset_presample else "")
        tag = f"{tag}{cs_tag}"
        out_csv = out_csv.with_name(out_csv.name.replace("_mvtec_metrics", f"{cs_tag}_mvtec_metrics"))
    if args.out is not None:
        out_csv = args.out
    elif len(cats) < len(MVTEC_CATEGORIES):
        out_csv = out_csv.with_name(out_csv.stem + f"_{'_'.join(cats)}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"ready: backbone={args.backbone} imgsz={ad.imgsz} device={ad.device} "
          f"feat_dim={ad.feat_dim} grid={ad.grid}"
          + (f" coreset={args.coreset}" if args.coreset else ""))

    rows: list[dict] = []
    for cat in cats:
        print(f"\n{'=' * 60}\n[{cat}] starting\n{'=' * 60}")
        try:
            row = val_one_category(cat, ad)
        except Exception as e:
            print(f"[{cat}] FAILED: {e!r}")
            row = {"category": cat, "image_auroc": float("nan"), "pixel_auroc": float("nan"),
                   "n_support": 0, "n_test": 0, "n_anom_with_gt": 0,
                   "build_s": 0.0, "val_s": 0.0}
        rows.append(row)
        print(f"[{cat}] image={row['image_auroc']:.4f} pixel={row['pixel_auroc']:.4f} "
              f"build={row['build_s']:.1f}s val={row['val_s']:.1f}s "
              f"(support={row['n_support']}, test={row['n_test']}, anom_w_gt={row['n_anom_with_gt']})")

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
