# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 validator — 2-pass (mask-on + mask-off) with AUROC.

Pass 1 (mask-on):  forward with GT bbox rendered mask.
Pass 2 (mask-off): forward with mask disabled (passthrough ≈ vanilla YOLO).

Metrics from pass 2 are prefixed ``mask_off/``. The canonical loss reported to
the trainer comes from pass 1.

In addition to standard detection metrics, accumulates image-AUROC and
pixel-AUROC from the model's stashed heatmap (``model._last_heatmap``) during
pass 1 only.

When ``prior_mode`` is set (standalone val), runs single-pass with that mode
and still computes AUROC.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from ._util import resolve_v2_model

_SENTINEL = object()  # "prior mode not snapshotted" marker (distinct from a real None)


class AnomalyV2Validator(DetectionValidator):
    """Detection validator: 2-pass during training (mask-on / mask-off), single-pass for prior modes."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None,
                 prior_mode: str | None = None) -> None:
        if isinstance(args, dict):
            prior_mode = args.pop("prior_mode", prior_mode)
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "anomaly_v2"
        self.prior_mode = prior_mode
        self._mask_mode = "on"
        self._model_ref = None

        # Heatmap-prior memory-bank lifecycle (single-pass OOD eval). When prior_mode
        # is "heatmap", build the BackboneMemoryBank from the dataset's train (normal)
        # split inside this validator, then restore prior_mode + drop the bank after —
        # so a shared/EMA model is left clean (no ckpt bloat, no stale prior).
        self._ood_bank_size = 10000
        self._saved_prior_mode = _SENTINEL
        self._built_bank = False

        # AUROC accumulators
        self._auroc_image_scores: list[float] = []
        self._auroc_image_labels: list[int] = []
        self._auroc_pixel_scores: list[float] = []
        self._auroc_pixel_labels: list[int] = []

        from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer
        self._eval_mask_renderer = BboxMaskRenderer(mask_size=256, mode="rect")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        self._model_ref = model
        m = resolve_v2_model(model)
        if m is not None and self.prior_mode is not None and hasattr(m, "set_prior_mode"):
            self._saved_prior_mode = getattr(m, "_prior_mode", None)  # snapshot for restore
            effective = self.prior_mode
            if self.prior_mode == "heatmap" and not self._ensure_memory_bank(m):
                # No usable bank (no bb_layers / no normal images / build failed): never run the
                # heatmap forward against a missing-or-empty bank — fall back to bare detection so
                # the run is honest (AUROC nan) instead of crashing or scoring against 0 vectors.
                LOGGER.warning("AnomalyV2Validator: heatmap prior unavailable -> running prior_mode='none'.")
                effective = "none"
            m.set_prior_mode(effective)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        model = resolve_v2_model(self._model_ref)
        if model is None or not hasattr(model, "set_mask_input"):
            return batch
        if self.prior_mode is not None:
            # Single-pass prior_mode: "mask" mode injects GT bboxes
            if self.prior_mode == "mask":
                bb = batch.get("bboxes")
                bi = batch.get("batch_idx")
                if bb is not None and bi is not None:
                    model.set_mask_input(bb, bi)
        elif self._mask_mode == "on":
            bb = batch.get("bboxes")
            bi = batch.get("batch_idx")
            if bb is not None and bi is not None:
                model.set_mask_input(bb, bi)
            else:
                model.disable_mask_once()
        else:
            model.disable_mask_once()
        return batch

    # ------------------------------------------------------------------
    # 2-pass __call__ (training val) or single-pass (prior_mode)
    # ------------------------------------------------------------------
    def __call__(self, trainer=None, model=None):
        self._built_bank = False
        self._saved_prior_mode = _SENTINEL
        try:
            if self.prior_mode is not None:
                return self._single_pass(trainer, model)
            return self._two_pass(trainer, model)
        finally:
            self._restore_prior_state()  # undo prior_mode + any bank we built (YOLOE-style)

    def _single_pass(self, trainer, model):
        self._reset_auroc()
        if self.prior_mode == "heatmap":
            self.args.rect = False
        return super().__call__(trainer=trainer, model=model)

    def _two_pass(self, trainer, model):
        # Pass 1: mask-on
        self._mask_mode = "on"
        self._reset_auroc()
        stats_on = super().__call__(trainer=trainer, model=model)

        # Snapshot mask-on metrics before pass 2 overwrites them
        metrics_on = self.metrics
        image_auroc = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
        pixel_auroc = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)

        # Pass 2: mask-off
        self._mask_mode = "off"
        try:
            stats_off = super().__call__(trainer=trainer, model=model)
        except Exception as e:
            LOGGER.warning(f"AnomalyV2Validator: mask-off pass failed: {e}")
            stats_off = {}

        self._mask_mode = "on"

        # Restore mask-on metrics so results_dict shows mask-on by default
        self.metrics = metrics_on

        if not isinstance(stats_on, dict):
            return stats_on
        merged = dict(stats_on)
        merged["image_auroc"] = image_auroc
        merged["pixel_auroc"] = pixel_auroc
        if isinstance(stats_off, dict):
            for k, v in stats_off.items():
                merged[f"mask_off/{k}"] = v
            # Stash mask-off stats for external access
            self._mask_off_stats = dict(stats_off)
        return merged

    # ------------------------------------------------------------------
    # Heatmap-prior memory-bank lifecycle (single-pass OOD eval)
    # ------------------------------------------------------------------
    def _collect_support_paths(self) -> list[str]:
        """Resolve normal (good) image paths from the dataset's train split for the bank."""
        from pathlib import Path

        data = self.data if isinstance(self.data, dict) else {}
        train = data.get("train")
        if train is None:
            return []
        if isinstance(train, (list, tuple)):
            return [str(p) for p in train]
        p = Path(train)
        if p.suffix.lower() == ".txt":  # image-list file (e.g. MVTec category train.txt)
            root = Path(data.get("path", p.parent))
            out = []
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    q = Path(ln)
                    out.append(str(q if q.is_absolute() else root / q))
            return out
        return [str(p)]  # a directory (load_support_set iterates it) or single image

    def _ensure_memory_bank(self, m) -> bool:
        """Build the bank from the train (normal) split if empty. Returns True iff a usable bank is ready."""
        mb = getattr(m, "memory_bank", None)
        if mb is None or getattr(m, "_bb_layers", None) is None:
            LOGGER.warning(
                "AnomalyV2Validator: prior_mode='heatmap' but no BackboneMemoryBank is configured "
                "(model YAML needs a bb_layers block)."
            )
            return False
        if mb.memory_bank is not None and mb.memory_bank.shape[0] > 0:
            return True  # caller already supplied a bank — use it, don't rebuild or clear it
        support = self._collect_support_paths()
        if not support:
            LOGGER.warning("AnomalyV2Validator: no train (normal) images found; cannot build memory bank.")
            return False
        imgsz = self.args.imgsz if isinstance(self.args.imgsz, int) else 640
        try:
            n = m.load_support_set(support, imgsz=imgsz, device=self.device,
                                   max_bank_size=self._ood_bank_size, verbose=False)
        except Exception as e:  # missing/corrupt images, OOM, etc. — never crash the val run
            LOGGER.warning(f"AnomalyV2Validator: memory bank build failed ({type(e).__name__}: {e}).")
            return False
        if not n:
            LOGGER.warning("AnomalyV2Validator: memory bank is empty after build.")
            return False
        LOGGER.info(f"AnomalyV2Validator: built memory bank ({n} features) from {len(support)} normal images.")
        self._built_bank = True
        return True

    def _restore_prior_state(self) -> None:
        """Undo prior_mode + any bank we built so a shared/EMA model is left clean."""
        if self._saved_prior_mode is _SENTINEL and not self._built_bank:
            return
        m = resolve_v2_model(self._model_ref)
        if m is None:
            return
        if self._saved_prior_mode is not _SENTINEL and hasattr(m, "set_prior_mode"):
            m.set_prior_mode(self._saved_prior_mode)
            self._saved_prior_mode = _SENTINEL
        if self._built_bank:
            mb = getattr(m, "memory_bank", None)
            if mb is not None and hasattr(mb, "reset_memory_bank"):
                mb.reset_memory_bank()  # free the bank; prevents EMA/ckpt bloat mid-training
            self._built_bank = False

    def plot_val_samples(self, batch, ni):
        """Slice the cached-prior channel off before plotting (plot_images expects 3-channel)."""
        if batch["img"].shape[1] == 4:
            batch = {**batch, "img": batch["img"][:, :3]}
        super().plot_val_samples(batch, ni)

    def plot_predictions(self, batch, preds, ni):
        """Slice the cached-prior channel off before plotting (plot_images expects 3-channel)."""
        if batch["img"].shape[1] == 4:
            batch = {**batch, "img": batch["img"][:, :3]}
        super().plot_predictions(batch, preds, ni)

    # ------------------------------------------------------------------
    # AUROC
    # ------------------------------------------------------------------
    def _reset_auroc(self):
        self._auroc_image_scores = []
        self._auroc_image_labels = []
        self._auroc_pixel_scores = []
        self._auroc_pixel_labels = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        m = resolve_v2_model(self._model_ref)
        if m is None:
            return
        heatmap = getattr(m, "_last_heatmap", None)
        if heatmap is None or heatmap.numel() == 0:
            return

        bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        bs = heatmap.shape[0]

        for b in range(bs):
            has_anom = 0
            if bboxes is not None and batch_idx is not None and batch_idx.numel() > 0:
                has_anom = int((batch_idx == b).any().item())
            img_score = heatmap[b].max().item()
            self._auroc_image_scores.append(img_score)
            self._auroc_image_labels.append(has_anom)

            if has_anom and bboxes is not None and batch_idx is not None:
                bb_per_img = bboxes[batch_idx == b]
                if bb_per_img.numel() > 0:
                    gt_mask = self._eval_mask_renderer(
                        bb_per_img,
                        torch.zeros(bb_per_img.shape[0], dtype=torch.long, device=bb_per_img.device),
                        1,
                    )
                    hmap_b = heatmap[b:b + 1]
                    if hmap_b.shape[2] != 256 or hmap_b.shape[3] != 256:
                        hmap_b = F.interpolate(hmap_b, size=(256, 256),
                                               mode="bilinear", align_corners=False)
                    self._auroc_pixel_scores.extend(hmap_b.flatten().cpu().tolist())
                    self._auroc_pixel_labels.extend(gt_mask.flatten().cpu().tolist())

    @staticmethod
    def _compute_auroc(scores: list[float], labels: list[int]) -> float:
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return float("nan")
        if len(scores) == 0 or len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    def finalize_metrics(self):
        image_auroc = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
        pixel_auroc = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)
        self.metrics.image_auroc = image_auroc
        self.metrics.pixel_auroc = pixel_auroc
        super().finalize_metrics()

    def get_stats(self):
        stats = super().get_stats()
        stats["image_auroc"] = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
        if any(self._auroc_pixel_labels):
            stats["pixel_auroc"] = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)
        else:
            stats["pixel_auroc"] = float("nan")
        return stats


# ----------------------------------------------------------------------
# MVTec cross-dataset OOD evaluation (3-mode sweep)
#
# Measures whether the trained fusion can USE a prior from an unseen distribution.
# Per category, runs this validator in three single-pass modes on the test split:
#   mask_off (prior_mode="none")    -> OOD bare-detection lower bound
#   heatmap  (prior_mode="heatmap") -> core: can fusion use a memory-bank prior?
#   mask_on  (prior_mode="mask")    -> GT-bbox upper bound (perfect prior)
# Shared by the standalone post-hoc path and the AnomalyV2Trainer callback.
# ----------------------------------------------------------------------
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
_MODE_TO_PRIOR = {"mask_off": "none", "heatmap": "heatmap", "mask_on": "mask"}
_OOD_CSV_FIELDS = ["epoch", "category", "mode", "mAP50", "mAP50_95", "P", "R", "image_auroc", "pixel_auroc"]

# Candidate dataset roots in priority order (env var wins, then ultra6, then laptop).
_MVTEC_ROOT_CANDIDATES = (
    "/data/shared-datasets/louis_data/MVTec-YOLO",
    "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO",
)


def resolve_mvtec_root(explicit: str | None = None) -> Path | None:
    """First existing directory among: explicit arg, ``$MVTEC_ROOT``, ultra6 path, laptop path."""
    for c in (explicit, os.environ.get("MVTEC_ROOT"), *_MVTEC_ROOT_CANDIDATES):
        if c and Path(c).is_dir():
            return Path(c)
    return None


def _category_yaml(root: Path, cat: str) -> Path | None:
    """Prefer the single-class ``<cat>_binary.yaml`` over the multi-class ``<cat>.yaml``."""
    for name in (f"{cat}_binary.yaml", f"{cat}.yaml"):
        p = root / cat / name
        if p.exists():
            return p
    return None


def run_mvtec_ood_eval(
    model,
    mvtec_root: str | Path,
    categories: list[str] | None = None,
    modes: tuple[str, ...] = ("mask_off", "heatmap", "mask_on"),
    imgsz: int = 320,
    batch: int = 8,
    device=None,
    bank_size: int = 10000,
    save_dir: str | Path | None = None,
    epoch: int | None = None,
) -> list[dict]:
    """Run the 3-mode MVTec OOD eval over ``categories``; ``model`` is a YOLOAnomalyV2Model.

    Each (category, mode) is isolated in try/except so one failure never aborts the sweep.
    Returns per-(category, mode) rows plus per-mode ``AVERAGE`` rows; appends ``mvtec_ood.csv``
    under ``save_dir`` when given. Reuses :class:`AnomalyV2Validator` via the ``model.val`` args
    path (the validator pops ``prior_mode`` from the dict and builds/drops the bank internally).
    """
    root = Path(mvtec_root)
    m = unwrap_model(model)
    rows: list[dict] = []

    for cat in (categories or MVTEC_CATEGORIES):
        yaml = _category_yaml(root, cat)
        if yaml is None:
            LOGGER.warning(f"MVTec OOD: no data yaml for category '{cat}' under {root}; skipping.")
            continue
        for mode in modes:
            try:
                overrides = {
                    "task": "anomaly_v2", "mode": "val", "data": str(yaml), "split": "val",
                    "imgsz": imgsz, "batch": batch, "device": str(device) if device is not None else None,
                    "rect": False, "plots": False, "verbose": False, "save_json": False,
                    "prior_mode": _MODE_TO_PRIOR[mode],  # popped by AnomalyV2Validator.__init__
                }
                sd = Path(save_dir) / "mvtec_ood_runs" if save_dir is not None else None
                validator = AnomalyV2Validator(args=overrides, save_dir=sd)
                validator._ood_bank_size = bank_size
                validator(model=m)  # builds bank (heatmap) + restores prior state internally
                box = validator.metrics.box
                rows.append({
                    "epoch": epoch, "category": cat, "mode": mode,
                    "mAP50": float(box.map50), "mAP50_95": float(box.map),
                    "P": float(box.mp), "R": float(box.mr),
                    "image_auroc": float(getattr(validator.metrics, "image_auroc", math.nan)),
                    "pixel_auroc": float(getattr(validator.metrics, "pixel_auroc", math.nan)),
                })
            except Exception as e:
                LOGGER.warning(f"MVTec OOD: {cat}/{mode} failed ({type(e).__name__}: {e}); recording NaN.")
                rows.append({"epoch": epoch, "category": cat, "mode": mode, "mAP50": math.nan,
                             "mAP50_95": math.nan, "P": math.nan, "R": math.nan,
                             "image_auroc": math.nan, "pixel_auroc": math.nan})

    rows.extend(_ood_macro_average(rows, epoch))
    if save_dir is not None:
        _append_ood_csv(rows, Path(save_dir) / "mvtec_ood.csv")
    return rows


def _ood_macro_average(rows: list[dict], epoch: int | None) -> list[dict]:
    """One ``AVERAGE`` row per mode, averaging each metric over categories (ignoring NaN)."""
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] != "AVERAGE":
            by_mode[r["mode"]].append(r)
    out = []
    for mode, rs in by_mode.items():
        def _avg(key: str) -> float:
            vals = [r[key] for r in rs if not math.isnan(r[key])]
            return sum(vals) / len(vals) if vals else math.nan
        avg = {"epoch": epoch, "category": "AVERAGE", "mode": mode,
               "mAP50": _avg("mAP50"), "mAP50_95": _avg("mAP50_95"), "P": _avg("P"), "R": _avg("R"),
               "image_auroc": _avg("image_auroc"), "pixel_auroc": _avg("pixel_auroc")}
        out.append(avg)
        LOGGER.info(f"MVTec OOD @ep{epoch} [{mode:8s}] mAP50={avg['mAP50']:.4f} "
                    f"mAP50-95={avg['mAP50_95']:.4f} image_auroc={avg['image_auroc']:.4f} "
                    f"pixel_auroc={avg['pixel_auroc']:.4f} (n={len(rs)})")
    return out


def _append_ood_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_OOD_CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _OOD_CSV_FIELDS})
