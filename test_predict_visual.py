#!/usr/bin/env python
"""All-category MVTec predict — per-model archive of 8-panel comparison grids.

Iterates every MVTec category (defect + normal test images, capped per category), runs 4 prior
modes (none / segment / memory-bank heatmap / GT mask) and saves a CompareGrid per image. The
trained weights load ONCE; each category's memory bank is built once and cached to disk
(banks/<category>.pt), so re-runs of the same model skip the slow feature extraction.

Grid layout (2 rows x 4 cols):
  Row 1: original | None Prior | seg heatmap | seg prior pred
  Row 2: mb heatmap | heatmap prior pred | GT mask | mask prior pred

Output (one dir per model, named by the ckpt's run id):
  runs/temp/predict_visual/<run_id>/
    banks/<category>.pt          # cached memory bank
    <category>/<type>__<stem>.jpg

Usage:
  python test_predict_visual.py --ckpt <best.pt> --yaml <model.yaml>                  # all 15 categories
  python test_predict_visual.py --ckpt ... --yaml ... --category zipper --n-per-category 3
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device
from compare_grid import CompareGrid

MVTEC_ROOT = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")


def collect_test_images(test_root: Path, n: int) -> list[tuple[str, str]]:
    """Return up to n [(path, defect_type), ...] randomly sampled across a category's test subdirs.

    Includes the ``good`` subdir (normal samples). ``n <= 0`` returns all images.
    """
    pairs = []
    for subdir in sorted(test_root.iterdir()):
        if subdir.is_dir():
            for p in sorted(subdir.glob("*.png")):
                pairs.append((str(p), subdir.name))
    random.shuffle(pairs)
    return pairs[:n] if n and n > 0 else pairs


def load_mask_tensor(mask_path: str | None, imgsz: int) -> torch.Tensor | None:
    """Load GT mask as (1, 1, H, W) float32 tensor."""
    if mask_path is None:
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
    m = m.astype(np.float32) / 255.0
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0)


def run_prior(y: YOLO, model: YOLOAnomalyV2Model, img_path: str, prior_mode: str,
              imgsz: int, conf: float = 0.1, iou: float = 0.1,
              external_mask: torch.Tensor | None = None, device=None):
    """Run predict with a prior mode, return (pred_rgb, n_det, heatmap_np)."""
    res = y.predict(img_path, imgsz=imgsz, prior_mode=prior_mode, conf=conf, iou=iou,
                    external_mask=external_mask, device=device, verbose=False)
    r = res[0]
    n_det = r.boxes.shape[0] if r.boxes is not None else 0
    pred_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
    hmap = getattr(model, "_last_heatmap", None)
    hmap_np = hmap.cpu().numpy().squeeze() if hmap is not None else None
    return pred_rgb, n_det, hmap_np


def save_bank(mb, path: Path) -> None:
    """Persist a frozen memory bank (tensor + feature_dim + temperature) for fast reload."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"memory_bank": mb.memory_bank.detach().cpu(),
         "feature_dim": mb.feature_dim,
         "temperature": float(mb.temperature)},
        path,
    )


def restore_bank(mb, path: Path) -> None:
    """Restore a bank saved by save_bank, skipping load_support_set feature extraction.

    Reuses BackboneMemoryBank.load_bank (sets the bank buffer + feature_dim), then restores the
    calibrated temperature and marks the bank frozen (update=False) so forward runs in scoring mode.
    """
    d = torch.load(path, map_location="cpu")
    mb.load_bank(d["memory_bank"])  # re-normalizes + sets feature_dim, onto the bank's device
    mb.temperature = d["temperature"]
    mb.update = False


def _report_load(matched: dict, ckpt_state: dict, model_state: dict) -> None:
    """Log which keys matched, which are missing in ckpt (random init), and
    which are missing in model (YAML mismatch / removed modules)."""
    ckpt_keys = set(ckpt_state.keys())
    model_keys = set(model_state.keys())
    matched_keys = set(matched.keys())
    missing_in_ckpt = model_keys - matched_keys
    missing_in_model = ckpt_keys - matched_keys

    LOGGER.info(
        f"Weight load: {len(matched)}/{len(ckpt_keys)} ckpt keys matched "
        f"(model has {len(model_keys)} total)"
    )
    if missing_in_model:
        grouped = _group_keys(missing_in_model)
        LOGGER.warning(
            f"  {len(missing_in_model)} keys in ckpt but NOT in model:\n"
            + "\n".join(f"    [{p}] {', '.join(sorted(ks)[:6])}{'...' if len(ks) > 6 else ''}"
                        for p, ks in grouped)
        )
    if missing_in_ckpt:
        grouped = _group_keys(missing_in_ckpt)
        LOGGER.warning(
            f"  {len(missing_in_ckpt)} keys in model but NOT in ckpt (RANDOM INIT):\n"
            + "\n".join(f"    [{p}] {', '.join(sorted(ks)[:6])}{'...' if len(ks) > 6 else ''}"
                        for p, ks in grouped)
        )


def _group_keys(keys: set[str]) -> list[tuple[str, list[str]]]:
    """Group state-dict keys by top-level prefix (e.g. 'model.2', 'seg_branch')."""
    groups: dict[str, list[str]] = {}
    for k in sorted(keys):
        prefix = k.split(".")[0]
        groups.setdefault(prefix, []).append(k)
    return sorted(groups.items())


MODEL_W = "runs/yoloa_v2/26m_yoloav2_film_gauss_v1/weights/best.pt"
YAML = "yolo26m-anomaly-v2-film-gauss.yaml"


def main():
    parser = argparse.ArgumentParser(description="All-category MVTec predict — per-model grid archive")
    parser.add_argument("--ckpt", type=str, default=MODEL_W)
    parser.add_argument("--yaml", type=str, default=YAML)
    parser.add_argument("--category", type=str, default=None,
                        help="Single category (default: all 15). 'all' = all.")
    parser.add_argument("--n-per-category", type=int, default=20,
                        help="Max test images per category (0 = all)")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.1, help="NMS IoU threshold (applied to all prior modes)")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_bank", type=int, default=10000)
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Cap on normal images for bank (0=all)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--nc", type=int, default=None,
                        help="Number of classes (default: auto-detect from ckpt; multiclass-safe)")
    parser.add_argument("--heat-norm", type=str, default="minmax", choices=["none", "minmax"],
                        help="Per-image prior normalization before fusion (minmax stretches to [0,1])")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: runs/temp/predict_visual/<run_id>)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for per-category sampling (same images across models; idempotent re-runs)")
    args = parser.parse_args()

    random.seed(args.seed)
    device = select_device(args.device or "cpu")  # handles cpu / mps / "0" / cuda:0

    # run_id from ckpt path: <run>/weights/best.pt -> <run>
    ckpt_path = Path(args.ckpt).resolve()
    run_id = ckpt_path.parents[1].name if ckpt_path.parent.name == "weights" else ckpt_path.stem
    out_root = Path(args.out) if args.out else Path(f"runs/temp/predict_visual/{run_id}")
    banks_dir = out_root / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Archive dir: {out_root}")

    # ---- Build model ONCE (weights shared across categories) ----
    # Load the ckpt FIRST so the head can be sized to its class count: a multiclass checkpoint
    # (e.g. nc=35) built with nc=1 would have its cls-head weights shape-mismatched and silently
    # dropped, leaving a random classifier. nc comes from --nc, else auto-detected from the ckpt.
    LOGGER.info("Building model...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        inner = ckpt.get("ema") or ckpt.get("model")
        ckpt_state = inner.state_dict() if hasattr(inner, "state_dict") else inner
    else:
        inner, ckpt_state = None, ckpt
    ckpt_names = getattr(inner, "names", None)
    nc = args.nc or getattr(inner, "nc", None) or (len(ckpt_names) if ckpt_names else None) or 1
    LOGGER.info(f"nc = {nc}" + (" (--nc)" if args.nc else " (auto-detected from ckpt)"))

    model = YOLOAnomalyV2Model(args.yaml, nc=nc, verbose=False)
    ms = model.state_dict()
    matched = {k: v for k, v in ckpt_state.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    _report_load(matched, ckpt_state, ms)
    if ckpt_names:
        model.names = ckpt_names  # plotted detections show the real class labels
    model.to(device)
    model.eval()
    model.heatmap_norm = args.heat_norm
    LOGGER.info(f"heatmap_norm = {model.heatmap_norm}")
    has_bank = model.memory_bank is not None
    if not has_bank:
        LOGGER.warning("Model has no memory bank (no bb_layers) — 'heatmap' prior falls back to no-prior; "
                       "'segment' also falls back if the model has no SegBranch. Only 'none'/'mask' are meaningful.")

    y = YOLO(args.yaml)
    y.model = model
    cg = CompareGrid()

    # ---- Categories ----
    if args.category and args.category.lower() != "all":
        cats = [args.category]
    else:
        cats = sorted(d.name for d in MVTEC_ROOT.iterdir() if d.is_dir())
    LOGGER.info(f"Categories ({len(cats)}): {cats}")

    total = 0
    for ci, cat in enumerate(cats, 1):
        test_root = MVTEC_ROOT / cat / "test"
        if not test_root.is_dir():
            LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: no test/ dir, skipping")
            continue

        # ---- Per-category memory bank: restore from cache, or build once + save ----
        if has_bank:
            mb = model.memory_bank
            bank_path = banks_dir / f"{cat}.pt"
            if bank_path.exists():
                restore_bank(mb, bank_path)
                LOGGER.info(f"[{ci}/{len(cats)}] {cat}: loaded cached bank ({mb.memory_bank.shape[0]} vecs)")
            else:
                train_dir = MVTEC_ROOT / cat / "train" / "good"
                if not train_dir.is_dir():
                    train_dir = MVTEC_ROOT / cat / "train"
                LOGGER.info(f"[{ci}/{len(cats)}] {cat}: building bank from {train_dir} ...")
                mb.reset_memory_bank()
                model.load_support_set(str(train_dir), imgsz=args.imgsz, device=device,
                                       batch=args.batch, max_bank_size=args.max_bank,
                                       max_images=args.max_images, verbose=False)
                save_bank(mb, bank_path)
                LOGGER.info(f"[{ci}/{len(cats)}] {cat}: bank built+saved ({mb.memory_bank.shape[0]} vecs) -> {bank_path}")

        # ---- Sample test images + save grids ----
        samples = collect_test_images(test_root, args.n_per_category)
        cat_out = out_root / cat
        for img_path, label in samples:
            original = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask_path = CompareGrid.find_mask(img_path)
            mask_tensor = load_mask_tensor(mask_path, args.imgsz)

            none_pred, n_none, _ = run_prior(y, model, img_path, "none", args.imgsz, args.conf, iou=args.iou, device=device)
            seg_pred, n_seg, seg_hmap = run_prior(y, model, img_path, "segment", args.imgsz, args.conf, iou=args.iou, device=device)
            seg_heat = CompareGrid.heatmap_panel(original, seg_hmap)
            heat_pred, n_heat, heat_hmap = run_prior(y, model, img_path, "heatmap", args.imgsz, args.conf, iou=args.iou, device=device)
            heat_heat = CompareGrid.heatmap_panel(original, heat_hmap)
            mask_pred, n_mask, _ = run_prior(y, model, img_path, "mask", args.imgsz, args.conf,
                                             iou=args.iou, external_mask=mask_tensor, device=device)
            mask_img = CompareGrid.mask_panel(original, mask_path)

            cg.save(
                original=original, none_pred=none_pred, seg_heat=seg_heat, seg_pred=seg_pred,
                heat_heat=heat_heat, heat_pred=heat_pred, mask_img=mask_img, mask_pred=mask_pred,
                out_path=cat_out / f"{label}__{Path(img_path).stem}.jpg",
                n_none=n_none, n_seg=n_seg, n_heat=n_heat, n_mask=n_mask,
            )
            total += 1
        LOGGER.info(f"[{ci}/{len(cats)}] {cat}: {len(samples)} grids -> {cat_out}")

    LOGGER.info(f"Done. {total} grids across {len(cats)} categories -> {out_root}")


if __name__ == "__main__":
    main()
