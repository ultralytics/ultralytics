#!/usr/bin/env python
"""Random-sample predict — save 4×2 comparison grids via CompareGrid.

Layout (2 rows × 4 cols):
  Row 1: original | None Prior | seg heatmap | seg prior pred
  Row 2: mb heatmap | heatmap prior pred | GT mask | mask prior pred

Usage:
    python test_predict_visual.py                         # leather, 4 images
    python test_predict_visual.py --category carpet --n 8
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
from compare_grid import CompareGrid


def collect_test_images(test_root: Path, n: int) -> list[tuple[str, str]]:
    """Return [(path, label_name), ...] randomly sampled from test subdirs."""
    pairs = []
    for subdir in sorted(test_root.iterdir()):
        if subdir.is_dir():
            label = subdir.name
            for p in sorted(subdir.glob("*.png")):
                pairs.append((str(p), label))
    random.shuffle(pairs)
    return pairs[:n]


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
              imgsz: int, external_mask: torch.Tensor | None = None):
    """Run predict with a prior mode, return (pred_rgb, n_det, heatmap_np)."""
    res = y.predict(img_path, imgsz=imgsz, prior_mode=prior_mode,conf=0.05,
                    external_mask=external_mask, verbose=False)
    r = res[0]
    n_det = r.boxes.shape[0] if r.boxes is not None else 0
    pred_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
    hmap = getattr(model, "_last_heatmap", None)
    hmap_np = hmap.cpu().numpy().squeeze() if hmap is not None else None
    return pred_rgb, n_det, hmap_np


MODEL_W="/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_seg_a1_v1/weights/best.pt"


# MODEL_W="/Users/louis/workspace/ultra_louis_work/ultralytics/runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_cm2_v1/weights/best.pt"

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


def main():
    parser = argparse.ArgumentParser(description="Random-sample predict — 4×2 comparison grids")
    parser.add_argument("--ckpt", type=str,
                        default=MODEL_W)
    parser.add_argument("--yaml", type=str, default="yolo26m-anomaly-v2.yaml")
    parser.add_argument("--category", type=str, default="leather")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--n", type=int, default=4, help="Number of random test images")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_bank", type=int, default=10000)
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Cap on normal images for bank (0=all)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: runs/temp/predict_visual/)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or "cpu"
    mvtec_root = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")
    train_dir = mvtec_root / args.category / "train" / "good"
    if not train_dir.is_dir():
        train_dir = mvtec_root / args.category / "train"
    test_root = mvtec_root / args.category / "test"

    out_root = Path(args.out or f"runs/temp/predict_visual/{args.category}")

    # ---- Build model + memory bank ----
    LOGGER.info("Building model...")
    model = YOLOAnomalyV2Model(args.yaml, nc=1, verbose=False)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt_state = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
    else:
        ckpt_state = ckpt
    ms = model.state_dict()
    matched = {k: v for k, v in ckpt_state.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)

    # ---- Report loading status ----
    _report_load(matched, ckpt_state, ms)
    model.to(device)
    model.eval()

    LOGGER.info("Building memory bank...")
    model.load_support_set(str(train_dir), imgsz=args.imgsz, device=device,
                           batch=args.batch, max_bank_size=args.max_bank,
                           max_images=args.max_images, verbose=True)

    # ---- Sample test images ----
    samples = collect_test_images(test_root, args.n)
    LOGGER.info(f"Sampled {len(samples)} test images")

    y = YOLO(args.yaml)
    y.model = model

    cg = CompareGrid()

    for img_path, label in samples:
        LOGGER.info(f"{img_path}  [{label}]")

        # Read original
        original_bgr = cv2.imread(img_path)
        original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        # Find GT mask
        mask_path = CompareGrid.find_mask(img_path)
        mask_tensor = load_mask_tensor(mask_path, args.imgsz)

        # ---- None Prior ----
        none_pred, n_none, _ = run_prior(y, model, img_path, "none", args.imgsz)

        # ---- Segment Prior ----
        seg_pred, n_seg, seg_hmap = run_prior(y, model, img_path, "segment", args.imgsz)
        seg_heat = CompareGrid.heatmap_panel(original, seg_hmap)

        # ---- Heatmap Prior (Memory Bank) ----
        heat_pred, n_heat, heat_hmap = run_prior(y, model, img_path, "heatmap", args.imgsz)
        heat_heat = CompareGrid.heatmap_panel(original, heat_hmap)

        # ---- Mask Prior (GT) ----
        mask_pred, n_mask, _ = run_prior(y, model, img_path, "mask", args.imgsz,
                                         external_mask=mask_tensor)
        mask_img = CompareGrid.mask_panel(original, mask_path)

        # ---- Save grid ----
        out_path = cg.save(
            original=original,
            none_pred=none_pred,
            seg_heat=seg_heat,
            seg_pred=seg_pred,
            heat_heat=heat_heat,
            heat_pred=heat_pred,
            mask_img=mask_img,
            mask_pred=mask_pred,
            out_path=out_root / f"{Path(img_path).stem}.jpg",
            n_none=n_none, n_seg=n_seg, n_heat=n_heat, n_mask=n_mask,
        )
        LOGGER.info(f"  -> {out_path}")

    LOGGER.info(f"Done. {len(samples)} images saved to {out_root}")


if __name__ == "__main__":
    main()
