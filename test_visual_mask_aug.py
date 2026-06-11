#!/usr/bin/env python
"""Visualize the bbox->prior mask rendering + augmentation menu (yoloa_v2).

For each defect image (with a YOLO bbox label) this renders the GT bbox into a prior mask and
shows every render/augmentation operation the training pipeline can apply, overlaid on the image,
so the effect of each knob is visible at a glance. Saves one composite per image to ``--out``.

Operations shown (the real code is reused where it matters — ``BboxMaskRenderer`` for rect/gauss,
``YOLOAnomalyV2Model._gaussian_blur`` for blur; magnitude/noise/softmax/remap mirror
``_augment_mask`` / ``_apply_spatial_softmax`` / ``_remap_mask_values`` in nn/tasks.py):

  Row 1: original+bbox | rect | gauss sf=0.15 (sharp) | gauss sf=0.35 | gauss sf=0.75 (diffuse)
  Row 2: gauss+blur | gauss+magnitude(peak) | gauss+noise | gauss+FULL augment | (p_drop -> empty)
  Row 3: gauss->spatial softmax | gauss->remap power | gauss->remap tanh_contrast

Usage:
  python test_visual_mask_aug.py --out runs/temp/mask_aug
  python test_visual_mask_aug.py --category leather --n-per-category 3 --out runs/temp/mask_aug
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import LOGGER
from ultra_ext.im import concat_samh

from compare_grid import CompareGrid

MVTEC_ROOT = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")


def read_yolo_bboxes(label_path: Path) -> torch.Tensor:
    """Read a YOLO label file into an (N, 4) tensor of normalized [cx, cy, w, h]."""
    rows = []
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            rows.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, 4)


def render(mode: str, sf, bboxes: torch.Tensor, mask_size: int) -> torch.Tensor:
    """Render bboxes -> (1, 1, H, H) mask with a fresh renderer in eval mode (deterministic)."""
    r = BboxMaskRenderer(mask_size=mask_size, mode=mode, sigma_factor=sf)
    r.eval()
    idx = torch.zeros(bboxes.shape[0], dtype=torch.long)
    return r(bboxes, idx, 1)


def overlay(cg: CompareGrid, img_rgb: np.ndarray, mask_2d: torch.Tensor, title: str,
            normalize: bool = False) -> np.ndarray:
    """Overlay a single-channel mask on the image as a heatmap, titled."""
    m = mask_2d.detach().cpu().float().numpy()
    if normalize:
        lo, hi = float(m.min()), float(m.max())
        m = (m - lo) / (hi - lo) if hi > lo else m
    return cg._title(CompareGrid.heatmap_overlay(img_rgb, m), title)


def main():
    ap = argparse.ArgumentParser(description="bbox->prior mask augmentation visualizer")
    ap.add_argument("--root", type=str, default=str(MVTEC_ROOT))
    ap.add_argument("--category", type=str, default=None, help="single category (default: all)")
    ap.add_argument("--n-per-category", type=int, default=3, help="defect images per category")
    ap.add_argument("--mask-size", type=int, default=256, help="render resolution")
    ap.add_argument("--panel", type=int, default=320, help="per-panel side length (px)")
    ap.add_argument("--blur-sigma", type=float, default=4.0, help="Gaussian blur sigma (px)")
    ap.add_argument("--mag-peak", type=float, default=0.5, help="magnitude scaling peak (<1 = weak)")
    ap.add_argument("--noise-std", type=float, default=0.2, help="additive noise std")
    ap.add_argument("--softmax-t", type=float, default=1.0, help="spatial softmax temperature")
    ap.add_argument("--power-gamma", type=float, default=2.0, help="remap power gamma")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cg = CompareGrid()
    LOGGER.info(f"Mask-aug viz -> {out_root}")

    cats = [args.category] if args.category else sorted(d.name for d in root.iterdir() if d.is_dir())
    total = 0
    for cat in cats:
        test_dir = root / cat / "test"
        if not test_dir.is_dir():
            continue
        # Collect defect images (skip 'good') that have a non-empty bbox label.
        samples = []
        for defect in sorted(d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"):
            for img in sorted(defect.glob("*.png")) + sorted(defect.glob("*.jpg")):
                lbl = img.with_suffix(".txt")
                if lbl.exists() and read_yolo_bboxes(lbl).numel() > 0:
                    samples.append((img, defect.name))
        random.shuffle(samples)
        samples = samples[: args.n_per_category] if args.n_per_category > 0 else samples

        for img_path, defect in samples:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            panel = cv2.resize(rgb, (args.panel, args.panel))
            bboxes = read_yolo_bboxes(img_path.with_suffix(".txt"))

            # original + drawn GT boxes
            orig = panel.copy()
            for cx, cy, w, h in bboxes.tolist():
                x1, y1 = int((cx - w / 2) * args.panel), int((cy - h / 2) * args.panel)
                x2, y2 = int((cx + w / 2) * args.panel), int((cy + h / 2) * args.panel)
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            orig = cg._title(orig, f"{cat}/{defect} (orig+bbox)")

            H = args.mask_size
            rect = render("rect", 0.25, bboxes, H)[0, 0]
            g15 = render("gauss", 0.15, bboxes, H)[0, 0]
            g35 = render("gauss", 0.35, bboxes, H)[0, 0]
            g75 = render("gauss", 0.75, bboxes, H)[0, 0]

            base = g35.clone()  # the working prior for the augment-effect panels
            # blur (real code path)
            blurred = YOLOAnomalyV2Model._gaussian_blur(base[None, None], args.blur_sigma)[0, 0]
            # magnitude scaling: mask * U(.,.) -> here fixed peak (mirrors _augment_mask)
            magged = (base * args.mag_peak)
            # additive noise then clamp (mirrors _augment_mask)
            noised = (base + torch.randn_like(base) * args.noise_std).clamp(0, 1)
            # full pipeline: blur -> magnitude -> noise -> clamp
            full = YOLOAnomalyV2Model._gaussian_blur(base[None, None], args.blur_sigma)[0, 0]
            full = (full * args.mag_peak + torch.randn_like(full) * args.noise_std).clamp(0, 1)
            # spatial softmax (mirrors _apply_spatial_softmax) — tiny per-pixel, shown normalized
            T = max(args.softmax_t, 1e-6)
            smax = torch.softmax(base.reshape(-1) / T, dim=-1).reshape(H, H)
            # value remaps (mirror _remap_mask_values)
            powered = base ** args.power_gamma
            tanh_c = (0.5 * torch.tanh(10.0 * (base - 0.5)) + 0.5)
            empty = torch.zeros_like(base)  # p_drop -> prior fully dropped

            row1 = concat_samh([
                orig,
                overlay(cg, panel, rect, "rect"),
                overlay(cg, panel, g15, "gauss sf=0.15 (sharp)"),
                overlay(cg, panel, g35, "gauss sf=0.35"),
                overlay(cg, panel, g75, "gauss sf=0.75 (diffuse)"),
            ], gap=10, gap_color=(255, 255, 255), cols=5)
            row2 = concat_samh([
                overlay(cg, panel, blurred, f"+blur sigma={args.blur_sigma}"),
                overlay(cg, panel, magged, f"+magnitude peak={args.mag_peak}"),
                overlay(cg, panel, noised, f"+noise std={args.noise_std}"),
                overlay(cg, panel, full, "+FULL augment"),
                overlay(cg, panel, empty, "p_drop -> empty"),
            ], gap=10, gap_color=(255, 255, 255), cols=5)
            row3 = concat_samh([
                overlay(cg, panel, smax, f"spatial softmax T={args.softmax_t}", normalize=True),
                overlay(cg, panel, powered, f"remap power g={args.power_gamma}"),
                overlay(cg, panel, tanh_c, "remap tanh_contrast"),
            ], gap=10, gap_color=(255, 255, 255), cols=5)

            w = max(row1.shape[1], row2.shape[1], row3.shape[1])
            rows = []
            for r in (row1, row2, row3):
                if r.shape[1] < w:
                    r = cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                                           value=(255, 255, 255))
                rows.append(r)
            sep = np.full((14, w, 3), 255, dtype=np.uint8)
            fig = rows[0]
            for r in rows[1:]:
                fig = np.vstack([fig, sep, r])

            out_path = out_root / cat / f"{defect}__{img_path.stem}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
            total += 1
        LOGGER.info(f"{cat}: {len(samples)} composites")

    LOGGER.info(f"Done. {total} mask-aug composites -> {out_root}")


if __name__ == "__main__":
    main()
