#!/usr/bin/env python
"""Visualize 'distractor-prior' noise: fuse other samples' GT masks into the prior.

The idea (robustness aug): instead of per-pixel Gaussian noise everywhere, inject a few OTHER
samples' rendered GT defect blobs into this image's prior via max-merge. The image's own defect
region stays intact, but extra defect-shaped hints appear at wrong locations (where THIS image
has no defect). The detector must then verify the prior against image content rather than blindly
trusting it. This mirrors how a real memory-bank heatmap fires spurious blobs on clean regions.

  noisy_prior = max( own_gauss,  max_k(donor_k_gauss) ),   donors ~ other samples in the batch

Each gauss blob uses a RANDOM width per box (sigma_factor ~ U(lo, hi)), like training, so the
prior width varies sample-to-sample. For each defect image this saves a row:
  original+bbox | own prior (clean) | noisy prior (n donors) for each n in --donor-counts

Usage:
  python test_visual_prior_noise.py --out runs/temp/prior_noise
  python test_visual_prior_noise.py --category leather --donor-counts 2,4,8 --out runs/temp/prior_noise
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer
from ultralytics.utils import LOGGER
from ultra_ext.im import concat_samh

from compare_grid import CompareGrid

MVTEC_ROOT = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")


def titled(panel: np.ndarray, text: str, bar_h: int = 30) -> np.ndarray:
    """Add a black title bar on top, auto-shrinking the font so the full text always fits."""
    h, w = panel.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    scale, thick = 0.7, 2
    while scale > 0.3:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw <= w - 12:
            break
        scale -= 0.05
        if scale <= 0.45:
            thick = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(bar, text, (max((w - tw) // 2, 4), (bar_h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return np.vstack([bar, panel])


def read_yolo_bboxes(label_path: Path) -> torch.Tensor:
    """Read a YOLO label file into an (N, 4) tensor of normalized [cx, cy, w, h]."""
    rows = []
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            rows.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, 4)


def collect_pool(root: Path, category: str | None) -> list[tuple[Path, str, torch.Tensor]]:
    """All defect images (with a non-empty bbox label) as (img_path, 'cat/defect', bboxes)."""
    cats = [category] if category else sorted(d.name for d in root.iterdir() if d.is_dir())
    pool = []
    for cat in cats:
        test_dir = root / cat / "test"
        if not test_dir.is_dir():
            continue
        for defect in sorted(d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"):
            for img in sorted(defect.glob("*.png")) + sorted(defect.glob("*.jpg")):
                bb = read_yolo_bboxes(img.with_suffix(".txt"))
                if bb.numel() > 0:
                    pool.append((img, f"{cat}/{defect.name}", bb))
    return pool


def draw_boxes(img: np.ndarray, bboxes: torch.Tensor, panel: int) -> None:
    """Draw normalized [cx,cy,w,h] boxes (green) on an image in-place."""
    for cx, cy, w, h in bboxes.tolist():
        x1, y1 = int((cx - w / 2) * panel), int((cy - h / 2) * panel)
        x2, y2 = int((cx + w / 2) * panel), int((cy + h / 2) * panel)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser(description="distractor-prior noise visualizer")
    ap.add_argument("--root", type=str, default=str(MVTEC_ROOT))
    ap.add_argument("--category", type=str, default=None, help="single category (default: all)")
    ap.add_argument("--n-per-category", type=int, default=5, help="target images per category")
    ap.add_argument("--donor-counts", type=str, default="2,4,8",
                    help="comma list of donor counts to visualize")
    ap.add_argument("--sigma-range", type=str, default="0.15,0.6",
                    help="gauss width sampled per box ~ U(lo,hi); pass 'v' for fixed v")
    ap.add_argument("--mask-size", type=int, default=256)
    ap.add_argument("--panel", type=int, default=380)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    counts = [int(x) for x in args.donor_counts.split(",") if x.strip()]
    sr = [float(x) for x in args.sigma_range.split(",") if x.strip()]
    sigma_factor = sr if len(sr) == 2 else sr[0]
    H = args.mask_size

    # Random per-box gauss width: a [lo,hi] range + train() mode makes the renderer sample
    # sigma_factor ~ U(lo,hi) on every forward (exactly like training-time prior-width aug).
    renderer = BboxMaskRenderer(mask_size=H, mode="gauss", sigma_factor=sigma_factor)
    renderer.train()

    def gauss(bboxes: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros(bboxes.shape[0], dtype=torch.long)
        return renderer(bboxes, idx, 1)[0, 0]  # (H, H), random width per call

    pool = collect_pool(root, args.category)
    LOGGER.info(f"Pool: {len(pool)} defect samples; donors {counts}; sigma~U{sigma_factor} -> {out_root}")

    by_cat: dict[str, list] = {}
    for item in pool:
        by_cat.setdefault(item[1].split("/")[0], []).append(item)
    targets = []
    for cat, items in by_cat.items():
        random.shuffle(items)
        targets += items if args.n_per_category <= 0 else items[: args.n_per_category]

    total = 0
    for img_path, tag, bboxes in targets:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        panel = cv2.resize(rgb, (args.panel, args.panel))

        own = gauss(bboxes)
        orig = panel.copy()
        draw_boxes(orig, bboxes, args.panel)

        donors_pool = [p for p in pool if p[0] != img_path]
        donor_masks = [gauss(d[2]) for d in random.sample(donors_pool, min(max(counts), len(donors_pool)))]

        panels = [titled(orig, f"{tag} (orig+bbox)"),
                  titled(CompareGrid.heatmap_overlay(panel, own.cpu().numpy()), "own prior (clean)")]
        for n in counts:
            fused = own.clone()
            for dm in donor_masks[:n]:
                fused = torch.maximum(fused, dm)
            ov = CompareGrid.heatmap_overlay(panel, fused.cpu().numpy())
            draw_boxes(ov, bboxes, args.panel)  # mark true region preserved among distractors
            panels.append(titled(ov, f"noisy prior (+{n} donors)"))

        fig = concat_samh(panels, gap=10, gap_color=(255, 255, 255), cols=len(panels))
        out_path = out_root / f"{tag.replace('/', '__')}__{img_path.stem}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
        total += 1

    LOGGER.info(f"Done. {total} distractor-prior composites -> {out_root}")


if __name__ == "__main__":
    main()
