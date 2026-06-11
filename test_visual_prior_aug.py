#!/usr/bin/env python
"""Visualize candidate prior-mask augmentations for robustness (beyond gauss width jitter).

For each defect image, renders the GT bbox into a gauss prior (random width) and shows a menu of
perturbations that mimic the ways a real inference heatmap is imperfect. The original GT box is
drawn (green) on every panel so the truth vs the perturbed prior is obvious.

Operations:
  orig+bbox | own prior (clean) | spatial jitter | per-box drop | partial erase | elastic warp
           | additive mixup (+donor) | distractor (+donors, max)

  - spatial jitter : bbox center randomly offset -> prior mis-localized (tolerance to slop)
  - per-box drop   : randomly drop a subset of boxes -> prior misses some defects (false negative)
  - partial erase  : zero a random sub-region of the blob -> prior covers only part of the defect
  - elastic warp   : random low-freq deformation -> irregular, non-elliptical blob
  - additive mixup : own + alpha * donor_blob (soft distractor, weighted)
  - distractor     : max(own, donor_blobs) (hard false-positive blobs at wrong places)

Usage:
  python test_visual_prior_aug.py --out runs/temp/prior_aug
  python test_visual_prior_aug.py --category leather --n-per-category 3 --out runs/temp/prior_aug
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
    rows = []
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            rows.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, 4)


def collect_pool(root: Path, category: str | None):
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
    for cx, cy, w, h in bboxes.tolist():
        x1, y1 = int((cx - w / 2) * panel), int((cy - h / 2) * panel)
        x2, y2 = int((cx + w / 2) * panel), int((cy + h / 2) * panel)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


# --- perturbations -----------------------------------------------------------

def jitter_bboxes(bboxes: torch.Tensor, j: float) -> torch.Tensor:
    """Randomly offset each box center by ~U(-j, j) of the image (mis-localized prior)."""
    b = bboxes.clone()
    b[:, 0] = (b[:, 0] + (torch.rand(b.shape[0]) * 2 - 1) * j).clamp(0, 1)
    b[:, 1] = (b[:, 1] + (torch.rand(b.shape[0]) * 2 - 1) * j).clamp(0, 1)
    return b


def drop_bboxes(bboxes: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly drop a subset of boxes (prior misses some defects). Keeps >=1 if any exist."""
    n = bboxes.shape[0]
    if n == 0:
        return bboxes
    keep = torch.rand(n) > p
    if not keep.any():
        keep[random.randrange(n)] = True
    return bboxes[keep]


def partial_erase(mask: torch.Tensor, frac=(0.2, 0.5)) -> torch.Tensor:
    """Zero a random rectangular sub-region (prior covers only part of the defect)."""
    H = mask.shape[-1]
    m = mask.clone()
    eh, ew = int(H * random.uniform(*frac)), int(H * random.uniform(*frac))
    y, x = random.randint(0, H - eh), random.randint(0, H - ew)
    m[y:y + eh, x:x + ew] = 0.0
    return m


def elastic_warp(mask: torch.Tensor, alpha: float = 18.0, sigma: float = 6.0) -> torch.Tensor:
    """Random low-frequency elastic deformation -> irregular blob shape."""
    H = mask.shape[-1]
    m = mask.cpu().numpy().astype(np.float32)
    dx = cv2.GaussianBlur((np.random.rand(H, H) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(H, H) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(H), np.arange(H))
    mapx = (xx + dx).astype(np.float32)
    mapy = (yy + dy).astype(np.float32)
    warped = cv2.remap(m, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return torch.from_numpy(warped)


def main():
    ap = argparse.ArgumentParser(description="prior-mask robustness augmentation visualizer")
    ap.add_argument("--root", type=str, default=str(MVTEC_ROOT))
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--n-per-category", type=int, default=5)
    ap.add_argument("--sigma-range", type=str, default="0.1,0.3", help="gauss width ~U(lo,hi)")
    ap.add_argument("--jitter", type=float, default=0.06, help="center offset frac of image")
    ap.add_argument("--box-drop-p", type=float, default=0.5, help="per-box drop prob")
    ap.add_argument("--mixup-alpha", type=float, default=0.5, help="additive donor weight")
    ap.add_argument("--distractor-n", type=int, default=3, help="hard distractor donors")
    ap.add_argument("--mask-size", type=int, default=256)
    ap.add_argument("--panel", type=int, default=380)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    sr = [float(x) for x in args.sigma_range.split(",") if x.strip()]
    H = args.mask_size
    renderer = BboxMaskRenderer(mask_size=H, mode="gauss", sigma_factor=(sr if len(sr) == 2 else sr[0]))
    renderer.train()  # random per-box width like training

    def gauss(bboxes: torch.Tensor) -> torch.Tensor:
        if bboxes.shape[0] == 0:
            return torch.zeros(H, H)
        idx = torch.zeros(bboxes.shape[0], dtype=torch.long)
        return renderer(bboxes, idx, 1)[0, 0]

    def ov(panel, mask, bboxes, title):
        o = CompareGrid.heatmap_overlay(panel, mask.detach().cpu().numpy())
        draw_boxes(o, bboxes, args.panel)
        return titled(o, title)

    pool = collect_pool(root, args.category)
    LOGGER.info(f"Pool: {len(pool)} defect samples -> {out_root}")
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
        panel = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (args.panel, args.panel))

        own = gauss(bboxes)
        donors_pool = [p for p in pool if p[0] != img_path]
        donor_items = random.sample(donors_pool, min(args.distractor_n, len(donors_pool)))
        donor_masks = [gauss(d[2]) for d in donor_items]

        # additive mixup with the first donor; hard distractor with all donors (max)
        mix = (own + args.mixup_alpha * donor_masks[0]).clamp(0, 1) if donor_masks else own
        distract = own.clone()
        for dm in donor_masks:
            distract = torch.maximum(distract, dm)

        orig = panel.copy()
        draw_boxes(orig, bboxes, args.panel)
        panels = [
            titled(orig, f"{tag} (orig+bbox)"),
            ov(panel, own, bboxes, "own prior (clean)"),
            ov(panel, gauss(jitter_bboxes(bboxes, args.jitter)), bboxes, f"spatial jitter ({args.jitter})"),
            ov(panel, gauss(drop_bboxes(bboxes, args.box_drop_p)), bboxes, f"per-box drop (p={args.box_drop_p})"),
            ov(panel, partial_erase(own), bboxes, "partial erase"),
            ov(panel, elastic_warp(own), bboxes, "elastic warp"),
            ov(panel, mix, bboxes, f"additive mixup (a={args.mixup_alpha})"),
            ov(panel, distract, bboxes, f"distractor (+{len(donor_masks)} max)"),
        ]
        fig = concat_samh(panels, gap=10, gap_color=(255, 255, 255), cols=len(panels))
        out_path = out_root / f"{tag.replace('/', '__')}__{img_path.stem}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
        total += 1

    LOGGER.info(f"Done. {total} prior-aug composites -> {out_root}")


if __name__ == "__main__":
    main()
