"""False-prompt evaluation for YOLOA v2 softhint.

For each anomalous image: predict with its GT-derived mask, record max conf (positive).
For each good image:      predict with a random rect mask,    record max conf (negative).
Report AUROC and percentiles. Optional matplotlib histogram.

Usage:
    python scripts/false_prompt_eval.py \
        --weights runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_v1/weights/best.pt \
        --data    /path/to/data.yaml \
        --out     runs/temp/false_prompt_softhint.json \
        [--no-plot]

Notes:
- The data.yaml is expected to have a 'val' (or 'test') split with YOLO labels.
- Images whose label file is empty or missing are treated as 'good'; otherwise 'anomalous'.
- Random mask: uniform center in [0.15, 0.85]^2, square side in [0.10, 0.40] of image.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo.anomaly_v2 import AnomalyV2Predictor  # noqa: F401  (ensures task registration)


def list_split_images(data_yaml: str, split: str) -> list[Path]:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg.get("path", "."))
    rel = cfg.get(split)
    if rel is None:
        raise SystemExit(f"split {split!r} not in {data_yaml}")
    p = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    if p.is_file():  # txt with image paths
        return [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([q for q in p.rglob("*") if q.suffix.lower() in exts])


def yolo_label_path(img: Path) -> Path:
    # Standard YOLO layout: .../images/<...>/x.jpg <-> .../labels/<...>/x.txt
    parts = list(img.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    lbl = Path(*parts).with_suffix(".txt")
    return lbl


def read_yolo_label(label_path: Path) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # cls cx cy w h  (normalized)
        cx, cy, w, h = (float(x) for x in parts[1:5])
        rows.append((cx, cy, w, h))
    return rows


def random_rect_mask(h: int = 80, w: int = 80, rng: random.Random | None = None) -> torch.Tensor:
    rng = rng or random
    cx = rng.uniform(0.15, 0.85)
    cy = rng.uniform(0.15, 0.85)
    side = rng.uniform(0.10, 0.40)
    x1 = max(0, int((cx - side / 2) * w))
    x2 = min(w, int((cx + side / 2) * w))
    y1 = max(0, int((cy - side / 2) * h))
    y2 = min(h, int((cy + side / 2) * h))
    m = torch.zeros(1, 1, h, w)
    m[0, 0, y1:y2, x1:x2] = 1.0
    return m


def gt_rect_mask(boxes: list[tuple], h: int = 80, w: int = 80) -> torch.Tensor:
    m = torch.zeros(1, 1, h, w)
    for cx, cy, bw, bh in boxes:
        x1 = max(0, int((cx - bw / 2) * w))
        x2 = min(w, int((cx + bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        y2 = min(h, int((cy + bh / 2) * h))
        m[0, 0, y1:y2, x1:x2] = 1.0
    return m


def max_conf(result) -> float:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
        return 0.0
    return float(boxes.conf.max().item())


def auroc(pos: list[float], neg: list[float]) -> float:
    # Mann-Whitney U based AUROC; robust to ties.
    scores = np.asarray(pos + neg, dtype=np.float64)
    labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int8)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - counts / 2 + 0.5  # mean rank within tie group
    tie_ranks = avg[inv]
    pos_rank_sum = tie_ranks[labels == 1].sum()
    n_pos = labels.sum()
    n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    images = list_split_images(args.data, args.split)
    print(f"Loaded {len(images)} images from {args.data}:{args.split}")

    model = YOLO(args.weights, task="anomaly_v2")

    pos_conf: list[float] = []  # anomaly + GT mask
    neg_conf: list[float] = []  # good + random mask

    for img in images:
        boxes = read_yolo_label(yolo_label_path(img))
        is_good = len(boxes) == 0
        mask = random_rect_mask(rng=rng) if is_good else gt_rect_mask(boxes)
        # Warm-up so predictor exists, then inject the mask and predict.
        model.predict(str(img), verbose=False, save=False)
        model.predictor.model.set_external_mask_once(mask.to(next(model.predictor.model.parameters()).device))
        res = model.predict(str(img), verbose=False, save=False)[0]
        c = max_conf(res)
        (neg_conf if is_good else pos_conf).append(c)

    out = {
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "n_pos": len(pos_conf),
        "n_neg": len(neg_conf),
        "auroc": round(auroc(pos_conf, neg_conf), 4),
        "pos_p50": round(float(np.percentile(pos_conf, 50)), 4) if pos_conf else None,
        "pos_p05": round(float(np.percentile(pos_conf, 5)), 4) if pos_conf else None,
        "neg_p50": round(float(np.percentile(neg_conf, 50)), 4) if neg_conf else None,
        "neg_p95": round(float(np.percentile(neg_conf, 95)), 4) if neg_conf else None,
        "neg_p99": round(float(np.percentile(neg_conf, 99)), 4) if neg_conf else None,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(neg_conf, bins=30, alpha=0.6, label=f"good + random mask (n={len(neg_conf)})")
            ax.hist(pos_conf, bins=30, alpha=0.6, label=f"anomaly + GT mask (n={len(pos_conf)})")
            ax.set_xlabel("max detection confidence")
            ax.set_ylabel("image count")
            ax.set_title(f"False-prompt eval | AUROC = {out['auroc']:.4f}")
            ax.legend()
            fig.tight_layout()
            fig_path = out_path.with_suffix(".png")
            fig.savefig(fig_path, dpi=120)
            print(f"Saved histogram to {fig_path}")
        except Exception as e:
            print(f"Plot skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
