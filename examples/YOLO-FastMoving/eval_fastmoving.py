# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate a YOLO-FastMoving detector with the TrackNet repo's protocol.

Per labelled val frame: take the single highest-confidence box, use its center, scale it to TrackNet's 512x288 grid and
count TP if within `tol` pixels of the GT center, else FP+FN; no prediction above conf is an FN; a prediction on an
unlabelled frame is an FP (the flyingball val set has none). Reports F1/precision/recall at tol 4, 5, 8, 10 for a fixed
conf and a conf sweep.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.temporal import FRAME_RE, stack_temporal

TOLS = (4, 5, 8, 10)
TN_W, TN_H = 512, 288


def history(f: str, k: int, exists) -> list[str | None]:
    """Newest-first history paths, identical to TemporalYOLODataset.history_paths."""
    m = FRAME_RE.match(f)
    paths = [f]
    for i in range(1, k):
        p = f"{m['stem']}{int(m['idx']) - i:0{len(m['idx'])}d}{m['suffix']}"
        paths.append(p if paths[-1] is not None and int(m["idx"]) - i >= 0 and exists(p) else None)
    return paths


def letterbox(im: np.ndarray, stride: int = 32):
    """Pad bottom/right to a stride multiple with 114, like rect val batches but corner-aligned."""
    h, w = im.shape[:2]
    H, W = -(-h // stride) * stride, -(-w // stride) * stride
    out = np.full((H, W, im.shape[2]), 114, dtype=np.uint8)
    out[:h, :w] = im
    return out


def metrics(rows, conf_thr):
    """rows: (gt_xy or None, pred_xy or None, conf). Returns {tol: dict}."""
    res = {}
    for t in TOLS:
        tp = fp = fn = 0
        for gt, pr, c in rows:
            has = pr is not None and c >= conf_thr
            if gt is None:
                fp += has
            elif not has:
                fn += 1
            elif np.hypot(pr[0] - gt[0], pr[1] - gt[1]) <= t:
                tp += 1
            else:
                fp += 1
                fn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        res[t] = {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r}
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="flyingball-<mode>.yaml from prepare_flyingball.py")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25, help="fixed reporting threshold")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.data))
    root = Path(cfg["path"])
    k, mode = cfg["temporal"]["frames"], cfg["temporal"]["mode"]
    files = [x for x in (root / cfg[args.split]).read_text().split() if x]

    dir_cache = {}

    def exists(p):
        d, n = p.rsplit("/", 1)
        if d not in dir_cache:
            dir_cache[d] = {q.name for q in Path(d).iterdir()}
        return n in dir_cache[d]

    model = YOLO(args.weights).model.to(args.device).eval().half()
    model.model[-1].end2end = True  # NMS-free one2one head, (B, 300, 6) output
    rows, dists = [], []
    for f in files:
        paths = history(f, k, exists)
        frames = [cv2.imread(p) if p else None for p in paths]
        h0, w0 = frames[0].shape[:2]
        r = args.imgsz / max(h0, w0)
        size = (math.ceil(w0 * r), math.ceil(h0 * r))
        frames = [cv2.resize(x, size, interpolation=cv2.INTER_LINEAR) if x is not None else None for x in frames]
        im = letterbox(stack_temporal(frames, mode))
        x = torch.from_numpy(im).to(args.device).permute(2, 0, 1)[None].half() / 255
        y = model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y  # (1, N, 6) xyxy, conf, cls for end2end heads
        det = y[0]
        if det.shape[-1] != 6:
            raise RuntimeError(f"unexpected output shape {tuple(y.shape)}; expected an end2end (NMS-free) head")
        j = int(det[:, 4].argmax())
        x1, y1, x2, y2, c = det[j, :5].float().tolist()
        pr = ((x1 + x2) / 2 / r * TN_W / w0, (y1 + y2) / 2 / r * TN_H / h0)
        lab = Path(f.replace("/images/", "/labels/")).with_suffix(".txt")
        gt = None
        if lab.exists() and lab.read_text().strip():
            _, cx, cy, *_ = map(float, lab.read_text().split()[:5])
            gt = (cx * TN_W, cy * TN_H)
        rows.append((gt, pr, c))
        if gt is not None:
            dists.append(np.hypot(pr[0] - gt[0], pr[1] - gt[1]))

    fixed = metrics(rows, args.conf)
    sweep = {round(t, 2): metrics(rows, t) for t in np.arange(0.05, 0.96, 0.05)}
    best_t = max(sweep, key=lambda t: sweep[t][10]["f1"])
    d = np.array(dists)
    report = {
        "weights": args.weights,
        "data": args.data,
        "imgsz": args.imgsz,
        "n_frames": len(rows),
        "conf_fixed": args.conf,
        "fixed": {f"f1@{t}": round(v["f1"], 4) for t, v in fixed.items()},
        "fixed_pr@4": (round(fixed[4]["precision"], 4), round(fixed[4]["recall"], 4)),
        "best_conf_by_f1@10": best_t,
        "best": {f"f1@{t}": round(v["f1"], 4) for t, v in sweep[best_t].items()},
        "top1_dist_512px": {"mean": float(d.mean()), "median": float(np.median(d)), "p90": float(np.percentile(d, 90))},
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
