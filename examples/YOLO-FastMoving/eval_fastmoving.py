# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate a YOLO-FastMoving detector (or a TrackNet checkpoint) with the TrackNet/WASB per-frame protocol.

Per frame: take the single highest-confidence box center (TrackNet: heatmap peak and weighted centroid of the last
frame of a causal 3-frame window), scale it to the `--grid` resolution (512x288 = TrackNet repo and WASB, 640x360 =
TrackNet paper) and count TP if within `tol` px of the GT center, else FP+FN; no prediction above conf on a ball frame is
an FN; a prediction on a frame without a ball is an FP, otherwise a TN. Reports precision/recall/F1/accuracy per tol for a
fixed conf and a conf sweep, and AP over the per-frame predictions ranked by confidence.
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


def gpu_temporal(frames, size, device, stride=32):
    """'diff' temporal input built on the GPU from raw BGR frames (newest first), padded like letterbox()."""
    gw = torch.tensor([0.114, 0.587, 0.299], device=device, dtype=torch.half).view(1, 3, 1, 1)
    w, h = size
    bgr, grays = None, []
    for f in frames:
        if f is None:
            grays.append(None)
            continue
        t = torch.from_numpy(f).to(device).permute(2, 0, 1)[None].half()
        t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        bgr = t if bgr is None else bgr
        grays.append((t * gw).sum(1, keepdim=True).round())
    diffs = [
        (a - b).abs() if a is not None and b is not None else torch.zeros_like(grays[0])
        for a, b in zip(grays[:-1], grays[1:])
    ]
    x = torch.full(
        (1, 3 + len(diffs), -(-h // stride) * stride, -(-w // stride) * stride), 114, device=device, dtype=torch.half
    )
    x[:, :, :h, :w] = torch.cat([bgr, *diffs], 1)
    return x / 255


def metrics(rows, conf_thr, tols):
    """rows: (gt_xy or None, pred_xy or None, conf). Returns {tol: dict}."""
    res = {}
    for t in tols:
        tp = fp = fn = tn = 0
        for gt, pr, c in rows:
            has = pr is not None and c >= conf_thr
            if gt is None:
                fp += has
                tn += not has
            elif not has:
                fn += 1
            elif np.hypot(pr[0] - gt[0], pr[1] - gt[1]) <= t:
                tp += 1
            else:
                fp += 1
                fn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        res[t] = {
            "f1": round(f1, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "acc": round((tp + tn) / len(rows), 4),
        }
    return res


def average_precision(rows, tol):
    """All-point AP of the per-frame predictions ranked by confidence; recall is over frames with a ball."""
    n_gt = sum(gt is not None for gt, _, _ in rows)
    preds = sorted(
        ((c, gt is not None and np.hypot(pr[0] - gt[0], pr[1] - gt[1]) <= tol) for gt, pr, c in rows if pr is not None),
        reverse=True,
    )
    if not preds or not n_gt:
        return 0.0
    hit = np.array([h for _, h in preds], dtype=float)
    tpc = np.cumsum(hit)
    prec, rec = tpc / np.arange(1, len(hit) + 1), tpc / n_gt
    prec = np.maximum.accumulate(prec[::-1])[::-1]  # interpolated precision envelope
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def recall_by_visibility(rows, files, root, conf_thr, tol):
    """Recall at the fixed conf per visibility class, if the dataset ships visibility.json (TrackNet paper dataset)."""
    vf = Path(root) / "visibility.json"
    if not vf.exists():
        return None
    vis = json.load(open(vf))
    hits, counts = {}, {}
    for f, (gt, pr, c) in zip(files, rows):
        if gt is None:
            continue
        v = str(vis.get(f, "?"))
        counts[v] = counts.get(v, 0) + 1
        hits[v] = hits.get(v, 0) + (pr is not None and c >= conf_thr and np.hypot(pr[0] - gt[0], pr[1] - gt[1]) <= tol)
    return {v: {"n": counts[v], "recall": round(hits[v] / counts[v], 4)} for v in sorted(counts)}


class TrackNetPredictor:
    """Causal TrackNet inference: window [t-2, t-1, t] (missing history repeats the oldest frame), heatmap of frame t."""

    def __init__(self, spec: str, device: str):
        import sys

        repo, ckpt, *arch = spec.split(":")
        sys.path.insert(0, repo)
        from model import TrackNet, TrackNetDeep

        self.net = (TrackNetDeep if arch and arch[0] == "deep" else TrackNet)(seq_len=3)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net.load_state_dict(sd.get("model", sd))
        self.net = self.net.to(device).eval().half()
        self.device = device

    def __call__(self, frames):
        """frames newest-first BGR; returns (x, y) in 512x288 and peak confidence."""
        frames = [f for f in frames if f is not None]
        frames = frames + [frames[-1]] * (3 - len(frames))
        small = [cv2.resize(f, (512, 288)) for f in frames[::-1]]  # oldest first, like FlyingBallDataset
        x = torch.from_numpy(np.concatenate(small, 2)).to(self.device).permute(2, 0, 1)[None].half() / 255
        hm = self.net(x)[0, -1].float().cpu().numpy()
        peak = float(hm.max())
        mask = hm > peak * 0.5
        ys, xs = np.nonzero(mask)
        w = hm[mask]
        return ((xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()), peak


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="YOLO weights")
    ap.add_argument("--tracknet", default="", help="evaluate TrackNet instead: <repo>:<ckpt>[:deep]")
    ap.add_argument("--data", required=True, help="data YAML from prepare_flyingball.py / prepare_tracknet_paper.py")
    ap.add_argument("--grid", default="512x288", help="evaluation resolution WxH")
    ap.add_argument("--tols", default="4,5,8,10", help="pixel tolerances at --grid; the first is primary")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25, help="fixed reporting threshold")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    ap.add_argument("--gpu-prep", action="store_true", help="build the 'diff' input on the GPU like bench_speed.py")
    args = ap.parse_args()
    gw_, gh_ = map(int, args.grid.split("x"))
    tols = tuple(float(t) for t in args.tols.split(","))

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

    if args.tracknet:
        tracknet, k = TrackNetPredictor(args.tracknet, args.device), 3
    else:
        model = YOLO(args.weights).model.to(args.device).eval().half()
        model.model[-1].end2end = True  # NMS-free one2one head, (B, 300, 6) output
    rows, dists = [], []
    for f in files:
        paths = history(f, k, exists)
        frames = [cv2.imread(p) if p else None for p in paths]
        h0, w0 = frames[0].shape[:2]
        lab = Path(f.replace("/images/", "/labels/")).with_suffix(".txt")
        gt = None
        if lab.exists() and lab.read_text().strip():
            _, cx, cy, *_ = map(float, lab.read_text().split()[:5])
            gt = (cx * gw_, cy * gh_)
        if args.tracknet:
            (px, py), c = tracknet(frames)
            pr = (px * gw_ / 512, py * gh_ / 288)
            rows.append((gt, pr, c))
            if gt is not None:
                dists.append(np.hypot(pr[0] - gt[0], pr[1] - gt[1]))
            continue
        r = args.imgsz / max(h0, w0)
        size = (math.ceil(w0 * r), math.ceil(h0 * r))
        if args.gpu_prep:  # the bench_speed.py deployment path: torch bilinear resize, gray and absdiff on the GPU
            x = gpu_temporal(frames, size, args.device)
        else:
            frames = [cv2.resize(x, size, interpolation=cv2.INTER_LINEAR) if x is not None else None for x in frames]
            im = stack_temporal(frames, mode)
            if im.shape[2] == 3:  # Format flips 3-channel training images BGR->RGB; 5/9-channel images stay BGR
                im = im[..., ::-1]
            im = letterbox(im)
            x = torch.from_numpy(im).to(args.device).permute(2, 0, 1)[None].half() / 255
        y = model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y  # (1, N, 6) xyxy, conf, cls for end2end heads
        det = y[0]
        if det.shape[-1] != 6:
            raise RuntimeError(f"unexpected output shape {tuple(y.shape)}; expected an end2end (NMS-free) head")
        j = int(det[:, 4].argmax())
        x1, y1, x2, y2, c = det[j, :5].float().tolist()
        pr = ((x1 + x2) / 2 / r * gw_ / w0, (y1 + y2) / 2 / r * gh_ / h0)
        rows.append((gt, pr, c))
        if gt is not None:
            dists.append(np.hypot(pr[0] - gt[0], pr[1] - gt[1]))

    fixed = metrics(rows, args.conf, tols)
    sweep = {round(t, 2): metrics(rows, t, tols) for t in np.arange(0.05, 0.96, 0.05)}
    best_t = max(sweep, key=lambda t: sweep[t][tols[0]]["f1"])
    d = np.array(dists)
    report = {
        "model": args.tracknet or args.weights,
        "data": args.data,
        "split": args.split,
        "imgsz": None if args.tracknet else args.imgsz,
        "grid": args.grid,
        "n_frames": len(rows),
        "n_ball_frames": sum(gt is not None for gt, _, _ in rows),
        "conf_fixed": args.conf,
        "fixed": fixed,
        "ap": {t: round(average_precision(rows, t), 4) for t in tols},
        "best_conf_by_primary_f1": best_t,
        "best": sweep[best_t],
        "sweep_primary_tol": {c: sweep[c][tols[0]] for c in sweep},
        "recall_by_visibility": recall_by_visibility(rows, files, root, args.conf, tols[0]),
        "top1_dist_grid_px": {
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "p90": float(np.percentile(d, 90)),
        },
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
