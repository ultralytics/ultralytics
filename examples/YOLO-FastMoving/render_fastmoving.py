# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render review videos: ground truth, YOLO-FastMoving and TrackNet predictions on the same clip.

Green ring = ground truth, red cross + trail = YOLO, cyan ring = TrackNet. A frame counts as a hit when the prediction
is above its threshold and within `--tol` px of the GT at the 512x288 grid (the WASB / TrackNet repo protocol). The
inset shows the first motion channel |g(t) - g(t-1)| that the diff model sees, gained x4. Writes <out>/<clip>.mp4 plus
<out>/clips.json with per-clip hit counts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import yaml
from eval_fastmoving import TrackNetPredictor, history, letterbox

from ultralytics import YOLO
from ultralytics.data.temporal import stack_temporal

GREEN, RED, CYAN, WHITE = (80, 220, 80), (60, 60, 255), (255, 220, 0), (255, 255, 255)


def hit(pr, gt, conf, thr, tol, w0, h0):
    """True if the prediction counts as a TP at the 512x288 grid."""
    if gt is None or pr is None or conf < thr:
        return False
    return math.hypot((pr[0] - gt[0]) * 512 / w0, (pr[1] - gt[1]) * 288 / h0) <= tol


def put(img, text, org, color=WHITE, scale=0.6):
    """Outlined label readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", required=True, help="YOLO-FastMoving weights matching --data's temporal mode")
    ap.add_argument("--tracknet", default="", help="<repo>:<ckpt>[:deep]")
    ap.add_argument("--clips", nargs="*", default=[], help="clip dir substrings to render (default: all)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tn-conf", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=4.0, help="hit tolerance in px at 512x288")
    ap.add_argument("--out-width", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data))
    root = Path(cfg["path"])
    k, mode = cfg["temporal"]["frames"], cfg["temporal"]["mode"]
    files = [x for x in (root / cfg[args.split]).read_text().split() if x]
    clips = defaultdict(list)
    for f in files:
        clips[str(Path(f).parent)].append(f)
    if args.clips:
        clips = {d: v for d, v in clips.items() if any(c in d for c in args.clips)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights).model.to(args.device).eval().half()
    model.model[-1].end2end = True
    tracknet = TrackNetPredictor(args.tracknet, args.device) if args.tracknet else None
    dir_cache = {}

    def exists(p):
        d, n = p.rsplit("/", 1)
        if d not in dir_cache:
            dir_cache[d] = {q.name for q in Path(d).iterdir()}
        return n in dir_cache[d]

    stats = {}
    for d, fs in sorted(clips.items()):
        fs = sorted(fs)
        name = "_".join(Path(d).parts[-2:])
        writer, trail = None, deque(maxlen=12)
        s = {"frames": len(fs), "ball_frames": 0, "yolo_hits": 0, "tracknet_hits": 0, "yolo_fp": 0, "tracknet_fp": 0}
        for f in fs:
            paths = history(f, k, exists)
            frames = [cv2.imread(p) if p else None for p in paths]
            cur = frames[0]
            h0, w0 = cur.shape[:2]
            r = args.imgsz / max(h0, w0)
            size = (math.ceil(w0 * r), math.ceil(h0 * r))
            small = [cv2.resize(x, size, interpolation=cv2.INTER_LINEAR) if x is not None else None for x in frames]
            im = stack_temporal(small, mode)
            x = letterbox(im[..., ::-1] if im.shape[2] == 3 else im)
            y = model(torch.from_numpy(x).to(args.device).permute(2, 0, 1)[None].half() / 255)
            det = (y[0] if isinstance(y, (list, tuple)) else y)[0]
            j = int(det[:, 4].argmax())
            x1, y1, x2, y2, yc = det[j, :5].float().tolist()
            yp = ((x1 + x2) / 2 / r, (y1 + y2) / 2 / r)
            tp, tc = None, 0.0
            if tracknet:
                (px, py), tc = tracknet(frames)
                tp = (px * w0 / 512, py * h0 / 288)
            lab = Path(f.replace("/images/", "/labels/")).with_suffix(".txt")
            t = lab.read_text().split() if lab.exists() else []
            gt = (float(t[1]) * w0, float(t[2]) * h0) if t else None

            yh, th = hit(yp, gt, yc, args.conf, args.tol, w0, h0), hit(tp, gt, tc, args.tn_conf, args.tol, w0, h0)
            s["ball_frames"] += gt is not None
            s["yolo_hits"] += yh
            s["tracknet_hits"] += th
            s["yolo_fp"] += gt is None and yc >= args.conf
            s["tracknet_fp"] += gt is None and tracknet is not None and tc >= args.tn_conf

            vis = cur.copy()
            if gt is not None:
                cv2.circle(vis, (round(gt[0]), round(gt[1])), 14, GREEN, 2, cv2.LINE_AA)
            if tp is not None and tc >= args.tn_conf:
                cv2.circle(vis, (round(tp[0]), round(tp[1])), 9, CYAN, 2, cv2.LINE_AA)
            trail.append(yp if yc >= args.conf else None)
            pts = [p for p in trail if p is not None]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(vis, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), RED, 1, cv2.LINE_AA)
            if yc >= args.conf:
                c = (round(yp[0]), round(yp[1]))
                cv2.drawMarker(vis, c, RED, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

            # motion inset: |g(t) - g(t-1)| x4, what the diff channels carry
            if frames[1] is not None:
                g0, g1 = (cv2.cvtColor(z, cv2.COLOR_BGR2GRAY) for z in frames[:2])
                diff = np.clip(cv2.absdiff(g0, g1).astype(np.int32) * 4, 0, 255).astype(np.uint8)
            else:
                diff = np.zeros((h0, w0), np.uint8)
            iw = w0 // 4
            ih = round(h0 * iw / w0)
            inset = cv2.cvtColor(cv2.resize(diff, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
            vis[h0 - ih - 10 : h0 - 10, w0 - iw - 10 : w0 - 10] = inset
            cv2.rectangle(vis, (w0 - iw - 11, h0 - ih - 11), (w0 - 9, h0 - 9), WHITE, 1)
            put(vis, "motion |g(t)-g(t-1)| x4", (w0 - iw - 6, h0 - ih - 18), WHITE, 0.5 * w0 / 1280)

            sc = w0 / 1280
            cv2.rectangle(vis, (0, 0), (w0, round(70 * sc)), (20, 20, 20), -1)
            put(
                vis,
                f"{name}  frame {Path(f).stem.rsplit('_', 1)[-1]}",
                (round(12 * sc), round(26 * sc)),
                WHITE,
                0.65 * sc,
            )
            ystat = "HIT" if yh else ("miss" if gt is not None else ("FP" if yc >= args.conf else "-"))
            put(vis, f"YOLO conf {yc:.2f} {ystat}", (round(12 * sc), round(56 * sc)), RED, 0.6 * sc)
            if tracknet:
                tstat = "HIT" if th else ("miss" if gt is not None else ("FP" if tc >= args.tn_conf else "-"))
                put(vis, f"TrackNet peak {tc:.2f} {tstat}", (round(330 * sc), round(56 * sc)), CYAN, 0.6 * sc)
            put(vis, "GT" if gt is not None else "GT: no ball", (round(640 * sc), round(56 * sc)), GREEN, 0.6 * sc)

            if args.out_width != w0:
                vis = cv2.resize(vis, (args.out_width, round(h0 * args.out_width / w0)), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = imageio_ffmpeg.write_frames(
                    str(out / f"{name}.mp4"),
                    vis.shape[1::-1],
                    fps=args.fps,
                    codec="libx264",
                    quality=None,
                    output_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart"],
                    macro_block_size=8,
                )
                writer.send(None)
            writer.send(np.ascontiguousarray(vis[..., ::-1]))
        writer.close()
        s["mp4_mb"] = round((out / f"{name}.mp4").stat().st_size / 1e6, 2)
        stats[name] = s
        print(name, s, flush=True)
    json.dump(stats, open(out / "clips.json", "w"), indent=1)


if __name__ == "__main__":
    main()
