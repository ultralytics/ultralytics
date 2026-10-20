# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Latency of YOLO-FastMoving vs. TrackNet on one GPU, batch 1, FP16.

YOLO streaming step per new frame: resize the new 1080p frame, build the temporal image (gray + absdiff), H2D, forward,
top-1 box. TrackNet step: one forward of 3 stacked RGB frames at 288x512 that emits 3 heatmaps, plus the weighted
centroid on the GPU; its per-frame cost is reported both per call and divided by 3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.temporal import TemporalStream


def timeit(fn, n=300, warmup=50):
    """Median and p90 wall time in ms of fn() with CUDA sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def top1(y):
    """Highest-confidence row of an end2end (B, N, 6) output; eval-mode models return (y, raw)."""
    y = y[0] if isinstance(y, (list, tuple)) else y
    return y[0, y[0, :, 4].argmax()]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", nargs="*", default=[], help="weights[:mode[:imgsz]] e.g. best.pt:diff:1280")
    ap.add_argument("--tracknet-repo", default="")
    ap.add_argument("--tracknet-ckpt", default="", help="optional; random weights give the same latency")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(8)]
    out = {"gpu": torch.cuda.get_device_name(dev)}

    for spec in args.yolo:
        w, mode, imgsz = (spec.split(":") + ["diff", "1280"])[:3]
        imgsz = int(imgsz)
        net = YOLO(w).model.to(dev).eval().fuse().half()
        net.model[-1].end2end = True  # NMS-free one2one head, (B, 300, 6) output
        W, H = imgsz, int(round(1080 * imgsz / 1920))
        Hp = -(-H // 32) * 32
        stream = TemporalStream(frames=3, mode=mode, size=(W, H))
        buf = torch.full((1, {"rgb": 3, "diff": 5, "stack": 9}[mode], Hp, W), 114 / 255, device=dev, dtype=torch.half)
        i = [0]

        def step():
            im = stream.push(frames[i[0] % 8])
            i[0] += 1
            buf[0, :, :H] = torch.from_numpy(im).to(dev, non_blocking=True).permute(2, 0, 1).half() / 255
            return top1(net(buf))

        def fwd():
            return net(buf)

        pinned = [torch.from_numpy(f).pin_memory() for f in frames]
        grays = deque(maxlen=3)
        gw = torch.tensor([0.114, 0.587, 0.299], device=dev, dtype=torch.half).view(1, 3, 1, 1)  # BGR -> gray

        def step_gpu():  # same temporal image built on the GPU from the raw 1080p frame
            raw = pinned[i[0] % 8].to(dev, non_blocking=True).permute(2, 0, 1)[None].half()
            i[0] += 1
            bgr = torch.nn.functional.interpolate(raw, size=(H, W), mode="bilinear", align_corners=False)
            grays.appendleft((bgr * gw).sum(1, keepdim=True).round())
            g = list(grays) + [None] * (3 - len(grays))
            diffs = [
                (a - b).abs() if a is not None and b is not None else torch.zeros_like(g[0])
                for a, b in zip(g[:-1], g[1:])
            ]
            buf[:, :, :H] = torch.cat([bgr, *diffs], 1)[:, : buf.shape[1]] / 255
            return top1(net(buf))

        e2e = timeit(step)
        e2e_gpu = timeit(step_gpu) if mode == "diff" else (float("nan"), float("nan"))
        f = timeit(fwd)
        params = sum(p.numel() for p in net.parameters())
        out[f"yolo:{w}:{mode}:{imgsz}"] = {
            "input": [buf.shape[1], Hp, W],
            "params_M": round(params / 1e6, 2),
            "e2e_ms_median": round(e2e[0], 2),
            "e2e_ms_p90": round(e2e[1], 2),
            "e2e_gpu_prep_ms_median": round(e2e_gpu[0], 2),
            "fwd_ms_median": round(f[0], 2),
            "fps_e2e": round(1000 / e2e[0], 1),
            "fps_e2e_gpu_prep": round(1000 / e2e_gpu[0], 1),
        }

    if args.tracknet_repo:
        sys.path.insert(0, args.tracknet_repo)
        from model import TrackNet, TrackNetDeep

        for name, cls in (("TrackNet", TrackNet), ("TrackNetDeep", TrackNetDeep)):
            net = cls(seq_len=3).to(dev).eval()
            if args.tracknet_ckpt and name == "TrackNet":
                sd = torch.load(args.tracknet_ckpt, map_location=dev)
                net.load_state_dict(sd.get("model", sd.get("model_state_dict", sd)))
            net = net.half()
            x = torch.rand(1, 9, 288, 512, device=dev).half()

            def tn_step():  # one call consumes 3 new 1080p frames
                small = [cv2.resize(f, (512, 288)) for f in frames[:3]]
                xin = torch.from_numpy(np.concatenate(small, 2)).to(dev).permute(2, 0, 1)[None].half() / 255
                hm = net(xin)[0]  # (3, H, W)
                flat = hm.flatten(1)
                return flat.argmax(1), flat.max(1).values

            e2e = timeit(tn_step)
            f = timeit(lambda: net(x))
            params = sum(p.numel() for p in net.parameters())
            out[name] = {
                "input": [9, 288, 512],
                "params_M": round(params / 1e6, 2),
                "e2e_ms_per_call_median": round(e2e[0], 2),
                "e2e_ms_per_frame_median": round(e2e[0] / 3, 2),
                "fwd_ms_per_call_median": round(f[0], 2),
                "fps_per_frame_e2e": round(3000 / e2e[0], 1),
            }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
