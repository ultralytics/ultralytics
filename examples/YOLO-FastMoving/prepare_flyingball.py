# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert the flyingball tennis-ball dataset into YOLO labels plus a TrackNet-compatible copy on the same split.

Layout produced under --out:
    images/            the extracted flyingball images (<source>/<video>/<video>__frame_XXXXXX.jpg), not moved
    labels/            one YOLO txt per labelled frame, class 0 = ball
    train.txt val.txt  labelled frames of each split (unlabelled frames stay on disk as diff history)
    flyingball-{rgb,diff,stack}.yaml
    tracknet/          annotations.json + splits.json + source symlinks for the TrackNet repo's FlyingBallDataset

Label alignment: Label Studio frame k describes image frame k + offset (TrackNet repo: `ls_frame - 3`). The
default `--offset auto` measures it per source (it differs: ZQW -2, Wimbledon -3).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load(ann_path):
    """Return {(source, video): {ls_frame: (bbox_pixel, w, h, image_path)}} and {(source, video): {frame: path}}."""
    data = json.load(open(ann_path))
    boxes, frames = defaultdict(dict), defaultdict(dict)
    for im in data["images"]:
        key = (im["source"], im["video_name"])
        frames[key][im["frame_index"]] = im["image_path"]
        for a in im.get("annotations", []):
            boxes[key][a["label_studio_frame_index"]] = (a["bbox_pixel"], im["width"], im["height"])
    return boxes, frames


def measure_offsets(boxes, frames, root, n=300, seed=0, candidates=range(-5, 2)):
    """Pick image_frame - ls_frame per source by where the ball box carries motion in both directions.

    For candidate offset o, image j = ls + o is scored by min(|g_j - g_{j-1}|, |g_j - g_{j+1}|) inside the GT box relative to
    the image mean; this is high only when the ball is in frame j at the box (a one-sided diff also fires at j + 1).
    """
    rng = random.Random(seed)
    items = [(k, f) for k, v in boxes.items() for f in v]
    rng.shuffle(items)
    best = {}
    for src in sorted({k[0] for k in boxes}):
        scores = {}
        for off in candidates:
            ratios = []
            for key, lsf in (it for it in items if it[0][0] == src):
                j = lsf + off
                if not all(i in frames[key] for i in (j - 1, j, j + 1)):
                    continue
                g = [
                    cv2.imread(str(root / frames[key][i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    for i in (j - 1, j, j + 1)
                ]
                d = np.minimum(np.abs(g[1] - g[0]), np.abs(g[1] - g[2]))
                bb = boxes[key][lsf][0]
                x0, y0 = max(int(bb["x"]), 0), max(int(bb["y"]), 0)
                x1, y1 = int(np.ceil(bb["x"] + bb["width"])), int(np.ceil(bb["y"] + bb["height"]))
                if d[y0:y1, x0:x1].size:
                    ratios.append(d[y0:y1, x0:x1].mean() / (d.mean() + 1e-3))
                if len(ratios) >= n:
                    break
            scores[off] = float(np.median(ratios)) if ratios else 0.0
        best[src] = max(scores, key=scores.get)
        print(f"{src}: offset {best[src]:+d}  median box/image motion ratio by offset {scores}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="flyingball_2026-03-15_annotations.json")
    ap.add_argument("--out", required=True, help="dataset dir; images must already be extracted to <out>/images")
    ap.add_argument("--offset", default="auto", help="image_frame = ls_frame + offset; int or 'auto' (per source)")
    ap.add_argument("--min-box", type=float, default=0.0, help="min box side in source pixels (0 = keep GT)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    root = out / "images"
    boxes, frames = load(args.ann)
    if args.offset == "auto":
        offsets = measure_offsets(boxes, frames, root)
    else:
        offsets = defaultdict(lambda: int(args.offset))

    # Video-level split, stratified by source
    rng = random.Random(args.seed)
    by_src = defaultdict(list)
    for src, vid in sorted(frames):
        by_src[src].append(vid)
    split = {"train": [], "val": []}
    for src, vids in sorted(by_src.items()):
        vids = sorted(vids)
        rng.shuffle(vids)
        nval = max(1, round(len(vids) * args.val_frac))
        split["val"] += [(src, v) for v in vids[:nval]]
        split["train"] += [(src, v) for v in vids[nval:]]

    lists = {"train": [], "val": []}
    tn_images = []
    n_lab = n_dropped = 0
    for sname, keys in split.items():
        for key in keys:
            for lsf, (bb, w, h) in sorted(boxes[key].items()):
                j = lsf + offsets[key[0]]
                if j not in frames[key]:
                    n_dropped += 1
                    continue
                rel = frames[key][j]
                bw, bh = max(bb["width"], args.min_box), max(bb["height"], args.min_box)
                cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
                lab = out / "labels" / Path(rel).with_suffix(".txt")
                lab.parent.mkdir(parents=True, exist_ok=True)
                lab.write_text(f"0 {cx / w:.6f} {cy / h:.6f} {min(bw / w, 1):.6f} {min(bh / h, 1):.6f}\n")
                lists[sname].append(str(root / rel))
                tn_images.append(
                    {
                        "image_path": rel,
                        "source": key[0],
                        "video_name": f"{key[0]}/{key[1]}",
                        "frame_index": j,
                        "width": w,
                        "height": h,
                        "annotations": [{"label": "flyingball", "bbox_pixel": bb}],
                    }
                )
                n_lab += 1
    for sname, files in lists.items():
        (out / f"{sname}.txt").write_text("\n".join(files) + "\n")
        print(f"{sname}: {len(split[sname])} videos, {len(files)} labelled frames")
    print(f"labelled {n_lab}, dropped {n_dropped} labels whose shifted frame has no image")

    for mode, ch in (("rgb", 3), ("diff", 5), ("stack", 9)):
        (out / f"flyingball-{mode}.yaml").write_text(
            f"path: {out}\ntrain: train.txt\nval: val.txt\nchannels: {ch}\n"
            f"temporal:\n  frames: 3\n  mode: {mode}\nnames:\n  0: ball\n"
        )

    tn = out / "tracknet"
    tn.mkdir(exist_ok=True)
    json.dump({"images": tn_images}, open(tn / "annotations.json", "w"))
    json.dump({s: [f"{k[0]}/{k[1]}" for k in v] for s, v in split.items()}, open(tn / "splits.json", "w"), indent=1)
    for src in by_src:
        link = tn / src
        if not link.exists():
            link.symlink_to(root / src)


if __name__ == "__main__":
    main()
