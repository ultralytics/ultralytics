# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert the TrackNet paper tennis dataset (Dataset/game*/Clip*/Label.csv) into YOLO-FastMoving layout.

Labels give the ball center and a visibility class (0 = no ball in frame, 1 = easy, 2 = hard, 3 = occluded). Frames with
visibility 0 get an empty label file (a negative); the others get a fixed `--box` px square around the center, since the
dataset has no ball extent (mean diameter ~10 px at 1280x720 per the paper).

Layout under --out:
    images/gameG/ClipC/gameG_ClipC_frame_NNNN.jpg   symlinks to the originals, renamed so diff history can find t-1
    labels/...                                      YOLO txt, class 0 = ball
    {wasb,paper}_{train,val}.txt, tennis-{wasb,paper}-{rgb,diff,stack}.yaml, visibility.json
    tracknet/annotations.json + splits.json          for the TrackNet repo's FlyingBallDataset (wasb split only)

Splits:
    wasb   games 1-7 train, games 8-10 test (WASB, BMVC 2023)
    paper  random 70/30 over frames, seed 0 (TrackNet, AVSS 2019); adjacent frames leak across the split
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the extracted Dataset/ dir with game1..game10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=float, default=10.0, help="box side in source pixels")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src).resolve(), Path(args.out).resolve()

    frames = []  # (game, clip, image path, label line or "", (cx, cy) or None, w, h)
    for lab in sorted(src.glob("game*/Clip*/Label.csv"), key=lambda p: (int(p.parts[-3][4:]), int(p.parts[-2][4:]))):
        game, clip = lab.parts[-3], lab.parts[-2]
        for row in csv.DictReader(open(lab)):
            name = row["file name"]
            f = lab.parent / name
            if not f.exists():
                continue
            vis = int(row["visibility"] or 0)
            xy = (float(row["x-coordinate"]), float(row["y-coordinate"])) if vis > 0 and row["x-coordinate"] else None
            frames.append((game, clip, f, xy, Path(name).stem, vis))
    w, h = 1280, 720
    rng = random.Random(args.seed)
    is_paper_train = [rng.random() < 0.7 for _ in frames]
    lists = {"wasb_train": [], "wasb_val": [], "paper_train": [], "paper_val": []}
    tn_images, visibility = [], {}
    for (game, clip, f, xy, stem, vis), ptrain in zip(frames, is_paper_train):
        img = out / "images" / game / clip / f"{game}_{clip}_frame_{stem}{f.suffix}"
        img.parent.mkdir(parents=True, exist_ok=True)
        if not img.is_symlink():
            img.symlink_to(f)
        lab = out / "labels" / game / clip / f"{img.stem}.txt"
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text(f"0 {xy[0] / w:.6f} {xy[1] / h:.6f} {args.box / w:.6f} {args.box / h:.6f}\n" if xy else "")
        visibility[str(img)] = vis
        lists[f"wasb_{'train' if int(game[4:]) <= 7 else 'val'}"].append(str(img))
        lists[f"paper_{'train' if ptrain else 'val'}"].append(str(img))
        ann = [{"label": "ball", "bbox_pixel": {"x": xy[0], "y": xy[1], "width": 0.0, "height": 0.0}}] if xy else []
        tn_images.append(
            {
                "image_path": str(img.relative_to(out)),
                "video_name": f"{game}/{clip}",
                "frame_index": int(stem),
                "width": w,
                "height": h,
                "annotations": ann,
            }
        )
    for name, files in lists.items():
        (out / f"{name}.txt").write_text("\n".join(files) + "\n")
        n_pos = sum(
            bool((out / "labels" / Path(f).relative_to(out / "images")).with_suffix(".txt").read_text()) for f in files
        )
        print(f"{name}: {len(files)} frames, {n_pos} with ball")
    for split in ("wasb", "paper"):
        for mode, ch in (("rgb", 3), ("diff", 5), ("stack", 9)):
            (out / f"tennis-{split}-{mode}.yaml").write_text(
                f"path: {out}\ntrain: {split}_train.txt\nval: {split}_val.txt\nchannels: {ch}\n"
                f"temporal:\n  frames: 3\n  mode: {mode}\nnames:\n  0: ball\n"
            )
    json.dump(visibility, open(out / "visibility.json", "w"))  # read by eval_fastmoving.py for recall per class
    tn = out / "tracknet"
    tn.mkdir(exist_ok=True)
    json.dump({"images": tn_images}, open(tn / "annotations.json", "w"))
    videos = sorted({i["video_name"] for i in tn_images})
    json.dump(
        {
            "train": [v for v in videos if int(v.split("/")[0][4:]) <= 7],
            "val": [v for v in videos if int(v.split("/")[0][4:]) > 7],
        },
        open(tn / "splits.json", "w"),
        indent=1,
    )
    if not (tn / "images").exists():
        (tn / "images").symlink_to(out / "images")


if __name__ == "__main__":
    main()
