# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Separate a global depth-scale error from a per-person one.

Absolute root depth can be wrong two ways: every person off by the same factor, which is a camera-calibration
problem, or each person wrong independently, which is a model problem. AbsRel conflates them. This fits one
scalar per model over the whole benchmark — the median of gt/pred — and reports AbsRel before and after.

If AbsRel collapses once that single number is applied, the model's depth *ordering* is fine and what is
broken is the scale it inherited from the teacher's assumed field of view.

Usage:
    python research/src/depth_scale_probe.py --models a.pt b.pt --data /home/rick/3dpw-pose3d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.pose3d import decode_z


def pairs_for(model_path: str, root: Path, limit: int) -> np.ndarray:
    """Return matched (pred_root_depth, gt_root_depth) in metres for one model over the benchmark."""
    m = YOLO(model_path)
    out = []
    files = [root / p.strip().lstrip("./") for p in (root / "test.txt").read_text().split()][:limit]
    for i in range(0, len(files), 32):
        batch = [str(f) for f in files[i : i + 32]]
        for path, r in zip(batch, m.predict(batch, conf=0.25, verbose=False)):
            lb = root / "labels" / (Path(path).stem + ".txt")
            if not lb.exists() or r.keypoints is None or len(r.keypoints.data) == 0:
                continue
            g = np.array([x.split() for x in lb.read_text().strip().splitlines()], dtype=np.float32)
            h, w = r.orig_shape
            gb = torch.tensor(
                np.stack(
                    [
                        (g[:, 1] - g[:, 3] / 2) * w,
                        (g[:, 2] - g[:, 4] / 2) * h,
                        (g[:, 1] + g[:, 3] / 2) * w,
                        (g[:, 2] + g[:, 4] / 2) * h,
                    ],
                    1,
                )
            )
            iou = box_iou(gb, r.boxes.xyxy.cpu())
            best_iou, best = iou.max(1)
            gk = g[:, 5:].reshape(len(g), -1, 4)
            pk = r.keypoints.data.cpu().numpy()
            for j in torch.nonzero(best_iou >= 0.5).flatten().tolist():
                _, gz = decode_z(gk[j, :17, 3], gk[j, 17, 3])
                _, pz = decode_z(pk[best[j], :17, 3], pk[best[j], 17, 3])
                out.append((float(pz), float(gz)))
    return np.array(out)


def main(models: list[str], data: Path, limit: int) -> None:
    """Report raw and globally scale-corrected depth error for each model."""
    for mp in models:
        p = pairs_for(mp, data, limit)
        if not len(p):
            print(f"{Path(mp).parent.parent.name}: no matches")
            continue
        pred, gt = p[:, 0], p[:, 1]
        raw = np.abs(pred - gt) / gt
        s = float(np.median(gt / pred))  # one scalar for the whole benchmark
        aligned = np.abs(pred * s - gt) / gt
        name = Path(mp).parent.parent.name
        print(
            f"{name:14s} n={len(p):5d}  AbsRel raw {raw.mean():.4f}  "
            f"global scale {s:.3f}  AbsRel scale-aligned {aligned.mean():.4f}  "
            f"(explained {100 * (1 - aligned.mean() / raw.mean()):.0f}%)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path("/home/rick/3dpw-pose3d"))
    parser.add_argument("--limit", type=int, default=1500)
    main(**vars(parser.parse_args()))
