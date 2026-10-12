# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Rewrite COCO's own 17x3 keypoint labels in the pose3d 18x4 layout, so `yolo val` can score a pose3d checkpoint
against human annotation with no harness of its own.

Every COCO-labelled joint is written with visibility 1 (labelled in 2D only, the pose3d tri-state), depth 0, and
the root as fully unlabelled. OKS masks on `visibility != 0`, so the 17 COCO joints score exactly as on the 17x3
ruler and the root never scores; `Pose3DValidator` gates its 3D metrics on `visibility == 2`, so those columns stay
empty rather than being computed against depth that does not exist.

Usage:
    python research/src/coco_gt_to_pose3d.py /data/shared-datasets/coco-pose /home/rick/coco-pose-gt-18x4
"""

import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
(dst / "labels/val2017").mkdir(parents=True, exist_ok=True)
(dst / "images").mkdir(exist_ok=True)
if not (dst / "images/val2017").exists():
    (dst / "images/val2017").symlink_to(src / "images/val2017")

for f in sorted((src / "labels/val2017").glob("*.txt")):
    rows = []
    for line in f.read_text().splitlines():
        v = line.split()
        kpts = [float(x) for x in v[5:]]
        joints = [f"{kpts[i]:g} {kpts[i + 1]:g} {int(kpts[i + 2] > 0)} 0" for i in range(0, 51, 3)]
        rows.append(" ".join([*v[:5], *joints, "0 0 0 0"]))
    (dst / "labels/val2017" / f.name).write_text("\n".join(rows) + "\n")

# Named val.txt, not val2017.txt: a COCO-looking val2017.txt sets `is_coco`, which forces save_json on and hands
# the metrics to pycocotools, so the two columns would be scored by different evaluators.
(dst / "val.txt").write_text((src / "val2017.txt").read_text())
print(f"wrote {len(list((dst / 'labels/val2017').glob('*.txt')))} label files to {dst}")
