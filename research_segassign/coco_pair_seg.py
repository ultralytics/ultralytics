"""Official COCO segm/bbox AP for seg-assign runs plus paired random-half-subset resampling of the mask - box delta.

Usage: python coco_pair_seg.py <runs_root>   (expects <runs_root>/{box,mask}_s{0,1}/predictions.json)
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

ANN = "/data/shared-datasets/coco/annotations/instances_val2017.json"
RUNS = ["box_s0", "box_s1", "mask_s0", "mask_s1"]
_gt, _dt = None, {}


def init(files):
    global _gt
    _gt = COCO(ANN)
    for n, f in files.items():
        _dt[n] = _gt.loadRes(f)


def ap(n, ids=None, t="segm"):
    e = COCOeval_faster(_gt, _dt[n], iouType=t, print_function=lambda *a, **k: None)
    if ids is not None:
        e.params.imgIds = list(ids)
    e.evaluate()
    e.accumulate()
    e.summarize()
    return e.stats_as_dict["AP_all"] * 100


def task(args):
    a, b, seed, ids_all = args
    ids = np.random.default_rng(seed).choice(ids_all, len(ids_all) // 2, replace=False)
    return ap(b, ids) - ap(a, ids)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    files = {n: str(root / n / "predictions.json") for n in RUNS}
    init(files)
    ids_all = sorted(_gt.getImgIds())
    for t in ("segm", "bbox"):
        for n in RUNS:
            print(f"FULL {n:8s} {t} AP {ap(n, t=t):.2f}")
    with ProcessPoolExecutor(16, initializer=init, initargs=(files,)) as ex:
        for a, b in (("box_s0", "mask_s0"), ("box_s1", "mask_s1"), ("box_s0", "box_s1")):
            d = np.array(list(ex.map(task, [(a, b, s, ids_all) for s in range(200)])))
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(
                f"PAIR {b} - {a}: full-set segm delta {ap(b) - ap(a):+.2f} | half-subsets n=200: mean {d.mean():+.2f} "
                f"95% [{lo:+.2f}, {hi:+.2f}] positive {(d > 0).mean() * 100:.1f}%"
            )
