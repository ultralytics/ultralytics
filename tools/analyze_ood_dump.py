"""Recompute the pooled OOD metrics offline, from a dump written by val_yoloa_mvtec_ultra.py --dump.

The point of this file is that it computes NOTHING itself. Matching is ultralytics'
``match_predictions``, scoring is ``YOLOAnomalyValidator._ood_map_metrics``, and the selection is
``GroupMeta`` -- the same three pieces the in-loop path uses. A number that differs from the run
that produced the dump is therefore a bug here, never a second opinion.

    PYTHONPATH=. python tools/analyze_ood_dump.py preds.jsonl --data <MVTec-Ultra root>

Once the reproduction is trusted, the slicing is free: any ``--groups`` query re-scores the same
dump in seconds instead of a GPU pass.
"""

import argparse
import json

import numpy as np
import torch

from ultralytics.models.yolo.anomaly.val import YOLOAnomalyValidator
from ultralytics.models.yolo.anomaly.val_rnd import GroupMeta, OODEvaluator

DECISIVE = ("mAP50", "mAP50@0.25", "R50@0.25", "P50@0.25")


def load(path: str) -> YOLOAnomalyValidator:
    """Replay a dump into a read-only validator holding the same flat stats the in-loop pass built."""
    v = YOLOAnomalyValidator.__new__(YOLOAnomalyValidator)
    v.iouv, v.names = torch.linspace(0.1, 0.5, 9), {0: "anomaly"}
    v.niou = v.iouv.numel()

    files, tp, conf, pcls, tcls, pimg, gimg = [], [], [], [], [], [], []
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        files.append(r["file"])
        p = {"bboxes": torch.tensor(r["bboxes"], dtype=torch.float32).reshape(-1, 4),
             "cls": torch.tensor(r["cls"], dtype=torch.float32)}
        g = {"bboxes": torch.tensor(r["gt_bboxes"], dtype=torch.float32).reshape(-1, 4),
             "cls": torch.tensor(r["gt_cls"], dtype=torch.float32)}
        tp.append(v._process_batch(p, g)["tp"])  # ultralytics' matcher, not a copy of it
        conf.append(np.asarray(r["conf"], dtype=np.float32))
        pcls.append(np.asarray(r["cls"], dtype=np.float32))
        tcls.append(np.asarray(r["gt_cls"], dtype=np.float32))
        pimg.append(np.full(len(r["conf"]), i, dtype=np.int64))
        gimg.append(np.full(len(r["gt_cls"]), i, dtype=np.int64))

    v._ood_files = files
    v._ood_stats = {"tp": np.concatenate(tp), "conf": np.concatenate(conf),
                    "pred_cls": np.concatenate(pcls), "target_cls": np.concatenate(tcls)}
    v._ood_img = {"pred": np.concatenate(pimg), "gt": np.concatenate(gimg)}
    return v


def select(v: YOLOAnomalyValidator, meta: GroupMeta, query: str) -> list[int]:
    """Image indices, in dump order, that the query pools over.

    All this does is split the flat dump back into per-product blocks; the rule for which images a
    query keeps is ``OODEvaluator.select_images`` and is not restated here -- a second copy of it
    would be the second ruler this whole file exists to avoid.
    """
    slugs = {slug: (ds, product) for ds, product, slug in meta.products()}
    per: dict[str, tuple[list[str], list[int]]] = {}
    for i, f in enumerate(v._ood_files):
        slug = next((p for p in f.split("/") if p in slugs), None)
        if slug is None:
            raise ValueError(f"cannot attribute {f} to a product under the given --data")
        files, where = per.setdefault(slug, ([], []))
        files.append(f)
        where.append(i)
    keys = list(per)
    blocks = [(*slugs[k], per[k][0]) for k in keys]
    kept = OODEvaluator.select_images(meta, query, blocks)
    return sorted(per[keys[b]][1][i] for b, idx in kept for i in idx)


def main() -> None:
    """Replay a dump and print the pooled decisive metrics for one group query."""
    ap = argparse.ArgumentParser(description="Recompute pooled OOD metrics from a prediction dump.")
    ap.add_argument("dump", help="jsonl written by val_yoloa_mvtec_ultra.py --dump")
    ap.add_argument("--data", required=True, help="the MVTec-Ultra root whose meta.yaml is the taxonomy")
    ap.add_argument("--groups", default="nature=structural", help="tag query, or a list of group ids")
    a = ap.parse_args()

    v = load(a.dump)
    sel = select(v, GroupMeta(f"{a.data}/meta.yaml"), a.groups)
    m = v._ood_map_metrics(images=sel)
    print(f"{len(v._ood_files)} images in dump · pooled over {len(sel)} · {len(v._ood_stats['conf'])} predictions")
    for k in DECISIVE:
        print(f"  pool_{k:<12} {m[k]:.4f}")


if __name__ == "__main__":
    main()
