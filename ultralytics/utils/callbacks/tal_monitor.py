# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""TAL label-assignment monitor.

When enabled with `tal_monitor=True` (train arg), appends one JSON record per epoch to
`{save_dir}/tal_stats.jsonl` with per-head assignment stats collected by TaskAlignedAssigner:
positives per GT, zero-positive rate, multi-GT conflict rate, chosen-anchor align/IoU/cls-score/
soft-label means, supervision mass (target_scores_sum per GT), and per-size-bucket breakdown.
The file can be shared for offline analysis of one-to-one vs one-to-many assignment behavior.
"""

import json
from pathlib import Path


def _summarize(a: dict) -> dict:
    """Reduce a TaskAlignedAssigner.mon accumulator to per-GT means."""
    n = max(a["n_gt"], 1)
    out = {
        "pos_per_gt": round(a["pos_post"] / n, 2),
        "candidates_pre_dedup": round(a["pos_pre"] / n, 2),
        "zero_pos_pct": round(100 * a["zero_pos"] / n, 3),
        "conflict_pct": round(100 * a["conflict"] / n, 2),
        "chosen_align": round(a["align"] / n, 5),
        "chosen_iou": round(a["iou"] / n, 4),
        "chosen_cls_score": round(a["score"] / n, 4),
        "chosen_soft_label": round(a["soft"] / n, 4),
        "tss_per_gt": round(a["tss"] / n, 3),
        "by_size": {},
    }
    for name, bs in a["by_size"].items():
        bn = max(bs["n_gt"], 1)
        out["by_size"][name] = {"zero_pos_pct": round(100 * bs["zero_pos"] / bn, 3),
                                "chosen_iou": round(bs["iou"] / bn, 4),
                                "chosen_cls_score": round(bs["score"] / bn, 4),
                                "chosen_soft_label": round(bs["soft"] / bn, 4)}
    return out


def on_train_epoch_end(trainer) -> None:
    """Dump per-epoch TAL assignment stats to tal_stats.jsonl and reset the accumulators."""
    from ultralytics.utils.torch_utils import unwrap_model

    criterion = getattr(unwrap_model(trainer.model), "criterion", None)
    if criterion is None:
        return
    heads = {"o2m": criterion.one2many, "o2o": criterion.one2one} if hasattr(criterion, "one2one") else {"head": criterion}
    rec = {"epoch": trainer.epoch + 1}
    for name, loss_fn in heads.items():
        assigner = getattr(loss_fn, "assigner", None)
        if assigner is None or not getattr(assigner, "monitor", False):
            return
        rec[name] = _summarize(assigner.mon)
    with open(Path(trainer.save_dir) / "tal_stats.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    for loss_fn in heads.values():
        loss_fn.assigner.mon = loss_fn.assigner._reset_mon()
