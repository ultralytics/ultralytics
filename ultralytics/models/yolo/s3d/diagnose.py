# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Network-level diagnostics for the s3d task.

Pure analysis functions over Stereo3DDetMetrics.stats (see val.py update_metrics for the stat schema), plus tensor
probes and a gradient-conflict callback. Literature basis: MonoDLE (CVPR 2021) oracle substitution, TIDE3D
(arXiv:2310.05447) ranking oracle, GradNorm (ICML 2018) / PCGrad (NeurIPS 2020) gradient diagnostics, GFLv2 (CVPR
2021) distribution-shape quality, van Dijk & de Croon (ICCV 2019) counterfactual probes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def match_stats(stats: list[dict[str, Any]], max_center_dist: float = 4.0) -> list[list[int]]:
    """Match predictions to GT per image for diagnostics (not for AP).

    Greedy on descending BEV IoU; a pred with zero IoU against every GT falls back to the nearest unmatched
    same-class GT by 3D center distance, capped at ``max_center_dist`` meters.

    Args:
        stats (list[dict]): Per-image stat dicts (Stereo3DDetMetrics.stats schema).
        max_center_dist (float): Fallback matching radius in meters.

    Returns:
        (list[list[int]]): Per image, matched GT indices aligned with pred_boxes (-1 = unmatched).
    """
    all_matches = []
    for stat in stats:
        preds, gts = stat["pred_boxes"], stat["gt_boxes"]
        bev = stat.get("bev_iou_matrix", np.zeros((len(preds), len(gts))))
        match = [-1] * len(preds)
        taken: set[int] = set()
        # Pass 1: greedy by BEV IoU
        if bev.size:
            order = np.dstack(np.unravel_index(np.argsort(-bev, axis=None), bev.shape))[0]
            for pi, gi in order:
                if bev[pi, gi] <= 0:
                    break
                if match[pi] == -1 and int(gi) not in taken:
                    match[pi] = int(gi)
                    taken.add(int(gi))
        # Pass 2: center-distance fallback for still-unmatched preds
        for pi, pb in enumerate(preds):
            if match[pi] != -1:
                continue
            best, best_d = -1, max_center_dist
            for gi, gb in enumerate(gts):
                if gi in taken or gb.class_id != pb.class_id:
                    continue
                d = float(np.linalg.norm(np.array(pb.center_3d) - np.array(gb.center_3d)))
                if d < best_d:
                    best, best_d = gi, d
            if best >= 0:
                match[pi] = best
                taken.add(best)
        all_matches.append(match)
    return all_matches


def error_records(
    stats: list[dict[str, Any]], matches: list[list[int]] | None = None, max_center_dist: float = 4.0
) -> list[dict[str, float]]:
    """Compute per-matched-prediction signed component errors (pred − GT).

    Args:
        stats (list[dict]): Per-image stat dicts.
        matches (list[list[int]], optional): Precomputed match_stats output; computed when None.
        max_center_dist (float): Fallback matching radius passed to match_stats.

    Returns:
        (list[dict]): One record per matched pred with keys img, cls, conf, z_gt, dx, dy, dz, dl, dw, dh,
            dtheta (wrapped to [-π, π]), iou3d, ioubev.
    """
    if matches is None:
        matches = match_stats(stats, max_center_dist)
    records = []
    for img_i, (stat, match) in enumerate(zip(stats, matches)):
        for pi, gi in enumerate(match):
            if gi < 0:
                continue
            pb, gb = stat["pred_boxes"][pi], stat["gt_boxes"][gi]
            dtheta = float(np.arctan2(np.sin(pb.orientation - gb.orientation), np.cos(pb.orientation - gb.orientation)))
            iou = stat["iou_matrix"]
            bev = stat["bev_iou_matrix"]
            records.append(
                {
                    "img": img_i,
                    "cls": int(pb.class_id),
                    "conf": float(pb.confidence),
                    "z_gt": float(gb.center_3d[2]),
                    "dx": float(pb.center_3d[0] - gb.center_3d[0]),
                    "dy": float(pb.center_3d[1] - gb.center_3d[1]),
                    "dz": float(pb.center_3d[2] - gb.center_3d[2]),
                    "dl": float(pb.dimensions[0] - gb.dimensions[0]),
                    "dw": float(pb.dimensions[1] - gb.dimensions[1]),
                    "dh": float(pb.dimensions[2] - gb.dimensions[2]),
                    "dtheta": dtheta,
                    "iou3d": float(iou[pi, gi]) if iou.size else 0.0,
                    "ioubev": float(bev[pi, gi]) if bev.size else 0.0,
                }
            )
    return records


def summarize_errors(records: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate error records into the stable per-component profile (the anti-spiky-AP eval)."""
    if not records:
        return {"n": 0}
    arr = {k: np.array([r[k] for r in records]) for k in ("dx", "dy", "dz", "dtheta", "iou3d", "ioubev")}
    return {
        "n": len(records),
        "mae_x": float(np.abs(arr["dx"]).mean()),
        "mae_y": float(np.abs(arr["dy"]).mean()),
        "mae_z": float(np.abs(arr["dz"]).mean()),
        "mae_theta": float(np.abs(arr["dtheta"]).mean()),
        "mean_iou3d": float(arr["iou3d"].mean()),
        "mean_ioubev": float(arr["ioubev"].mean()),
        "frac_iou3d_ge_50": float((arr["iou3d"] >= 0.5).mean()),
        "frac_iou3d_ge_70": float((arr["iou3d"] >= 0.7).mean()),
    }


def depth_bias_fit(records: list[dict[str, float]]) -> tuple[float, float, float]:
    """Fit dz = a·z_gt + b over matched predictions.

    A significant slope ``a`` is a systematic depth scale bias (calibration/fusion bug signature); a near-zero
    slope with large residual std is a resolution/matching limit.

    Args:
        records (list[dict]): error_records output.

    Returns:
        (tuple[float, float, float]): Slope a, intercept b, residual standard deviation.
    """
    if len(records) < 3:
        return 0.0, 0.0, 0.0
    z = np.array([r["z_gt"] for r in records])
    dz = np.array([r["dz"] for r in records])
    a, b = np.polyfit(z, dz, 1)
    resid = dz - (a * z + b)
    return float(a), float(b), float(resid.std())
