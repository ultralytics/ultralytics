# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Network-level diagnostics for the s3d task.

Pure analysis functions over Stereo3DDetMetrics.stats (see val.py update_metrics for the stat schema), plus tensor
probes and a gradient-conflict callback. Literature basis: MonoDLE (CVPR 2021) oracle substitution, TIDE3D
(arXiv:2310.05447) ranking oracle, GradNorm (ICML 2018) / PCGrad (NeurIPS 2020) gradient diagnostics, GFLv2 (CVPR
2021) distribution-shape quality, van Dijk & de Croon (ICCV 2019) counterfactual probes.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import torch

ORACLE_COMPONENTS = ("z", "y", "xy", "location", "dims", "orientation", "score")


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


def oracle_swap(
    stats: list[dict[str, Any]], component: str, matches: list[list[int]] | None = None
) -> list[dict[str, Any]]:
    """Return a copy of stats with one predicted component replaced by GT (MonoDLE-style oracle).

    Matched preds get the GT value for ``component``; IoU matrices are recomputed — except for the "score"
    ranking oracle, which only re-scores confidence by achieved 3D IoU (TIDE3D style). Component semantics:
    "z" replaces depth AND rescales x,y by z_gt/z_pred (preserves the camera ray, i.e. the image-plane center);
    "y" replaces the vertical center only (height/Y hypothesis); "xy" replaces the lateral/vertical center
    keeping depth; "location" replaces the full 3D center; "dims"/"orientation" replace those fields.

    Args:
        stats (list[dict]): Per-image stat dicts.
        component (str): One of ORACLE_COMPONENTS.
        matches (list[list[int]], optional): Precomputed match_stats output.

    Returns:
        (list[dict]): New stat dicts; input stats and Box3D objects are not mutated.
    """
    from ultralytics.models.yolo.s3d.val import compute_3d_iou_batch, compute_bev_iou_batch

    if component not in ORACLE_COMPONENTS:
        raise ValueError(f"Unknown oracle component {component!r}; expected one of {ORACLE_COMPONENTS}")
    if matches is None:
        matches = match_stats(stats)

    out = []
    for stat, match in zip(stats, matches):
        preds, gts = stat["pred_boxes"], stat["gt_boxes"]
        new_preds = []
        for pi, pb in enumerate(preds):
            gi = match[pi]
            if component == "score":
                iou = stat["iou_matrix"]
                q = float(iou[pi].max()) if iou.size else 0.0
                new_preds.append(replace(pb, confidence=min(max(q, 0.0), 1.0)))
                continue
            if gi < 0:
                new_preds.append(pb)
                continue
            gb = gts[gi]
            x, y, z = pb.center_3d
            if component == "z":
                zg = gb.center_3d[2]
                s = zg / z if z > 0 else 1.0
                new_preds.append(replace(pb, center_3d=(x * s, y * s, zg)))
            elif component == "y":
                new_preds.append(replace(pb, center_3d=(x, gb.center_3d[1], z)))
            elif component == "xy":
                new_preds.append(replace(pb, center_3d=(gb.center_3d[0], gb.center_3d[1], z)))
            elif component == "location":
                new_preds.append(replace(pb, center_3d=gb.center_3d))
            elif component == "dims":
                new_preds.append(replace(pb, dimensions=gb.dimensions))
            elif component == "orientation":
                new_preds.append(replace(pb, orientation=gb.orientation))
        new_stat = dict(stat)
        new_stat["pred_boxes"] = new_preds
        if component != "score":
            new_stat["iou_matrix"] = compute_3d_iou_batch(new_preds, gts)
            new_stat["bev_iou_matrix"] = compute_bev_iou_batch(new_preds, gts)
        out.append(new_stat)
    return out


def run_metrics(stats: list[dict[str, Any]], names: dict[int, str]) -> dict[str, Any]:
    """Run Stereo3DDetMetrics over a stats list and return its results_dict."""
    from ultralytics.models.yolo.s3d.metrics import Stereo3DDetMetrics

    m = Stereo3DDetMetrics(names=names)
    for stat in stats:
        m.update_stats(stat)
    m.process()
    return m.results_dict


def oracle_ladder(
    stats: list[dict[str, Any]],
    names: dict[int, str],
    components: tuple[str, ...] = ORACLE_COMPONENTS,
    keys: tuple[str, ...] = ("ap3d_50", "ap3d_70", "apbev_50", "apbev_70"),
) -> dict[str, dict[str, float]]:
    """Compute the one-at-a-time GT-substitution ladder: ΔAP per component quantifies its headroom.

    Args:
        stats (list[dict]): Per-image stat dicts from a completed validation run.
        names (dict[int, str]): Class id → name mapping for the metrics.
        components (tuple[str, ...]): Oracle components to evaluate.
        keys (tuple[str, ...]): results_dict keys to report per rung.

    Returns:
        (dict[str, dict[str, float]]): {"baseline": {key: val}, "<component>": {key: val}, ...}.
    """
    matches = match_stats(stats)
    baseline_rd = run_metrics(stats, names)
    ladder = {"baseline": {k: float(baseline_rd.get(k, 0.0)) for k in keys}}
    for comp in components:
        rd = run_metrics(oracle_swap(stats, comp, matches), names)
        ladder[comp] = {k: float(rd.get(k, 0.0)) for k in keys}
    return ladder


def bootstrap_ap(
    stats: list[dict[str, Any]],
    names: dict[int, str],
    n: int = 200,
    seed: int = 0,
    keys: tuple[str, ...] = ("ap3d_50", "ap3d_70"),
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap-resample images to a (2.5%, 50%, 97.5%) confidence interval per metric key.

    Separates "training is unstable" from "AP on a small val set is a coin flip": any A/B delta smaller than
    the CI width is noise. n=200 keeps runtime tolerable (metrics.process() is O(images) per resample).

    Args:
        stats (list[dict]): Per-image stat dicts.
        names (dict[int, str]): Class id → name mapping.
        n (int): Number of bootstrap resamples.
        seed (int): RNG seed for reproducibility.
        keys (tuple[str, ...]): results_dict keys to report.

    Returns:
        (dict[str, tuple[float, float, float]]): {key: (p2.5, median, p97.5)}.
    """
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(stats), size=len(stats))
        rd = run_metrics([stats[i] for i in idx], names)
        for k in keys:
            samples[k].append(float(rd.get(k, 0.0)))
    return {k: tuple(float(np.percentile(v, p)) for p in (2.5, 50, 97.5)) for k, v in samples.items()}


def dist_stats(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Softmax distribution-shape stats for [N, n_bins] logits (GFLv2: shape predicts localization quality).

    Entropy is normalized by log(n_bins) so 1.0 = uniform (uninformative) and 0.0 = one-hot. Applies to both
    DepthDFL depth-bin logits and cost-volume disparity slices.

    Args:
        logits (torch.Tensor): [..., n_bins] raw logits.

    Returns:
        (dict[str, torch.Tensor]): entropy, top1 (peak mass), top2_mass, argmax — each [...].
    """
    p = logits.float().softmax(dim=-1)
    n_bins = p.shape[-1]
    entropy = -(p * (p + 1e-12).log()).sum(-1) / math.log(n_bins)
    top2 = p.topk(min(2, n_bins), dim=-1).values
    return {"entropy": entropy, "top1": top2[..., 0], "top2_mass": top2.sum(-1), "argmax": p.argmax(-1)}


def expected_dz(z: float, dpx: float, fx: float, baseline: float) -> float:
    """Exact depth change when the measured disparity changes by dpx pixels (d = fx·b/z).

    Args:
        z (float): Base depth in meters.
        dpx (float): Disparity reduction in pixels: the perturbed disparity is d − dpx, so positive dpx
            (less disparity) yields z' > z.
        fx (float): Focal length in pixels (original-image space).
        baseline (float): Stereo baseline in meters.

    Returns:
        (float): z' − z; inf when the perturbed disparity is non-positive.
    """
    d = fx * baseline / z
    if d - dpx <= 0:
        return float("inf")
    return fx * baseline / (d - dpx) - z


def stereo_sensitivity(
    base_z: np.ndarray, shifted_z: np.ndarray, z_gt: np.ndarray, dpx: float, fx: float, baseline: float
) -> float:
    """Median ratio of observed to geometrically-expected depth shift under a right-image shift.

    1.0 → depth is fully stereo-driven; 0.0 → the network ignores stereo (monocular shortcut, e.g. the
    apparent-size cue on constant-dimension objects). van Dijk & de Croon (ICCV 2019)-style probe.

    Args:
        base_z (np.ndarray): Predicted depths on the unmodified batch.
        shifted_z (np.ndarray): Predicted depths on the right-image-shifted batch (matched objects).
        z_gt (np.ndarray): GT depths (unused in the ratio; kept for per-range breakdowns by callers).
        dpx (float): Disparity change implied by the shift (right image rolled +k px → dpx = −k).
        fx (float): Focal length in pixels.
        baseline (float): Stereo baseline in meters.

    Returns:
        (float): Median sensitivity ratio, or nan when no valid objects.
    """
    expected = np.array([expected_dz(float(z), dpx, fx, baseline) for z in base_z])
    valid = np.isfinite(expected) & (np.abs(expected) > 1e-9)
    if not valid.any():
        return float("nan")
    return float(np.median((shifted_z[valid] - base_z[valid]) / expected[valid]))


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
