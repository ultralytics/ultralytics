# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the s3d network diagnostics toolkit (ultralytics/models/yolo/s3d/diagnose.py)."""

import numpy as np
import pytest

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.models.yolo.s3d.diagnose import (
    depth_bias_fit,
    error_records,
    match_stats,
    summarize_errors,
)
from ultralytics.models.yolo.s3d.val import compute_3d_iou_batch, compute_bev_iou_batch


def _box(x=0.0, y=1.0, z=20.0, dims=(4.0, 1.8, 1.5), ry=0.0, cls_id=0, conf=0.9):
    """Build a synthetic Box3D with car-like defaults."""
    return Box3D(
        center_3d=(x, y, z),
        dimensions=dims,
        orientation=ry,
        class_label="Car",
        class_id=cls_id,
        confidence=conf,
        truncated=0.0,
        occluded=0,
    )


def _stat(pred_boxes, gt_boxes):
    """Build a stats dict exactly like val.py update_metrics does."""
    return {
        "pred_boxes": pred_boxes,
        "gt_boxes": gt_boxes,
        "iou_matrix": compute_3d_iou_batch(pred_boxes, gt_boxes),
        "bev_iou_matrix": compute_bev_iou_batch(pred_boxes, gt_boxes),
        "gt_difficulties": np.zeros(len(gt_boxes), dtype=int),  # all Easy
        "pred_heights_2d": np.full(len(pred_boxes), 50.0, dtype=np.float32),
    }


def test_match_stats_greedy_bev():
    """Two preds, two GTs: each pred matches its overlapping GT, not the other."""
    gt = [_box(z=20.0), _box(x=10.0, z=40.0)]
    pred = [_box(z=20.3, conf=0.9), _box(x=10.1, z=40.5, conf=0.8)]
    matches = match_stats([_stat(pred, gt)])
    assert matches[0] == [0, 1]


def test_match_stats_center_fallback():
    """A pred with zero IoU everywhere matches the nearest same-class GT under the cap."""
    gt = [_box(z=20.0)]
    pred = [_box(z=23.0)]  # 3 m off: zero 3D IoU but within 4 m fallback
    matches = match_stats([_stat(pred, gt)])
    assert matches[0] == [0]
    far = [_box(z=30.0)]  # 10 m off: beyond cap → unmatched
    assert match_stats([_stat(far, gt)])[0] == [-1]


def test_error_records_signed():
    """Errors are signed pred − GT."""
    gt = [_box(z=20.0)]
    pred = [_box(z=21.0, x=0.5)]
    recs = error_records([_stat(pred, gt)])
    assert len(recs) == 1
    assert recs[0]["dz"] == pytest.approx(1.0, abs=1e-6)
    assert recs[0]["dx"] == pytest.approx(0.5, abs=1e-6)
    assert recs[0]["z_gt"] == pytest.approx(20.0)


def test_summarize_errors():
    """Summary aggregates MAE and IoU-threshold fractions."""
    recs = [
        {"dx": 0.5, "dy": 0.0, "dz": 1.0, "dtheta": 0.0, "iou3d": 0.6, "ioubev": 0.7},
        {"dx": -0.5, "dy": 0.0, "dz": -2.0, "dtheta": 0.1, "iou3d": 0.4, "ioubev": 0.5},
    ]
    s = summarize_errors(recs)
    assert s["n"] == 2
    assert s["mae_z"] == pytest.approx(1.5)
    assert s["frac_iou3d_ge_50"] == pytest.approx(0.5)


def test_oracle_swap_z_restores_iou():
    """A pred that only errs in depth becomes a perfect box under the z oracle."""
    from ultralytics.models.yolo.s3d.diagnose import oracle_swap

    # y=0 keeps the box on the optical axis so the ray-preserving rescale is identity in x,y
    gt = [_box(y=0.0, z=20.0)]
    pred = [_box(y=0.0, z=21.5)]
    swapped = oracle_swap([_stat(pred, gt)], "z")
    sb = swapped[0]["pred_boxes"][0]
    assert sb.center_3d[2] == pytest.approx(20.0)
    assert swapped[0]["iou_matrix"][0, 0] == pytest.approx(1.0, abs=1e-3)
    # original stats untouched
    assert _stat(pred, gt)["pred_boxes"][0].center_3d[2] == pytest.approx(21.5)


def test_oracle_swap_z_rescales_xy():
    """Depth oracle preserves the camera ray: x scales by z_gt/z_pred."""
    from ultralytics.models.yolo.s3d.diagnose import oracle_swap

    gt = [_box(x=2.0, z=20.0)]
    pred = [_box(x=2.2, z=22.0)]
    sb = oracle_swap([_stat(pred, gt)], "z")[0]["pred_boxes"][0]
    assert sb.center_3d[2] == pytest.approx(20.0)
    assert sb.center_3d[0] == pytest.approx(2.2 * 20.0 / 22.0)


def test_oracle_score_reranks():
    """Ranking oracle sets confidence to achieved 3D IoU."""
    from ultralytics.models.yolo.s3d.diagnose import oracle_swap

    gt = [_box(z=20.0)]
    pred = [_box(z=20.1, conf=0.3), _box(x=15.0, z=50.0, conf=0.95)]  # good box low conf, bad box high conf
    swapped = oracle_swap([_stat(pred, gt)], "score")
    confs = [b.confidence for b in swapped[0]["pred_boxes"]]
    assert confs[0] > confs[1]


def test_oracle_ladder_orders_headroom():
    """With depth as the only error source, the z oracle recovers more AP than the dims oracle."""
    from ultralytics.models.yolo.s3d.diagnose import oracle_ladder

    stats = []
    rng = np.random.default_rng(1)
    for _ in range(12):
        gt = [_box(z=float(rng.uniform(10, 40)))]
        # depth off by ~8%, dims off by 2% → depth is the binding error
        pred = [_box(z=gt[0].center_3d[2] * 1.08, dims=(4.08, 1.836, 1.53))]
        stats.append(_stat(pred, gt))
    ladder = oracle_ladder(stats, {0: "Car"}, components=("z", "dims"))
    assert ladder["z"]["ap3d_70"] >= ladder["dims"]["ap3d_70"]
    assert ladder["z"]["ap3d_70"] >= ladder["baseline"]["ap3d_70"]


def _mixed_stats(n_img=20, seed=0):
    """Images with ~5% depth-noised single-object predictions."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_img):
        z = float(rng.uniform(10, 40))
        gt = [_box(z=z)]
        pred = [_box(z=z * float(rng.normal(1.0, 0.05)))]
        stats.append(_stat(pred, gt))
    return stats


def test_bootstrap_ap_deterministic_and_ordered():
    """Seeded bootstrap is reproducible and returns ordered (lo, med, hi) in [0, 1]."""
    from ultralytics.models.yolo.s3d.diagnose import bootstrap_ap

    stats = _mixed_stats()
    ci1 = bootstrap_ap(stats, {0: "Car"}, n=25, seed=7)
    ci2 = bootstrap_ap(stats, {0: "Car"}, n=25, seed=7)
    assert ci1 == ci2  # seeded → reproducible
    lo, med, hi = ci1["ap3d_50"]
    assert lo <= med <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ap_tightens_with_perfect_preds():
    """All-perfect predictions → zero-width CI at AP=1 regardless of resample."""
    from ultralytics.models.yolo.s3d.diagnose import bootstrap_ap

    stats = [_stat([_box(z=z)], [_box(z=z)]) for z in (10.0, 20.0, 30.0, 40.0)]
    ci = bootstrap_ap(stats, {0: "Car"}, n=10, seed=0)
    lo, med, hi = ci["ap3d_50"]
    assert lo == pytest.approx(hi)


def test_depth_bias_fit_recovers_slope():
    """dz = 0.05*z + 0.1 exactly → fit recovers (0.05, 0.1, ~0)."""
    rng = np.random.default_rng(0)
    recs = [{"z_gt": z, "dz": 0.05 * z + 0.1} for z in rng.uniform(5, 60, 50)]
    a, b, resid = depth_bias_fit(recs)
    assert a == pytest.approx(0.05, abs=1e-6)
    assert b == pytest.approx(0.1, abs=1e-6)
    assert resid == pytest.approx(0.0, abs=1e-6)
