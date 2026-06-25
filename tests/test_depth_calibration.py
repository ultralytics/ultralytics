"""Unit tests for scale-only depth calibration (cal_a/cal_b log-affine)."""

import math

import numpy as np
import torch

from ultralytics.models.yolo.depth.calibrate import (
    _depth_head,
    fit_calibration,
    fit_calibration_selective,
    lstsq_affine,
    select_calibration,
    select_calibration_cv,
)
from ultralytics.nn.modules.head import Depth
from ultralytics.nn.tasks import DepthModel


def test_lstsq_affine_recovers_known_affine():
    """A clean affine log_gt = a·log_pred + b is recovered to high precision."""
    rng = np.random.default_rng(0)
    log_pred = rng.normal(size=20000)
    a0, b0 = 0.7, 1.3
    log_gt = a0 * log_pred + b0
    a, b = lstsq_affine(log_pred, log_gt)
    assert abs(a - a0) < 1e-6 and abs(b - b0) < 1e-6


def test_lstsq_affine_distance_weighted_runs():
    """The optional gt-weighted fit (dist_power>0) returns a finite result."""
    rng = np.random.default_rng(1)
    log_pred = rng.normal(size=5000)
    log_gt = 1.1 * log_pred - 0.4 + rng.normal(scale=0.05, size=5000)
    a, b = lstsq_affine(log_pred, log_gt, dist_power=1.0)
    assert math.isfinite(a) and math.isfinite(b)


def test_calibration_identity_by_default():
    """A fresh log head carries identity calibration buffers (no-op)."""
    head = Depth(c_mid=16, mode="log", ch=(16, 32, 64))
    assert float(head.cal_a) == 1.0 and float(head.cal_b) == 0.0


def test_calibration_applied_in_forward():
    """Setting cal_a/cal_b applies d' = exp(a·log d + b) = d**a · exp(b) to the output."""
    torch.manual_seed(0)
    head = Depth(c_mid=16, mode="log", ch=(16, 32, 64)).eval()
    feats = [torch.randn(1, 16, 32, 32), torch.randn(1, 32, 16, 16), torch.randn(1, 64, 8, 8)]
    with torch.no_grad():
        raw = head(feats)  # identity calibration
        a, b = 2.0, 0.5
        head.cal_a.fill_(a)
        head.cal_b.fill_(b)
        cal = head(feats)
    assert torch.allclose(cal, raw.pow(a) * math.exp(b), rtol=1e-4, atol=1e-5)


def test_fit_calibration_sets_buffers():
    """fit_calibration returns (a, b), writes them into the head, and is finite."""
    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    batches = [
        {"img": (torch.rand(2, 3, 64, 64) * 255).to(torch.uint8), "depth": torch.rand(2, 64, 64) * 5 + 0.5}
        for _ in range(2)
    ]
    res = fit_calibration(model, batches, device="cpu", max_images=4)
    assert res is not None
    a, b = res
    assert math.isfinite(a) and math.isfinite(b)
    head = model.model[-1]
    assert abs(float(head.cal_a) - a) < 1e-4 and abs(float(head.cal_b) - b) < 1e-4


def test_fit_calibration_no_valid_pixels_returns_none():
    """All-invalid ground truth → no fit; buffers restored to identity."""
    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    batches = [{"img": (torch.rand(1, 3, 64, 64) * 255).to(torch.uint8), "depth": torch.zeros(1, 64, 64)}]
    res = fit_calibration(model, batches, device="cpu", max_images=2)
    assert res is None
    head = model.model[-1]
    assert float(head.cal_a) == 1.0 and float(head.cal_b) == 0.0


def test_depth_head_finder_ignores_non_depth_head():
    """_depth_head returns None when the last module lacks calibration buffers."""
    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    model.model[-1] = torch.nn.Identity()  # strip the Depth head
    assert _depth_head(model) is None


def test_select_calibration_keeps_identity_when_already_aligned():
    """When predictions already match GT scale, no transform beats identity → identity is kept."""
    rng = np.random.default_rng(0)
    lp = rng.normal(size=20000)
    lg = lp.copy()  # exact alignment: identity is optimal
    res = select_calibration(lp, lg, lp.copy(), lg.copy())
    assert res["name"] == "identity"
    assert res["a"] == 1.0 and res["b"] == 0.0


def test_select_calibration_picks_scale_only_for_pure_offset():
    """A pure global log-offset is best fixed by scale-only (a=1, b=offset), not the affine slope."""
    rng = np.random.default_rng(1)
    lp = rng.normal(size=20000)
    offset = 0.7
    lg = lp + offset
    res = select_calibration(lp, lg, lp.copy(), lg.copy())
    assert res["name"] == "scale-only"
    assert res["a"] == 1.0 and abs(res["b"] - offset) < 1e-6


def test_select_calibration_picks_affine_when_slope_helps():
    """When GT is a sloped log-affine of preds, only the affine candidate recovers it."""
    rng = np.random.default_rng(2)
    lp = rng.normal(size=20000)
    lg = 0.5 * lp + 0.3  # slope != 1, scale-only cannot match this
    res = select_calibration(lp, lg, lp.copy(), lg.copy())
    assert res["name"] == "affine"
    assert abs(res["a"] - 0.5) < 1e-6 and abs(res["b"] - 0.3) < 1e-6


def test_select_calibration_does_no_harm_when_fit_does_not_generalize():
    """A transform fit on the fit-split that hurts the held-out score-split is rejected for identity."""
    rng = np.random.default_rng(3)
    lp_fit = rng.normal(size=20000)
    lg_fit = lp_fit + 0.5  # fit-split suggests a +0.5 offset
    lp_score = rng.normal(size=20000)
    lg_score = lp_score.copy()  # but the held-out split is already aligned → offset would hurt
    res = select_calibration(lp_fit, lg_fit, lp_score, lg_score)
    assert res["name"] == "identity"


def _aligned(rng, n=4000, slope=1.0, offset=0.0):
    """Return one per-image (log_pred, log_gt) pair with log_gt = slope·log_pred + offset."""
    lp = rng.normal(size=n)
    return lp, slope * lp + offset


def test_cv_keeps_identity_when_all_folds_aligned():
    """Cross-validated selection keeps identity when predictions already match GT across folds."""
    rng = np.random.default_rng(0)
    pairs = [_aligned(rng) for _ in range(6)]
    res = select_calibration_cv(pairs, folds=2)
    assert res["name"] == "identity"


def test_cv_picks_scale_only_for_consistent_offset():
    """A consistent global offset across folds is captured by scale-only and refit on all data."""
    rng = np.random.default_rng(1)
    pairs = [_aligned(rng, offset=0.7) for _ in range(6)]
    res = select_calibration_cv(pairs, folds=2)
    assert res["name"] == "scale-only"
    assert res["a"] == 1.0 and abs(res["b"] - 0.7) < 1e-6


def test_cv_falls_back_to_scale_only_for_slope_by_default():
    """By default affine is NOT auto-selected (slope overfits within-dataset CV) → scale-only fallback."""
    rng = np.random.default_rng(2)
    pairs = [_aligned(rng, slope=0.5, offset=0.3) for _ in range(6)]
    res = select_calibration_cv(pairs, folds=2)
    assert res["name"] == "scale-only"


def test_cv_picks_affine_when_slope_consistent_and_opted_in():
    """With allow_affine=True a consistent log-slope is captured by the affine candidate."""
    rng = np.random.default_rng(2)
    pairs = [_aligned(rng, slope=0.5, offset=0.3) for _ in range(6)]
    res = select_calibration_cv(pairs, folds=2, allow_affine=True)
    assert res["name"] == "affine"
    assert abs(res["a"] - 0.5) < 1e-6


def test_cv_rejects_transform_that_does_not_help_across_folds():
    """A transform that only helps some images (not held-out on average) is rejected for identity."""
    rng = np.random.default_rng(3)
    # interleave aligned and offset images: no single global transform helps both folds on average
    pairs = []
    for i in range(6):
        pairs.append(_aligned(rng) if i % 2 == 0 else _aligned(rng, offset=1.0))
    res = select_calibration_cv(pairs, folds=2)
    assert res["name"] == "identity"


def test_calibrate_checkpoint_applies_selective_policy(tmp_path):
    """calibrate_checkpoint writes exactly what the selective policy chooses (not the old affine fit)."""
    import copy

    from ultralytics.models.yolo.depth.calibrate import calibrate_checkpoint

    torch.manual_seed(0)
    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    batches = [
        {"img": (torch.rand(2, 3, 64, 64) * 255).to(torch.uint8), "depth": torch.rand(2, 64, 64) * 5 + 0.5}
        for _ in range(4)
    ]
    expected = fit_calibration_selective(copy.deepcopy(model), batches, device="cpu")

    path = tmp_path / "ckpt.pt"
    torch.save({"model": copy.deepcopy(model)}, path)
    calibrate_checkpoint(path, batches, device="cpu")

    head = _depth_head(torch.load(path, weights_only=False)["model"])
    assert abs(float(head.cal_a) - expected["a"]) < 1e-6 and abs(float(head.cal_b) - expected["b"]) < 1e-6


def test_fit_calibration_selective_runs_and_sets_buffers():
    """The model-level selective fit returns a chosen candidate and writes finite buffers."""
    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    batches = [
        {"img": (torch.rand(2, 3, 64, 64) * 255).to(torch.uint8), "depth": torch.rand(2, 64, 64) * 5 + 0.5}
        for _ in range(4)
    ]
    res = fit_calibration_selective(model, batches, device="cpu", max_images=8)
    assert res is not None
    assert res["name"] in {"identity", "scale-only", "affine"}
    head = model.model[-1]
    assert math.isfinite(float(head.cal_a)) and math.isfinite(float(head.cal_b))
    assert abs(float(head.cal_a) - res["a"]) < 1e-4 and abs(float(head.cal_b) - res["b"]) < 1e-4
