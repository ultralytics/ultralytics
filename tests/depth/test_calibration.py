"""Unit tests for scale-only depth calibration (cal_a/cal_b log-affine)."""

import math

import numpy as np
import torch

from ultralytics.models.yolo.depth.calibrate import _depth_head, fit_calibration, lstsq_affine
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
