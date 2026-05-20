"""Tests for Dice and Boundary-IoU metric helpers and SegmentMetrics integration.

Covers:
  1. mask_dice / mask_biou degenerate cases (identical masks → 1, disjoint masks → 0).
  2. mask_boundary_rim shape and emptiness rules.
  3. SegmentMetrics aggregates per-class Dice / BIoU from matched-pair stats.
  4. SegmentMetrics.keys / mean_results / class_result widen consistently.
  5. fitness_weight length 10 routes the last two weights into Dice / BIoU contributions.
  6. End-to-end: SegmentationValidator._matched_mask_quality picks the correct pairs and
     reports Dice == 1.0 / BIoU == 1.0 when pred == GT.

Run all:
    pytest tests/test_boundary_metrics.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics.utils.metrics import (
    SegmentMetrics,
    mask_biou,
    mask_boundary_rim,
    mask_dice,
)


def _circle(size: int = 64, cy: int = 32, cx: int = 32, r: int = 16) -> torch.Tensor:
    """Return a (1, H, W) binary mask with a filled circle."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).float().unsqueeze(0)


# ─────────────────────────────── helpers: mask_dice / mask_biou ───────────────────────────────


def test_mask_dice_identical_returns_one():
    """Dice between identical masks is 1.0 (within float tolerance)."""
    g = _circle().flatten(1)
    out = mask_dice(g, g)
    assert out.shape == (1, 1)
    assert pytest.approx(out.item(), abs=1e-5) == 1.0


def test_mask_dice_disjoint_returns_zero():
    """Dice between disjoint non-empty masks is 0 (intersection = 0, denom > 0)."""
    a = _circle(cy=16, cx=16, r=8).flatten(1)
    b = _circle(cy=48, cx=48, r=8).flatten(1)
    assert mask_dice(a, b).item() == pytest.approx(0.0, abs=1e-6)


def test_mask_dice_matches_iou_relationship():
    """Sanity check: Dice = 2*IoU / (1+IoU) holds for partially overlapping masks."""
    a = _circle(cy=32, cx=28, r=14).flatten(1)
    b = _circle(cy=32, cx=36, r=14).flatten(1)
    inter = (a * b).sum().item()
    union = ((a + b) > 0).float().sum().item()
    iou = inter / union
    expected = 2 * iou / (1 + iou)
    assert mask_dice(a, b).item() == pytest.approx(expected, abs=1e-5)


def test_mask_boundary_rim_empty_input():
    """Empty masks produce empty rims."""
    g = torch.zeros(1, 16, 16)
    assert mask_boundary_rim(g, kernel=3).sum().item() == 0.0


def test_mask_boundary_rim_thickness_grows_with_kernel():
    """Larger kernel ⇒ thicker rim (more 1-pixels), at least until rim swallows the whole mask."""
    g = _circle(size=80, r=24)
    rims = [mask_boundary_rim(g, k).sum().item() for k in (3, 5, 9)]
    assert rims[0] < rims[1] < rims[2]


def test_mask_biou_identical_masks_is_one():
    """BIoU between identical non-trivial masks is exactly 1.0."""
    g = _circle()
    assert mask_biou(g, g, kernel=3).item() == pytest.approx(1.0, abs=1e-6)


def test_mask_biou_disjoint_masks_is_zero():
    """Disjoint masks ⇒ rims disjoint ⇒ BIoU = 0."""
    a = _circle(cy=16, cx=16, r=8)
    b = _circle(cy=48, cx=48, r=8)
    assert mask_biou(a, b, kernel=3).item() == pytest.approx(0.0, abs=1e-6)


def test_mask_biou_more_sensitive_than_iou_to_shift():
    """Small offset should drop BIoU faster than mask IoU (the whole point of the metric)."""
    from ultralytics.utils.metrics import mask_iou

    g = _circle(cy=32, cx=32, r=20)
    p = _circle(cy=32, cx=34, r=20)  # 2-pixel shift
    iou_val = mask_iou(g.flatten(1), p.flatten(1)).item()
    biou_val = mask_biou(g, p, kernel=3).item()
    assert biou_val < iou_val, f"Expected BIoU ({biou_val:.3f}) < mask IoU ({iou_val:.3f}) under small shift"


# ───────────────────────────── SegmentMetrics integration ────────────────────────────────────


def _build_segment_metrics(fitness_weight=None, nc=2, boundary_kernel=3):
    names = {i: f"c{i}" for i in range(nc)}
    return SegmentMetrics(names=names, fitness_weight=fitness_weight, boundary_kernel=boundary_kernel)


def test_segment_metrics_keys_include_dice_biou():
    """`keys` widens by exactly two entries for Dice and BIoU."""
    sm = _build_segment_metrics()
    assert "metrics/dice(M)" in sm.keys
    assert "metrics/biou(M)" in sm.keys
    # 4 box + 4 mask + 2 boundary = 10
    assert len(sm.keys) == 10


def test_segment_metrics_mean_results_widens_to_match_keys():
    """`mean_results()` length must match `len(keys)` so print_results / save_metrics line up."""
    sm = _build_segment_metrics()
    assert len(sm.mean_results()) == len(sm.keys)


def test_segment_metrics_process_aggregates_per_class():
    """Per-pair scalars get bucketed by class id and averaged on `process()`.

    We seed `self.stats` with synthetic per-batch arrays for both the base detection stats
    (so `DetMetrics.process` doesn't crash on empty concatenation) and the new boundary stats,
    then assert the per-class / mean aggregation matches by-hand averages.
    """
    sm = _build_segment_metrics(nc=2)
    # Base detection stats: three predictions across two classes, all TP at IoU 0.5+.
    niou = 10  # match _process_batch tp width
    tp_arr = np.ones((3, niou), dtype=bool)
    sm.stats["tp"].extend([tp_arr])
    sm.stats["tp_m"].extend([tp_arr])
    sm.stats["conf"].extend([np.array([0.9, 0.8, 0.7], dtype=np.float32)])
    sm.stats["pred_cls"].extend([np.array([0, 0, 1], dtype=np.float32)])
    sm.stats["target_cls"].extend([np.array([0, 0, 1], dtype=np.float32)])
    sm.stats["target_img"].extend([np.array([0, 0, 1], dtype=np.float32)])
    # Boundary stats: two matched pairs for class 0, one for class 1.
    sm.stats["mask_dice"].extend([np.array([0.8, 0.6], dtype=np.float32), np.array([0.4], dtype=np.float32)])
    sm.stats["mask_biou"].extend([np.array([0.5, 0.3], dtype=np.float32), np.array([0.9], dtype=np.float32)])
    sm.stats["matched_cls"].extend([np.array([0, 0], dtype=np.int64), np.array([1], dtype=np.int64)])

    sm.process(save_dir=Path("."), plot=False)

    assert sm.dice_per_class[0] == pytest.approx(0.7)  # mean(0.8, 0.6)
    assert sm.dice_per_class[1] == pytest.approx(0.4)
    assert sm.biou_per_class[0] == pytest.approx(0.4)  # mean(0.5, 0.3)
    assert sm.biou_per_class[1] == pytest.approx(0.9)
    assert sm.dice_mean == pytest.approx((0.7 + 0.4) / 2)
    assert sm.biou_mean == pytest.approx((0.4 + 0.9) / 2)
    # Sanity: keys / mean_results lengths still aligned post-process.
    assert len(sm.mean_results()) == len(sm.keys)


def test_segment_metrics_fitness_weight_length_10_adds_boundary_terms():
    """fitness_weight = [box×4, mask×4, dice, biou] adds dice/biou contribution to fitness()."""
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0]
    sm = _build_segment_metrics(fitness_weight=weights)
    assert sm.dice_fitness_weight == 2.0
    assert sm.biou_fitness_weight == 3.0
    sm.dice_mean = 0.5
    sm.biou_mean = 0.25
    # Box & mask AP metrics are empty/zero so their fitness contributions are 0.
    assert sm.fitness == pytest.approx(2.0 * 0.5 + 3.0 * 0.25)


def test_segment_metrics_fitness_weight_length_8_skips_boundary_terms():
    """Backward compat: 8-value weights ⇒ dice/biou weights stay at 0 and don't affect fitness."""
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sm = _build_segment_metrics(fitness_weight=weights)
    assert sm.dice_fitness_weight == 0.0
    assert sm.biou_fitness_weight == 0.0
    sm.dice_mean = 0.5
    sm.biou_mean = 0.25
    # Boundary terms must not contribute under the 8-value layout.
    assert sm.fitness == 0.0


# ───────────────────────── SegmentationValidator._matched_mask_quality ───────────────────────


def _make_validator_with_metrics(boundary_kernel=3):
    """Construct a minimal SegmentationValidator-like object exposing only what the helper needs.

    We avoid the full constructor (which requires a dataloader / save_dir) and instead bind the
    SegmentMetrics instance + the helper method onto a SimpleNamespace.
    """
    from ultralytics.models.yolo.segment.val import SegmentationValidator

    metrics = _build_segment_metrics(boundary_kernel=boundary_kernel)
    # SimpleNamespace gives us attribute access; methods need an explicit self via __get__.
    ns = SimpleNamespace(metrics=metrics)
    ns._matched_mask_quality = SegmentationValidator._matched_mask_quality.__get__(ns)
    return ns


def test_matched_mask_quality_identical_masks_score_one():
    """When pred == GT and classes agree, Dice and BIoU are 1.0 for every matched pair."""
    val = _make_validator_with_metrics()
    gt = torch.stack([_circle(cy=20, cx=20, r=8).squeeze(0), _circle(cy=44, cx=44, r=10).squeeze(0)])
    pred = gt.clone()
    gt_cls = torch.tensor([0, 1])
    pred_cls = torch.tensor([0, 1])
    iou = torch.eye(2)  # perfect IoU diagonal so both pairs match at 0.5
    dice, biou, cls = val._matched_mask_quality(gt, pred, gt_cls, pred_cls, iou)
    assert dice.shape == (2,)
    assert np.allclose(dice, 1.0, atol=1e-5)
    assert np.allclose(biou, 1.0, atol=1e-5)
    assert set(cls.tolist()) == {0, 1}


def test_matched_mask_quality_class_mismatch_yields_no_pairs():
    """Class disagreement zeros the IoU matrix ⇒ no pairs reported."""
    val = _make_validator_with_metrics()
    gt = _circle().repeat(1, 1, 1)
    pred = gt.clone()
    gt_cls = torch.tensor([0])
    pred_cls = torch.tensor([1])
    iou = torch.tensor([[1.0]])
    dice, biou, cls = val._matched_mask_quality(gt, pred, gt_cls, pred_cls, iou)
    assert dice.size == 0
    assert biou.size == 0
    assert cls.size == 0


def test_matched_mask_quality_below_threshold_no_pairs():
    """IoU strictly below 0.5 ⇒ no match, no scores."""
    val = _make_validator_with_metrics()
    gt = _circle().repeat(1, 1, 1)
    pred = gt.clone()
    gt_cls = torch.tensor([0])
    pred_cls = torch.tensor([0])
    iou = torch.tensor([[0.3]])  # below default 0.5
    dice, biou, cls = val._matched_mask_quality(gt, pred, gt_cls, pred_cls, iou)
    assert dice.size == 0
    assert biou.size == 0
    assert cls.size == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
