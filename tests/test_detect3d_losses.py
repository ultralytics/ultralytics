# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Focused tests for the final Mono3D quality and depth objectives."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.nn.tasks import Detection3DModel
from ultralytics.utils.loss import (
    E2ELoss,
    quality_focal_loss_with_logits,
    v8Detection3DLoss,
)


class _SingleTargetAssigner:
    """Assign the only prediction to the only target for deterministic 3D loss tests."""

    def __call__(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        del pd_bboxes, anchor_points, mask_gt
        batch_size, num_anchors, num_classes = pd_scores.shape
        target_labels = gt_labels[:, :1].expand(batch_size, num_anchors, 1)
        target_bboxes = gt_bboxes[:, :1].expand(batch_size, num_anchors, 4)
        target_scores = pd_scores.new_zeros((batch_size, num_anchors, num_classes))
        target_scores[..., 0] = 1.0
        fg_mask = torch.ones((batch_size, num_anchors), dtype=torch.bool, device=pd_scores.device)
        target_gt_idx = torch.zeros((batch_size, num_anchors), dtype=torch.long, device=pd_scores.device)
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx


def _synthetic_batch(image: torch.Tensor | None = None) -> dict:
    """Return one finite KITTI-style target and its augmented projection matrix."""
    p2 = np.array([[4.0, 0.0, 4.0, 0.0], [0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    batch = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25, 10.0, 0.0, 0.75, 1.6, 1.5, 3.9, 0.0]]),
        "d3_valid": torch.tensor([[True]]),
        "p2s_aug": [p2],
    }
    if image is not None:
        batch["img"] = image
    return batch


def _controlled_loss(
    overrides: dict | None = None,
    predicted_depth: float = 12.0,
    *,
    nonzero_geometry: bool = False,
):
    """Compute deterministic loss items for one prediction."""
    model = Detection3DModel("yolo11n-3d.yaml", ch=3, nc=1, verbose=False)
    model.args = get_cfg(overrides=overrides or {})
    criterion = v8Detection3DLoss(model)
    criterion.assigner = _SingleTargetAssigner()
    criterion.bbox_loss = lambda pred_distri, *args: (
        pred_distri.sum() * 0.0,
        pred_distri.sum() * 0.0,
    )

    stride = float(criterion.stride[0])
    predicted_box = (torch.tensor([3.0, 3.0, 5.0, 5.0]) / stride).reshape(1, 1, 4)
    criterion.bbox_decode = lambda anchor_points, pred_dist: predicted_box

    head = model.model[-1]
    raw = torch.zeros((1, head.nr, 1))
    raw[:, 2] = math.log(predicted_depth)
    if nonzero_geometry:
        raw[:, 0:2] = torch.tensor([0.2, -0.1]).reshape(1, 2, 1)
        raw[:, 3:6] = torch.tensor([0.1, -0.1, 0.05]).reshape(1, 3, 1)
    raw = raw.requires_grad_()
    predictions = {
        "boxes": torch.zeros((1, 64, 1), requires_grad=True),
        "scores": torch.zeros((1, 1, 1), requires_grad=True),
        "feats": [torch.zeros((1, 1, 1, 1))],
        "d3_params": raw,
    }
    if nonzero_geometry:
        predictions["d3_aux"] = torch.full((1, head.aux_channels, 1), 0.1, requires_grad=True)
    loss = criterion.get_assigned_targets_and_loss(predictions, _synthetic_batch())[1]
    return loss, raw, criterion


def test_release_defaults_use_iou_quality_and_bounded_direct_depth_behavior():
    assert DEFAULT_CFG.depth_z == 0.1
    assert DEFAULT_CFG.depth_z_tau == 2.0
    assert DEFAULT_CFG.quality3d_power == 0.5
    assert DEFAULT_CFG.d3_geometry_gain == 2.0
    assert DEFAULT_CFG.kitti_eval == "off"

    default_loss, _, default_criterion = _controlled_loss()
    explicit_loss, _, explicit_criterion = _controlled_loss({"depth_z": 0.1, "depth_z_tau": 2.0})

    assert default_criterion.depth_z == explicit_criterion.depth_z == pytest.approx(0.1)
    torch.testing.assert_close(default_loss, explicit_loss)


def test_geometry_gain_scales_geometry_only_without_changing_2d_alpha_or_quality():
    baseline, _, baseline_criterion = _controlled_loss({"d3_geometry_gain": 1.0}, nonzero_geometry=True)
    scaled, _, scaled_criterion = _controlled_loss({"d3_geometry_gain": 2.0}, nonzero_geometry=True)

    active_geometry_indices = torch.tensor([3, 4, 6, 7, 8])
    unchanged_indices = torch.tensor([0, 1, 2, 5, 9])
    assert baseline_criterion.d3_geometry_gain == pytest.approx(1.0)
    assert scaled_criterion.d3_geometry_gain == pytest.approx(2.0)
    assert torch.all(baseline[active_geometry_indices] > 0.0)
    torch.testing.assert_close(scaled[active_geometry_indices], baseline[active_geometry_indices] * 2.0)
    torch.testing.assert_close(scaled[unchanged_indices], baseline[unchanged_indices])


def test_quality3d_uses_exact_paired_iou_target():
    def fixed_iou(centers_a, dims_a, yaws_a, centers_b, dims_b, yaws_b):
        del dims_a, yaws_a, centers_b, dims_b, yaws_b
        return centers_a.new_full(centers_a.shape[:-1], 0.37)

    with patch("ultralytics.utils.loss.paired_boxes3d_iou_torch", side_effect=fixed_iou) as iou_mock:
        iou_loss, _, _ = _controlled_loss()

    expected = quality_focal_loss_with_logits(torch.tensor([[0.0]]), torch.tensor([[0.37]])).item()
    assert iou_mock.call_count == 1
    assert iou_loss[9].item() == pytest.approx(expected, rel=1e-6)


def test_bounded_absolute_z_term_is_positive_finite_and_backpropagates_without_a_new_slot():
    baseline, _, _ = _controlled_loss({"depth_z": 0.0}, predicted_depth=12.0)
    augmented, raw, criterion = _controlled_loss({"depth_z": 0.2, "depth_z_tau": 2.0}, predicted_depth=12.0)

    assert criterion.depth_z == pytest.approx(0.2)
    assert augmented.numel() == baseline.numel() == 10
    assert torch.isfinite(augmented).all()
    assert augmented[4] > baseline[4]
    augmented[4].backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad[0, 2, 0].abs() > 0.0


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"depth_z": -0.1}, "depth_z"),
        ({"depth_z": float("inf")}, "depth_z"),
        ({"depth_z": float("nan")}, "depth_z"),
        ({"depth_z_tau": 0.0}, "depth_z_tau"),
        ({"depth_z_tau": -1.0}, "depth_z_tau"),
        ({"depth_z_tau": float("inf")}, "depth_z_tau"),
        ({"depth_z_tau": float("nan")}, "depth_z_tau"),
        ({"d3_geometry_gain": 0.0}, "d3_geometry_gain"),
        ({"d3_geometry_gain": -1.0}, "d3_geometry_gain"),
        ({"d3_geometry_gain": float("inf")}, "d3_geometry_gain"),
        ({"d3_geometry_gain": float("nan")}, "d3_geometry_gain"),
    ),
)
def test_detect3d_loss_configuration_rejects_invalid_values(override: dict, message: str):
    model = Detection3DModel("yolo11n-3d.yaml", ch=3, nc=1, verbose=False)
    model.args = get_cfg(overrides=override)
    with pytest.raises(ValueError, match=message):
        v8Detection3DLoss(model)


@pytest.mark.parametrize("config", ("yolo11n-3d.yaml", "yolo26n-3d.yaml"))
def test_release_losses_support_yolo11_and_yolo26_end_to_end_backward(config: str):
    torch.manual_seed(0)
    model = Detection3DModel(config, ch=3, nc=1, verbose=False)
    model.args = get_cfg(overrides={"depth_z": 0.1, "depth_z_tau": 2.0})
    image = torch.randn(1, 3, 64, 64)
    model.train()

    predictions = model.predict(image)
    loss, loss_items = model.loss(_synthetic_batch(image), predictions)

    assert torch.isfinite(loss).all() and all(torch.isfinite(item) for item in loss_items.values())
    assert "depth_loss" in loss_items and "quality3d_loss" in loss_items
    loss.sum().backward()
    head = model.model[-1]
    assert torch.count_nonzero(head.cv4[0].primary.weight.grad[2]) > 0
    assert torch.count_nonzero(head.cv4[0].primary.weight.grad[6]) > 0
    if head.end2end:
        assert isinstance(model.criterion, E2ELoss)
        assert torch.count_nonzero(head.one2one_cv4[0].primary.weight.grad[2]) > 0
        assert torch.count_nonzero(head.one2one_cv4[0].primary.weight.grad[6]) > 0
