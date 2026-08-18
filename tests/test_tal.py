# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math

import pytest
import torch

from ultralytics.utils.tal import bbox2dist, dist2bbox, dist2rbox, make_anchors, rbox2dist


def test_make_anchors_single_level():
    """Anchor points are pixel centers of the feature grid in (x, y) order."""
    anchors, strides = make_anchors([torch.zeros(1, 1, 2, 2)], [8])
    assert torch.equal(anchors, torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]))
    assert torch.equal(strides, torch.full((4, 1), 8.0))


def test_make_anchors_non_square_grid():
    """A (h=1, w=3) grid varies x fastest and keeps h/w unswapped."""
    anchors, strides = make_anchors([torch.zeros(1, 1, 1, 3)], [8])
    assert torch.equal(anchors, torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]]))
    assert strides.shape == (3, 1)


def test_make_anchors_multi_level():
    """Levels are concatenated in order and each carries its own stride."""
    feats = [torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 1)]
    anchors, strides = make_anchors(feats, torch.tensor([8.0, 16.0]))
    assert anchors.shape == (5, 2)
    assert torch.equal(strides.flatten(), torch.tensor([8.0, 8.0, 8.0, 8.0, 16.0]))


def test_make_anchors_grid_cell_offset():
    """grid_cell_offset shifts every anchor point uniformly."""
    anchors, _ = make_anchors([torch.zeros(1, 1, 2, 2)], [4], grid_cell_offset=0.0)
    assert torch.equal(anchors, torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_make_anchors_preserves_dtype(dtype):
    """Anchors and strides inherit the feature map dtype."""
    anchors, strides = make_anchors([torch.zeros(1, 1, 2, 2, dtype=dtype)], [8])
    assert anchors.dtype is dtype and strides.dtype is dtype


def test_dist2bbox_xyxy():
    """Ltrb distances decode to xyxy by subtracting lt and adding rb."""
    out = dist2bbox(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([[10.0, 10.0]]), xywh=False)
    assert torch.equal(out, torch.tensor([[9.0, 8.0, 13.0, 14.0]]))


def test_dist2bbox_xywh():
    """The xywh form is the center and size of the same decoded box."""
    out = dist2bbox(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([[10.0, 10.0]]), xywh=True)
    assert torch.equal(out, torch.tensor([[11.0, 11.0, 4.0, 6.0]]))


def test_dist2bbox_custom_dim():
    """Splitting along a non-default dim decodes each channel pair independently."""
    dist = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # (1, 4, 1)
    anchors = torch.tensor([[[10.0], [10.0]]])  # (1, 2, 1)
    out = dist2bbox(dist, anchors, xywh=False, dim=1)
    assert torch.equal(out, torch.tensor([[[9.0], [8.0], [13.0], [14.0]]]))


def test_bbox2dist_inverts_dist2bbox():
    """Bbox2dist is the exact inverse of dist2bbox(xywh=False)."""
    torch.manual_seed(0)
    anchors = torch.rand(16, 2) * 100
    dist = torch.rand(16, 4) * 10
    bbox = dist2bbox(dist, anchors, xywh=False)
    assert torch.allclose(bbox2dist(anchors, bbox), dist, atol=1e-5)


def test_bbox2dist_clamps_to_reg_max():
    """reg_max clamps distances into [0, reg_max - 0.01]."""
    anchors = torch.zeros(1, 2)
    bbox = torch.tensor([[5.0, 5.0, 100.0, 100.0]])  # negative lt, oversized rb
    dist = bbox2dist(anchors, bbox, reg_max=16)
    assert torch.equal(dist, torch.tensor([[0.0, 0.0, 15.99, 15.99]]))


def test_bbox2dist_does_not_mutate_inputs():
    """The in-place clamp must not leak into the caller's tensors."""
    anchors = torch.zeros(1, 2)
    bbox = torch.tensor([[5.0, 5.0, 100.0, 100.0]])
    bbox_before = bbox.clone()
    bbox2dist(anchors, bbox, reg_max=16)
    assert torch.equal(bbox, bbox_before)
    assert torch.equal(anchors, torch.zeros(1, 2))


def test_dist2rbox_zero_angle_matches_dist2bbox():
    """At angle 0 a rotated box reduces to the axis-aligned xywh box."""
    torch.manual_seed(0)
    dist = torch.rand(1, 8, 4) * 10
    anchors = torch.rand(8, 2) * 50
    rotated = dist2rbox(dist, torch.zeros(1, 8, 1), anchors)
    assert torch.allclose(rotated, dist2bbox(dist, anchors, xywh=True), atol=1e-5)


def test_dist2rbox_quarter_turn():
    """A +90 degree angle maps the box offset from +x onto +y."""
    dist = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    angle = torch.full((1, 1, 1), math.pi / 2)
    out = dist2rbox(dist, angle, torch.tensor([[10.0, 10.0]]))
    assert torch.allclose(out, torch.tensor([[[9.0, 11.0, 4.0, 6.0]]]), atol=1e-5)


def test_rbox2dist_inverts_dist2rbox():
    """Rbox2dist recovers the ltrb distances produced by dist2rbox."""
    torch.manual_seed(0)
    dist = torch.rand(2, 16, 4) * 10
    angle = (torch.rand(2, 16, 1) - 0.5) * math.pi
    anchors = torch.rand(16, 2) * 50
    rbox = dist2rbox(dist, angle, anchors)
    assert torch.allclose(rbox2dist(rbox, anchors, angle), dist, atol=1e-4)


def test_rbox2dist_clamps_to_reg_max():
    """reg_max bounds the rotated distances the same way as bbox2dist."""
    rbox = torch.tensor([[[0.0, 0.0, 100.0, 100.0]]])
    dist = rbox2dist(rbox, torch.zeros(1, 2), torch.zeros(1, 1, 1), reg_max=16)
    assert torch.equal(dist, torch.full((1, 1, 4), 15.99))
