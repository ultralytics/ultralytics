# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ultralytics.utils.geometry3d import paired_boxes3d_iou_torch
from ultralytics.utils.kitti_eval import build_kitti_predictions, paired_d3_box_overlap


def _kitti_annotation(centers: np.ndarray, dims: np.ndarray, yaws: np.ndarray):
    """Build bottom-centered KITTI annotations from geometric-center boxes."""
    locations = centers.copy()
    locations[:, 1] += dims[:, 0] * 0.5
    count = len(centers)
    return build_kitti_predictions(
        ["Car"] * count,
        np.zeros((count, 4)),
        np.ones(count),
        np.zeros(count),
        dims,
        locations,
        yaws,
    )


def test_paired_boxes3d_iou_known_geometries():
    centers_a = torch.tensor([[0.0, 0.0, 20.0]] * 7)
    dims_a = torch.tensor([[2.0, 2.0, 4.0]] * 5 + [[4.0, 4.0, 8.0], [1.5, 1.6, 4.0]])
    yaws_a = torch.zeros(7)
    centers_b = centers_a.clone()
    centers_b[2, 0] = 10.0  # Separated in BEV.
    centers_b[3, 0] = 4.0  # Footprints touch along one edge.
    centers_b[4, 1] = 3.0  # Footprints overlap but vertical intervals do not.
    dims_b = dims_a.clone()
    dims_b[5] = torch.tensor([2.0, 2.0, 4.0])  # Fully contained, one eighth of the volume.
    yaws_b = torch.tensor([0.0, math.pi, 0.0, 0.0, 0.0, 0.0, math.pi / 2])

    result = paired_boxes3d_iou_torch(centers_a, dims_a, yaws_a, centers_b, dims_b, yaws_b)

    torch.testing.assert_close(result, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.125, 0.25]), atol=2e-6, rtol=0)


def test_paired_boxes3d_iou_matches_opencv_kitti_evaluator():
    rng = np.random.default_rng(20260731)
    count = 512
    centers_a = np.column_stack(
        (
            rng.uniform(-8.0, 8.0, count),
            rng.uniform(-1.0, 2.0, count),
            rng.uniform(5.0, 80.0, count),
        )
    )
    dims_a = np.column_stack(
        (
            rng.uniform(0.5, 3.0, count),
            rng.uniform(0.5, 3.0, count),
            rng.uniform(0.5, 6.0, count),
        )
    )
    yaws_a = rng.uniform(-math.pi, math.pi, count)
    centers_b = centers_a + rng.normal(0.0, (2.0, 0.8, 2.0), (count, 3))
    dims_b = dims_a * rng.uniform(0.6, 1.4, (count, 3))
    yaws_b = yaws_a + rng.normal(0.0, 1.0, count)

    expected = paired_d3_box_overlap(
        _kitti_annotation(centers_a, dims_a, yaws_a),
        _kitti_annotation(centers_b, dims_b, yaws_b),
    )
    result = paired_boxes3d_iou_torch(
        torch.from_numpy(centers_a),
        torch.from_numpy(dims_a),
        torch.from_numpy(yaws_a),
        torch.from_numpy(centers_b),
        torch.from_numpy(dims_b),
        torch.from_numpy(yaws_b),
    )

    np.testing.assert_allclose(result.numpy(), expected, atol=2e-5, rtol=2e-4)


def test_paired_boxes3d_iou_handles_batched_shapes_and_invalid_boxes():
    centers = torch.zeros((2, 3, 3))
    dims = torch.ones((2, 3, 3))
    dims[0, 0, 0] = 0.0
    dims[0, 1, 1] = -1.0
    dims[0, 2, 2] = torch.nan
    dims[1, 0, 0] = torch.inf
    centers[1, 1, 2] = torch.nan
    yaws = torch.zeros((2, 3, 1))
    yaws[1, 2, 0] = torch.inf

    result = paired_boxes3d_iou_torch(centers, dims, yaws, torch.zeros_like(centers), torch.ones_like(dims), yaws)

    assert result.shape == (2, 3)
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.zeros_like(result))


def test_paired_boxes3d_iou_stays_fp32_under_cpu_autocast():
    centers = torch.tensor([[40.0, 1.0, 103.6], [0.0, 0.0, 20.0]])
    dims = torch.tensor([[1.5, 1.6, 4.0], [2.0, 2.0, 4.0]])
    yaws = torch.tensor([[0.25], [-0.7]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = paired_boxes3d_iou_torch(centers, dims, yaws, centers, dims, yaws + math.pi)

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.ones(2), atol=5e-5, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_paired_boxes3d_iou_uses_cuda_under_amp():
    centers = torch.tensor([[0.0, 0.0, 20.0]], device="cuda", dtype=torch.float16)
    dims = torch.tensor([[1.5, 1.6, 4.0]], device="cuda", dtype=torch.float16)
    yaws = torch.tensor([0.0], device="cuda", dtype=torch.float16)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = paired_boxes3d_iou_torch(centers, dims, yaws, centers, dims, yaws + math.pi / 2)

    assert result.is_cuda and result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([0.25], device="cuda"), atol=2e-4, rtol=0)
