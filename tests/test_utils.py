# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the per-image property analysis module."""

import cv2
import numpy as np
import pytest

from ultralytics import YOLO
from ultralytics.utils.analysis import ImagePropertyAnalyzer, _softmin1d, compute_objectlab_scores
from ultralytics.utils.ops import xywhn2xyxy


@pytest.fixture
def synthetic_gray():
    """Return a deterministic 320x480 grayscale image with mid-range intensity and structure."""
    rng = np.random.default_rng(42)
    return rng.integers(20, 200, size=(320, 480), dtype=np.uint8)


@pytest.fixture
def synthetic_bgr(synthetic_gray):
    """Return a 320x480 BGR image derived from the synthetic gray fixture."""
    return cv2.cvtColor(synthetic_gray, cv2.COLOR_GRAY2BGR)


def test_brightness_in_unit_interval(synthetic_bgr):
    """Brightness is HSP perceptual [0,1] (Finley 2006)."""
    v = ImagePropertyAnalyzer._brightness(synthetic_bgr)
    assert 0.0 <= v <= 1.0


def test_blurriness_inverse_relationship():
    """A sharp checkerboard returns a lower blurriness than a uniform image (Pech-Pacheco 2000)."""
    sharp = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (160, 240))
    flat = np.full((320, 480), 128, dtype=np.uint8)
    assert ImagePropertyAnalyzer._blurriness(sharp) < ImagePropertyAnalyzer._blurriness(flat)


def test_contrast_zero_for_flat_image():
    """A constant-intensity image has zero contrast."""
    flat = np.full((320, 480), 128, dtype=np.uint8)
    assert ImagePropertyAnalyzer._contrast(flat) == 0.0


def test_dark_and_bright_pixel_ratios():
    """Hand-constructed image has known dark-pixel and bright-pixel fractions."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:5] = 5  # half the image is dark
    img[5:] = 250  # other half bright
    assert ImagePropertyAnalyzer._dark_pixel_ratio(img) == pytest.approx(0.5)
    assert ImagePropertyAnalyzer._bright_pixel_ratio(img) == pytest.approx(0.5)


def test_entropy_uniform_higher_than_constant():
    """Uniform random has higher Shannon entropy than constant image (Shannon 1948)."""
    uniform = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
    flat = np.full((64, 64), 128, dtype=np.uint8)
    assert ImagePropertyAnalyzer._entropy(uniform) > ImagePropertyAnalyzer._entropy(flat)
    assert ImagePropertyAnalyzer._entropy(flat) == 0.0


def test_edge_density_higher_for_checkerboard():
    """Checkerboard has higher Canny edge density than flat (Canny 1986)."""
    sharp = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (160, 240))
    flat = np.full((320, 480), 128, dtype=np.uint8)
    assert ImagePropertyAnalyzer._edge_density(sharp) > ImagePropertyAnalyzer._edge_density(flat)


def test_sharpness_higher_for_step_edge():
    """A step edge has higher Tenengrad sharpness than a flat image (Krotkov 1988)."""
    edge = np.zeros((320, 480), dtype=np.uint8)
    edge[:, 240:] = 255  # vertical step edge in the middle
    flat = np.full((320, 480), 128, dtype=np.uint8)
    assert ImagePropertyAnalyzer._sharpness(edge) > ImagePropertyAnalyzer._sharpness(flat)


def test_pairwise_iou_stats_three_box_example():
    """Two overlapping boxes yield max IoU in (0.1, 0.5), a third disjoint box doesn't increase it."""
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]], dtype=np.float32)
    max_iou, _ = ImagePropertyAnalyzer._pairwise_iou_stats(boxes)
    assert 0.1 < max_iou < 0.5


def test_pairwise_iou_stats_zero_for_disjoint_boxes():
    """Three pairwise-disjoint boxes have mean IoU == 0."""
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]], dtype=np.float32)
    _, mean_iou = ImagePropertyAnalyzer._pairwise_iou_stats(boxes)
    assert mean_iou == 0.0


def test_xywhn_to_xyxy_pixels_round_trip():
    """Xywh-normalized boxes round-trip cleanly to xyxy pixels."""
    xywhn = np.array([[0.5, 0.5, 0.4, 0.2]])
    xyxy = xywhn2xyxy(xywhn, w=100, h=200)
    np.testing.assert_allclose(xyxy[0], [30, 80, 70, 120])


def test_softmin1d_bounded_by_min_max():
    """Softmin1d returns a value bounded by min and max of input scores."""
    scores = np.array([0.2, 0.5, 0.8])
    s = _softmin1d(scores, T=0.1)
    assert scores.min() <= s <= scores.max()
    assert s == pytest.approx(scores.min(), abs=0.05)  # low T -> near min


def test_softmin1d_uniform_returns_constant():
    """Equal scores collapse to the shared value regardless of T."""
    s = _softmin1d(np.array([0.7, 0.7, 0.7]), T=0.1)
    assert s == pytest.approx(0.7)


def test_objectlab_clean_image_returns_high_quality():
    """An image where every GT has a matched same-class high-conf prediction returns near-1.0 quality."""
    out = compute_objectlab_scores(
        iou=np.array([[1.0, 0.0], [0.0, 1.0]]),
        pred_bb=np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
        pred_cls=np.array([0, 1]),
        pred_conf=np.array([0.99, 0.99]),
        gt_bb=np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
        gt_cls=np.array([0, 1]),
    )
    assert out["overlooked_score"] == pytest.approx(1.0, abs=0.05)
    assert out["badloc_score"] == pytest.approx(1.0, abs=0.05)
    assert out["swap_score"] == pytest.approx(1.0, abs=0.05)
    assert out["label_quality_score"] == pytest.approx(1.0, abs=0.05)


def test_objectlab_swap_drops_quality():
    """A high-confidence different-class prediction overlapping a GT triggers a low swap score."""
    out = compute_objectlab_scores(
        iou=np.array([[1.0]]),
        pred_bb=np.array([[0, 0, 10, 10]], dtype=np.float32),
        pred_cls=np.array([1]),
        pred_conf=np.array([0.99]),
        gt_bb=np.array([[0, 0, 10, 10]], dtype=np.float32),
        gt_cls=np.array([0]),
    )
    assert out["swap_score"] < 0.1, f"expected low swap score, got {out['swap_score']}"


def test_run_dataset_only_path(tmp_path):
    """Dataset-only path produces a CSV, JSON, and summary on COCO128 with no model."""
    out_dir = tmp_path / "ds_only"
    report = ImagePropertyAnalyzer(data="coco128.yaml", save_dir=out_dir).run()
    assert report.has_predictions is False
    assert len(report.per_image) > 0
    for fname in ("per_image_analysis.csv", "correlations.json", "worst_images.json", "summary.md"):
        p = out_dir / fname
        assert p.exists() and p.stat().st_size > 0
    # No prediction-quality columns when there is no model
    sample = next(iter(report.per_image.values()))
    assert sample.get("f1") is None
    assert np.isnan(sample.get("overlooked_score"))


def test_run_from_metrics_path(tmp_path):
    """From_metrics reuses an existing model.val() result and produces full ObjectLab output."""
    model = YOLO("yolo11n.pt")
    metrics = model.val(
        data="coco128.yaml",
        batch=8,
        plots=False,
        save_json=False,
        imgsz=320,
        verbose=False,
        score_labels=True,
        device="cpu",
    )
    out_dir = tmp_path / "from_metrics"
    report = ImagePropertyAnalyzer.from_metrics(
        metrics, dataset=model.validator.dataloader.dataset, save_dir=out_dir
    ).run()
    assert report.has_predictions is True
    sample = next(iter(report.per_image.values()))
    assert sample.get("f1") is not None
    assert "overlooked_score" in sample  # full ObjectLab populated
    for fname in ("per_image_analysis.csv", "correlations.json", "summary.md", "correlation_scatter.png"):
        p = out_dir / fname
        assert p.exists() and p.stat().st_size > 0


def test_lazy_matplotlib_import():
    """Importing the analysis module must not pull matplotlib (lazy-loaded inside plot)."""
    import subprocess
    import sys

    code = "import sys, ultralytics.utils.analysis\nsys.exit('matplotlib' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
