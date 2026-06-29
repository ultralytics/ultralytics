# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the per-image property analysis module."""

import cv2
import numpy as np
import pytest

from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.analysis import CorrelationAnalysis, ImagePropertyExtractor
from ultralytics.utils.ops import xywhn2xyxy


@pytest.fixture
def synthetic_bgr():
    """Return a deterministic 320x480 BGR image with mid-range intensity and structure."""
    gray = np.random.default_rng(42).integers(20, 200, size=(320, 480), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_brightness_in_unit_interval(synthetic_bgr):
    """Brightness is HSP perceptual [0,1] (Finley 2006)."""
    v = ImagePropertyExtractor._brightness(synthetic_bgr)
    assert 0.0 <= v <= 1.0


def test_blurriness_inverse_relationship():
    """A sharp checkerboard returns a lower blurriness than a uniform image (Pech-Pacheco 2000)."""
    sharp = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (160, 240))
    flat = np.full((320, 480), 128, dtype=np.uint8)
    assert ImagePropertyExtractor._blurriness(sharp) < ImagePropertyExtractor._blurriness(flat)


def test_contrast_zero_for_flat_image():
    """A constant-intensity image has zero contrast."""
    flat = np.full((320, 480), 128, dtype=np.uint8)
    assert ImagePropertyExtractor._contrast(flat) == 0.0


def test_dark_and_bright_pixel_ratios():
    """Hand-constructed image has known dark-pixel and bright-pixel fractions."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:5] = 5  # half the image is dark
    img[5:] = 250  # other half bright
    assert ImagePropertyExtractor._dark_pixel_ratio(img) == pytest.approx(0.5)
    assert ImagePropertyExtractor._bright_pixel_ratio(img) == pytest.approx(0.5)


def test_entropy_uniform_higher_than_constant():
    """Uniform random has higher Shannon entropy than constant image (Shannon 1948)."""
    uniform = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
    flat = np.full((64, 64), 128, dtype=np.uint8)
    assert ImagePropertyExtractor._entropy(uniform) > ImagePropertyExtractor._entropy(flat)
    assert ImagePropertyExtractor._entropy(flat) == 0.0


def test_edge_density_higher_for_checkerboard():
    """Checkerboard has higher Canny edge density than flat (Canny 1986)."""
    sharp = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (160, 240))
    flat = np.full((320, 480), 128, dtype=np.uint8)
    assert ImagePropertyExtractor._edge_density(sharp) > ImagePropertyExtractor._edge_density(flat)


def test_sharpness_higher_for_step_edge():
    """A step edge has higher Tenengrad sharpness than a flat image (Krotkov 1988)."""
    edge = np.zeros((320, 480), dtype=np.uint8)
    edge[:, 240:] = 255  # vertical step edge in the middle
    flat = np.full((320, 480), 128, dtype=np.uint8)
    assert ImagePropertyExtractor._sharpness(edge) > ImagePropertyExtractor._sharpness(flat)


def test_pairwise_iou_stats_three_box_example():
    """Two overlapping boxes yield max IoU in (0.1, 0.5), a third disjoint box doesn't increase it."""
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]], dtype=np.float32)
    max_iou, _ = ImagePropertyExtractor._pairwise_iou_stats(boxes)
    assert 0.1 < max_iou < 0.5


def test_pairwise_iou_stats_zero_for_disjoint_boxes():
    """Three pairwise-disjoint boxes have mean IoU == 0."""
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]], dtype=np.float32)
    _, mean_iou = ImagePropertyExtractor._pairwise_iou_stats(boxes)
    assert mean_iou == 0.0


def test_xywhn_to_xyxy_pixels_round_trip():
    """Xywh-normalized boxes round-trip cleanly to xyxy pixels."""
    xywhn = np.array([[0.5, 0.5, 0.4, 0.2]])
    xyxy = xywhn2xyxy(xywhn, w=100, h=200)
    np.testing.assert_allclose(xyxy[0], [30, 80, 70, 120])


def test_extractor_augments_labels():
    """ImagePropertyExtractor mutates dataset.labels in place with property keys, no model or I/O."""
    from ultralytics.cfg import get_cfg

    data = check_det_dataset("coco128.yaml")
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 320})
    ds = build_yolo_dataset(cfg, data["val"], 1, data, mode="val", rect=False, stride=32)
    extractor = ImagePropertyExtractor(ds)
    assert extractor.labels is ds.labels, "extractor.labels must be the same list as dataset.labels"
    lbl = ds.labels[0]
    assert "im_file" in lbl  # original key survives
    for key in ("brightness", "blurriness", "num_small", "num_medium", "num_large", "max_pairwise_iou"):
        assert key in lbl, f"missing property key {key!r} after extraction"
    assert 0.0 <= lbl["brightness"] <= 1.0


def test_run_correlation_analysis_path(tmp_path):
    """CorrelationAnalysis joins extractor labels with validator metrics and writes the full report."""
    model = YOLO("yolo11n.pt")
    metrics = model.val(
        data="coco128.yaml",
        batch=8,
        plots=False,
        save_json=False,
        imgsz=320,
        verbose=False,
        device="cpu",
    )
    labels = ImagePropertyExtractor(model.validator.dataloader.dataset).labels
    out_dir = tmp_path / "corr"
    report = CorrelationAnalysis(labels, metrics).run(save_dir=out_dir)
    sample = next(iter(report.per_image.values()))
    assert sample.get("f1") is not None
    for fname in ("per_image_analysis.csv", "correlations.json", "summary.md", "correlation_scatter.png"):
        p = out_dir / fname
        assert p.exists() and p.stat().st_size > 0


def test_lazy_matplotlib_import():
    """Importing the analysis module must not pull matplotlib (lazy-loaded inside plot)."""
    import subprocess
    import sys

    code = "import sys, ultralytics.utils.analysis\nsys.exit('matplotlib' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
