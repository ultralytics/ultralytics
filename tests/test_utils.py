# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the per-image property analysis module."""

import cv2
import numpy as np
import pytest

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.analysis import ImagePropertyExtractor


def test_pixel_properties():
    """Pixel metrics separate structured from flat images and stay in range (Pech-Pacheco/Canny/Krotkov/Shannon)."""
    flat = np.full((320, 480), 128, dtype=np.uint8)
    checker = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (160, 240))
    uniform = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
    step = np.zeros((320, 480), dtype=np.uint8)
    step[:, 240:] = 255
    halves = np.full((10, 10), 5, dtype=np.uint8)
    halves[5:] = 250
    assert 0.0 <= ImagePropertyExtractor._brightness(cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)) <= 1.0
    assert ImagePropertyExtractor._contrast(flat) == 0.0
    assert ImagePropertyExtractor._entropy(flat) == 0.0
    assert ImagePropertyExtractor._entropy(uniform) > 0.0
    assert ImagePropertyExtractor._blurriness(checker) < ImagePropertyExtractor._blurriness(flat)
    assert ImagePropertyExtractor._edge_density(checker) > ImagePropertyExtractor._edge_density(flat)
    assert ImagePropertyExtractor._sharpness(step) > ImagePropertyExtractor._sharpness(flat)
    assert ImagePropertyExtractor._dark_pixel_ratio(halves) == pytest.approx(0.5)
    assert ImagePropertyExtractor._bright_pixel_ratio(halves) == pytest.approx(0.5)


def test_pairwise_iou_stats():
    """Overlapping boxes give max IoU in (0.1, 0.5); disjoint boxes give mean IoU 0 (CrowdHuman 2018)."""
    overlap = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]], dtype=np.float32)
    disjoint = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]], dtype=np.float32)
    assert 0.1 < ImagePropertyExtractor._pairwise_iou_stats(overlap)[0] < 0.5
    assert ImagePropertyExtractor._pairwise_iou_stats(disjoint)[1] == 0.0


def test_extractor_augments_labels():
    """ImagePropertyExtractor adds an im_properties dict to each label in place, no model or I/O."""
    data = check_det_dataset("coco128.yaml")
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 320})
    ds = build_yolo_dataset(cfg, data["val"], 1, data, mode="val", rect=False, stride=32)
    labels = ImagePropertyExtractor(ds).labels
    assert labels is ds.labels and "im_file" in labels[0]
    props = labels[0]["im_properties"]
    for key in ("brightness", "blurriness", "num_small", "num_medium", "num_large", "max_pairwise_iou"):
        assert key in props, f"missing property {key!r}"
    assert 0.0 <= props["brightness"] <= 1.0
