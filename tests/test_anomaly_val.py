# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression test for YOLOA anomaly detection with the AnomalyDetect head."""

from pathlib import Path

import pytest

from ultralytics import YOLOA

DATASET_ROOT = Path("/home/laughing/codes/datasets/MVTec-YOLO")
CHECKPOINT = Path("/home/laughing/Downloads/best_converted.pt")


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="MVTec-YOLO dataset not found")
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Converted anomaly checkpoint not found")
def test_yoloa_anomaly_val_bottle():
    """Build a memory bank on bottle-good images and validate on bottle test set."""
    yoloa = YOLOA(str(CHECKPOINT))
    train_good = DATASET_ROOT / "bottle" / "train" / "good"
    bank_size = yoloa.model.build_memory_bank(str(train_good), imgsz=640, batch=8, max_images=50)
    assert bank_size > 0, "memory bank should contain features"

    metrics = yoloa.val(
        data=str(DATASET_ROOT / "bottle" / "bottle.yaml"),
        imgsz=640,
        batch=8,
        device="0" if yoloa.device.type == "cuda" else "cpu",
    )
    # Validator maps the coarse IoU grid to mAP10/mAP25/mAP50/mAP10-50.
    assert 0.0 <= metrics.results_dict.get("metrics/precision(B)", 0.0) <= 1.0
    assert 0.0 <= metrics.results_dict.get("metrics/recall(B)", 0.0) <= 1.0
    assert 0.0 <= metrics.results_dict.get("metrics/mAP50(B)", 0.0) <= 1.0
    assert 0.0 <= metrics.results_dict.get("metrics/mAP50-95(B)", 0.0) <= 1.0


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="MVTec-YOLO dataset not found")
def test_anomaly_detect_head_build():
    """Ensure an AnomalyDetect head can be instantiated from the canonical YAML."""
    yoloa = YOLOA("yolo26-anomaly.yaml")
    head = yoloa.model.model[-1]
    assert type(head).__name__ == "AnomalyDetect"
    assert hasattr(head, "heatmap_bias_fusion")
    assert hasattr(head, "heatmap_processor")
