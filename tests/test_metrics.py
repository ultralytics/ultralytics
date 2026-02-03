import numpy as np

from ultralytics.utils.metrics import DetMetrics, SegmentMetrics


def test_det_metrics_tp_rmse():
    """Test TP-RMSE (True Positive-RMSE) metric. Tests that center_rmse does not crash and is unique for (detect, OBB)
    models (DetMetrics).
    """
    met = DetMetrics({0: "a", 1: "b"})
    met.update_stats(
        {
            "tp": np.array([[1]]),
            "conf": np.array([0.9]),
            "pred_cls": np.array([0]),
            "target_cls": np.array([0]),
            "target_img": np.array([0]),
            "tp_center_offset": np.array([0.5]),
        }
    )

    keys = met.keys
    assert "metrics/center_rmse(B)" in keys
    assert keys.count("metrics/center_rmse(B)") == 1

    results = met.box.mean_results()
    assert isinstance(results[-1], float)
    assert results[-1] == 0.0


def test_segm_metrics_tp_rmse():
    """Test TP-RMSE metric. Tests center_rmse does not exist when using SegmentMetrics."""
    met = SegmentMetrics({0: "a", 1: "b"})
    keys = met.keys
    assert "metrics/center_rmse(B)" not in keys
