import numpy as np

from ultralytics.utils.metrics import DetMetrics, SegmentMetrics
from ultralytics.models.yolo.detect.val import _compute_box_centers 


def test_det_metrics_center_rmse_single_tp():
    """Deterministic unit test for center_rmse in DetMetrics. One TP with zero offset → RMSE must be 0.0."""
    met = DetMetrics({0: "a", 1: "b"})
    met.update_stats(
        {
            "tp": np.array([[1]]),
            "conf": np.array([0.9]),
            "pred_cls": np.array([0]),
            "target_cls": np.array([0]),
            "target_img": np.array([0]),
            "tp_center_offset": np.array([0.0]),
        }
    )

    met.process()

    # Metric must be registered exactly once
    assert met.keys.count("metrics/center_rmse(B)") == 1

    # center_rmse is stored per class
    assert len(met.box.center_rmse) == met.box.nc
    assert isinstance(met.box.center_rmse[0], float)
    assert met.box.center_rmse[0] == 0.0
    assert met.results_dict["metrics/center_rmse(B)"] == 0.0
    assert met.box.image_metrics["image_0"]["center_rmse"] == 0.0


def test_det_metrics_center_rmse_empty():
    """Center_rmse should not crash with empty preds / empty labels."""
    met = DetMetrics({0: "a"})

    met.update_stats(
        {
            "tp": np.zeros((0, 1)),
            "conf": np.array([]),
            "pred_cls": np.array([]),
            "target_cls": np.array([]),
            "target_img": np.array([]),
            "tp_center_offset": np.array([]),
        }
    )

    met.process()

    # Metric still exists and is well-formed
    assert "metrics/center_rmse(B)" in met.keys
    assert len(met.box.center_rmse) == met.box.nc
    assert isinstance(met.box.center_rmse[0], float)
    assert met.results_dict["metrics/center_rmse(B)"] == 0.0
    assert met.box.image_metrics["image_0"]["center_rmse"] == 0.0


def test_det_metrics_center_rmse_ignores_non_tp_offsets():
    """center_rmse must use only TP offsets and expose the value in results_dict."""
    met = DetMetrics({0: "a"})
    met.update_stats(
        {
            "tp": np.array([[1], [0]]),
            "conf": np.array([0.9, 0.8]),
            "pred_cls": np.array([0, 0]),
            "target_cls": np.array([0]),
            "target_img": np.array([0]),
            "tp_center_offset": np.array([0.2, 10.0]),
        }
    )

    met.process()

    assert np.isclose(met.box.center_rmse[0], 0.2)
    assert np.isclose(met.results_dict["metrics/center_rmse(B)"], 0.2)
    assert np.isclose(met.box.image_metrics["image_0"]["center_rmse"], 0.2)


def test_segment_metrics_no_center_rmse():
    """SegmentMetrics must not expose center_rmse."""
    met = SegmentMetrics({0: "a", 1: "b"})
    assert "metrics/center_rmse(B)" not in met.keys

def test_compute_box_centers_xyxy():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    centers = _compute_box_centers(boxes)
    np.testing.assert_allclose(centers, [[5.0, 5.0]])


def test_compute_box_centers_xywhr():
    boxes = np.array([[10.0, 20.0, 4.0, 2.0, 0.5]])
    centers = _compute_box_centers(boxes)
    np.testing.assert_allclose(centers, [[10.0, 20.0]])


def test_compute_box_centers_xywhr_shape():
    boxes = np.random.rand(6, 5)
    centers = _compute_box_centers(boxes)
    assert centers.shape == (6, 2)


def test_compute_box_centers_polygon():
    # unitary square
    boxes = np.array([[0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
    centers = _compute_box_centers(boxes)
    np.testing.assert_allclose(centers, [[0.5, 0.5]])
