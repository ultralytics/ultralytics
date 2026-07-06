# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly predictor with optional heatmap prior."""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionPredictor


class YOLOAnomalyPredictor(DetectionPredictor):
    """YOLO Anomaly predictor.

    Uses the memory-bank heatmap prior automatically when a non-empty bank has been fitted.
    Otherwise it falls back to regular detection inference.
    """

    pass
