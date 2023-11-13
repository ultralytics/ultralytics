from typing import List

import numpy as np

from supervision.detection.core import Detections


def mock_detections(
    xyxy: List[List[float]],
    confidence: List[float] = None,
    class_id: List[int] = None,
    tracker_id: List[int] = None,
) -> Detections:
    return Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=confidence
        if confidence is None
        else np.array(confidence, dtype=np.float32),
        class_id=class_id if class_id is None else np.array(class_id, dtype=int),
        tracker_id=tracker_id
        if tracker_id is None
        else np.array(tracker_id, dtype=int),
    )


def assert_almost_equal(actual, expected, tolerance=1e-5):
    assert abs(actual - expected) < tolerance, f"Expected {expected}, but got {actual}."
