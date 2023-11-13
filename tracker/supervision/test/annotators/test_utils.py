from contextlib import ExitStack as DoesNotRaise
from test.utils import mock_detections
from typing import Optional

import numpy as np
import pytest

from supervision.annotators.utils import ColorLookup, resolve_color_idx
from supervision.detection.core import Detections


@pytest.mark.parametrize(
    "detections, detection_idx, color_lookup, expected_result, exception",
    [
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorLookup.INDEX,
            0,
            DoesNotRaise(),
        ),  # multiple detections; index lookup
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorLookup.CLASS,
            5,
            DoesNotRaise(),
        ),  # multiple detections; class lookup
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorLookup.TRACK,
            2,
            DoesNotRaise(),
        ),  # multiple detections; track lookup
        (
            Detections.empty(),
            0,
            ColorLookup.INDEX,
            None,
            pytest.raises(ValueError),
        ),  # no detections; index lookup; out of bounds
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            2,
            ColorLookup.INDEX,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; index lookup; out of bounds
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            ColorLookup.CLASS,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; class lookup; no class_id
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            ColorLookup.TRACK,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; class lookup; no track_id
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            np.array([1, 0]),
            1,
            DoesNotRaise(),
        ),  # multiple detections; custom lookup; correct length
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            np.array([1]),
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; custom lookup; wrong length
    ],
)
def test_resolve_color_idx(
    detections: Detections,
    detection_idx: int,
    color_lookup: ColorLookup,
    expected_result: Optional[int],
    exception: Exception,
) -> None:
    with exception:
        result = resolve_color_idx(
            detections=detections,
            detection_idx=detection_idx,
            color_lookup=color_lookup,
        )
        assert result == expected_result
