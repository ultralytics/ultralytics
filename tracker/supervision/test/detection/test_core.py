from contextlib import ExitStack as DoesNotRaise
from test.utils import mock_detections
from typing import List, Optional, Union

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.geometry.core import Position

PREDICTIONS = np.array(
    [
        [2254, 906, 2447, 1353, 0.90538, 0],
        [2049, 1133, 2226, 1371, 0.59002, 56],
        [727, 1224, 838, 1601, 0.51119, 39],
        [808, 1214, 910, 1564, 0.45287, 39],
        [6, 52, 1131, 2133, 0.45057, 72],
        [299, 1225, 512, 1663, 0.45029, 39],
        [529, 874, 645, 945, 0.31101, 39],
        [8, 47, 1935, 2135, 0.28192, 72],
        [2265, 813, 2328, 901, 0.2714, 62],
    ],
    dtype=np.float32,
)

DETECTIONS = Detections(
    xyxy=PREDICTIONS[:, :4],
    confidence=PREDICTIONS[:, 4],
    class_id=PREDICTIONS[:, 5].astype(int),
)


@pytest.mark.parametrize(
    "detections, index, expected_result, exception",
    [
        (
            DETECTIONS,
            DETECTIONS.class_id == 0,
            mock_detections(
                xyxy=[[2254, 906, 2447, 1353]], confidence=[0.90538], class_id=[0]
            ),
            DoesNotRaise(),
        ),  # take only detections with class_id = 0
        (
            DETECTIONS,
            DETECTIONS.confidence > 0.5,
            mock_detections(
                xyxy=[
                    [2254, 906, 2447, 1353],
                    [2049, 1133, 2226, 1371],
                    [727, 1224, 838, 1601],
                ],
                confidence=[0.90538, 0.59002, 0.51119],
                class_id=[0, 56, 39],
            ),
            DoesNotRaise(),
        ),  # take only detections with confidence > 0.5
        (
            DETECTIONS,
            np.array(
                [True, True, True, True, True, True, True, True, True], dtype=bool
            ),
            DETECTIONS,
            DoesNotRaise(),
        ),  # take all detections
        (
            DETECTIONS,
            np.array(
                [False, False, False, False, False, False, False, False, False],
                dtype=bool,
            ),
            Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=int),
            ),
            DoesNotRaise(),
        ),  # take no detections
        (
            DETECTIONS,
            [0, 2],
            mock_detections(
                xyxy=[[2254, 906, 2447, 1353], [727, 1224, 838, 1601]],
                confidence=[0.90538, 0.51119],
                class_id=[0, 39],
            ),
            DoesNotRaise(),
        ),  # take only first and third detection using List[int] index
        (
            DETECTIONS,
            np.array([0, 2]),
            mock_detections(
                xyxy=[[2254, 906, 2447, 1353], [727, 1224, 838, 1601]],
                confidence=[0.90538, 0.51119],
                class_id=[0, 39],
            ),
            DoesNotRaise(),
        ),  # take only first and third detection using np.ndarray index
        (
            DETECTIONS,
            0,
            mock_detections(
                xyxy=[[2254, 906, 2447, 1353]], confidence=[0.90538], class_id=[0]
            ),
            DoesNotRaise(),
        ),  # take only first detection by index
        (
            DETECTIONS,
            slice(1, 3),
            mock_detections(
                xyxy=[[2049, 1133, 2226, 1371], [727, 1224, 838, 1601]],
                confidence=[0.59002, 0.51119],
                class_id=[56, 39],
            ),
            DoesNotRaise(),
        ),  # take only first detection by index slice (1, 3)
        (DETECTIONS, 10, None, pytest.raises(IndexError)),  # index out of range
        (DETECTIONS, [0, 2, 10], None, pytest.raises(IndexError)),  # index out of range
        (DETECTIONS, np.array([0, 2, 10]), None, pytest.raises(IndexError)),
        (
            DETECTIONS,
            np.array(
                [True, True, True, True, True, True, True, True, True, True, True]
            ),
            None,
            pytest.raises(IndexError),
        ),
    ],
)
def test_getitem(
    detections: Detections,
    index: Union[int, slice, List[int], np.ndarray],
    expected_result: Optional[Detections],
    exception: Exception,
) -> None:
    with exception:
        result = detections[index]
        assert result == expected_result


@pytest.mark.parametrize(
    "detections_list, expected_result, exception",
    [
        ([], Detections.empty(), DoesNotRaise()),  # empty detections list
        (
            [Detections.empty()],
            Detections.empty(),
            DoesNotRaise(),
        ),  # single empty detections
        (
            [mock_detections(xyxy=[[10, 10, 20, 20]])],
            mock_detections(xyxy=[[10, 10, 20, 20]]),
            DoesNotRaise(),
        ),  # single detection with xyxy field
        (
            [mock_detections(xyxy=[[10, 10, 20, 20]]), Detections.empty()],
            mock_detections(xyxy=[[10, 10, 20, 20]]),
            DoesNotRaise(),
        ),  # single detection with xyxy field + empty detection
        (
            [
                mock_detections(xyxy=[[10, 10, 20, 20]]),
                mock_detections(xyxy=[[20, 20, 30, 30]]),
            ],
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            DoesNotRaise(),
        ),  # two detections with xyxy field
        (
            [
                mock_detections(xyxy=[[10, 10, 20, 20]], class_id=[0]),
                mock_detections(xyxy=[[20, 20, 30, 30]]),
            ],
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            DoesNotRaise(),
        ),  # detection with xyxy, class_id fields + detection with xyxy field
        (
            [
                mock_detections(xyxy=[[10, 10, 20, 20]], class_id=[0]),
                mock_detections(xyxy=[[20, 20, 30, 30]], class_id=[1]),
            ],
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]], class_id=[0, 1]),
            DoesNotRaise(),
        ),  # two detections with xyxy, class_id fields
    ],
)
def test_merge(
    detections_list: List[Detections],
    expected_result: Optional[Detections],
    exception: Exception,
) -> None:
    with exception:
        result = Detections.merge(detections_list=detections_list)
        assert result == expected_result


@pytest.mark.parametrize(
    "detections, anchor, expected_result, exception",
    [
        (
            Detections.empty(),
            Position.CENTER,
            np.empty((0, 2), dtype=np.float32),
            DoesNotRaise(),
        ),  # empty detections
        (
            mock_detections(xyxy=[[10, 10, 20, 20]]),
            Position.CENTER,
            np.array([[15, 15]], dtype=np.float32),
            DoesNotRaise(),
        ),  # single detection; center anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER,
            np.array([[15, 15], [25, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER_LEFT,
            np.array([[10, 15], [20, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center left anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER_RIGHT,
            np.array([[20, 15], [30, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center right anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_CENTER,
            np.array([[15, 10], [25, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top center anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_LEFT,
            np.array([[10, 10], [20, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top left anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_RIGHT,
            np.array([[20, 10], [30, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top right anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.BOTTOM_CENTER,
            np.array([[15, 20], [25, 30]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; bottom center anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.BOTTOM_LEFT,
            np.array([[10, 20], [20, 30]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; bottom left anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.BOTTOM_RIGHT,
            np.array([[20, 20], [30, 30]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; bottom right anchor
    ],
)
def test_get_anchor_coordinates(
    detections: Detections,
    anchor: Position,
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    result = detections.get_anchor_coordinates(anchor)
    with exception:
        assert np.array_equal(result, expected_result)
