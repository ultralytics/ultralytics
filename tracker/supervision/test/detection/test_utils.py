from contextlib import ExitStack as DoesNotRaise
from typing import List, Optional, Tuple

import numpy as np
import pytest

from supervision.detection.utils import (
    clip_boxes,
    filter_polygons_by_area,
    move_boxes,
    non_max_suppression,
    process_roboflow_result,
)

TEST_MASK = np.zeros((1, 1000, 1000), dtype=bool)
TEST_MASK[:, 300:351, 200:251] = True


@pytest.mark.parametrize(
    "predictions, iou_threshold, expected_result, exception",
    [
        (
            np.empty(shape=(0, 5)),
            0.5,
            np.array([]),
            DoesNotRaise(),
        ),  # single box with no category
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.8]]),
            0.5,
            np.array([True]),
            DoesNotRaise(),
        ),  # single box with no category
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.8, 0]]),
            0.5,
            np.array([True]),
            DoesNotRaise(),
        ),  # single box with category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8],
                    [15.0, 15.0, 40.0, 40.0, 0.9],
                ]
            ),
            0.5,
            np.array([False, True]),
            DoesNotRaise(),
        ),  # two boxes with no category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8, 0],
                    [15.0, 15.0, 40.0, 40.0, 0.9, 1],
                ]
            ),
            0.5,
            np.array([True, True]),
            DoesNotRaise(),
        ),  # two boxes with different category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8, 0],
                    [15.0, 15.0, 40.0, 40.0, 0.9, 0],
                ]
            ),
            0.5,
            np.array([False, True]),
            DoesNotRaise(),
        ),  # two boxes with same category
        (
            np.array(
                [
                    [0.0, 0.0, 30.0, 40.0, 0.8],
                    [5.0, 5.0, 35.0, 45.0, 0.9],
                    [10.0, 10.0, 40.0, 50.0, 0.85],
                ]
            ),
            0.5,
            np.array([False, True, False]),
            DoesNotRaise(),
        ),  # three boxes with no category
        (
            np.array(
                [
                    [0.0, 0.0, 30.0, 40.0, 0.8, 0],
                    [5.0, 5.0, 35.0, 45.0, 0.9, 1],
                    [10.0, 10.0, 40.0, 50.0, 0.85, 2],
                ]
            ),
            0.5,
            np.array([True, True, True]),
            DoesNotRaise(),
        ),  # three boxes with same category
        (
            np.array(
                [
                    [0.0, 0.0, 30.0, 40.0, 0.8, 0],
                    [5.0, 5.0, 35.0, 45.0, 0.9, 0],
                    [10.0, 10.0, 40.0, 50.0, 0.85, 1],
                ]
            ),
            0.5,
            np.array([False, True, True]),
            DoesNotRaise(),
        ),  # three boxes with different category
    ],
)
def test_non_max_suppression(
    predictions: np.ndarray,
    iou_threshold: float,
    expected_result: Optional[np.ndarray],
    exception: Exception,
) -> None:
    with exception:
        result = non_max_suppression(
            predictions=predictions, iou_threshold=iou_threshold
        )
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "xyxy, resolution_wh, expected_result",
    [
        (
            np.empty(shape=(0, 4)),
            (1280, 720),
            np.empty(shape=(0, 4)),
        ),
        (
            np.array([[1.0, 1.0, 1279.0, 719.0]]),
            (1280, 720),
            np.array([[1.0, 1.0, 1279.0, 719.0]]),
        ),
        (
            np.array([[-1.0, 1.0, 1279.0, 719.0]]),
            (1280, 720),
            np.array([[0.0, 1.0, 1279.0, 719.0]]),
        ),
        (
            np.array([[1.0, -1.0, 1279.0, 719.0]]),
            (1280, 720),
            np.array([[1.0, 0.0, 1279.0, 719.0]]),
        ),
        (
            np.array([[1.0, 1.0, 1281.0, 719.0]]),
            (1280, 720),
            np.array([[1.0, 1.0, 1280.0, 719.0]]),
        ),
        (
            np.array([[1.0, 1.0, 1279.0, 721.0]]),
            (1280, 720),
            np.array([[1.0, 1.0, 1279.0, 720.0]]),
        ),
    ],
)
def test_clip_boxes(
    xyxy: np.ndarray,
    resolution_wh: Tuple[int, int],
    expected_result: np.ndarray,
) -> None:
    result = clip_boxes(xyxy=xyxy, resolution_wh=resolution_wh)
    assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "polygons, min_area, max_area, expected_result, exception",
    [
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            None,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # single polygon without area constraints
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            50,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # single polygon with min_area constraint
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            None,
            50,
            [],
            DoesNotRaise(),
        ),  # single polygon with max_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            200,
            None,
            [np.array([[0, 0], [0, 20], [20, 20], [20, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            None,
            200,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # two polygons with max_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            200,
            200,
            [],
            DoesNotRaise(),
        ),  # two polygons with both area constraints
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            100,
            100,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area and
        # max_area equal to the area of the first polygon
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            400,
            400,
            [np.array([[0, 0], [0, 20], [20, 20], [20, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area and
        # max_area equal to the area of the second polygon
    ],
)
def test_filter_polygons_by_area(
    polygons: List[np.ndarray],
    min_area: Optional[float],
    max_area: Optional[float],
    expected_result: List[np.ndarray],
    exception: Exception,
) -> None:
    with exception:
        result = filter_polygons_by_area(
            polygons=polygons, min_area=min_area, max_area=max_area
        )
        assert len(result) == len(expected_result)
        for result_polygon, expected_result_polygon in zip(result, expected_result):
            assert np.array_equal(result_polygon, expected_result_polygon)


@pytest.mark.parametrize(
    "roboflow_result, expected_result, exception",
    [
        (
            {"predictions": [], "image": {"width": 1000, "height": 1000}},
            (np.empty((0, 4)), np.empty(0), np.empty(0), None, None),
            DoesNotRaise(),
        ),  # empty result
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                    }
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),  # single correct object detection result
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "tracker_id": 1,
                    },
                    {
                        "x": 500.0,
                        "y": 500.0,
                        "width": 100.0,
                        "height": 100.0,
                        "confidence": 0.8,
                        "class_id": 7,
                        "class": "truck",
                        "tracker_id": 2,
                    },
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.array([[175.0, 275.0, 225.0, 325.0], [450.0, 450.0, 550.0, 550.0]]),
                np.array([0.9, 0.8]),
                np.array([0, 7]),
                None,
                np.array([1, 2]),
            ),
            DoesNotRaise(),
        ),  # two correct object detection result
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "points": [],
                        "tracker_id": None,
                    }
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (np.empty((0, 4)), np.empty(0), np.empty(0), None, None),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no points
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "points": [{"x": 200.0, "y": 300.0}, {"x": 250.0, "y": 300.0}],
                    }
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (np.empty((0, 4)), np.empty(0), np.empty(0), None, None),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no enough points
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "points": [
                            {"x": 200.0, "y": 300.0},
                            {"x": 250.0, "y": 300.0},
                            {"x": 250.0, "y": 350.0},
                            {"x": 200.0, "y": 350.0},
                        ],
                    }
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                TEST_MASK,
                None,
            ),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no enough points
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "points": [
                            {"x": 200.0, "y": 300.0},
                            {"x": 250.0, "y": 300.0},
                            {"x": 250.0, "y": 350.0},
                            {"x": 200.0, "y": 350.0},
                        ],
                    },
                    {
                        "x": 500.0,
                        "y": 500.0,
                        "width": 100.0,
                        "height": 100.0,
                        "confidence": 0.8,
                        "class_id": 7,
                        "class": "truck",
                        "points": [],
                    },
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                TEST_MASK,
                None,
            ),
            DoesNotRaise(),
        ),  # two instance segmentation results - one correct, one incorrect
    ],
)
def test_process_roboflow_result(
    roboflow_result: dict,
    expected_result: Tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray
    ],
    exception: Exception,
) -> None:
    with exception:
        result = process_roboflow_result(roboflow_result=roboflow_result)
        assert np.array_equal(result[0], expected_result[0])
        assert np.array_equal(result[1], expected_result[1])
        assert np.array_equal(result[2], expected_result[2])
        assert (result[3] is None and expected_result[3] is None) or (
            np.array_equal(result[3], expected_result[3])
        )
        assert (result[4] is None and expected_result[4] is None) or (
            np.array_equal(result[4], expected_result[4])
        )


@pytest.mark.parametrize(
    "xyxy, offset, expected_result, exception",
    [
        (
            np.empty(shape=(0, 4)),
            np.array([0, 0]),
            np.empty(shape=(0, 4)),
            DoesNotRaise(),
        ),  # empty xyxy array
        (
            np.array([[0, 0, 10, 10]]),
            np.array([0, 0]),
            np.array([[0, 0, 10, 10]]),
            DoesNotRaise(),
        ),  # single box with zero offset
        (
            np.array([[0, 0, 10, 10]]),
            np.array([10, 10]),
            np.array([[10, 10, 20, 20]]),
            DoesNotRaise(),
        ),  # single box with non-zero offset
        (
            np.array([[0, 0, 10, 10], [0, 0, 10, 10]]),
            np.array([10, 10]),
            np.array([[10, 10, 20, 20], [10, 10, 20, 20]]),
            DoesNotRaise(),
        ),  # two boxes with non-zero offset
        (
            np.array([[0, 0, 10, 10], [0, 0, 10, 10]]),
            np.array([-10, -10]),
            np.array([[-10, -10, 0, 0], [-10, -10, 0, 0]]),
            DoesNotRaise(),
        ),  # two boxes with negative offset
    ],
)
def test_move_boxes(
    xyxy: np.ndarray,
    offset: np.ndarray,
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    result = move_boxes(xyxy=xyxy, offset=offset)
    assert np.array_equal(result, expected_result)
