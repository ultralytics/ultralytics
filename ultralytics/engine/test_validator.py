import numpy as np
from numba import njit
from validator import BaseValidator


def test_numba_match_predictions():
    iouv = np.array([0.5, 0.7, 0.9])
    iou = np.array([[0.4, 0.6, 0.8],
                    [0.6, 0.7, 0.9],
                    [0.3, 0.5, 0.7]])
    correct = np.zeros_like(iou, dtype=bool)

    expected_result = np.array([[False, False, False],
                                [True, True, True],
                                [False, False, False]])

    result = BaseValidator._numba_match_predictions(iouv, iou, correct)

    assert np.array_equal(result, expected_result), "Test failed"

test_numba_match_predictions()