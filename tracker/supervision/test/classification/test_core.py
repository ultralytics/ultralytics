from contextlib import ExitStack as DoesNotRaise
from typing import Optional, Tuple

import numpy as np
import pytest

from supervision.classification.core import Classifications


@pytest.mark.parametrize(
    "class_id, confidence, k, expected_result, exception",
    [
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0.1, 0.2, 0.9, 0.4, 0.5]),
            5,
            (np.array([2, 4, 3, 1, 0]), np.array([0.9, 0.5, 0.4, 0.2, 0.1])),
            DoesNotRaise(),
        ),  # class_id with 5 numbers and 5 confidences
        (
            np.array([5, 1, 2, 3, 4]),
            np.array([0.1, 0.2, 0.9, 0.4, 0.5]),
            1,
            (np.array([2]), np.array([0.9])),
            DoesNotRaise(),
        ),  # class_id with 5 numbers and 5 confidences, retrieve where k = 1
        (
            np.array([4, 1, 2, 3, 6, 5]),
            np.array([0.8, 0.2, 0.9, 0.4, 0.5, 0.1]),
            2,
            (np.array([2, 4]), np.array([0.9, 0.8])),
            DoesNotRaise(),
        ),  # class_id with 5 numbers and 5 confidences, retrieve where k = 3
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([]),
            5,
            None,
            pytest.raises(ValueError),
        ),  # class_id with 5 numbers and 0 confidences
        (
            [0, 1, 2, 3, 4],
            [0.1, 0.2, 0.3, 0.4],
            5,
            None,
            pytest.raises(ValueError),
        ),  # class_id with 5 numbers and 4 confidences
    ],
)
def test_top_k(
    class_id: np.ndarray,
    confidence: Optional[np.ndarray],
    k: int,
    expected_result: Optional[Tuple[np.ndarray, np.ndarray]],
    exception: Exception,
) -> None:
    with exception:
        result = Classifications(
            class_id=np.array(class_id), confidence=np.array(confidence)
        ).get_top_k(k)

        assert np.array_equal(result[0], expected_result[0])
        assert np.array_equal(result[1], expected_result[1])
