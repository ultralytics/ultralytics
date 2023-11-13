from contextlib import ExitStack as DoesNotRaise
from typing import Dict, List, Tuple

import numpy as np
import pytest

from supervision import Detections
from supervision.dataset.formats.coco import (
    build_coco_class_index_mapping,
    classes_to_coco_categories,
    coco_annotations_to_detections,
    coco_categories_to_classes,
    group_coco_annotations_by_image_id,
)


def mock_cock_coco_annotation(
    annotation_id: int = 0,
    image_id: int = 0,
    category_id: int = 0,
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    area: float = 0.0,
) -> dict:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "area": area,
        "iscrowd": 0,
    }


@pytest.mark.parametrize(
    "coco_categories, expected_result, exception",
    [
        ([], [], DoesNotRaise()),  # empty coco categories
        (
            [{"id": 0, "name": "fashion-assistant", "supercategory": "none"}],
            ["fashion-assistant"],
            DoesNotRaise(),
        ),  # single coco category with supercategory == "none"
        (
            [
                {"id": 0, "name": "fashion-assistant", "supercategory": "none"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["fashion-assistant", "baseball cap"],
            DoesNotRaise(),
        ),  # two coco categories; one with supercategory == "none" and
        # one with supercategory != "none"
        (
            [
                {"id": 0, "name": "fashion-assistant", "supercategory": "none"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
                {"id": 2, "name": "hoodie", "supercategory": "fashion-assistant"},
            ],
            ["fashion-assistant", "baseball cap", "hoodie"],
            DoesNotRaise(),
        ),  # three coco categories; one with supercategory == "none" and
        # two with supercategory != "none"
        (
            [
                {"id": 0, "name": "fashion-assistant", "supercategory": "none"},
                {"id": 2, "name": "hoodie", "supercategory": "fashion-assistant"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["fashion-assistant", "baseball cap", "hoodie"],
            DoesNotRaise(),
        ),  # three coco categories; one with supercategory == "none" and
        # two with supercategory != "none" (different order)
    ],
)
def test_coco_categories_to_classes(
    coco_categories: List[dict], expected_result: List[str], exception: Exception
) -> None:
    with exception:
        result = coco_categories_to_classes(coco_categories=coco_categories)
        assert result == expected_result


@pytest.mark.parametrize(
    "classes, exception",
    [
        ([], DoesNotRaise()),  # empty classes
        (["baseball cap"], DoesNotRaise()),  # single class
        (["baseball cap", "hoodie"], DoesNotRaise()),  # two classes
    ],
)
def test_classes_to_coco_categories_and_back_to_classes(
    classes: List[str], exception: Exception
) -> None:
    with exception:
        coco_categories = classes_to_coco_categories(classes=classes)
        result = coco_categories_to_classes(coco_categories=coco_categories)
        assert result == classes


@pytest.mark.parametrize(
    "coco_annotations, expected_result, exception",
    [
        ([], {}, DoesNotRaise()),  # empty coco annotations
        (
            [mock_cock_coco_annotation(annotation_id=0, image_id=0, category_id=0)],
            {
                0: [
                    mock_cock_coco_annotation(
                        annotation_id=0, image_id=0, category_id=0
                    )
                ]
            },
            DoesNotRaise(),
        ),  # single coco annotation
        (
            [
                mock_cock_coco_annotation(annotation_id=0, image_id=0, category_id=0),
                mock_cock_coco_annotation(annotation_id=1, image_id=1, category_id=0),
            ],
            {
                0: [
                    mock_cock_coco_annotation(
                        annotation_id=0, image_id=0, category_id=0
                    )
                ],
                1: [
                    mock_cock_coco_annotation(
                        annotation_id=1, image_id=1, category_id=0
                    )
                ],
            },
            DoesNotRaise(),
        ),  # two coco annotations
        (
            [
                mock_cock_coco_annotation(annotation_id=0, image_id=0, category_id=0),
                mock_cock_coco_annotation(annotation_id=1, image_id=1, category_id=1),
                mock_cock_coco_annotation(annotation_id=2, image_id=1, category_id=2),
                mock_cock_coco_annotation(annotation_id=3, image_id=2, category_id=3),
                mock_cock_coco_annotation(annotation_id=4, image_id=3, category_id=1),
                mock_cock_coco_annotation(annotation_id=5, image_id=3, category_id=2),
                mock_cock_coco_annotation(annotation_id=5, image_id=3, category_id=3),
            ],
            {
                0: [
                    mock_cock_coco_annotation(
                        annotation_id=0, image_id=0, category_id=0
                    ),
                ],
                1: [
                    mock_cock_coco_annotation(
                        annotation_id=1, image_id=1, category_id=1
                    ),
                    mock_cock_coco_annotation(
                        annotation_id=2, image_id=1, category_id=2
                    ),
                ],
                2: [
                    mock_cock_coco_annotation(
                        annotation_id=3, image_id=2, category_id=3
                    ),
                ],
                3: [
                    mock_cock_coco_annotation(
                        annotation_id=4, image_id=3, category_id=1
                    ),
                    mock_cock_coco_annotation(
                        annotation_id=5, image_id=3, category_id=2
                    ),
                    mock_cock_coco_annotation(
                        annotation_id=5, image_id=3, category_id=3
                    ),
                ],
            },
            DoesNotRaise(),
        ),  # two coco annotations
    ],
)
def test_group_coco_annotations_by_image_id(
    coco_annotations: List[dict], expected_result: dict, exception: Exception
) -> None:
    with exception:
        result = group_coco_annotations_by_image_id(coco_annotations=coco_annotations)
        assert result == expected_result


@pytest.mark.parametrize(
    "image_annotations, resolution_wh, with_masks, expected_result, exception",
    [
        (
            [],
            (1000, 1000),
            False,
            Detections.empty(),
            DoesNotRaise(),
        ),  # empty image annotations
        (
            [
                mock_cock_coco_annotation(
                    category_id=0, bbox=(0, 0, 100, 100), area=100 * 100
                )
            ],
            (1000, 1000),
            False,
            Detections(
                xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            ),
            DoesNotRaise(),
        ),  # single image annotations
        (
            [
                mock_cock_coco_annotation(
                    category_id=0, bbox=(0, 0, 100, 100), area=100 * 100
                ),
                mock_cock_coco_annotation(
                    category_id=0, bbox=(100, 100, 100, 100), area=100 * 100
                ),
            ],
            (1000, 1000),
            False,
            Detections(
                xyxy=np.array(
                    [[0, 0, 100, 100], [100, 100, 200, 200]], dtype=np.float32
                ),
                class_id=np.array([0, 0], dtype=int),
            ),
            DoesNotRaise(),
        ),  # two image annotations
    ],
)
def test_coco_annotations_to_detections(
    image_annotations: List[dict],
    resolution_wh: Tuple[int, int],
    with_masks: bool,
    expected_result: Detections,
    exception: Exception,
) -> None:
    with exception:
        result = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=resolution_wh,
            with_masks=with_masks,
        )
        assert result == expected_result


@pytest.mark.parametrize(
    "coco_categories, target_classes, expected_result, exception",
    [
        ([], [], {}, DoesNotRaise()),  # empty coco categories
        (
            [{"id": 0, "name": "fashion-assistant", "supercategory": "none"}],
            ["fashion-assistant"],
            {0: 0},
            DoesNotRaise(),
        ),  # single coco category starting from 0
        (
            [{"id": 1, "name": "fashion-assistant", "supercategory": "none"}],
            ["fashion-assistant"],
            {1: 0},
            DoesNotRaise(),
        ),  # single coco category starting from 1
        (
            [
                {"id": 0, "name": "fashion-assistant", "supercategory": "none"},
                {"id": 2, "name": "hoodie", "supercategory": "fashion-assistant"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["fashion-assistant", "baseball cap", "hoodie"],
            {0: 0, 1: 1, 2: 2},
            DoesNotRaise(),
        ),  # three coco categories
        (
            [
                {"id": 2, "name": "hoodie", "supercategory": "fashion-assistant"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["baseball cap", "hoodie"],
            {2: 1, 1: 0},
            DoesNotRaise(),
        ),  # two coco categories
        (
            [
                {"id": 3, "name": "hoodie", "supercategory": "fashion-assistant"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["baseball cap", "hoodie"],
            {3: 1, 1: 0},
            DoesNotRaise(),
        ),  # two coco categories with missing category
    ],
)
def test_build_coco_class_index_mapping(
    coco_categories: List[dict],
    target_classes: List[str],
    expected_result: Dict[int, int],
    exception: Exception,
) -> None:
    with exception:
        result = build_coco_class_index_mapping(
            coco_categories=coco_categories, target_classes=target_classes
        )
        assert result == expected_result
