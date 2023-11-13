import copy
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.detection.utils import (
    approximate_polygon,
    filter_polygons_by_area,
    mask_to_polygons,
)

T = TypeVar("T")


def approximate_mask_with_polygons(
    mask: np.ndarray,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> List[np.ndarray]:
    height, width = mask.shape
    image_area = height * width
    minimum_detection_area = min_image_area_percentage * image_area
    maximum_detection_area = max_image_area_percentage * image_area

    polygons = mask_to_polygons(mask=mask)
    if len(polygons) == 1:
        polygons = filter_polygons_by_area(
            polygons=polygons, min_area=None, max_area=maximum_detection_area
        )
    else:
        polygons = filter_polygons_by_area(
            polygons=polygons,
            min_area=minimum_detection_area,
            max_area=maximum_detection_area,
        )
    return [
        approximate_polygon(polygon=polygon, percentage=approximation_percentage)
        for polygon in polygons
    ]


def merge_class_lists(class_lists: List[List[str]]) -> List[str]:
    unique_classes = set()

    for class_list in class_lists:
        for class_name in class_list:
            unique_classes.add(class_name.lower())

    return sorted(list(unique_classes))


def build_class_index_mapping(
    source_classes: List[str], target_classes: List[str]
) -> Dict[int, int]:
    index_mapping = {}

    for i, class_name in enumerate(source_classes):
        if class_name not in target_classes:
            raise ValueError(
                f"Class {class_name} not found in target classes. "
                "source_classes must be a subset of target_classes."
            )
        corresponding_index = target_classes.index(class_name)
        index_mapping[i] = corresponding_index

    return index_mapping


def map_detections_class_id(
    source_to_target_mapping: Dict[int, int], detections: Detections
) -> Detections:
    if detections.class_id is None:
        raise ValueError("Detections must have class_id attribute.")
    if set(np.unique(detections.class_id)) - set(source_to_target_mapping.keys()):
        raise ValueError(
            "Detections class_id must be a subset of source_to_target_mapping keys."
        )

    detections_copy = copy.deepcopy(detections)

    if len(detections) > 0:
        detections_copy.class_id = np.vectorize(source_to_target_mapping.get)(
            detections_copy.class_id
        )

    return detections_copy


def save_dataset_images(
    images_directory_path: str, images: Dict[str, np.ndarray]
) -> None:
    Path(images_directory_path).mkdir(parents=True, exist_ok=True)

    for image_path, image in images.items():
        image_name = Path(image_path).name
        target_image_path = os.path.join(images_directory_path, image_name)
        cv2.imwrite(target_image_path, image)


def train_test_split(
    data: List[T],
    train_ratio: float = 0.8,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[List[T], List[T]]:
    """
    Splits the data into two parts using the provided train_ratio.

    Args:
        data (List[T]): The data to split.
        train_ratio (float): The ratio of the training set to the entire dataset.
        random_state (Optional[int]): The seed for the random number generator.
        shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
        Tuple[List[T], List[T]]: The split data.
    """
    if random_state is not None:
        random.seed(random_state)

    if shuffle:
        random.shuffle(data)

    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]
