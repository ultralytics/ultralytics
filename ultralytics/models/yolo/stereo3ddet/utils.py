# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Utility functions for class filtering and reindexing in stereo 3D detection.

This module provides functions to filter KITTI dataset classes from 8 classes to 3 classes
(Car, Pedestrian, Cyclist) matching the Stereo CenterNet paper, and reindex class IDs
from original KITTI indices (0, 3, 5) to paper indices (0, 1, 2).
"""

from __future__ import annotations

# Original KITTI class mapping (8 classes)
# 0: Car, 1: Van, 2: Truck, 3: Pedestrian, 4: Person_sitting, 5: Cyclist, 6: Tram, 7: Misc

# Paper class mapping (3 classes)
# Car (original 0) -> 0
# Pedestrian (original 3) -> 1
# Cyclist (original 5) -> 2

# Mapping from original KITTI class ID to paper class ID
ORIGINAL_TO_PAPER: dict[int, int] = {
    0: 0,  # Car -> Car
    3: 1,  # Pedestrian -> Pedestrian
    5: 2,  # Cyclist -> Cyclist
}

# Reverse mapping from paper class ID to original KITTI class ID
PAPER_TO_ORIGINAL: dict[int, int] = {v: k for k, v in ORIGINAL_TO_PAPER.items()}

# Paper class names
PAPER_CLASS_NAMES: dict[int, str] = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist",
}


def get_paper_class_mapping() -> tuple[dict[int, int], dict[int, int]]:
    """Get class mapping dictionaries for filtering and reindexing.

    Returns:
        tuple[dict[int, int], dict[int, int]]: A tuple containing:
            - original_to_paper: Mapping from original KITTI class ID to paper class ID
            - paper_to_original: Reverse mapping from paper class ID to original KITTI class ID

    Examples:
        >>> original_to_paper, paper_to_original = get_paper_class_mapping()
        >>> original_to_paper[0]  # Car
        0
        >>> original_to_paper[3]  # Pedestrian
        1
        >>> paper_to_original[1]  # Pedestrian -> original ID
        3
    """
    return ORIGINAL_TO_PAPER.copy(), PAPER_TO_ORIGINAL.copy()


def filter_and_remap_class_id(original_class_id: int) -> int | None:
    """Filter and remap a class ID from original KITTI to paper indices.

    Args:
        original_class_id: Original KITTI class ID (0-7).

    Returns:
        int | None: Remapped paper class ID (0-2) if the class is in the paper set,
            None if the class should be filtered out.

    Examples:
        >>> filter_and_remap_class_id(0)  # Car
        0
        >>> filter_and_remap_class_id(3)  # Pedestrian
        1
        >>> filter_and_remap_class_id(5)  # Cyclist
        2
        >>> filter_and_remap_class_id(1)  # Van (filtered out)
        None
    """
    return ORIGINAL_TO_PAPER.get(original_class_id)


def is_paper_class(original_class_id: int) -> bool:
    """Check if a class ID is in the paper class set.

    Args:
        original_class_id: Original KITTI class ID (0-7).

    Returns:
        bool: True if the class is in the paper set (Car, Pedestrian, Cyclist),
            False otherwise.

    Examples:
        >>> is_paper_class(0)  # Car
        True
        >>> is_paper_class(1)  # Van
        False
    """
    return original_class_id in ORIGINAL_TO_PAPER


def get_paper_class_names() -> dict[int, str]:
    """Get paper class names mapping.

    Returns:
        dict[int, str]: Mapping from paper class ID to class name.

    Examples:
        >>> names = get_paper_class_names()
        >>> names[0]
        'Car'
        >>> names[1]
        'Pedestrian'
    """
    return PAPER_CLASS_NAMES.copy()

