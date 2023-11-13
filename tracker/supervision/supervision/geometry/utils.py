import numpy as np

from supervision.geometry.core import Point


def get_polygon_center(polygon: np.ndarray) -> Point:
    """
    Calculate the center of a polygon.

    This function takes in a polygon as a 2-dimensional numpy ndarray and
    returns the center of the polygon as a Point object.
    The center is calculated as the mean of the polygon's vertices along each axis,
    and is rounded down to the nearest integer.

    Parameters:
        polygon (np.ndarray): A 2-dimensional numpy ndarray representing the
            vertices of the polygon.

    Returns:
        Point: The center of the polygon, represented as a
            Point object with x and y attributes.

    Examples:
        ```python
        >>> from supervision.geometry.utils import get_polygon_center

        >>> vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> get_center(vertices)
        Point(x=0.5, y=0.5)
        ```
    """
    center = np.mean(polygon, axis=0).astype(int)
    return Point(x=center[0], y=center[1])
