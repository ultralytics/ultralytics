from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image(
    image: np.ndarray, size: Tuple[int, int] = (12, 12), cmap: Optional[str] = "gray"
) -> None:
    """
    Plots image using matplotlib.

    Args:
        image (np.ndarray): The frame to be displayed.
        size (Tuple[int, int]): The size of the plot.
        cmap (str): the colormap to use for single channel images.

    Examples:
        ```python
        >>> import cv2
        >>> import supervision as sv

        >>> image = cv2.imread("path/to/image.jpg")

        %matplotlib inline
        >>> sv.plot_image(image=image, size=(16, 16))
        ```
    """
    plt.figure(figsize=size)

    if image.ndim == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis("off")
    plt.show()


def plot_images_grid(
    images: List[np.ndarray],
    grid_size: Tuple[int, int],
    titles: Optional[List[str]] = None,
    size: Tuple[int, int] = (12, 12),
    cmap: Optional[str] = "gray",
) -> None:
    """
    Plots images in a grid using matplotlib.

    Args:
       images (List[np.ndarray]): A list of images as numpy arrays.
       grid_size (Tuple[int, int]): A tuple specifying the number
            of rows and columns for the grid.
       titles (Optional[List[str]]): A list of titles for each image.
            Defaults to None.
       size (Tuple[int, int]): A tuple specifying the width and
            height of the entire plot in inches.
       cmap (str): the colormap to use for single channel images.

    Raises:
       ValueError: If the number of images exceeds the grid size.

    Examples:
        ```python
        >>> import cv2
        >>> import supervision as sv

        >>> image1 = cv2.imread("path/to/image1.jpg")
        >>> image2 = cv2.imread("path/to/image2.jpg")
        >>> image3 = cv2.imread("path/to/image3.jpg")

        >>> images = [image1, image2, image3]
        >>> titles = ["Image 1", "Image 2", "Image 3"]

        %matplotlib inline
        >>> plot_images_grid(images, grid_size=(2, 2), titles=titles, size=(16, 16))
        ```
    """
    nrows, ncols = grid_size

    if len(images) > nrows * ncols:
        raise ValueError(
            "The number of images exceeds the grid size. Please increase the grid size"
            " or reduce the number of images."
        )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            if images[idx].ndim == 2:
                ax.imshow(images[idx], cmap=cmap)
            else:
                ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])

        ax.axis("off")
    plt.show()
