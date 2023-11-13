import os
import shutil
from typing import Optional

import cv2
import numpy as np


def crop_image(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Crops the given image based on the given bounding box.

    Args:
        image (np.ndarray): The image to be cropped, represented as a numpy array.
        xyxy (np.ndarray): A numpy array containing the bounding box coordinates
            in the format (x1, y1, x2, y2).

    Returns:
        (np.ndarray): The cropped image as a numpy array.

    Examples:
        ```python
        >>> import supervision as sv

        >>> detection = sv.Detections(...)
        >>> with sv.ImageSink(target_dir_path='target/directory/path') as sink:
        ...     for xyxy in detection.xyxy:
        ...         cropped_image = sv.crop_image(image=image, xyxy=xyxy)
        ...         sink.save_image(image=image)
        ```
    """

    xyxy = np.round(xyxy).astype(int)
    x1, y1, x2, y2 = xyxy
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img


class ImageSink:
    def __init__(
        self,
        target_dir_path: str,
        overwrite: bool = False,
        image_name_pattern: str = "image_{:05d}.png",
    ):
        """
        Initialize a context manager for saving images.

        Args:
            target_dir_path (str): The target directory where images will be saved.
            overwrite (bool, optional): Whether to overwrite the existing directory.
                Defaults to False.
            image_name_pattern (str, optional): The image file name pattern.
                Defaults to "image_{:05d}.png".

        Examples:
            ```python
            >>> import supervision as sv

            >>> with sv.ImageSink(target_dir_path='target/directory/path',
            ...                   overwrite=True) as sink:
            ...     for image in sv.get_video_frames_generator(
            ...         source_path='source_video.mp4', stride=2):
            ...         sink.save_image(image=image)
            ```
        """

        self.target_dir_path = target_dir_path
        self.overwrite = overwrite
        self.image_name_pattern = image_name_pattern
        self.image_count = 0

    def __enter__(self):
        if os.path.exists(self.target_dir_path):
            if self.overwrite:
                shutil.rmtree(self.target_dir_path)
                os.makedirs(self.target_dir_path)
        else:
            os.makedirs(self.target_dir_path)

        return self

    def save_image(self, image: np.ndarray, image_name: Optional[str] = None):
        """
        Save a given image in the target directory.

        Args:
            image (np.ndarray): The image to be saved.
            image_name (str, optional): The name to use for the saved image.
                If not provided, a name will be
                generated using the `image_name_pattern`.
        """
        if image_name is None:
            image_name = self.image_name_pattern.format(self.image_count)

        image_path = os.path.join(self.target_dir_path, image_name)
        cv2.imwrite(image_path, image)
        self.image_count += 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
