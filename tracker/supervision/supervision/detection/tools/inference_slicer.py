from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Tuple

import numpy as np

from supervision.detection.core import Detections, validate_inference_callback
from supervision.detection.utils import move_boxes
from supervision.utils.image import crop_image


def move_detections(detections: Detections, offset: np.array) -> Detections:
    """
    Args:
        detections (sv.Detections): Detections object to be moved.
        offset (np.array): An array of shape `(2,)` containing offset values in format
            is `[dx, dy]`.
    Returns:
        (sv.Detections) repositioned Detections object.
    """
    detections.xyxy = move_boxes(xyxy=detections.xyxy, offset=offset)
    return detections


class InferenceSlicer:
    """
    InferenceSlicer performs slicing-based inference for small target detection. This
    method, often referred to as
    [Slicing Adaptive Inference (SAHI)](https://ieeexplore.ieee.org/document/9897990),
    involves dividing a larger image into smaller slices, performing inference on each
    slice, and then merging the detections.

    Args:
        slice_wh (Tuple[int, int]): Dimensions of each slice in the format
            `(width, height)`.
        overlap_ratio_wh (Tuple[float, float]): Overlap ratio between consecutive
            slices in the format `(width_ratio, height_ratio)`.
        iou_threshold (Optional[float]): Intersection over Union (IoU) threshold
            used for non-max suppression.
        callback (Callable): A function that performs inference on a given image
            slice and returns detections.
        thread_workers (int): Number of threads for parallel execution.

    Note:
        The class ensures that slices do not exceed the boundaries of the original
        image. As a result, the final slices in the row and column dimensions might be
        smaller than the specified slice dimensions if the image's width or height is
        not a multiple of the slice's width or height minus the overlap.
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], Detections],
        slice_wh: Tuple[int, int] = (320, 320),
        overlap_ratio_wh: Tuple[float, float] = (0.2, 0.2),
        iou_threshold: Optional[float] = 0.5,
        thread_workers: int = 1,
    ):
        self.slice_wh = slice_wh
        self.overlap_ratio_wh = overlap_ratio_wh
        self.iou_threshold = iou_threshold
        self.callback = callback
        self.thread_workers = thread_workers
        validate_inference_callback(callback=callback)

    def __call__(self, image: np.ndarray) -> Detections:
        """
        Performs slicing-based inference on the provided image using the specified
            callback.

        Args:
            image (np.ndarray): The input image on which inference needs to be
                performed. The image should be in the format
                `(height, width, channels)`.

        Returns:
            Detections: A collection of detections for the entire image after merging
                results from all slices and applying NMS.

        Example:
            ```python
            >>> import cv2
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> image = cv2.imread(SOURCE_IMAGE_PATH)
            >>> model = YOLO(...)

            >>> def callback(image_slice: np.ndarray) -> sv.Detections:
            ...     result = model(image_slice)[0]
            ...     return sv.Detections.from_ultralytics(result)

            >>> slicer = sv.InferenceSlicer(callback = callback)

            >>> detections = slicer(image)
            ```
        """
        detections_list = []
        resolution_wh = (image.shape[1], image.shape[0])
        offsets = self._generate_offset(
            resolution_wh=resolution_wh,
            slice_wh=self.slice_wh,
            overlap_ratio_wh=self.overlap_ratio_wh,
        )

        with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
            futures = [
                executor.submit(self._run_callback, image, offset) for offset in offsets
            ]
            for future in as_completed(futures):
                detections_list.append(future.result())

        return Detections.merge(detections_list=detections_list).with_nms(
            threshold=self.iou_threshold
        )

    def _run_callback(self, image, offset) -> Detections:
        """
        Run the provided callback on a slice of an image.

        Args:
            image (np.ndarray): The input image on which inference needs to run
            offset (np.ndarray): An array of shape `(4,)` containing coordinates
                for the slice.

        Returns:
            Detections: A collection of detections for the slice.
        """
        image_slice = crop_image(image=image, xyxy=offset)
        detections = self.callback(image_slice)
        detections = move_detections(detections=detections, offset=offset[:2])

        return detections

    @staticmethod
    def _generate_offset(
        resolution_wh: Tuple[int, int],
        slice_wh: Tuple[int, int],
        overlap_ratio_wh: Tuple[float, float],
    ) -> np.ndarray:
        """
        Generate offset coordinates for slicing an image based on the given resolution,
        slice dimensions, and overlap ratios.

        Args:
            resolution_wh (Tuple[int, int]): A tuple representing the width and height
                of the image to be sliced.
            slice_wh (Tuple[int, int]): A tuple representing the desired width and
                height of each slice.
            overlap_ratio_wh (Tuple[float, float]): A tuple representing the desired
                overlap ratio for width and height between consecutive slices. Each
                value should be in the range [0, 1), where 0 means no overlap and a
                value close to 1 means high overlap.

        Returns:
            np.ndarray: An array of shape `(n, 4)` containing coordinates for each
                slice in the format `[xmin, ymin, xmax, ymax]`.

        Note:
            The function ensures that slices do not exceed the boundaries of the
                original image. As a result, the final slices in the row and column
                dimensions might be smaller than the specified slice dimensions if the
                image's width or height is not a multiple of the slice's width or
                height minus the overlap.
        """
        slice_width, slice_height = slice_wh
        image_width, image_height = resolution_wh
        overlap_ratio_width, overlap_ratio_height = overlap_ratio_wh

        width_stride = slice_width - int(overlap_ratio_width * slice_width)
        height_stride = slice_height - int(overlap_ratio_height * slice_height)

        ws = np.arange(0, image_width, width_stride)
        hs = np.arange(0, image_height, height_stride)

        xmin, ymin = np.meshgrid(ws, hs)
        xmax = np.clip(xmin + slice_width, 0, image_width)
        ymax = np.clip(ymin + slice_height, 0, image_height)

        offsets = np.stack([xmin, ymin, xmax, ymax], axis=-1).reshape(-1, 4)

        return offsets
