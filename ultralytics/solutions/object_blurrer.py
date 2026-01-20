# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class ObjectBlurrer(BaseSolution):
    """A class to manage the blurring of detected objects in a real-time video stream.

    This class extends the BaseSolution class and provides functionality for blurring objects based on detected bounding
    boxes. The blurred areas are updated directly in the input image, allowing for privacy preservation or other effects.

    Attributes:
        blur_ratio (int): The intensity of the blur effect applied to detected objects (higher values create more blur).
        iou (float): Intersection over Union threshold for object detection.
        conf (float): Confidence threshold for object detection.

    Methods:
        process: Apply a blurring effect to detected objects in the input image.
        extract_tracks: Extract tracking information from detected objects.
        display_output: Display the processed output image.

    Examples:
        >>> blurrer = ObjectBlurrer()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_results = blurrer.process(frame)
        >>> print(f"Total blurred objects: {processed_results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the ObjectBlurrer class for applying a blur effect to objects detected in video streams or images.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent class and for configuration including:
                - blur_ratio (float): Intensity of the blur effect (0.1-1.0, default=0.5).
        """
        super().__init__(**kwargs)
        blur_ratio = self.CFG["blur_ratio"]
        if blur_ratio < 0.1:
            LOGGER.warning("blur ratio cannot be less than 0.1, updating it to default value 0.5")
            blur_ratio = 0.5
        self.blur_ratio = int(blur_ratio * 100)

    def process(self, im0) -> SolutionResults:
        """Apply a blurring effect to detected objects in the input image.

        This method extracts tracking information, applies blur to regions corresponding to detected objects, and
        annotates the image with bounding boxes.

        Args:
            im0 (np.ndarray): The input image containing detected objects.

        Returns:
            (SolutionResults): Object containing the processed image and number of tracked objects.
                - plot_im (np.ndarray): The annotated output image with blurred objects.
                - total_tracks (int): The total number of tracked objects in the frame.

        Examples:
            >>> blurrer = ObjectBlurrer()
            >>> frame = cv2.imread("image.jpg")
            >>> results = blurrer.process(frame)
            >>> print(f"Blurred {results.total_tracks} objects")
        """
        self.extract_tracks(im0)  # Extract tracks
        annotator = SolutionAnnotator(im0, self.line_width)

        # Iterate over bounding boxes and classes
        for box, cls, conf in zip(self.boxes, self.clss, self.confs):
            # Crop and blur the detected object
            blur_obj = cv2.blur(
                im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])],
                (self.blur_ratio, self.blur_ratio),
            )
            # Update the blurred area in the original image
            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf), color=colors(cls, True)
            )  # Annotate bounding box

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display the output using the base class function

        # Return a SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
