# Ultralytics YOLO 🚀, AGPL-3.0 license


import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectBlurrer(BaseSolution):
    """
    A class to manage the blurring of detected objects in a real-time video stream.

    This class extends the BaseSolution class and provides functionality for blurring objects based on detected bounding boxes. The blurred areas are updated directly in the input image, allowing for privacy preservation or other effects.

    Attributes:
        blur_ratio (int): The intensity of the blur effect applied to detected objects.
        iou (float): Intersection over Union threshold for object detection.
        conf (float): Confidence threshold for object detection.

    Methods:
        blur: Applies a blurring effect to detected objects in the input image.

    Examples:
        >>> blurrer = ObjectBlurrer()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = blurrer.blur(frame)
        >>> print(f"Total blurred objects: {processed_frame['total_tracks']}")
    """

    def __init__(self, **kwargs):
        """
        Initializes the ObjectBlurrer class for applying a blur effect to objects detected in video streams or images.

        Attributes:
            blur_ratio (int): Intensity of the blur effect, derived from the configuration.
        """
        super().__init__(**kwargs)

        self.blur_ratio = int(self.CFG["blur_ratio"] * 100)  # Blur intensity multiplier

    def blur(self, im0):
        """
        Applies a blurring effect to detected objects in the input image.

        Args:
            im0 (numpy.ndarray): The input image containing detected objects.

        This method uses the bounding box coordinates from the model's predictions to apply a blurring effect to the regions corresponding to detected objects. The processed image is updated in place.

        Returns:
            results (dict): A summary dictionary containing the total number of blurred objects.

        Examples:
            >>> blurrer = ObjectBlurrer()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = blurrer.blur(frame)
            >>> print(summary)
        """
        annotator = SolutionAnnotator(im0, self.line_width)

        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes and classes
        for box, cls in zip(self.boxes, self.clss):
            # Crop and blur the detected object
            blur_obj = cv2.blur(
                im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])],
                (self.blur_ratio, self.blur_ratio),
            )
            # Update the blurred area in the original image
            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj
            annotator.box_label(box, label=self.names[cls], color=colors(cls, True))  # Annotate bounding box

        self.display_output(im0)  # Display the output using the base class function

        # Return a summary dictionary for usage
        return SolutionResults(im0=im0, total_tracks=len(self.track_ids)).summary(verbose=self.verbose)
