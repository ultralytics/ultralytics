# Ultralytics YOLO 🚀, AGPL-3.0 license

import os

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectCropper(BaseSolution):
    """
    A class to manage the cropping of detected objects in a real-time video stream.

    This class extends the BaseSolution class and provides functionality for cropping objects based on detected bounding boxes. The cropped images are saved to a specified directory for further analysis or usage.

    Attributes:
        crop_directory (str): Directory where cropped object images are stored.
        crop_idx (int): Counter for the total number of cropped objects.

    Methods:
        crop: Crops detected objects from the input image and saves them to the output directory.

    Examples:
        >>> cropper = ObjectCropper()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = cropper.crop(frame)
        >>> print(f"Total cropped objects: {cropper.crop_idx}")
    """

    def __init__(self, **kwargs):
        """
        Initializes the ObjectCropper class for cropping objects from detected bounding boxes in video streams or
        images.

        Attributes:
            crop_directory (str): Path to the directory for saving cropped object images.
            crop_idx (int): Counter for the number of cropped objects, initialized to zero.
        """
        super().__init__(**kwargs)

        self.crop_directory = "cropped-detections"  # Directory for storing cropped detections
        if not os.path.exists(self.crop_directory):
            os.mkdir(self.crop_directory)  # Create directory if it does not exist

        self.crop_idx = 0  # Initialize counter for total cropped objects
        self.iou = self.CFG["iou"]
        self.conf = self.CFG["conf"] if self.CFG["conf"] is not None else 0.25

    def crop(self, im0):
        """
        Crops detected objects from the input image and saves them as separate images.

        Args:
            im0 (numpy.ndarray): The input image containing detected objects.

        This method uses the bounding box coordinates from the model's predictions to extract regions corresponding to detected objects. The cropped objects are saved as individual images in the `crop_directory`.

        Returns:
            results (dict): A summary dictionary containing the total number of cropped objects and processed image `im0`.

        Examples:
            >>> cropper = ObjectCropper()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = cropper.crop(frame)
            >>> print(summary)
        """
        annotator = SolutionAnnotator(im0)

        results = self.model.predict(im0, classes=self.classes, conf=self.conf, iou=self.iou, device="cpu")[0]
        boxes = results.boxes.xyxy.cpu().tolist()  # Detected bounding boxes list
        clss = results.boxes.cls.cpu().tolist()  # Detected classes list

        for box, cls in zip(boxes, clss):
            self.crop_idx += 1
            crop_object = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]  # Crop the detected object
            cv2.imwrite(
                os.path.join(self.crop_directory, f"crop-{self.crop_idx}.jpg"), crop_object
            )  # Save cropped image
            annotator.box_label(box, label=self.names[cls], color=colors(cls, True))  # Bounding box plot

        self.display_output(im0)  # display output with base class function

        # Return output dictionary with summary for usage
        return SolutionResults(
            im0=im0,
            total_crop_objects=self.crop_idx,
        ).summary(verbose=self.verbose)
