# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

from ultralytics.solutions.solutions import BaseSolution, SolutionResults
from ultralytics.utils.plotting import save_one_box

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
            crop_dir (str): Path to the directory for saving cropped object images.
            crop_idx (int): Counter for the number of cropped objects, initialized to zero.
        """
        super().__init__(**kwargs)

        self.crop_dir = kwargs.get("crop_dir", "cropped-detections")  # Directory for storing cropped detections
        if not os.path.exists(self.crop_dir):
            os.mkdir(self.crop_dir)  # Create directory if it does not exist
        if self.CFG["show"]:
            self.LOGGER.info(f"⚠️ show=True disabled for crop solution, results will be saved in the directory named: "
                    f"{self.crop_dir}")
        self.crop_idx = 0  # Initialize counter for total cropped objects
        self.iou = self.CFG["iou"]
        self.conf = self.CFG["conf"] if self.CFG["conf"] is not None else 0.25

    def process(self, im0):
        """
        Crops detected objects from the input image and saves them as separate images.

        Args:
            im0 (numpy.ndarray): The input image containing detected objects.

        This method uses the bounding box coordinates from the model's predictions to extract regions corresponding to detected objects. The cropped objects are saved as individual images in the `crop_directory`.

        Returns:
            results (SolutionResults): A SolutionResults object containing the total number of cropped objects and processed image `im0`.

        Examples:
            >>> cropper = ObjectCropper()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = cropper.crop(frame)
            >>> print(summary)
        """
        results = self.model.predict(
            im0, classes=self.classes, conf=self.conf, iou=self.iou, device=self.CFG["device"]
        )[0]

        for box in results.boxes:
            self.crop_idx += 1
            save_one_box(
                box.xyxy,
                im0,
                file=Path(self.crop_dir) / f"crop_{self.crop_idx}.jpg",
                BGR=True,
            )

        # Return SolutionResults
        return SolutionResults(plot_im=im0, total_crop_objects=self.crop_idx)
