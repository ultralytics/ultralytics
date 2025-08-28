# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.engine.results import Results
from ultralytics.solutions.solutions import BaseSolution, SolutionResults


class InstanceSegmentation(BaseSolution):
    """
    A class to manage instance segmentation in images or video streams.

    This class extends the BaseSolution class and provides functionality for performing instance segmentation, including
    drawing segmented masks with bounding boxes and labels.

    Attributes:
        model (str): The segmentation model to use for inference.
        line_width (int): Width of the bounding box and text lines.
        names (Dict[int, str]): Dictionary mapping class indices to class names.
        clss (List[int]): List of detected class indices.
        track_ids (List[int]): List of track IDs for detected instances.
        masks (List[np.ndarray]): List of segmentation masks for detected instances.
        show_conf (bool): Whether to display confidence scores.
        show_labels (bool): Whether to display class labels.
        show_boxes (bool): Whether to display bounding boxes.

    Methods:
        process: Process the input image to perform instance segmentation and annotate results.
        extract_tracks: Extract tracks including bounding boxes, classes, and masks from model predictions.

    Examples:
        >>> segmenter = InstanceSegmentation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = segmenter.process(frame)
        >>> print(f"Total segmented instances: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the InstanceSegmentation class for detecting and annotating segmented instances.

        Args:
            **kwargs (Any): Keyword arguments passed to the BaseSolution parent class.
                model (str): Model name or path, defaults to "yolo11n-seg.pt".
        """
        kwargs["model"] = kwargs.get("model", "yolo11n-seg.pt")
        super().__init__(**kwargs)

        self.show_conf = self.CFG.get("show_conf", True)
        self.show_labels = self.CFG.get("show_labels", True)
        self.show_boxes = self.CFG.get("show_boxes", True)

    def process(self, im0) -> SolutionResults:
        """
        Perform instance segmentation on the input image and annotate the results.

        Args:
            im0 (np.ndarray): The input image for segmentation.

        Returns:
            (SolutionResults): Object containing the annotated image and total number of tracked instances.

        Examples:
            >>> segmenter = InstanceSegmentation()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = segmenter.process(frame)
            >>> print(summary)
        """
        self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)
        self.masks = getattr(self.tracks, "masks", None)

        # Iterate over detected classes, track IDs, and segmentation masks
        if self.masks is None:
            self.LOGGER.warning("No masks detected! Ensure you're using a supported Ultralytics segmentation model.")
            plot_im = im0
        else:
            results = Results(im0, path=None, names=self.names, boxes=self.track_data.data, masks=self.masks.data)
            plot_im = results.plot(
                line_width=self.line_width,
                boxes=self.show_boxes,
                conf=self.show_conf,
                labels=self.show_labels,
                color_mode="instance",
            )

        self.display_output(plot_im)  # Display the annotated output using the base class function

        # Return SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
