# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class InstanceSegmentation(BaseSolution):
    """
    A class to manage instance segmentation in images or video streams.

    This class extends the BaseSolution class and provides functionality for performing instance segmentation, including drawing segmented masks with bounding boxes and labels.

    Methods:
        segment: Processes the input image to perform instance segmentation and annotate results.

    Examples:
        >>> segmenter = InstanceSegmentation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = segmenter.segment(frame)
        >>> print(f"Total segmented instances: {results['total_tracks']}")
    """

    def __init__(self, **kwargs):
        """
        Initializes the InstanceSegmentation class for detecting and annotating segmented instances.

        Attributes are inherited from the BaseSolution class and initialized using the provided configuration.
        """
        super().__init__(**kwargs)

    def segment(self, im0):
        """
        Performs instance segmentation on the input image and annotates the results.

        Args:
            im0 (numpy.ndarray): The input image for segmentation.

        This method processes the input image, applies segmentation masks, and annotates each segmented instance with a bounding box and label.

        Returns:
            results (dict): A summary dictionary containing the total number of tracked instances.

        Examples:
            >>> segmenter = InstanceSegmentation()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = segmenter.segment(frame)
            >>> print(summary)
        """
        annotator = SolutionAnnotator(im0, self.line_width)

        self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)

        # Iterate over detected classes, track IDs, and segmentation masks
        if self.masks is None:
            self.LOGGER.warning("⚠️ No masks detected! Ensure you're using a supported Ultralytics segmentation model.")
        else:
            for cls, t_id, mask in zip(self.clss, self.track_ids, self.masks):
                # Annotate the image with segmentation mask, mask color, and label
                annotator.seg_bbox(mask=mask, mask_color=colors(t_id, True), label=self.names[cls])

        self.display_output(im0)  # Display the annotated output using the base class function

        # Return a summary dictionary with the total count of tracks
        return SolutionResults(im0=im0, total_tracks=len(self.track_ids)).summary(verbose=self.verbose)
