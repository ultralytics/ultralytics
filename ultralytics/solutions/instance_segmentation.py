# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class InstanceSegmentation(BaseSolution):
    """
    A class to manage instance segmentation in images or video streams.

    This class extends the BaseSolution class and provides functionality for performing instance segmentation, including
    drawing segmented masks with bounding boxes and labels.

    Attributes:
        model (str): The segmentation model to use for inference.

    Methods:
        process: Processes the input image to perform instance segmentation and annotate results.

    Examples:
        >>> segmenter = InstanceSegmentation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = segmenter.segment(frame)
        >>> print(f"Total segmented instances: {results['total_tracks']}")
    """

    def __init__(self, **kwargs):
        """
        Initializes the InstanceSegmentation class for detecting and annotating segmented instances.

        Args:
            **kwargs (Any): Keyword arguments passed to the BaseSolution parent class.
                model (str): Model name or path, defaults to "yolo11n-seg.pt".
        """
        kwargs["model"] = kwargs.get("model", "yolo11n-seg.pt")
        super().__init__(**kwargs)

    def process(self, im0):
        """
        Performs instance segmentation on the input image and annotates the results.

        Args:
            im0 (numpy.ndarray): The input image for segmentation.

        Returns:
            (SolutionResults): Object containing the annotated image and total number of tracked instances.

        Examples:
            >>> segmenter = InstanceSegmentation()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = segmenter.segment(frame)
            >>> print(summary)
        """
        self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)
        annotator = SolutionAnnotator(im0, self.line_width)

        # Iterate over detected classes, track IDs, and segmentation masks
        if self.masks is None:
            self.LOGGER.warning("⚠️ No masks detected! Ensure you're using a supported Ultralytics segmentation model.")
        else:
            for cls, t_id, mask in zip(self.clss, self.track_ids, self.masks):
                # Annotate the image with segmentation mask, mask color, and label
                annotator.segmentation_mask(mask=mask, mask_color=colors(t_id, True), label=self.names[cls])

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display the annotated output using the base class function

        # Return SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
