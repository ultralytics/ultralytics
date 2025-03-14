# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class VisionEye(BaseSolution):
    """
    A class to manage object detection and vision mapping in images or video streams.

    This class extends the BaseSolution class and provides functionality for detecting objects,
    mapping vision points, and annotating results with bounding boxes and labels.

    Attributes:
        vision_point (Tuple[int, int]): Coordinates (x, y) where vision will view objects and draw tracks.

    Methods:
        process: Process the input image to detect objects, annotate them, and apply vision mapping.

    Examples:
        >>> vision_eye = VisionEye()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = vision_eye.process(frame)
        >>> print(f"Total detected instances: {results.total_tracks}")
    """

    def __init__(self, **kwargs):
        """
        Initialize the VisionEye class for detecting objects and applying vision mapping.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent class and for configuring vision_point.
        """
        super().__init__(**kwargs)
        # Set the vision point where the system will view objects and draw tracks
        self.vision_point = kwargs.get("vision_point", (30, 30))

    def process(self, im0):
        """
        Perform object detection, vision mapping, and annotation on the input image.

        Args:
            im0 (numpy.ndarray): The input image for detection and annotation.

        Returns:
            (SolutionResults): Object containing the annotated image and tracking statistics.
                - plot_im: Annotated output image with bounding boxes and vision mapping
                - total_tracks: Number of tracked objects in the frame

        Examples:
            >>> vision_eye = VisionEye()
            >>> frame = cv2.imread("image.jpg")
            >>> results = vision_eye.process(frame)
            >>> print(f"Detected {results.total_tracks} objects")
        """
        self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)
        annotator = SolutionAnnotator(im0, self.line_width)

        for cls, t_id, box in zip(self.clss, self.track_ids, self.boxes):
            # Annotate the image with bounding boxes, labels, and vision mapping
            annotator.box_label(box, label=self.names[cls], color=colors(int(t_id), True))
            annotator.visioneye(box, self.vision_point)

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display the annotated output using the base class function

        # Return a SolutionResults object with the annotated image and tracking statistics
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
