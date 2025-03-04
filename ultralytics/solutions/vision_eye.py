# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class VisionEye(BaseSolution):
    """
    A class to manage object detection and vision mapping in images or video streams.

    This class extends the BaseSolution class and provides functionality for detecting objects,
    mapping vision points, and annotating results with bounding boxes and labels.

    Methods:
        mapping: Processes the input image to detect objects, annotate them, and apply vision mapping.

    Examples:
        >>> vision_eye = VisionEye()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = vision_eye.mapping(frame)
        >>> print(f"Total detected instances: {results['total_tracks']}")
    """

    def __init__(self, **kwargs):
        """
        Initializes the VisionEye class for detecting objects and applying vision mapping.

        Attributes are inherited from the BaseSolution class and initialized using the provided configuration.
        """
        super().__init__(**kwargs)
        self.vision_point = kwargs.get(
            "vision_point", (30, 30)
        )  # visioneye point value, the point, where vision will view objects and draw tracks

    def process(self, im0):
        """
        Performs object detection, vision mapping, and annotation on the input image.

        Args:
            im0 (numpy.ndarray): The input image for detection and annotation.

        This method processes the input image, detects objects, applies vision mapping,
        and annotates each detected instance with a bounding box and label.

        Returns:
            results (SolutionResults): A summary dictionary containing the total number of tracked instances.

        Examples:
            >>> vision_eye = VisionEye()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = vision_eye.mapping(frame)
            >>> print(summary)
        """
        self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)
        annotator = SolutionAnnotator(im0, self.line_width)

        for cls, t_id, box in zip(self.clss, self.track_ids, self.boxes):
            # Annotate the image with bounding boxes, labels, and vision mapping
            annotator.box_label(box, label=self.names[cls], color=colors(int(t_id), True))
            annotator.visioneye(box, self.vision_point)
        plot_im = annotator.result()
        self.display_output(plot_im)  # Display the annotated output using the base class function

        # Return a SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
