# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class TrackZone(BaseSolution):
    """A class to manage region-based object tracking in a video stream.

    This class extends the BaseSolution class and provides functionality for tracking objects within a specific region
    defined by a polygonal area. Objects outside the region are excluded from tracking.

    Attributes:
        region (np.ndarray): The polygonal region for tracking, represented as a convex hull of points.
        line_width (int): Width of the lines used for drawing bounding boxes and region boundaries.
        names (list[str]): List of class names that the model can detect.
        boxes (list[np.ndarray]): Bounding boxes of tracked objects.
        track_ids (list[int]): Unique identifiers for each tracked object.
        clss (list[int]): Class indices of tracked objects.

    Methods:
        process: Process each frame of the video, applying region-based tracking.
        extract_tracks: Extract tracking information from the input frame.
        display_output: Display the processed output.

    Examples:
        >>> tracker = TrackZone()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = tracker.process(frame)
        >>> cv2.imshow("Tracked Frame", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the TrackZone class for tracking objects within a defined region in video streams.

        Args:
            **kwargs (Any): Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        default_region = [(75, 75), (565, 75), (565, 285), (75, 285)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))
        self.mask = None

    def process(self, im0: np.ndarray) -> SolutionResults:
        """Process the input frame to track objects within a defined region.

        This method initializes the annotator, creates a mask for the specified region, extracts tracks only from the
        masked area, and updates tracking information. Objects outside the region are ignored.

        Args:
            im0 (np.ndarray): The input image or frame to be processed.

        Returns:
            (SolutionResults): Contains processed image `plot_im` and `total_tracks` (int) representing the total number
                of tracked objects within the defined region.

        Examples:
            >>> tracker = TrackZone()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = tracker.process(frame)
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        if self.mask is None:  # Create a mask for the region
            self.mask = np.zeros_like(im0[:, :, 0])
            cv2.fillPoly(self.mask, [self.region], 255)
        masked_frame = cv2.bitwise_and(im0, im0, mask=self.mask)
        self.extract_tracks(masked_frame)

        # Draw the region boundary
        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)

        # Iterate over boxes, track ids, classes indexes list and draw bounding boxes
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf, track_id=track_id), color=colors(track_id, True)
            )

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return a SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
