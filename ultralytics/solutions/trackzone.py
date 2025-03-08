# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class TrackZone(BaseSolution):
    """
    A class to manage region-based object tracking in a video stream.

    This class extends the BaseSolution class and provides functionality for tracking objects within a specific region
    defined by a polygonal area. Objects outside the region are excluded from tracking. It supports dynamic initialization
    of the region, allowing either a default region or a user-specified polygon.

    Attributes:
        region (ndarray): The polygonal region for tracking, represented as a convex hull.

    Methods:
        trackzone: Processes each frame of the video, applying region-based tracking.

    Examples:
        >>> tracker = TrackZone()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = tracker.trackzone(frame)
        >>> cv2.imshow("Tracked Frame", processed_frame)
    """

    def __init__(self, **kwargs):
        """Initializes the TrackZone class for tracking objects within a defined region in video streams."""
        super().__init__(**kwargs)
        default_region = [(150, 150), (1130, 150), (1130, 570), (150, 570)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))

    def trackzone(self, im0):
        """
        Processes the input frame to track objects within a defined region.

        This method initializes the annotator, creates a mask for the specified region, extracts tracks
        only from the masked area, and updates tracking information. Objects outside the region are ignored.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed.

        Returns:
            (numpy.ndarray): The processed image with tracking id and bounding boxes annotations.

        Examples:
            >>> tracker = TrackZone()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> tracker.trackzone(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        # Create a mask for the region and extract tracks from the masked image
        masked_frame = cv2.bitwise_and(im0, im0, mask=cv2.fillPoly(np.zeros_like(im0[:, :, 0]), [self.region], 255))
        self.extract_tracks(masked_frame)

        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)

        # Iterate over boxes, track ids, classes indexes list and draw bounding boxes
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=f"{self.names[cls]}:{track_id}", color=colors(track_id, True))

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
