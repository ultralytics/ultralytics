# Ultralytics YOLO 🚀, AGPL-3.0 license

from time import time

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, LineString
from ultralytics.utils.plotting import Annotator, colors


class SpeedEstimator(BaseSolution):
    """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

    def __init__(self, **kwargs):
        """Initializes the SpeedEstimator with the given parameters."""
        super().__init__(**kwargs)

        self.initialize_region()  # Initialize speed region

        self.spd = {}  # set for speed data
        self.trkd_ids = []  # list for already speed_estimated and tracked ID's
        self.trk_pt = {}  # set for tracks previous time
        self.trk_pp = {}  # set for tracks previous point

    def estimate_speed(self, im0):
        """
        Estimates the speed of objects based on tracking data.

        Args:
            im0 (ndarray): The input image that will be used for processing
        Returns
            im0 (ndarray): The processed image for more usage
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # Draw region

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # Store track history

            # Check if track_id is already in self.trk_pp or trk_pt initialize if not
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
            self.annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # Draw bounding box

            # Draw tracks of objects
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            # Calculate object speed and direction based on region intersection
            if LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.l_s):
                direction = "known"
            else:
                direction = "unknown"

            # Perform speed calculation and tracking updates if direction is valid
            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    self.spd[track_id] = np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1]) / time_difference

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
