# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from time import time

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.solutions.solutions import BaseSolution


class SpeedEstimator(BaseSolution):
    """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

    def __init__(self, **kwargs):
        """Initializes the SpeedEstimator with the given parameters."""
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

        for box, t_id, cls in zip(self.boxes, self.tracks, self.clss):
            self.store_tracking_history(t_id, box)  # Store track history

            trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            if t_id not in self.trk_pt:
                self.trk_pt[t_id] = 0

            speed_label = f"{int(self.spd[t_id])} km/h" if t_id in self.spd else self.names[int(cls)]
            bbox_color = colors(int(t_id), True)

            annotator.box_label(box, speed_label, bbox_color)
            cv2.polylines(im0, [trk_pts], isClosed=False, color=bbox_color, thickness=self.tf)
            cv2.circle(im0, (int(track[-1][0]), int(track[-1][1])), self.tf * 2, bbox_color, -1)

            # Calculation of object speed
            if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
                return
            if self.reg_pts[1][1] - 10 < track[-1][1] < self.reg_pts[1][1] + 10:
                direction = "known"
            elif self.reg_pts[0][1] - 10 < track[-1][1] < self.reg_pts[0][1] + 10:
                direction = "known"
            else:
                direction = "unknown"

            if self.trk_pt.get(t_id) != 0 and direction != "unknown" and t_id not in self.trkd_ids:
                self.trkd_ids.append(t_id)

                time_difference = time() - self.trk_pt[t_id]
                if time_difference > 0:
                    self.spd[t_id] = np.abs(track[-1][1] - self.trk_pp[t_id][1]) / time_difference

            self.trk_pt[t_id] = time()
            self.trk_pp[t_id] = track[-1]

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
