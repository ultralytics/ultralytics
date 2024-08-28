# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from time import time

import cv2
import numpy as np

from ultralytics import YOLO, solutions
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class SpeedEstimator:
    """A class to estimate the speed of objects in a real-time video stream using tracking data."""

    def __init__(self, **kwargs):
        """
        Initializes an instance of the SpeedEstimator class, setting up configurations for analyzing object speeds in
        real-time video streams.

        Args:
            kwargs (dict): Dictionary of arguments for configuring the speed estimation process, such as frame rate, calibration data, and measurement units.
        """
        DEFAULT_CFG_DICT.update(kwargs)
        self.model = YOLO(DEFAULT_CFG_DICT["model"])
        self.im0 = None
        self.annotator = None
        self.clss = None
        self.boxes = None
        self.trk_ids = None
        self.trk_pts = None
        self.trk_history = defaultdict(list)

        # Speed estimation information
        self.dist_data = {}
        self.trk_idslist = []
        self.trk_previous_times = {}
        self.trk_previous_points = {}
        self.env_check = check_imshow(warn=True)  # Check if the environment supports imshow
        print(f"Ultralytics Solutions ✅ {DEFAULT_CFG_DICT}")

    def store_track_info(self, track_id, box):
        """
        Saves tracking data.

        Args:
            track_id (int): The identifier for the object's track.
            box (list): The bounding box data for the object.

        Returns:
            track (list): The updated tracking history for the specified track_id.
        """
        track = self.trk_history[track_id]
        track.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))

        if len(track) > 30:
            track.pop(0)

        self.trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        return track

    def plot_box_and_track(self, track_id, box, cls, track):
        """
        Draws the track and bounding box.

        Args:
            track_id (int): The identifier for the object's track.
            box (list): The bounding box data for the object.
            cls (str): The name of the object's class.
            track (list): The tracking history used to draw the track path.
        """
        speed_label = (
            f"{int(self.dist_data[track_id])} km/h" if track_id in self.dist_data else self.model.names[int(cls)]
        )
        bbox_color = colors(int(track_id))

        self.annotator.box_label(box, speed_label, bbox_color)
        cv2.polylines(
            self.im0, [self.trk_pts], isClosed=False, color=bbox_color, thickness=DEFAULT_CFG_DICT["line_width"]
        )
        cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), DEFAULT_CFG_DICT["line_width"] * 2, bbox_color, -1)

    def calculate_speed(self, trk_id, track):
        """
        Computes the speed of an object.

        Args:
            trk_id (int): The identifier for the object's track.
            track (list): The tracking history used to plot the track path.
        """
        if not DEFAULT_CFG_DICT["reg_pts"][0][0] < track[-1][0] < DEFAULT_CFG_DICT["reg_pts"][1][0]:
            return
        if (
            DEFAULT_CFG_DICT["reg_pts"][1][1] - 10
            < track[-1][1]
            < DEFAULT_CFG_DICT["reg_pts"][1][1] + 10
        ):
            direction = "known"
        elif (
            DEFAULT_CFG_DICT["reg_pts"][0][1] - 10
            < track[-1][1]
            < DEFAULT_CFG_DICT["reg_pts"][0][1] + 10
        ):
            direction = "known"
        else:
            direction = "unknown"

        if self.trk_previous_times.get(trk_id) != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
            self.trk_idslist.append(trk_id)

            time_difference = time() - self.trk_previous_times[trk_id]
            if time_difference > 0:
                dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                self.dist_data[trk_id] = dist_difference / time_difference

        self.trk_previous_times[trk_id] = time()
        self.trk_previous_points[trk_id] = track[-1]

    def estimate_speed(self, im0):
        """
        Estimates the speed of objects using tracking data.

        Args:
            im0 (ndarray): The image.

        Returns:
            im0 (ndarray): The image with annotated boxes and tracks.
        """
        self.im0 = im0
        tracks = self.model.track(
            source=im0,
            persist=True,
            tracker=DEFAULT_CFG_DICT["tracker"],
            classes=DEFAULT_CFG_DICT["classes"],
            iou=DEFAULT_CFG_DICT["iou"],
            conf=DEFAULT_CFG_DICT["conf"],
        )
        self.boxes, self.clss, self.trk_ids = solutions.extract_tracks(tracks)
        if self.trk_ids is not None:
            self.annotator = Annotator(self.im0, line_width=DEFAULT_CFG_DICT["line_width"])
            self.annotator.draw_region(
                reg_pts=DEFAULT_CFG_DICT["reg_pts"],
                color=(104, 31, 17),
                thickness=int(DEFAULT_CFG_DICT["line_width"]) * 2,
            )

            for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
                track = self.store_track_info(trk_id, box)

                if trk_id not in self.trk_previous_times:
                    self.trk_previous_times[trk_id] = 0

                self.plot_box_and_track(trk_id, box, cls, track)
                self.calculate_speed(trk_id, track)

        if DEFAULT_CFG_DICT["show"] and self.env_check:
            cv2.imshow("Ultralytics Solutions", self.im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return self.im0


if __name__ == "__main__":
    names = {0: "person", 1: "car"}  # example class names
    speed_estimator = SpeedEstimator(names=names)
