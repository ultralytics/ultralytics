# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from pathlib import Path
from time import time

import cv2
import numpy as np

from ultralytics.cfg import get_cfg
from ultralytics.solutions.cfg import extract_cfg_data
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

FILE = Path(__file__).resolve()  # get path of file


class SpeedEstimator:
    """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

    def __init__(self, **kwargs):
        """Initializes the SpeedEstimator with the kwargs arguments."""
        self.args = get_cfg(extract_cfg_data(FILE))
        if "names" not in kwargs:
            raise ValueError("Error: Classes names 'names' argument is required")
        for key, value in kwargs.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            else:
                print(f"Warning: Unknown argument Skipping!!! {key}")

        self.im0 = None
        self.annotator = None
        self.clss = None
        self.boxes = None
        self.trk_ids = None
        self.trk_pts = None
        self.trk_history = defaultdict(list)

        # Speed estimation information
        self.current_time = 0
        self.dist_data = {}
        self.trk_idslist = []
        self.trk_previous_times = {}
        self.trk_previous_points = {}
        self.env_check = check_imshow(warn=True)  # Check if the environment supports imshow

    def extract_tracks(self, tracks):
        """
        Extracts results from the provided tracking data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def store_track_info(self, track_id, box):
        """
        Stores track data.

        Args:
            track_id (int): Object track id.
            box (list): Object bounding box data.

        Returns:
            (list): Updated tracking history for the given track_id.
        """
        track = self.trk_history[track_id]
        bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
        track.append(bbox_center)

        if len(track) > 30:
            track.pop(0)

        self.trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        return track

    def plot_box_and_track(self, track_id, box, cls, track):
        """
        Plots track and bounding box.

        Args:
            track_id (int): Object track id.
            box (list): Object bounding box data.
            cls (str): Object class name.
            track (list): Tracking history for drawing tracks path.
        """
        speed_label = (
            f"{int(self.dist_data[track_id])} km/h" if track_id in self.dist_data else self.args.names[int(cls)]
        )
        bbox_color = colors(int(track_id)) if track_id in self.dist_data else (255, 0, 255)

        self.annotator.box_label(box, speed_label, bbox_color)
        cv2.polylines(self.im0, [self.trk_pts], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)

    def calculate_speed(self, trk_id, track):
        """
        Calculates the speed of an object.

        Args:
            trk_id (int): Object track id.
            track (list): Tracking history for drawing tracks path.
        """
        if not self.args.reg_pts[0][0] < track[-1][0] < self.args.reg_pts[1][0]:
            return
        if (
            self.args.reg_pts[1][1] - self.args.spdl_dist_thresh
            < track[-1][1]
            < self.args.reg_pts[1][1] + self.args.spdl_dist_thresh
        ):
            direction = "known"
        elif (
            self.args.reg_pts[0][1] - self.args.spdl_dist_thresh
            < track[-1][1]
            < self.args.reg_pts[0][1] + self.args.spdl_dist_thresh
        ):
            direction = "known"
        else:
            direction = "unknown"

        if self.trk_previous_times.get(trk_id) != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
            self.trk_idslist.append(trk_id)

            time_difference = time() - self.trk_previous_times[trk_id]
            if time_difference > 0:
                dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                speed = dist_difference / time_difference
                self.dist_data[trk_id] = speed

        self.trk_previous_times[trk_id] = time()
        self.trk_previous_points[trk_id] = track[-1]

    def estimate_speed(self, im0, tracks, region_color=(255, 0, 0)):
        """
        Estimates the speed of objects based on tracking data.

        Args:
            im0 (ndarray): Image.
            tracks (list): List of tracks obtained from the object tracking process.
            region_color (tuple, optional): Color to use when drawing regions. Defaults to (255, 0, 0).

        Returns:
            (ndarray): The image with annotated boxes and tracks.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.args.view_img and self.env_check:
                self.display_frames()
            return im0

        self.extract_tracks(tracks)
        self.annotator = Annotator(self.im0, line_width=self.args.line_thickness)
        self.annotator.draw_region(reg_pts=self.args.reg_pts, color=region_color, thickness=self.args.region_thickness)

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            track = self.store_track_info(trk_id, box)

            if trk_id not in self.trk_previous_times:
                self.trk_previous_times[trk_id] = 0

            self.plot_box_and_track(trk_id, box, cls, track)
            self.calculate_speed(trk_id, track)

        if self.args.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self):
        """Displays the current frame."""
        cv2.imshow("Ultralytics Speed Estimation", self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    names = {0: "person", 1: "car"}  # example class names
    speed_estimator = SpeedEstimator(names=names)
