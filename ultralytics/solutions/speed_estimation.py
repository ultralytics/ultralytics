# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from time import time

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class SpeedEstimator:
    """A class to estimation speed of objects in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the speed-estimator class with default values for Visual, Image, track and speed parameters."""

        # Visual & im0 information
        self.im0 = None
        self.annotator = None
        self.view_img = False

        # Region information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.region_thickness = 3

        # Predict/track information
        self.clss = None
        self.names = None
        self.boxes = None
        self.trk_ids = None
        self.trk_pts = None
        self.line_thickness = 2
        self.trk_history = defaultdict(list)

        # Speed estimator information
        self.current_time = 0
        self.dist_data = {}
        self.trk_idslist = []
        self.spdl_dist_thresh = 10
        self.trk_previous_times = {}
        self.trk_previous_points = {}

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        reg_pts,
        names,
        view_img=False,
        line_thickness=2,
        region_thickness=5,
        spdl_dist_thresh=10,
    ):
        """
        Configures the speed estimation and display parameters.

        Args:
            reg_pts (list): Initial list of points defining the speed calculation region.
            names (dict): object detection classes names
            view_img (bool): Flag indicating frame display
            line_thickness (int): Line thickness for bounding boxes.
            region_thickness (int): Speed estimation region thickness
            spdl_dist_thresh (int): Euclidean distance threshold for speed line
        """
        if reg_pts is None:
            print("Region points not provided, using default values")
        else:
            self.reg_pts = reg_pts
        self.names = names
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.region_thickness = region_thickness
        self.spdl_dist_thresh = spdl_dist_thresh

    def extract_tracks(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def store_track_info(self, track_id, box):
        """
        Store track data.

        Args:
            track_id (int): object track id.
            box (list): object bounding box data
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
        Plot track and bounding box.

        Args:
            track_id (int): object track id.
            box (list): object bounding box data
            cls (str): object class name
            track (list): tracking history for tracks path drawing
        """
        speed_label = (
            str(int(self.dist_data[track_id])) + "km/ph" if track_id in self.dist_data else self.names[int(cls)]
        )
        bbox_color = colors(int(track_id)) if track_id in self.dist_data else (255, 0, 255)

        self.annotator.box_label(box, speed_label, bbox_color)

        cv2.polylines(self.im0, [self.trk_pts], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)

    def calculate_speed(self, trk_id, track):
        """
        Calculation of object speed
        Args:
            trk_id (int): object track id.
            track (list): tracking history for tracks path drawing
        """

        if self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
            if self.reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[1][1] + self.spdl_dist_thresh:
                direction = "known"

            elif self.reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[0][1] + self.spdl_dist_thresh:
                direction = "known"

            else:
                direction = "unknown"

            if self.trk_previous_times[trk_id] != 0 and direction != "unknown":
                if trk_id not in self.trk_idslist:
                    self.trk_idslist.append(trk_id)

                    time_difference = time() - self.trk_previous_times[trk_id]
                    if time_difference > 0:
                        dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                        speed = dist_difference / time_difference
                        self.dist_data[trk_id] = speed

            self.trk_previous_times[trk_id] = time()
            self.trk_previous_points[trk_id] = track[-1]

    def estimate_speed(self, im0, tracks):
        """
        Calculate object based on tracking data
        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img and self.env_check:
                self.display_frames()
                return
            else:
                return

        self.extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=2)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=(255, 0, 0), thickness=self.region_thickness)

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            track = self.store_track_info(trk_id, box)

            if trk_id not in self.trk_previous_times:
                self.trk_previous_times[trk_id] = 0

            self.plot_box_and_track(trk_id, box, cls, track)
            self.calculate_speed(trk_id, track)

        if self.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self):
        """Display frame."""
        cv2.imshow("Ultralytics Speed Estimation", self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    SpeedEstimator()
