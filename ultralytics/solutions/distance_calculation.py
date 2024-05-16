# Ultralytics YOLO 🚀, AGPL-3.0 license

import math

import cv2

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class DistanceCalculation:
    """A class to calculate distance between two objects in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the distance calculation class with default values for Visual, Image, track and distance
        parameters.
        """
        self.im0 = None
        self.annotator = None
        self.window_name = "Ultralytics Distance Calculator"

    def extract_tracks(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def calculate_centroid(self, box):
        """
        Calculate the centroid of bounding box.

        Args:
            box (list): Bounding box data
        """
        return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

    def calculate_distance(self, centroid1, centroid2):
        """
        Calculate distance between two centroids.

        Args:
            centroid1 (point): First bounding box data
            centroid2 (point): Second bounding box data
        """
        pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
        return pixel_distance / self.pixel_per_meter, (pixel_distance / self.pixel_per_meter) * 1000

    def start_process(self, im0, tracks):
        """
        Calculate distance between two bounding boxes based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img:
                self.display_frames()
            return
        self.extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=2)

        for box, cls, track_id in zip(self.boxes, self.clss, self.trk_ids):
            self.annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

            if len(self.selected_boxes) == 2:
                for trk_id, _ in self.selected_boxes.items():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            for trk_id, box in self.selected_boxes.items():
                centroid = self.calculate_centroid(self.selected_boxes[trk_id])
                self.centroids.append(centroid)

            distance_m, distance_mm = self.calculate_distance(self.centroids[0], self.centroids[1])
            self.annotator.plot_distance_and_line(
                distance_m, distance_mm, self.centroids, self.line_color, self.line_color
            )

        self.centroids = []

        if self.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self):
        """Display frame."""
        cv2.namedWindow("Ultralytics Distance Estimation")
        cv2.setMouseCallback("Ultralytics Distance Estimation", self.mouse_event_for_distance)
        cv2.imshow("Ultralytics Distance Estimation", self.im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    DistanceCalculation()
