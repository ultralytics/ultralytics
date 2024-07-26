# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
from pathlib import Path

import cv2

from ultralytics.cfg import get_cfg
from ultralytics.solutions.cfg import extract_cfg_data
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

FILE = Path(__file__).resolve()  # get path of file


class DistanceCalculation:
    """A class to calculate distance between two objects in a real-time video stream based on their tracks."""

    def __init__(self, **kwargs):
        """Initializes the DistanceCalculation class with the kwargs arguments."""
        import ast

        self.args = get_cfg(extract_cfg_data(FILE))
        if "names" not in kwargs:
            raise ValueError("Error: Classes names 'names' argument is required")
        for key, value in kwargs.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            else:
                print(f"Warning: Unknown argument Skipping!!! {key}")
        self.annotator = None

        # Prediction & tracking information
        self.clss = None
        self.boxes = None
        self.trk_ids = None
        self.centroids = []

        # Mouse event information
        self.left_mouse_count = 0
        self.selected_boxes = {}

        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow
        self.args.line_color = ast.literal_eval(self.args.line_color)
        self.args.centroid_color = ast.literal_eval(self.args.centroid_color)

    def mouse_event_for_distance(self, event, x, y, flags, param):
        """
        Handles mouse events to select regions in a real-time video stream.

        Args:
            event (int): Type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): X-coordinate of the mouse pointer.
            y (int): Y-coordinate of the mouse pointer.
            flags (int): Flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY, cv2.EVENT_FLAG_SHIFTKEY, etc.).
            param (dict): Additional parameters passed to the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.trk_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = box

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def extract_tracks(self, tracks):
        """
        Extracts tracking results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    @staticmethod
    def calculate_centroid(box):
        """
        Calculates the centroid of a bounding box.

        Args:
            box (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            (tuple): Centroid coordinates (x, y).
        """
        return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

    def calculate_distance(self, centroid1, centroid2):
        """
        Calculates the distance between two centroids.

        Args:
            centroid1 (tuple): Coordinates of the first centroid (x, y).
            centroid2 (tuple): Coordinates of the second centroid (x, y).

        Returns:
            (tuple): Distance in meters and millimeters.
        """
        pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
        distance_m = pixel_distance / self.args.pixels_per_meter
        distance_mm = distance_m * 1000
        return distance_m, distance_mm

    def start_process(self, im0, tracks):
        """
        Processes the video frame and calculates the distance between two bounding boxes.

        Args:
            im0 (ndarray): The image frame.
            tracks (list): List of tracks obtained from the object tracking process.

        Returns:
            (ndarray): The processed image frame.
        """
        if tracks[0].boxes.id is not None:
            self.extract_tracks(tracks)
            self.annotator = Annotator(im0, line_width=self.args.line_thickness)

            for box, cls, track_id in zip(self.boxes, self.clss, self.trk_ids):
                self.annotator.box_label(box, color=colors(int(cls), True), label=self.args.names[int(cls)])

                if len(self.selected_boxes) == 2:
                    for trk_id in self.selected_boxes.keys():
                        if trk_id == track_id:
                            self.selected_boxes[track_id] = box

            if len(self.selected_boxes) == 2:
                self.centroids = [
                    self.calculate_centroid(self.selected_boxes[trk_id]) for trk_id in self.selected_boxes
                ]

                distance_m, distance_mm = self.calculate_distance(self.centroids[0], self.centroids[1])
                self.annotator.plot_distance_and_line(
                    distance_m, distance_mm, self.centroids, self.args.line_color, self.args.centroid_color
                )

            self.centroids = []

        # Displays the current frame with annotations
        if self.args.view_img and self.env_check:
            cv2.namedWindow(self.args.window_name)
            cv2.setMouseCallback(self.args.window_name, self.mouse_event_for_distance)
            cv2.imshow(self.args.window_name, im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0


if __name__ == "__main__":
    names = {0: "person", 1: "car"}  # example class names
    distance_calculation = DistanceCalculation(names=names)
