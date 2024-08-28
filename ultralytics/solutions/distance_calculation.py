# Ultralytics YOLO 🚀, AGPL-3.0 license

import math

import cv2

from ultralytics import YOLO, solutions
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class DistanceCalculation:
    """A class for calculating the distance between two objects in a real-time video stream using their tracking
    data.
    """

    def __init__(self, **kwargs):
        """
        Initializes an instance of the DistanceCalculation class, setting up configurations for tracking and calculating
        distances between objects.

        Args:
            kwargs (dict): Dictionary of arguments for configuring the distance calculation process, such as parameters for object detection, tracking precision, and measurement units.
        """
        import ast

        DEFAULT_CFG_DICT.update(kwargs)
        self.model = YOLO(DEFAULT_CFG_DICT["model"])
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
        print(f"Ultralytics Solutions ✅ {DEFAULT_CFG_DICT}")

    def mouse_event_for_distance(self, event, x, y, flags, param):
        """
        Manages mouse events for selecting regions in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN).
            x (int): The X-coordinate of the mouse pointer.
            y (int): The Y-coordinate of the mouse pointer.
            flags (int): Flags related to the event (e.g., cv2.EVENT_FLAG_CTRLKEY, cv2.EVENT_FLAG_SHIFTKEY).
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

    def start_process(self, im0):
        """
        Processes a video frame to compute the distance between two bounding boxes.

        Args:
            im0 (ndarray): The image frame.

        Returns:
            im0 (ndarray): The processed image frame.
        """
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
            self.annotator = Annotator(im0, line_width=DEFAULT_CFG_DICT["line_width"])
            for box, cls, track_id in zip(self.boxes, self.clss, self.trk_ids):
                self.annotator.box_label(box, color=colors(int(cls), True), label=self.model.names[int(cls)])

                if len(self.selected_boxes) == 2:
                    for trk_id in self.selected_boxes.keys():
                        if trk_id == track_id:
                            self.selected_boxes[track_id] = box

            if len(self.selected_boxes) == 2:
                self.centroids = [
                    (
                        int((self.selected_boxes[trk_id][0] + self.selected_boxes[trk_id][2]) // 2),
                        int((self.selected_boxes[trk_id][1] + self.selected_boxes[trk_id][3]) // 2),
                    )
                    for trk_id in self.selected_boxes
                ]

                pixel_distance = math.sqrt((self.centroids[0][0] - self.centroids[1][0]) ** 2 + (
                            self.centroids[1][0] - self.centroids[1][1]) ** 2)
                self.annotator.plot_pixel_distance(pixel_distance, self.centroids,)

            self.centroids = []

        # Display the image if the environment supports it and view_img is set to True
        if DEFAULT_CFG_DICT["show"] and self.env_check:
            cv2.namedWindow("Ultralytics Solutions")
            cv2.setMouseCallback("Ultralytics Solutions", self.mouse_event_for_distance)
            cv2.imshow("Ultralytics Solutions", im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0


if __name__ == "__main__":
    names = {0: "person", 1: "car"}  # example class names
    distance_calculation = DistanceCalculation(names=names)
