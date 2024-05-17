# Ultralytics YOLO 🚀, AGPL-3.0 license

import math

import cv2

from . import Annotator, bg_color_rgb, cls_names, colors, display_frames, extract_tracks, pxl_m, tf, txt_color_rgb


class DistanceCalculation:
    """A class to calculate distance between two objects in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the distance calculation class with default values for Visual, Image, track and distance
        parameters.
        """
        self.boxes = None
        self.trk_ids = None
        self.im0 = None
        self.annotator = None
        self.centroids = []
        self.left_mouse_count = 0
        self.selected_boxes = {}
        self.window_name = "Ultralytics Distance Calculator"
        print("Distance calculation app initialized...")

    def mouse_event_for_distance(self, event, x, y, flags, param):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            param (dict): Additional parameters you may want to pass to the function.
        """
        global selected_boxes
        global left_mouse_count
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, trk_id in zip(self.boxes, self.trk_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and trk_id not in self.selected_boxes:
                        self.selected_boxes[trk_id] = []
                        self.selected_boxes[trk_id] = box

        if event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def distance(self, centroid1, centroid2):
        """
        Calculate distance between two centroids.

        Args:
            centroid1 (point): First bounding box data
            centroid2 (point): Second bounding box data
        """
        pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
        return pixel_distance / pxl_m, (pixel_distance / pxl_m) * 1000

    def calculate_distance(self, im0, tracks):
        """
        Calculate distance between two bounding boxes based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """

        self.im0 = im0

        self.boxes, clss, self.trk_ids = extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=2)

        if self.trk_ids is not None:
            for box, cls, trk_id in zip(self.boxes, clss, self.trk_ids):
                self.annotator.box_label(box, color=colors(int(cls), True), label=cls_names[int(cls)])

                if len(self.selected_boxes) == 2:
                    for trk_ids, _ in self.selected_boxes.items():
                        if trk_ids == trk_id:
                            self.selected_boxes[trk_id] = box

            if len(self.selected_boxes) == 2:
                for trk_id, box in self.selected_boxes.items():
                    box = self.selected_boxes[trk_id]
                    centroid = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)
                    self.centroids.append(centroid)

                distance_m, distance_mm = self.distance(self.centroids[0], self.centroids[1])
                self.annotator.plot_distance_and_line(
                    distance_m, distance_mm, self.centroids, bg_color_rgb, txt_color_rgb
                )

        self.centroids = []
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_event_for_distance)
        display_frames(self.im0, self.window_name)

        return im0
