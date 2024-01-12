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

        # Visual & im0 information
        self.im0 = None
        self.annotator = None
        self.view_img = False
        self.line_color = (255, 255, 0)
        self.centroid_color = (255, 0, 255)

        # Predict/track information
        self.clss = None
        self.names = None
        self.boxes = None
        self.line_thickness = 2
        self.trk_ids = None

        # Distance calculation information
        self.centroids = []
        self.pixel_per_meter = 10

        # Mouse event
        self.left_mouse_count = 0
        self.selected_boxes = {}

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        names,
        pixels_per_meter=10,
        view_img=False,
        line_thickness=2,
        line_color=(255, 255, 0),
        centroid_color=(255, 0, 255),
    ):
        """
        Configures the distance calculation and display parameters.

        Args:
            names (dict): object detection classes names
            pixels_per_meter (int): Number of pixels in meter
            view_img (bool): Flag indicating frame display
            line_thickness (int): Line thickness for bounding boxes.
            line_color (RGB): color of centroids line
            centroid_color (RGB): colors of bbox centroids
        """
        self.names = names
        self.pixel_per_meter = pixels_per_meter
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.centroid_color = centroid_color

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
                for box, track_id in zip(self.boxes, self.trk_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = []
                        self.selected_boxes[track_id] = box

        if event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

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
        Calculate the centroid of bounding box
        Args:
            box (list): Bounding box data
        """
        return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

    def calculate_distance(self, centroid1, centroid2):
        """
        Calculate distance between two centroids
        Args:
            centroid1 (point): First bounding box data
            centroid2 (point): Second bounding box data
        """
        pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
        return pixel_distance / self.pixel_per_meter

    def plot_distance_and_line(self, distance):
        """
        Plot the distance and line on frame
        Args:
            distance (float): Distance between two centroids
        """
        cv2.rectangle(self.im0, (15, 25), (280, 70), (255, 255, 255), -1)
        cv2.putText(
            self.im0, f"Distance : {distance:.2f}m", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.line(self.im0, self.centroids[0], self.centroids[1], self.line_color, 3)
        cv2.circle(self.im0, self.centroids[0], 6, self.centroid_color, -1)
        cv2.circle(self.im0, self.centroids[1], 6, self.centroid_color, -1)

    def start_process(self, im0, tracks):
        """
        Calculate distance between two bounding boxes based on tracking data
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

            distance = self.calculate_distance(self.centroids[0], self.centroids[1])
            self.plot_distance_and_line(distance)

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
