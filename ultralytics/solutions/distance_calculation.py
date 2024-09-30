# Ultralytics YOLO 🚀, AGPL-3.0 license

import math

import cv2

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class DistanceCalculation:
    """A class to calculate distance between two objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names,
        view_img=False,
        line_thickness=2,
        line_color=(255, 0, 255),
        centroid_color=(104, 31, 17),
    ):
        """
        Initializes the DistanceCalculation class with the given parameters.

        Args:
            names (dict): Dictionary of classes names.
            view_img (bool, optional): Flag to indicate if the video stream should be displayed. Defaults to False.
            line_thickness (int, optional): Thickness of the lines drawn on the image. Defaults to 2.
            line_color (tuple, optional): Color of the lines drawn on the image (BGR format). Defaults to (255, 255, 0).
            centroid_color (tuple, optional): Color of the centroids drawn (BGR format). Defaults to (255, 0, 255).
        """
        # Visual & image information
        self.im0 = None
        self.annotator = None
        self.view_img = view_img
        self.line_color = line_color
        self.centroid_color = centroid_color

        # Prediction & tracking information
        self.names = names
        self.boxes = None
        self.line_thickness = line_thickness
        self.trk_ids = None

        # Distance calculation information
        self.centroids = []

        # Mouse event information
        self.left_mouse_count = 0
        self.selected_boxes = {}

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)
        self.window_name = "Ultralytics Solutions"

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

    def start_process(self, im0, tracks):
        """
        Processes the video frame and calculates the distance between two bounding boxes.

        Args:
            im0 (ndarray): The image frame.
            tracks (list): List of tracks obtained from the object tracking process.

        Returns:
            (ndarray): The processed image frame.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img:
                self.display_frames()
            return im0

        self.boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

        self.annotator = Annotator(self.im0, line_width=self.line_thickness)

        for box, cls, track_id in zip(self.boxes, clss, self.trk_ids):
            self.annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes.keys():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            # Store user selected boxes in centroids list
            self.centroids.extend(
                [[int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)] for box in self.selected_boxes.values()]
            )
            # Calculate pixels distance
            pixels_distance = math.sqrt(
                (self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2
            )
            self.annotator.plot_distance_and_line(pixels_distance, self.centroids, self.line_color, self.centroid_color)

        self.centroids = []

        if self.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self):
        """Displays the current frame with annotations."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_event_for_distance)
        cv2.imshow(self.window_name, self.im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    names = {0: "person", 1: "car"}  # example class names
    distance_calculation = DistanceCalculation(names)
