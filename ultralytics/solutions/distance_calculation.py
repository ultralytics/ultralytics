# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class DistanceCalculation(BaseSolution):
    """
    A class to calculate distance between two objects in a real-time video stream based on their tracks.

    This class extends BaseSolution to provide functionality for selecting objects and calculating the distance
    between them in a video stream using YOLO object detection and tracking.

    Attributes:
        left_mouse_count (int): Counter for left mouse button clicks.
        selected_boxes (Dict[int, List[float]]): Dictionary to store selected bounding boxes and their track IDs.
        centroids (List[List[int]]): List to store centroids of selected bounding boxes.

    Methods:
        mouse_event_for_distance: Handles mouse events for selecting objects in the video stream.
        process: Processes video frames and calculates the distance between selected objects.

    Examples:
        >>> distance_calc = DistanceCalculation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = distance_calc.process(frame)
        >>> cv2.imshow("Distance Calculation", results.plot_im)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs):
        """Initializes the DistanceCalculation class for measuring object distances in video streams."""
        super().__init__(**kwargs)

        # Mouse event information
        self.left_mouse_count = 0
        self.selected_boxes = {}
        self.centroids = []  # Store centroids of selected objects

    def mouse_event_for_distance(self, event, x, y, flags, param):
        """
        Handles mouse events to select regions in a real-time video stream for distance calculation.

        Args:
            event (int): Type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN).
            x (int): X-coordinate of the mouse pointer.
            y (int): Y-coordinate of the mouse pointer.
            flags (int): Flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY, cv2.EVENT_FLAG_SHIFTKEY).
            param (Any): Additional parameters passed to the function.

        Examples:
            >>> # Assuming 'dc' is an instance of DistanceCalculation
            >>> cv2.setMouseCallback("window_name", dc.mouse_event_for_distance)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.track_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = box

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def process(self, im0):
        """
        Processes a video frame and calculates the distance between two selected bounding boxes.

        This method extracts tracks from the input frame, annotates bounding boxes, and calculates the distance
        between two user-selected objects if they have been chosen.

        Args:
            im0 (numpy.ndarray): The input image frame to process.

        Returns:
            (SolutionResults): Contains processed image `plot_im`, `total_tracks` (int) representing the total number
                of tracked objects, and `pixels_distance` (float) representing the distance between selected objects
                in pixels.

        Examples:
            >>> import numpy as np
            >>> from ultralytics.solutions import DistanceCalculation
            >>> dc = DistanceCalculation()
            >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = dc.process(frame)
            >>> print(f"Distance: {results.pixels_distance:.2f} pixels")
        """
        self.extract_tracks(im0)  # Extract tracks
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        pixels_distance = 0
        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

            # Update selected boxes if they're being tracked
            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes.keys():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            # Calculate centroids of selected boxes
            self.centroids.extend(
                [[int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)] for box in self.selected_boxes.values()]
            )
            # Calculate Euclidean distance between centroids
            pixels_distance = math.sqrt(
                (self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2
            )
            annotator.plot_distance_and_line(pixels_distance, self.centroids)

        self.centroids = []  # Reset centroids for next frame
        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function
        cv2.setMouseCallback("Ultralytics Solutions", self.mouse_event_for_distance)

        # Return SolutionResults with processed image and calculated metrics
        return SolutionResults(plot_im=plot_im, pixels_distance=pixels_distance, total_tracks=len(self.track_ids))
