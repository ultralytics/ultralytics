# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import Point, Polygon


class QueueManager:
    """A class to manage the queue in a real-time video stream based on object tracks."""

    def __init__(
        self,
        names,
        reg_pts=None,
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        region_color=(255, 0, 255),
        view_queue_counts=True,
        draw_tracks=False,
        count_txt_color=(255, 255, 255),
        track_color=None,
        region_thickness=5,
        fontsize=0.7,
    ):
        """
        Initializes the QueueManager with specified parameters for tracking and counting objects.

        Args:
            names (dict): A dictionary mapping class IDs to class names.
            reg_pts (list of tuples, optional): Points defining the counting region polygon. Defaults to a predefined
                rectangle.
            line_thickness (int, optional): Thickness of the annotation lines. Defaults to 2.
            track_thickness (int, optional): Thickness of the track lines. Defaults to 2.
            view_img (bool, optional): Whether to display the image frames. Defaults to False.
            region_color (tuple, optional): Color of the counting region lines (BGR). Defaults to (255, 0, 255).
            view_queue_counts (bool, optional): Whether to display the queue counts. Defaults to True.
            draw_tracks (bool, optional): Whether to draw tracks of the objects. Defaults to False.
            count_txt_color (tuple, optional): Color of the count text (BGR). Defaults to (255, 255, 255).
            track_color (tuple, optional): Color of the tracks. If None, different colors will be used for different
                tracks. Defaults to None.
            region_thickness (int, optional): Thickness of the counting region lines. Defaults to 5.
            fontsize (float, optional): Font size for the text annotations. Defaults to 0.7.
        """

        # Mouse events state
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = reg_pts if reg_pts is not None else [(20, 60), (20, 680), (1120, 680), (1120, 60)]
        self.counting_region = (
            Polygon(self.reg_pts) if len(self.reg_pts) >= 3 else Polygon([(20, 60), (20, 680), (1120, 680), (1120, 60)])
        )
        self.region_color = region_color
        self.region_thickness = region_thickness

        # Image and annotation Information
        self.im0 = None
        self.tf = line_thickness
        self.view_img = view_img
        self.view_queue_counts = view_queue_counts
        self.fontsize = fontsize

        self.names = names  # Class names
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Queue Manager"

        # Object counting Information
        self.counts = 0
        self.count_txt_color = count_txt_color

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.track_color = track_color

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for queue management in a video stream."""

        # Initialize annotator and draw the queue region
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Update track history
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails if enabled
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color or colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Check if the object is inside the counting region
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        self.counts += 1

        # Display queue counts
        label = f"Queue Counts : {str(self.counts)}"
        if label is not None:
            self.annotator.queue_counts_display(
                label,
                points=self.reg_pts,
                region_color=self.region_color,
                txt_color=self.count_txt_color,
            )

        self.counts = 0  # Reset counts after displaying
        self.display_frames()

    def display_frames(self):
        """Displays the current frame with annotations."""
        if self.env_check and self.view_img:
            self.annotator.draw_region(reg_pts=self.reg_pts, thickness=self.region_thickness, color=self.region_color)
            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, self.im0)
            # Close window on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def process_queue(self, im0, tracks):
        """
        Main function to start the queue management process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # Store the current frame
        self.extract_and_process_tracks(tracks)  # Extract and process tracks

        if self.view_img:
            self.display_frames()  # Display the frame if enabled
        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    queue_manager = QueueManager(classes_names)
