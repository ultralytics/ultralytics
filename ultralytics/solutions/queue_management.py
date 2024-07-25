# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from pathlib import Path

import cv2

from ultralytics.cfg import get_cfg
from ultralytics.solutions.cfg import extract_cfg_data
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import Point, Polygon

FILE = Path(__file__).resolve()  # get path of file


class QueueManager:
    """A class to manage the queue in a real-time video stream based on object tracks."""

    def __init__(self, **kwargs):
        """Initialize the queuemanager class with kwargs arguments."""
        import ast

        self.args = get_cfg(extract_cfg_data(FILE))
        for key, value in kwargs.items():
            if "names" not in kwargs:
                raise ValueError("Error: Classes names 'names' argument is required")
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            else:
                print(f"Warning: Unknown argument Skipping!!! {key}")

        # Region & Line Information
        self.counting_region = (
            Polygon(self.args.reg_pts)
            if len(self.args.reg_pts) >= 3
            else Polygon([(20, 60), (20, 680), (1120, 680), (1120, 60)])
        )
        self.im0 = None
        self.annotator = None  # Annotator
        self.counts = 0
        self.track_history = defaultdict(list)
        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow
        self.args.count_txt_color = ast.literal_eval(self.args.count_txt_color)
        self.args.count_reg_color = ast.literal_eval(self.args.count_reg_color)

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for queue management in a video stream."""

        # Initialize annotator and draw the queue region
        self.annotator = Annotator(self.im0, self.args.line_thickness, self.args.names)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                self.annotator.box_label(
                    box, label=f"{self.args.names[cls]}#{track_id}", color=colors(int(track_id), True)
                )

                # Update track history
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails if enabled
                if self.args.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.args.track_color or colors(int(track_id), True),
                        track_thickness=self.args.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Check if the object is inside the counting region
                if len(self.args.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        self.counts += 1

        # Display queue counts
        label = f"Queue Counts : {str(self.counts)}"
        if label is not None:
            self.annotator.queue_counts_display(
                label,
                points=self.args.reg_pts,
                region_color=self.args.count_reg_color,
                txt_color=self.args.count_txt_color,
            )

        self.counts = 0  # Reset counts after displaying
        self.display_frames()

    def display_frames(self):
        """Displays the current frame with annotations."""
        if self.env_check and self.args.view_img:
            self.annotator.draw_region(
                reg_pts=self.args.reg_pts, thickness=self.args.region_thickness, color=self.args.count_reg_color
            )
            cv2.namedWindow(self.args.window_name)
            cv2.imshow(self.args.window_name, self.im0)
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

        if self.args.view_img:
            self.display_frames()  # Display the frame if enabled
        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    queue_manager = QueueManager(names=classes_names)
