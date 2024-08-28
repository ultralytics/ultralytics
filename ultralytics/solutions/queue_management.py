# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics import YOLO, solutions
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import Point, Polygon


class QueueManager:
    """A class to manage queues in a real-time video stream using object tracking data."""

    def __init__(self, **kwargs):
        """
        Initializes an instance of the QueueManager class, setting up configurations for monitoring and managing queues
        in real-time video streams.

        Args:
            kwargs (dict): Dictionary of arguments for configuring the queue management process, such as detection thresholds, regions of interest, and analysis logic parameters.
        """
        DEFAULT_CFG_DICT.update(kwargs)
        self.model = YOLO(DEFAULT_CFG_DICT["model"])
        # Region & Line Information
        self.counting_region = (
            Polygon(DEFAULT_CFG_DICT["reg_pts"])
            if len(DEFAULT_CFG_DICT["reg_pts"]) >= 3
            else Polygon([(20, 60), (20, 680), (1120, 680), (1120, 60)])
        )
        self.im0 = None
        self.annotator = None  # Annotator
        self.counts = 0
        self.track_history = defaultdict(list)
        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow
        print(f"Ultralytics Solutions ✅ {DEFAULT_CFG_DICT}")

    def process_tracks(self):
        """Extracts and processes tracking data for queue management in a video stream."""
        # Initialize annotator and draw the queue region
        self.annotator = Annotator(self.im0, DEFAULT_CFG_DICT["line_width"])

        tracks = self.model.track(
            source=self.im0,
            persist=True,
            tracker=DEFAULT_CFG_DICT["tracker"],
            classes=DEFAULT_CFG_DICT["classes"],
            iou=DEFAULT_CFG_DICT["iou"],
            conf=DEFAULT_CFG_DICT["conf"],
        )

        boxes, clss, track_ids = solutions.extract_tracks(tracks)

        if track_ids is not None:
            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                color = colors(int(track_id), True)
                self.annotator.box_label(
                    box, label=f"{self.model.names[cls]}", color=color, txt_color=self.annotator.get_txt_color(color)
                )

                # Update track history
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails if enabled
                if DEFAULT_CFG_DICT["draw_tracks"]:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=colors(int(track_id), True),
                        track_thickness=DEFAULT_CFG_DICT["line_width"],
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Check if the object is inside the counting region
                if len(DEFAULT_CFG_DICT["reg_pts"]) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        self.counts += 1

        # Display queue counts
        label = f"Queue Counts : {str(self.counts)}"
        if label is not None:
            self.annotator.queue_counts_display(
                label,
                points=DEFAULT_CFG_DICT["reg_pts"],
                region_color=(104, 31, 17),
                txt_color=(255, 255, 255),
            )

        self.counts = 0  # Reset counts after displaying

    def process_queue(self, im0):
        """
        Main function to start the queue management process.

        Args:
            im0 (ndarray): Current frame from the video stream.

        Returns:
            im0 (ndarray): The processed image frame.
        """
        self.im0 = im0  # Store the current frame
        self.process_tracks()  # Extract and process tracks

        if DEFAULT_CFG_DICT["show"] and self.env_check:
            self.annotator.draw_region(
                reg_pts=DEFAULT_CFG_DICT["reg_pts"],
                thickness=int(DEFAULT_CFG_DICT["line_width"]) * 2,
                color=(104, 31, 17),
            )
            cv2.imshow("Ultralytics Solutions", self.im0)
            # Close window on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    queue_manager = QueueManager(names=classes_names)
