# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics import solutions
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
        import ast

        DEFAULT_CFG_DICT.update(kwargs)

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
        DEFAULT_CFG_DICT["txt_color"] = ast.literal_eval(DEFAULT_CFG_DICT["txt_color"])
        DEFAULT_CFG_DICT["reg_color"] = ast.literal_eval(DEFAULT_CFG_DICT["reg_color"])
        print(f"Ultralytics Solutions ✅ {DEFAULT_CFG_DICT}")

    def process_tracks(self, tracks):
        """
        Extracts and processes tracking data for queue management in a video stream.

        Args:
            tracks (list): A list of track objects representing detected objects in the video stream, each containing information such as position and movement.
        """
        # Initialize annotator and draw the queue region
        self.annotator = Annotator(self.im0, DEFAULT_CFG_DICT["line_width"], DEFAULT_CFG_DICT["names"])

        boxes, clss, track_ids = solutions.extract_tracks(tracks)

        if track_ids is not None:
            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                self.annotator.box_label(
                    box, label=f"{DEFAULT_CFG_DICT['names'][cls]}#{track_id}", color=colors(int(track_id), True)
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
                region_color=DEFAULT_CFG_DICT["reg_color"],
                txt_color=DEFAULT_CFG_DICT["txt_color"],
            )

        self.counts = 0  # Reset counts after displaying

    def process_queue(self, im0, tracks):
        """
        Main function to start the queue management process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.

        Returns:
            im0 (ndarray): The processed image frame.
        """
        self.im0 = im0  # Store the current frame
        self.process_tracks(tracks)  # Extract and process tracks

        if DEFAULT_CFG_DICT["show"] and self.env_check:
            self.annotator.draw_region(
                reg_pts=DEFAULT_CFG_DICT["reg_pts"],
                thickness=int(DEFAULT_CFG_DICT["line_width"])*2,
                color=DEFAULT_CFG_DICT["reg_color"],
            )
            cv2.imshow("Ultralytics Solutions", self.im0)
            # Close window on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    queue_manager = QueueManager(names=classes_names)
