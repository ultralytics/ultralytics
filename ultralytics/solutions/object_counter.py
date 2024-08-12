# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics import YOLO, solutions
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class for managing object counting in a real-time video stream using tracking data."""

    def __init__(self, **kwargs):
        """
        Initializes an instance of the ObjectCounter class, setting up configurations for tracking and counting objects
        in real-time video streams.

        Args:
            kwargs (dict): Dictionary of arguments for configuring the object counting process, such as regions of interest, and counting logic parameters.
        """
        import ast

        DEFAULT_CFG_DICT.update(kwargs)
        self.model = YOLO(DEFAULT_CFG_DICT["model"])
        self.im0 = None
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.track_history = defaultdict(list)  # Tracks info
        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow
        DEFAULT_CFG_DICT["reg_color"] = ast.literal_eval(DEFAULT_CFG_DICT["reg_color"])
        DEFAULT_CFG_DICT["txt_color"] = ast.literal_eval(DEFAULT_CFG_DICT["txt_color"])
        DEFAULT_CFG_DICT["bg_color"] = ast.literal_eval(DEFAULT_CFG_DICT["bg_color"])

        if isinstance(DEFAULT_CFG_DICT["reg_pts"], str):
            DEFAULT_CFG_DICT["reg_pts"] = ast.literal_eval(DEFAULT_CFG_DICT["reg_pts"])
        print(f"Ultralytics Solutions ✅ {DEFAULT_CFG_DICT}")

        # Initialize counting region
        if len(DEFAULT_CFG_DICT["reg_pts"]) == 2:
            print("Line Counter Initiated.")
            self.counting_region = LineString(DEFAULT_CFG_DICT["reg_pts"])
        elif len(DEFAULT_CFG_DICT["reg_pts"]) >= 3:
            print("Polygon Counter Initiated.")
            self.counting_region = Polygon(DEFAULT_CFG_DICT["reg_pts"])
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region = LineString(DEFAULT_CFG_DICT["reg_pts"])

    def process_tracks(self):
        """Extracts and processes tracking data for counting objects in a video stream."""

        # Initialize annotator and draw regions
        annotator = Annotator(self.im0, DEFAULT_CFG_DICT["line_width"], self.model.names)
        annotator.draw_region(
            reg_pts=DEFAULT_CFG_DICT["reg_pts"],
            color=DEFAULT_CFG_DICT["reg_color"],
            thickness=int(DEFAULT_CFG_DICT["line_width"]) * 2,
        )
        tracks = self.model.track(
            source=self.im0,
            persist=True,
            tracker=DEFAULT_CFG_DICT["tracker"],
            classes=DEFAULT_CFG_DICT["classes"],
            iou=DEFAULT_CFG_DICT["iou"],
            conf=DEFAULT_CFG_DICT["conf"],
        )
        boxes, clss, track_ids = solutions.extract_tracks(tracks)  # Extract tracks
        if track_ids is not None:
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                annotator.box_label(box, label=f"{self.model.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Store class info
                if self.model.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.model.names[cls]] = {"IN": 0, "OUT": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if DEFAULT_CFG_DICT["draw_tracks"]:
                    annotator.draw_centroid_and_tracks(
                        track_line,
                        color=colors(int(track_id), True),
                        track_thickness=DEFAULT_CFG_DICT["line_width"],
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(DEFAULT_CFG_DICT["reg_pts"]) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)

                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.model.names[cls]]["IN"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.model.names[cls]]["OUT"] += 1

                # Count objects using line
                elif len(DEFAULT_CFG_DICT["reg_pts"]) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        distance = Point(track_line[-1]).distance(self.counting_region)
                        if distance < DEFAULT_CFG_DICT["line_dist_thresh"] and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.model.names[cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.model.names[cls]]["OUT"] += 1

        labels_dict = {}

        for key, value in self.class_wise_count.items():
            if value["IN"] != 0 or value["OUT"] != 0:
                if not DEFAULT_CFG_DICT["show_in_counts"] and not DEFAULT_CFG_DICT["show_out_counts"]:
                    continue
                elif not DEFAULT_CFG_DICT["show_in_counts"]:
                    labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                elif not DEFAULT_CFG_DICT["show_out_counts"]:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                else:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

        if labels_dict:
            annotator.display_analytics(
                self.im0, labels_dict, (DEFAULT_CFG_DICT["txt_color"]), (DEFAULT_CFG_DICT["bg_color"]), 10
            )

    def start_counting(self, im0):
        """
        Primary function to initiate the object counting process.

        Args:
            im0 (ndarray): The current frame from the video stream.

        Returns:
            im0 (ndarray): The processed image frame.
        """
        self.im0 = im0  # store image
        self.process_tracks()  # draw region even if no objects

        if DEFAULT_CFG_DICT["show"] and self.env_check:
            cv2.imshow("Ultralytics Solutions", self.im0)

            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    ObjectCounter(names=classes_names)
