# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from pathlib import Path

import cv2

from ultralytics.cfg import get_cfg
from ultralytics.solutions.cfg import extract_cfg_data
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon

FILE = Path(__file__).resolve()  # get path of file


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self, **kwargs):
        """Initialize the object counter class with kwargs arguments."""
        import ast

        self.args = get_cfg(extract_cfg_data(FILE))
        if "names" not in kwargs:
            raise ValueError("Error: Classes names 'names' argument is required")
        for key, value in kwargs.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            else:
                print(f"Warning: Unknown argument Skipping!!! {key}")

        self.im0 = None
        self.annotator = None
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.track_history = defaultdict(list)  # Tracks info
        self.args.count_txt_color = ast.literal_eval(self.args.count_txt_color)
        self.args.count_bg_color = ast.literal_eval(self.args.count_bg_color)
        self.args.count_reg_color = ast.literal_eval(self.args.count_reg_color)

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

        # Initialize counting region
        if len(self.args.reg_pts) == 2:
            print("Line Counter Initiated.")
            self.counting_region = LineString(self.args.reg_pts)
        elif len(self.args.reg_pts) >= 3:
            print("Polygon Counter Initiated.")
            self.counting_region = Polygon(self.args.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.args.reg_pts)

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.args.line_thickness, self.args.names)

        # Draw region or line
        self.annotator.draw_region(
            reg_pts=self.args.reg_pts, color=self.args.count_reg_color, thickness=self.args.region_thickness
        )

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

                # Store class info
                if self.args.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.args.names[cls]] = {"IN": 0, "OUT": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.args.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.args.track_color or colors(int(track_id), True),
                        track_thickness=self.args.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(self.args.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)

                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.args.names[cls]]["IN"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.args.names[cls]]["OUT"] += 1

                # Count objects using line
                elif len(self.args.reg_pts) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        distance = Point(track_line[-1]).distance(self.counting_region)
                        if distance < self.args.line_dist_thresh and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.args.names[cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.args.names[cls]]["OUT"] += 1

        labels_dict = {}

        for key, value in self.class_wise_count.items():
            if value["IN"] != 0 or value["OUT"] != 0:
                if not self.args.view_in_counts and not self.args.view_out_counts:
                    continue
                elif not self.args.view_in_counts:
                    labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                elif not self.args.view_out_counts:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                else:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

        if labels_dict:
            self.annotator.display_analytics(
                self.im0, labels_dict, self.args.count_txt_color, self.args.count_bg_color, 10
            )

    def display_frames(self):
        """Displays the current frame with annotations and regions in a window."""
        if self.env_check:
            cv2.namedWindow(self.args.window_name)
            cv2.imshow(self.args.window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        if self.args.view_img:
            self.display_frames()
        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    ObjectCounter(names=classes_names)
