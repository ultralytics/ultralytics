# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ultralytics.cfg import get_cfg
from ultralytics.solutions.cfg import extract_cfg_data
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon

FILE = Path(__file__).resolve()  # get path of file


class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(self, **kwargs):
        """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""
        import ast

        self.args = get_cfg(extract_cfg_data(FILE))
        if "names" not in kwargs:
            raise ValueError("Error: Classes names 'names' argument is required")
        for key, value in kwargs.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            else:
                print(f"Warning: Unknown argument Skipping!!! {key}")

        self.annotator = None
        self.initialized = False
        self.heatmap = None

        # Predict/track information
        self.boxes = []
        self.track_ids = []
        self.clss = []
        self.track_history = defaultdict(list)

        # Object Counting Information
        self.counting_region = None
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.args.count_txt_color = ast.literal_eval(self.args.count_txt_color)
        self.args.count_bg_color = ast.literal_eval(self.args.count_bg_color)
        self.args.count_reg_color = ast.literal_eval(self.args.count_reg_color)
        self.args.count_reg_pts = ast.literal_eval(self.args.count_reg_pts) if self.args.count_reg_pts else None

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

        # Region and line selection
        print(self.args.count_reg_pts)
        if self.args.count_reg_pts is not None:
            if len(self.args.count_reg_pts) == 2:
                print("Line Counter Initiated.")
                self.counting_region = LineString(self.args.count_reg_pts)
            elif len(self.args.count_reg_pts) >= 3:
                print("Polygon Counter Initiated.")
                self.counting_region = Polygon(self.args.count_reg_pts)
            else:
                print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
                print("Using Line Counter Now")
                self.counting_region = LineString(self.args.count_reg_pts)

        # Shape of heatmap, if not selected
        if self.args.shape not in {"circle", "rect"}:
            print("Unknown shape value provided, 'circle' & 'rect' supported")
            print("Using Circular shape now")
            self.args.shape = "circle"

    def extract_results(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        if tracks[0].boxes.id is not None:
            self.boxes = tracks[0].boxes.xyxy.cpu()
            self.clss = tracks[0].boxes.cls.tolist()
            self.track_ids = tracks[0].boxes.id.int().tolist()

    def generate_heatmap(self, im0, tracks):
        """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """

        # Initialize heatmap only once
        if not self.initialized:
            self.heatmap = np.zeros((int(im0.shape[0]), int(im0.shape[1])), dtype=np.float32)
            self.initialized = True

        self.heatmap *= self.args.decay_factor  # decay factor
        self.extract_results(tracks)

        self.annotator = Annotator(im0, self.args.line_thickness, None)

        if self.track_ids:
            # Draw counting region
            if self.args.count_reg_pts is not None:
                self.annotator.draw_region(
                    reg_pts=self.args.count_reg_pts,
                    color=self.args.count_reg_color,
                    thickness=self.args.region_thickness,
                )

            for box, cls, track_id in zip(self.boxes, self.clss, self.track_ids):
                # Store class info
                if self.args.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.args.names[cls]] = {"IN": 0, "OUT": 0}

                if self.args.shape == "circle":
                    x0, y0, x1, y1 = map(int, [box[0], box[1], box[2], box[3]])
                    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
                    radius = min(x1 - x0, y1 - y0) // 2
                    y_indices, x_indices = np.ogrid[y0:y1, x0:x1]  # Create a coordinate grid for the bounding box area
                    dist_sq = (x_indices - center_x) ** 2 + (
                        y_indices - center_y
                    ) ** 2  # Calculate squared distances from the center
                    mask = dist_sq <= radius**2  # Create a mask for the circle
                    self.heatmap[y0:y1, x0:x1] += 2 * mask  # Update the heatmap using the mask

                else:
                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += 2

                # Store tracking hist
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                if self.args.count_reg_pts is not None:
                    # Count objects in any polygon
                    if len(self.args.count_reg_pts) >= 3:
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
                    elif len(self.args.count_reg_pts) == 2:
                        if prev_position is not None and track_id not in self.count_ids:
                            distance = Point(track_line[-1]).distance(self.counting_region)
                            if distance < self.args.line_dist_thresh and track_id not in self.count_ids:
                                self.count_ids.append(track_id)

                                if (box[0] - prev_position[0]) * (
                                    self.counting_region.centroid.x - prev_position[0]
                                ) > 0:
                                    self.in_counts += 1
                                    self.class_wise_count[self.args.names[cls]]["IN"] += 1
                                else:
                                    self.out_counts += 1
                                    self.class_wise_count[self.args.names[cls]]["OUT"] += 1

        else:
            for box, cls in zip(self.boxes, self.clss):
                if self.args.shape == "circle":
                    x0, y0, x1, y1 = map(int, [box[0], box[1], box[2], box[3]])
                    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
                    radius = min(x1 - x0, y1 - y0) // 2
                    y_indices, x_indices = np.ogrid[y0:y1, x0:x1]  # Create a coordinate grid for the bounding box area
                    dist_sq = (x_indices - center_x) ** 2 + (
                        y_indices - center_y
                    ) ** 2  # Calculate squared distances from the center
                    mask = dist_sq <= radius**2  # Create a mask for the circle
                    self.heatmap[y0:y1, x0:x1] += 2 * mask  # Update the heatmap using the mask
                else:
                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += 2

        if self.args.count_reg_pts is not None:
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

            if labels_dict is not None:
                self.annotator.display_analytics(
                    im0, labels_dict, self.args.count_txt_color, self.args.count_bg_color, 10
                )

        # Normalize, apply colormap to heatmap and combine with original image
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), self.args.colormap)
        im0 = cv2.addWeighted(im0, 1 - self.args.heatmap_alpha, heatmap_colored, self.args.heatmap_alpha, 0)

        if self.env_check and self.args.view_img:
            cv2.imshow(self.args.window_name, im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    heatmap = Heatmap(names=classes_names)
