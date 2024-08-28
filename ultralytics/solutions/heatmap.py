# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO, solutions
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class Heatmap:
    """A class for drawing heatmaps in a real-time video stream using tracking data."""

    def __init__(self, **kwargs):
        """
        Initializes an instance of the Heatmaps class with specified parameters for visualizing and tracking data in
        real-time video streams.

        Args:
            kwargs (dict): Dictionary of arguments that allow customization of heatmap properties, such as visual settings, image dimensions, tracking configurations, and heatmap display options.
        """
        import ast

        DEFAULT_CFG_DICT.update(kwargs)
        self.model = YOLO(DEFAULT_CFG_DICT["model"])
        self.annotator = None
        self.initialized = False
        self.heatmap = None

        # track information
        self.track_history = defaultdict(list)

        # Object Counting Information
        self.counting_region = None
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        if isinstance(DEFAULT_CFG_DICT["reg_pts"], str):
            DEFAULT_CFG_DICT["reg_pts"] = ast.literal_eval(DEFAULT_CFG_DICT["reg_pts"])

        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow
        print(f"Ultralytics Solutions ✅ {DEFAULT_CFG_DICT}")
        # Region and line selection
        if DEFAULT_CFG_DICT["reg_pts"] is not None:
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

    def generate_heatmap(self, im0):
        """
        Generate a heatmap using tracking data.

        Args:
            im0 (ndarray): The image.

        Returns:
            im0 (ndarray): The processed image frame.
        """
        # Initialize heatmap only once
        if not self.initialized:
            self.heatmap = np.zeros((int(im0.shape[0]), int(im0.shape[1])), dtype=np.float32)
            self.initialized = True

        self.heatmap *= DEFAULT_CFG_DICT["decay_factor"]  # decay factor
        tracks = self.model.track(
            source=im0,
            persist=True,
            tracker=DEFAULT_CFG_DICT["tracker"],
            classes=DEFAULT_CFG_DICT["classes"],
            iou=DEFAULT_CFG_DICT["iou"],
            conf=DEFAULT_CFG_DICT["conf"],
        )
        boxes, clss, track_ids = solutions.extract_tracks(tracks)

        self.annotator = Annotator(im0, DEFAULT_CFG_DICT["line_width"], None)

        if track_ids is not None:
            # Draw counting region
            if DEFAULT_CFG_DICT["reg_pts"] is not None:
                self.annotator.draw_region(
                    reg_pts=DEFAULT_CFG_DICT["reg_pts"],
                    color=(104, 31, 17),
                    thickness=int(DEFAULT_CFG_DICT["line_width"]) * 2,
                )

            for box, cls, track_id in zip(boxes, clss, track_ids):
                # Store class info
                if self.model.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.model.names[cls]] = {"IN": 0, "OUT": 0}

                x0, y0, x1, y1 = map(int, [box[0], box[1], box[2], box[3]])
                center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
                y_indices, x_indices = np.ogrid[y0:y1, x0:x1]  # Create a coordinate grid for the bounding box area
                dist_sq = (x_indices - center_x) ** 2 + (
                    y_indices - center_y
                ) ** 2  # Calculate squared distances from the center
                mask = dist_sq <= (min(x1 - x0, y1 - y0) // 2) ** 2  # Create a mask for the circle
                self.heatmap[y0:y1, x0:x1] += 2 * mask  # Update the heatmap using the mask

                # Store tracking hist
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                if DEFAULT_CFG_DICT["reg_pts"] is not None:
                    # Count objects in any polygon
                    if len(DEFAULT_CFG_DICT["reg_pts"]) >= 3:
                        print(self.counting_region)
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

                                if (box[0] - prev_position[0]) * (
                                    self.counting_region.centroid.x - prev_position[0]
                                ) > 0:
                                    self.in_counts += 1
                                    self.class_wise_count[self.model.names[cls]]["IN"] += 1
                                else:
                                    self.out_counts += 1
                                    self.class_wise_count[self.model.names[cls]]["OUT"] += 1

        else:
            for box, cls in zip(boxes, clss):
                x0, y0, x1, y1 = map(int, [box[0], box[1], box[2], box[3]])
                center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
                y_indices, x_indices = np.ogrid[y0:y1, x0:x1]  # Create a coordinate grid for the bounding box area
                dist_sq = (x_indices - center_x) ** 2 + (
                    y_indices - center_y
                ) ** 2  # Calculate squared distances from the center
                # Create a mask for the circle and update the heatmap
                self.heatmap[y0:y1, x0:x1] += 2 * dist_sq <= (min(x1 - x0, y1 - y0) // 2) ** 2

        if DEFAULT_CFG_DICT["reg_pts"] is not None:
            labels_dict = {}

            for key, value in self.class_wise_count.items():
                if value["IN"] != 0 or value["OUT"] != 0:
                    if not DEFAULT_CFG_DICT["show_in_counts"] and not DEFAULT_CFG_DICT["show_in_counts"]:
                        continue
                    elif not DEFAULT_CFG_DICT["show_in_counts"]:
                        labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                    elif not DEFAULT_CFG_DICT["show_out_counts"]:
                        labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                    else:
                        labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

            if labels_dict is not None:
                self.annotator.display_analytics(
                    im0, labels_dict, (255, 255, 255), (104, 31, 17), 10 + DEFAULT_CFG_DICT["line_width"]
                )

        # Normalize, apply colormap to heatmap and combine with original image
        heatmap_colored = cv2.applyColorMap(
            cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), DEFAULT_CFG_DICT["colormap"]
        )
        im0 = cv2.addWeighted(
            im0, 1 - DEFAULT_CFG_DICT["heatmap_alpha"], heatmap_colored, DEFAULT_CFG_DICT["heatmap_alpha"], 0
        )

        if self.env_check and DEFAULT_CFG_DICT["show"]:
            cv2.imshow("Ultralytics Solutions", im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    heatmap = Heatmap(names=classes_names)
