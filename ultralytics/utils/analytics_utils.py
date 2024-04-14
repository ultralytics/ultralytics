# Ultralytics YOLO 🚀, AGPL-3.0 license
# Documentation: https://docs.ultralytics.com/analytics/index.md
# Example usage: yolo dashboard "path/to/custom_data.yaml"

import ast
import os
from collections import defaultdict

import yaml

from ultralytics import YOLO
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.plotting import colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class Analytics:
    def __init__(self):
        self.total_detections = 0  # total detections count

        self.modes = {"predict", "count", "track"}  # modes names

        self.video_path = None  # Video information

        self.mode = None  # init mode name

        # Object counting information
        self.in_counts = 0  # store in_counts
        self.out_counts = 0  # store out_counts
        self.count_ids = []  # store count_ids
        self.region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]  # region points
        self.counting_region = None  # counting region

        # Model processed data information
        self.model_file = None  # store model file name
        self.model = None  # store model data

        # Bounding box and tracks information
        self.track_history = defaultdict(lambda: [])  # tracking history dict
        self.track_ids = None  # store tracks id for tracking
        self.clss = None  # store model classes data
        self.boxes = None  # store boxes data for  detect task

        # Classwise Detections Bar
        self.classwise_detections = {}

    def load_and_proces_yaml(self, yaml_path):
        """
        Load and process yaml file data.

        Args:
            yaml_path (str): Path to YAML file
        Returns:
            task (str): task name to be performed
            model (str): path to model file
            mode (str): mode name
        """
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        self.mode = config.get("mode")
        if self.mode not in self.modes or self.mode is None:
            print("Warning! mode is none or invalid")
            print("using predict mode now")
            self.mode = "predict"

        self.model_file = config.get("model")
        if self.model_file is None:
            print("Using yolov8n.pt now")
            self.model_file = "yolov8n.pt"
        self.model = YOLO(self.model_file)

        if self.mode == "count":
            self.region_points = config.get("region_points")
            if self.region_points:
                points = ast.literal_eval(self.region_points)
                if len(points) == 2:
                    self.counting_region = LineString(points)
                elif len(points) >= 3:
                    self.counting_region = Polygon(points)
            else:
                self.counting_region = Polygon(self.region_points)

        self.video_path = config.get("video_path")
        if not os.path.exists(self.video_path):
            from ultralytics.utils.downloads import safe_download

            safe_download("https://ultralytics.com/assets/dashboard_sample.mp4")
            self.video_path = "dashboard_sample.mp4"

        return self.model, self.mode, self.video_path

    def object_counting(self, track_id, box):
        """
        Object counting module for counting the objects with region or line coordinates.

        Args:
            track_id (int): tracking id of an object
            box (bbox): bounding box data
        """
        track_line = self.track_history[track_id]
        track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
        track_line.pop(0) if len(track_line) > 30 else None

        prev_position = None
        if len(self.track_history[track_id]) > 1:
            prev_position = self.track_history[track_id][-2]

        if len(self.region_points) >= 3:
            is_inside = self.counting_region.contains(Point(track_line[-1]))

            if prev_position is not None and is_inside and track_id not in self.count_ids:
                self.count_ids.append(track_id)

                if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                    self.in_counts += 1
                else:
                    self.out_counts += 1

        elif len(self.region_points) == 2:
            if prev_position is not None and track_id not in self.count_ids:
                distance = Point(track_line[-1]).distance(self.counting_region)

                if distance < 15 and track_id not in self.count_ids:
                    self.count_ids.append(track_id)
                    if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                        self.in_counts += 1
                    else:
                        self.out_counts += 1

    def store_classwise(self, cls):
        """
        Store classwise detections.

        Args:
            cls (float): class index
        Returns:
            classwise_detections (dict): Classwise detection dictionary
        """
        if self.model.names[int(cls)] not in self.classwise_detections:
            self.classwise_detections[self.model.names[int(cls)]] = 1
        else:
            self.classwise_detections[self.model.names[int(cls)]] += 1
        return self.classwise_detections

    def process_results(self, results, annotator):
        """
        Process the results for object detection, segmentation and pose.

        Args:
            results (list): segment, detect or pose data
            annotator (object): object for plotting bounding boxes
        Returns:
            total_detections (int): total detection counts
            in_counts (int, optional): total objects in counts
            out_counts (int, optional): total objects out counts
        """
        self.classwise_detections = {}

        self.boxes = results[0].boxes.xyxy.cpu().tolist()
        self.clss = results[0].boxes.cls.cpu().tolist()

        self.total_detections = len(self.boxes)

        if self.mode == "count" or self.mode == "track":
            if results[0].boxes.id is not None:
                self.track_ids = results[0].boxes.id.int().cpu().tolist()

                if self.mode == "count":
                    annotator.draw_region(
                        reg_pts=ast.literal_eval(self.region_points), color=(255, 0, 255), thickness=2
                    )

                    for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
                        self.store_classwise(cls)
                        annotator.box_label(box, color=colors(int(track_id), True), label=str(track_id))
                        self.object_counting(track_id, box)
                    return self.total_detections, self.in_counts, self.out_counts

                elif self.mode == "track":
                    for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
                        self.store_classwise(cls)
                        annotator.box_label(box, color=colors(int(track_id), True), label=str(track_id))
                    return self.total_detections

            else:
                if self.mode == "count":
                    return 0, 0, 0
                if self.mode == "track":
                    return 0
        else:
            for box, cls in zip(self.boxes, self.clss):
                self.store_classwise(cls)
                annotator.box_label(box, color=colors(int(cls), True), label=self.model.names[int(cls)])
            return self.total_detections
