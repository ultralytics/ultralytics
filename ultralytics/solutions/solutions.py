# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER, yaml_load
from ultralytics.utils.checks import check_imshow, check_requirements

check_requirements("shapely>=2.0.0")
from shapely.geometry import LineString, Polygon

DEFAULT_SOL_CFG_PATH = Path(__file__).resolve().parents[1] / "cfg/solutions/default.yaml"


class BaseSolution:
    """A class to manage all the Ultralytics Solutions: https://docs.ultralytics.com/solutions/."""

    def __init__(self, **kwargs):
        """
        Base initializer for all solutions.

        Child classes should call this with necessary parameters.
        """
        # Load config and update with args
        self.CFG = yaml_load(DEFAULT_SOL_CFG_PATH)
        self.CFG.update(kwargs)
        LOGGER.info(f"Ultralytics Solutions: ✅ {self.CFG}")

        self.region = self.CFG["region"]  # Store region data for other classes usage
        self.line_width = self.CFG["line_width"]  # Store line_width for usage

        # Load Model and store classes names
        self.model = YOLO(self.CFG["model"])
        self.names = self.model.names

        # Initialize environment and region setup
        self.env_check = check_imshow(warn=True)
        self.track_history = defaultdict(list)

    def extract_tracks(self, im0):
        """
        Apply object tracking and extract tracks.

        Args:
            im0 (ndarray): The input image or frame
        """
        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"])

        # Extract tracks for OBB or object detection
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
        else:
            LOGGER.warning("WARNING ⚠️ no tracks found!")
            self.boxes, self.clss, self.track_ids = [], [], []

    def store_tracking_history(self, track_id, box):
        """
        Store object tracking history.

        Args:
            track_id (int): The track ID of the object
            box (list): Bounding box coordinates of the object
        """
        # Store tracking history
        self.track_line = self.track_history[track_id]
        self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    def initialize_region(self):
        """Initialize the counting region and line segment based on config."""
        self.region = [(20, 400), (1080, 404), (1080, 360), (20, 360)] if self.region is None else self.region
        self.r_s = Polygon(self.region) if len(self.region) >= 3 else LineString(self.region)  # region segment
        self.l_s = LineString(
            [(self.region[0][0], self.region[0][1]), (self.region[1][0], self.region[1][1])]
        )  # line segment

    def display_output(self, im0):
        """
        Display the results of the processing, which could involve showing frames, printing counts, or saving results.

        Args:
            im0 (ndarray): The input image or frame
        """
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Ultralytics Solutions", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
