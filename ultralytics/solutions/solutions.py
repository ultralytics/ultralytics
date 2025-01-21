# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict

import cv2

from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER
from ultralytics.utils.checks import check_imshow, check_requirements


class BaseSolution:
    """
    A base class for managing Ultralytics Solutions.

    This class provides core functionality for various Ultralytics Solutions, including model loading, object tracking,
    and region initialization.

    Attributes:
        LineString (shapely.geometry.LineString): Class for creating line string geometries.
        Polygon (shapely.geometry.Polygon): Class for creating polygon geometries.
        Point (shapely.geometry.Point): Class for creating point geometries.
        CFG (Dict): Configuration dictionary loaded from a YAML file and updated with kwargs.
        region (List[Tuple[int, int]]): List of coordinate tuples defining a region of interest.
        line_width (int): Width of lines used in visualizations.
        model (ultralytics.YOLO): Loaded YOLO model instance.
        names (Dict[int, str]): Dictionary mapping class indices to class names.
        env_check (bool): Flag indicating whether the environment supports image display.
        track_history (collections.defaultdict): Dictionary to store tracking history for each object.

    Methods:
        extract_tracks: Apply object tracking and extract tracks from an input image.
        store_tracking_history: Store object tracking history for a given track ID and bounding box.
        initialize_region: Initialize the counting region and line segment based on configuration.
        display_output: Display the results of processing, including showing frames or saving results.

    Examples:
        >>> solution = BaseSolution(model="yolo11n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> solution.initialize_region()
        >>> image = cv2.imread("image.jpg")
        >>> solution.extract_tracks(image)
        >>> solution.display_output(image)
    """

    def __init__(self, IS_CLI=False, **kwargs):
        """
        Initializes the `BaseSolution` class with configuration settings and the YOLO model for Ultralytics solutions.

        IS_CLI (optional): Enables CLI mode if set.
        """
        check_requirements("shapely>=2.0.0")
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        self.annotator = None  # Initialize annotator
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.r_s = None

        # Load config and update with args
        DEFAULT_SOL_DICT.update(kwargs)
        DEFAULT_CFG_DICT.update(kwargs)
        self.CFG = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}
        LOGGER.info(f"Ultralytics Solutions: ✅ {DEFAULT_SOL_DICT}")

        self.region = self.CFG["region"]  # Store region data for other classes usage
        self.line_width = (
            self.CFG["line_width"] if self.CFG["line_width"] is not None else 2
        )  # Store line_width for usage

        # Load Model and store classes names
        if self.CFG["model"] is None:
            self.CFG["model"] = "yolo11n.pt"
        self.model = YOLO(self.CFG["model"])
        self.names = self.model.names

        self.track_add_args = {  # Tracker additional arguments for advance configuration
            k: self.CFG[k] for k in ["verbose", "iou", "conf", "device", "max_det", "half", "tracker"]
        }

        if IS_CLI and self.CFG["source"] is None:
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            LOGGER.warning(f"⚠️ WARNING: source not provided. using default source {ASSETS_URL}/{d_s}")
            from ultralytics.utils.downloads import safe_download

            safe_download(f"{ASSETS_URL}/{d_s}")  # download source from ultralytics assets
            self.CFG["source"] = d_s  # set default source

        # Initialize environment and region setup
        self.env_check = check_imshow(warn=True)
        self.track_history = defaultdict(list)

    def extract_tracks(self, im0):
        """
        Applies object tracking and extracts tracks from an input image or frame.

        Args:
            im0 (ndarray): The input image or frame.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)

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
        Stores the tracking history of an object.

        This method updates the tracking history for a given object by appending the center point of its
        bounding box to the track line. It maintains a maximum of 30 points in the tracking history.

        Args:
            track_id (int): The unique identifier for the tracked object.
            box (List[float]): The bounding box coordinates of the object in the format [x1, y1, x2, y2].

        Examples:
            >>> solution = BaseSolution()
            >>> solution.store_tracking_history(1, [100, 200, 300, 400])
        """
        # Store tracking history
        self.track_line = self.track_history[track_id]
        self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    def initialize_region(self):
        """Initialize the counting region and line segment based on configuration settings."""
        if self.region is None:
            self.region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )  # region or line

    def display_output(self, im0):
        """
        Display the results of the processing, which could involve showing frames, printing counts, or saving results.

        This method is responsible for visualizing the output of the object detection and tracking process. It displays
        the processed frame with annotations, and allows for user interaction to close the display.

        Args:
            im0 (numpy.ndarray): The input image or frame that has been processed and annotated.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.display_output(frame)

        Notes:
            - This method will only display output if the 'show' configuration is set to True and the environment
              supports image display.
            - The display can be closed by pressing the 'q' key.
        """
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Ultralytics Solutions", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
