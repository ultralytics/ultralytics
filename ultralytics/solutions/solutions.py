# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator


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

        self.LOGGER = LOGGER  # Store logger object to be used in multiple solution classes

        # Load config and update with args
        DEFAULT_SOL_DICT.update(kwargs)
        DEFAULT_CFG_DICT.update(kwargs)
        self.CFG = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}
        self.LOGGER.info(f"Ultralytics Solutions: ✅ {DEFAULT_SOL_DICT}")

        self.region = self.CFG["region"]  # Store region data for other classes usage
        self.line_width = (
            self.CFG["line_width"] if self.CFG["line_width"] is not None else 2
        )  # Store line_width for usage

        # Load Model and store classes names
        if self.CFG["model"] is None:
            self.CFG["model"] = "yolo11n.pt"
        self.model = YOLO(self.CFG["model"])
        self.names = self.model.names
        self.classes = self.CFG["classes"]
        self.verbose = self.CFG["verbose"]

        self.track_add_args = {  # Tracker additional arguments for advance configuration
            k: self.CFG[k] for k in ["verbose", "iou", "conf", "device", "max_det", "half", "tracker"]
        }

        if IS_CLI and self.CFG["source"] is None:
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            self.LOGGER.warning(f"⚠️ WARNING: source not provided. using default source {ASSETS_URL}/{d_s}")
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
        self.tracks = self.model.track(source=im0, persist=True, classes=self.classes, **self.track_add_args)

        # Extract tracks for OBB or object detection
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        self.masks = (
            self.tracks[0].masks.xy if hasattr(self.tracks[0], "masks") and self.tracks[0].masks is not None else None
        )

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
        else:
            self.LOGGER.warning("WARNING ⚠️ no tracks found!")
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


class SolutionAnnotator(Annotator):
    """
    A specialized annotator class for visualizing and analyzing computer vision tasks.

    This class extends the base Annotator class, providing additional methods for drawing regions, centroids, tracking
    trails, and visual annotations for Ultralytics Solutions: https://docs.ultralytics.com/solutions/.
    and parking management.

    Attributes:
        im (np.ndarray): The image being annotated.
        line_width (int): Thickness of lines used in annotations.
        font_size (int): Size of the font used for text annotations.
        font (str): Path to the font file used for text rendering.
        pil (bool): Whether to use PIL for text rendering.
        example (str): An example attribute for demonstration purposes.

    Methods:
        draw_region: Draws a region using specified points, colors, and thickness.
        draw_centroid_and_tracks: Draws the centroid of an object and its tracking trail.
        queue_counts_display: Displays queue counts in the specified region.
        display_analytics: Displays overall statistics for parking lot management.
        estimate_pose_angle: Calculates the angle between three points in an object pose.
        draw_specific_points: Draws specific keypoints on the image.
        plot_workout_information: Draws a labeled text box on the image.
        plot_angle_and_count_and_stage: Visualizes angle, step count, and stage for workout monitoring.
        plot_distance_and_line: Displays the distance between centroids and connects them with a line.
        display_objects_labels: Annotates bounding boxes with object class labels.
        seg_bbox: Draws contours for segmented objects and optionally labels them.
        sweep_annotator: Visualizes a vertical sweep line and optional label.
        visioneye: Maps and connects object centroids to a visual "eye" point.
        circle_label: Draws a circular label within a bounding box.
        text_label: Draws a rectangular label within a bounding box.

    Examples:
        >>> annotator = SolutionAnnotator(image)
        >>> annotator.draw_region([(0, 0), (100, 100)], color=(0, 255, 0), thickness=5)
        >>> annotator.draw_centroid_and_tracks(track=[[10, 20], [30, 40]], color=(255, 0, 255))
        >>> annotator.display_analytics(
        ...     image, text={"Available Spots": 5}, txt_color=(0, 0, 0), bg_color=(255, 255, 255), margin=10
        ... )
    """

    # Overrides the `__init__` method from Annotator class
    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """
        Initializes the `SolutionAnnotator` class with an image for annotation.

        Args:
            im (np.ndarray): The image to be annotated.
            line_width (int, optional): Line thickness for drawing on the image. Defaults to None.
            font_size (int, optional): Font size for text annotations. Defaults to None.
            font (str, optional): Path to the font file. Defaults to "Arial.ttf".
            pil (bool, optional): Indicates whether to use PIL for rendering text. Defaults to False.
            example (str, optional): An example parameter for demonstration purposes. Defaults to "abc".
        """
        super().__init__(im, line_width, font_size, font, pil, example)

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw region line.

        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # Draw small circles at the corner points
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        Draw centroid point and track trails.

        Args:
            track (list): object tracking points for trails display
            color (tuple): tracks line color
            track_thickness (int): track line thickness value
        """
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0)):
        """
        Displays queue counts on an image centered at the points with customizable font size and colors.

        Args:
            label (str): Queue counts label.
            points (tuple): Region points for center point calculation to display text.
            region_color (tuple): RGB queue region color.
            txt_color (tuple): RGB text display color.
        """
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Draw text
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_analytics(self, im0, text, txt_color, bg_color, margin):
        """
        Display the overall statistics for parking lots.

        Args:
            im0 (ndarray): Inference image.
            text (dict): Labels dictionary.
            txt_color (tuple): Display color for text foreground.
            bg_color (tuple): Display color for text background.
            margin (int): Gap between text and rectangle for better display.
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """
        Calculate the pose angle for object.

        Args:
            a (float) : The value of pose point a
            b (float): The value of pose point b
            c (float): The value o pose point c

        Returns:
            angle (degree): Degree value of angle between three points
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_specific_points(self, keypoints, indices=None, radius=2, conf_thres=0.25):
        """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list): Keypoints data to be plotted.
            indices (list, optional): Keypoint indices to be plotted. Defaults to [2, 5, 7].
            radius (int, optional): Keypoint radius. Defaults to 2.
            conf_thres (float, optional): Confidence threshold for keypoints. Defaults to 0.25.

        Returns:
            (numpy.ndarray): Image with drawn keypoints.

        Note:
            Keypoint format: [x, y] or [x, y, confidence].
            Modifies self.im in-place.
        """
        indices = indices or [2, 5, 7]
        points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thres]

        # Draw lines between consecutive points
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Draw circles for keypoints
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im

    def plot_workout_information(self, display_text, position, color=(104, 31, 17), txt_color=(255, 255, 255)):
        """
        Draw text with a background on the image.

        Args:
            display_text (str): The text to be displayed.
            position (tuple): Coordinates (x, y) on the image where the text will be placed.
            color (tuple, optional): Text background color
            txt_color (tuple, optional): Text foreground color
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, self.sf, self.tf)

        # Draw background rectangle
        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
            color,
            -1,
        )
        # Draw text
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)

        return text_height

    def plot_angle_and_count_and_stage(
        self, angle_text, count_text, stage_text, center_kpt, color=(104, 31, 17), txt_color=(255, 255, 255)
    ):
        """
        Plot the pose angle, count value, and step stage.

        Args:
            angle_text (str): Angle value for workout monitoring
            count_text (str): Counts value for workout monitoring
            stage_text (str): Stage decision for workout monitoring
            center_kpt (list): Centroid pose index for workout monitoring
            color (tuple, optional): Text background color
            txt_color (tuple, optional): Text foreground color
        """
        # Format text
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

        # Draw angle, count and stage text
        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
        )
        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
        )
        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
        )

    def plot_distance_and_line(
        self, pixels_distance, centroids, line_color=(104, 31, 17), centroid_color=(255, 0, 255)
    ):
        """
        Plot the distance and line on frame.

        Args:
            pixels_distance (float): Pixels distance between two bbox centroids.
            centroids (list): Bounding box centroids data.
            line_color (tuple, optional): Distance line color.
            centroid_color (tuple, optional): Bounding box centroid color.
        """
        # Get the text size
        text = f"Pixels Distance: {pixels_distance:.2f}"
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)

        # Define corners with 10-pixel margin and draw rectangle
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

        # Calculate the position for the text with a 10-pixel margin and draw text
        text_position = (25, 25 + text_height_m + 10)
        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )

        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
        """
        Display the bounding boxes labels in parking management app.

        Args:
            im0 (ndarray): Inference image.
            text (str): Object/class name.
            txt_color (tuple): Display color for text foreground.
            bg_color (tuple): Display color for text background.
            x_center (float): The x position center point for bounding box.
            y_center (float): The y position center point for bounding box.
            margin (int): The gap between text and rectangle for better display.
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)

    def seg_bbox(self, mask, mask_color=(255, 0, 255), label=None, txt_color=(255, 255, 255)):
        """
        Function for drawing segmented object in bounding box shape.

        Args:
            mask (np.ndarray): A 2D array of shape (N, 2) containing the contour points of the segmented object.
            mask_color (tuple): RGB color for the contour and label background.
            label (str, optional): Text label for the object. If None, no label is drawn.
            txt_color (tuple): RGB color for the label text.
        """
        txt_color = self.get_txt_color(mask_color)
        if mask.size == 0:  # no masks to plot
            return

        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)
        text_size, _ = cv2.getTextSize(label, 0, self.sf, self.tf)

        if label:
            cv2.rectangle(
                self.im,
                (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
                (int(mask[0][0]) + text_size[0] // 2 + 10, int(mask[0][1] + 10)),
                mask_color,
                -1,
            )
            cv2.putText(
                self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1])), 0, self.sf, txt_color, self.tf
            )

    def sweep_annotator(self, line_x=0, line_y=0, label=None, color=(221, 0, 186), txt_color=(255, 255, 255)):
        """
        Function for drawing a sweep annotation line and an optional label.

        Args:
            line_x (int): The x-coordinate of the sweep line.
            line_y (int): The y-coordinate limit of the sweep line.
            label (str, optional): Text label to be drawn in center of sweep line. If None, no label is drawn.
            color (tuple): RGB color for the line and label background.
            txt_color (tuple): RGB color for the label text.
        """
        # Draw the sweep line
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

        # Draw label, if provided
        if label:
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),
                (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
                color,
                -1,
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf,
            )

    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255)):
        """
        Function for pinpoint human-vision eye mapping and plotting.

        Args:
            box (list): Bounding box coordinates
            center_point (tuple): center point for vision eye view
            color (tuple): object centroid and line color value
            pin_color (tuple): visioneye point color value
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)

    def circle_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=2):
        """
        Draws a label with a background circle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        # If label have more than 3 characters, skip other characters, due to circle size
        if len(label) > 3:
            print(
                f"Length of label is {len(label)}, initial 3 label characters will be considered for circle annotation!"
            )
            label = label[:3]

        # Calculate the center of the box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the text size
        text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]
        # Calculate the required radius to fit the text with the margin
        required_radius = int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin
        # Draw the circle with the required radius
        cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
        # Calculate the position for the text
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Draw the text
        cv2.putText(
            self.im,
            str(label),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def text_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=5):
        """
        Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        # Calculate the center of the bounding box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the size of the text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        # Calculate the top-left corner of the text (to center it)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Calculate the coordinates of the background rectangle
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # Draw the background rectangle
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.1,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )


class SolutionResults:
    """
    A class to encapsulate the results of Ultralytics Solutions.

    This class is designed to store and manage various outputs generated by the solution pipeline, including counts,
    angles, and workout stages.

    Methods:
        __init__: Initializes the attributes with default or provided values.
        summary: Returns the summary of Ultralytics Solutions: https://docs.ultralytics.com/solutions/
    """

    def __init__(self, **kwargs):
        """
        Initializes a SolutionResults object with default or user-specified values for its attributes.

        This method sets up the various attributes used to store the results generated by the Ultralytics solution pipeline.
        It allows flexible initialization with optional arguments, enabling customization for specific use cases.

        Args:
            im0 (ndarray): Preprocessed image with counts, blurred or other effect from solutions.
            in_count (int, optional): The total number of "in" counts in a video stream. Default is 0.
            out_count (int, optional): The total number of "out" counts in a video stream. Default is 0.
            classwise_count (Dict[str, int], optional): A dictionary containing counts of objects categorized by class. Default is None.
            queue_count (int, optional): The count of objects in a queue or waiting area. Default is 0.
            workout_count (int, optional): The count of workout repetitions (e.g., 10). Default is 0.
            workout_angle (float, optional): The angle calculated during a workout exercise (e.g., 55 degrees). Default is 0.
            workout_stage (str, optional): The current stage of the workout (e.g., "up", "down"). Default is None.
            pixels_distance (float, optional): The calculated distance in pixels between two points or objects. Default is 0.
            available_slots (int, optional): The number of available slots in a monitored area (e.g., parking slots). Default is 0.
            filled_slots (int, optional): The number of filled slots in a monitored area. Default is 0.
            email_sent (bool, optional): A flag indicating whether an email notification was sent. Default is False.
            total_tracks (int, optional): The total number of tracked objects. Default is 0.
            region_counts (int, optional): The count of objects within a specific region. Default is 0.
            speed_dict (Dict[str, float], optional): A dictionary containing speed information for tracked objects. Default is None.
            total_crop_objects (int, optional): Total number of cropped objects using ObjectCropper class.

        Examples:
            >>> results = SolutionResults(
            ...     im0=np.zeros((256, 256, 3), dtype=np.uint8),
            ...     in_count=5,
            ...     out_count=3,
            ...     classwise_count={"person": 10, "car": 2},
            ...     queue_count=4,
            ...     workout_count=15,
            ...     workout_angle=90.0,
            ...     workout_stage="down",
            ...     pixels_distance=150.5,
            ...     available_slots=5,
            ...     filled_slots=10,
            ...     email_sent=True,
            ...     total_tracks=50,
            ...     region_counts=20,
            ...     speed_dict={"object_1": 3.5, "object_2": 2.8},
            ...     total_crop_objects=5,
            ... )
        """
        defaults = {
            "im0": None,
            "in_count": 0,
            "out_count": 0,
            "classwise_count": None,
            "queue_count": 0,
            "workout_count": 0,
            "workout_angle": 0,
            "workout_stage": None,
            "pixels_distance": 0,
            "available_slots": 0,
            "filled_slots": 0,
            "email_sent": False,
            "total_tracks": 0,
            "region_counts": 0,
            "speed_dict": None,
            "total_crop_objects": 0,
        }
        self.__dict__.update({**defaults, **kwargs})

    def summary(self, verbose=False):
        """
        Generates a summary of the current state of the SolutionResults object.

        This method consolidates all attributes of the object into a single dictionary, providing an organized view of the
        results stored in the instance. It is particularly useful for reporting, serialization, or debugging purposes.

        Returns:
            summary (dict): A dictionary containing key-value pairs representing the current state of all attributes in the object.
        """
        # Get the dictionary of the attributes, convert im0 large array to ndarray string for Logging
        self.result_summary = {
            k: ("np.ndarray" if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items() if k != "summary"
        }

        if verbose:
            LOGGER.info(self.result_summary)

        return {k: v for k, v in self.__dict__.items() if k != "summary"}
