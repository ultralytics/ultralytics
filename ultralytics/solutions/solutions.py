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

    def __init__(self, is_cli=False, **kwargs):
        """
        Initializes the BaseSolution class with configuration settings and the YOLO model.

        Args:
            is_cli (bool): Enables CLI mode if set to True.
            **kwargs (Any): Additional configuration parameters that override defaults.
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
        self.masks = None
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

        self.track_add_args = {  # Tracker additional arguments for advance configuration
            k: self.CFG[k] for k in ["iou", "conf", "device", "max_det", "half", "tracker", "device", "verbose"]
        }  # verbose must be passed to track method; setting it False in YOLO still logs the track information.

        if is_cli and self.CFG["source"] is None:
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
            im0 (np.ndarray): The input image or frame.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        self.tracks = self.model.track(source=im0, persist=True, classes=self.classes, **self.track_add_args)
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes  # Extract tracks for OBB or object detection

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

    def display_output(self, plot_im):
        """
        Display the results of the processing, which could involve showing frames, printing counts, or saving results.

        This method is responsible for visualizing the output of the object detection and tracking process. It displays
        the processed frame with annotations, and allows for user interaction to close the display.

        Args:
            plot_im (numpy.ndarray): The image or frame that has been processed and annotated.

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
            cv2.imshow("Ultralytics Solutions", plot_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()  # Closes current frame window
                return

    def process(self, *args, **kwargs):
        """Process method should be implemented by each Solution subclass."""

    def __call__(self, *args, **kwargs):
        """Allow instances to be called like a function with flexible arguments."""
        result = self.process(*args, **kwargs)  # Call the subclass-specific process method
        if self.CFG["verbose"]:  # extract verbose value to display the output logs if True
            LOGGER.info(f"🚀 Results: {result}")
        return result


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
        queue_counts_display: Displays queue counts in the specified region.
        display_analytics: Displays overall statistics for parking lot management.
        estimate_pose_angle: Calculates the angle between three points in an object pose.
        draw_specific_points: Draws specific keypoints on the image.
        plot_workout_information: Draws a labeled text box on the image.
        plot_angle_and_count_and_stage: Visualizes angle, step count, and stage for workout monitoring.
        plot_distance_and_line: Displays the distance between centroids and connects them with a line.
        display_objects_labels: Annotates bounding boxes with object class labels.
        segmentation_mask: Draws mask for segmented objects and optionally labels them.
        sweep_annotator: Visualizes a vertical sweep line and optional label.
        visioneye: Maps and connects object centroids to a visual "eye" point.
        circle_label: Draws a circular label within a bounding box.
        text_label: Draws a rectangular label within a bounding box.

    Examples:
        >>> annotator = SolutionAnnotator(image)
        >>> annotator.draw_region([(0, 0), (100, 100)], color=(0, 255, 0), thickness=5)
        >>> annotator.display_analytics(
        ...     image, text={"Available Spots": 5}, txt_color=(0, 0, 0), bg_color=(255, 255, 255), margin=10
        ... )
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """
        Initializes the SolutionAnnotator class with an image for annotation.

        Args:
            im (np.ndarray): The image to be annotated.
            line_width (int, optional): Line thickness for drawing on the image.
            font_size (int, optional): Font size for text annotations.
            font (str, optional): Path to the font file.
            pil (bool, optional): Indicates whether to use PIL for rendering text.
            example (str, optional): An example parameter for demonstration purposes.
        """
        super().__init__(im, line_width, font_size, font, pil, example)

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw a region or line on the image.

        Args:
            reg_pts (List[Tuple[int, int]]): Region points (for line 2 points, for region 4+ points).
            color (Tuple[int, int, int]): RGB color value for the region.
            thickness (int): Line thickness for drawing the region.
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # Draw small circles at the corner points
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle

    def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0)):
        """
        Displays queue counts on an image centered at the points with customizable font size and colors.

        Args:
            label (str): Queue counts label.
            points (List[Tuple[int, int]]): Region points for center point calculation to display text.
            region_color (Tuple[int, int, int]): RGB queue region color.
            txt_color (Tuple[int, int, int]): RGB text display color.
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
        Display the overall statistics for parking lots, object counter etc.

        Args:
            im0 (np.ndarray): Inference image.
            text (Dict[str, Any]): Labels dictionary.
            txt_color (Tuple[int, int, int]): Display color for text foreground.
            bg_color (Tuple[int, int, int]): Display color for text background.
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
        Calculate the angle between three points for workout monitoring.

        Args:
            a (List[float]): The coordinates of the first point.
            b (List[float]): The coordinates of the second point (vertex).
            c (List[float]): The coordinates of the third point.

        Returns:
            (float): The angle in degrees between the three points.
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_specific_kpts(self, keypoints, indices=None, radius=2, conf_thresh=0.25):
        """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (List[List[float]]): Keypoints data to be plotted, each in format [x, y, confidence].
            indices (List[int], optional): Keypoint indices to be plotted.
            radius (int, optional): Keypoint radius.
            conf_thresh (float, optional): Confidence threshold for keypoints.

        Returns:
            (np.ndarray): Image with drawn keypoints.

        Note:
            Keypoint format: [x, y] or [x, y, confidence].
            Modifies self.im in-place.
        """
        indices = indices or [2, 5, 7]
        points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thresh]

        # Draw lines between consecutive points
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Draw circles for keypoints
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im

    def plot_workout_information(self, display_text, position, color=(104, 31, 17), txt_color=(255, 255, 255)):
        """
        Draw workout text with a background on the image.

        Args:
            display_text (str): The text to be displayed.
            position (Tuple[int, int]): Coordinates (x, y) on the image where the text will be placed.
            color (Tuple[int, int, int], optional): Text background color.
            txt_color (Tuple[int, int, int], optional): Text foreground color.

        Returns:
            (int): The height of the text.
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
        Plot the pose angle, count value, and step stage for workout monitoring.

        Args:
            angle_text (str): Angle value for workout monitoring.
            count_text (str): Counts value for workout monitoring.
            stage_text (str): Stage decision for workout monitoring.
            center_kpt (List[int]): Centroid pose index for workout monitoring.
            color (Tuple[int, int, int], optional): Text background color.
            txt_color (Tuple[int, int, int], optional): Text foreground color.
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
        Plot the distance and line between two centroids on the frame.

        Args:
            pixels_distance (float): Pixels distance between two bbox centroids.
            centroids (List[Tuple[int, int]]): Bounding box centroids data.
            line_color (Tuple[int, int, int], optional): Distance line color.
            centroid_color (Tuple[int, int, int], optional): Bounding box centroid color.
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
            im0 (np.ndarray): Inference image.
            text (str): Object/class name.
            txt_color (Tuple[int, int, int]): Display color for text foreground.
            bg_color (Tuple[int, int, int]): Display color for text background.
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
        cv2.rectangle(
            im0,
            (int(rect_x1), int(rect_y1)),
            (int(rect_x2), int(rect_y2)),
            tuple(map(int, bg_color)),  # Ensure color values are int
            -1,
        )

        cv2.putText(
            im0,
            text,
            (int(text_x), int(text_y)),
            0,
            self.sf,
            tuple(map(int, txt_color)),  # Ensure color values are int
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def segmentation_mask(self, mask, mask_color=(255, 0, 255), label=None, alpha=0.5):
        """
        Draw an optimized segmentation mask with smooth corners, highlighted edge, and dynamic text box size.

        Args:
            mask (np.ndarray): A 2D array of shape (N, 2) containing the object mask.
            mask_color (Tuple[int, int, int]): RGB color for the mask.
            label (str, optional): Text label for the object.
            alpha (float): Transparency level (0 = fully transparent, 1 = fully opaque).
        """
        if mask.size == 0:
            return

        overlay = self.im.copy()
        mask = np.int32([mask])

        # Approximate polygon for smooth corners with epsilon
        refined_mask = cv2.approxPolyDP(mask, 0.002 * cv2.arcLength(mask, True), True)

        # Apply a highlighter effect by drawing a thick outer shadow
        cv2.polylines(overlay, [refined_mask], isClosed=True, color=mask_color, thickness=self.lw * 3)
        cv2.fillPoly(overlay, [refined_mask], mask_color)  # draw mask with primary color

        # Apply an inner glow effect for extra clarity
        cv2.polylines(overlay, [refined_mask], isClosed=True, color=mask_color, thickness=self.lw)

        self.im = cv2.addWeighted(overlay, alpha, self.im, 1 - alpha, 0)  # blend overlay with the original image

        # Draw label if provided
        if label:
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            text_x, text_y = refined_mask[0][0][0], refined_mask[0][0][1]
            rect_start, rect_end = (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5)
            cv2.rectangle(self.im, rect_start, rect_end, mask_color, -1)
            cv2.putText(
                self.im,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                self.get_txt_color(mask_color),
                self.tf,
            )

    def sweep_annotator(self, line_x=0, line_y=0, label=None, color=(221, 0, 186), txt_color=(255, 255, 255)):
        """
        Draw a sweep annotation line and an optional label.

        Args:
            line_x (int): The x-coordinate of the sweep line.
            line_y (int): The y-coordinate limit of the sweep line.
            label (str, optional): Text label to be drawn in center of sweep line. If None, no label is drawn.
            color (Tuple[int, int, int]): RGB color for the line and label background.
            txt_color (Tuple[int, int, int]): RGB color for the label text.
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
        Perform pinpoint human-vision eye mapping and plotting.

        Args:
            box (List[float]): Bounding box coordinates in format [x1, y1, x2, y2].
            center_point (Tuple[int, int]): Center point for vision eye view.
            color (Tuple[int, int, int]): Object centroid and line color.
            pin_color (Tuple[int, int, int]): Visioneye point color.
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)

    def circle_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=2):
        """
        Draw a label with a background circle centered within a given bounding box.

        Args:
            box (Tuple[float, float, float, float]): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (Tuple[int, int, int]): The background color of the circle (B, G, R).
            txt_color (Tuple[int, int, int]): The color of the text (R, G, B).
            margin (int): The margin between the text and the circle border.
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
        Draw a label with a background rectangle centered within a given bounding box.

        Args:
            box (Tuple[float, float, float, float]): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (Tuple[int, int, int]): The background color of the rectangle (B, G, R).
            txt_color (Tuple[int, int, int]): The color of the text (R, G, B).
            margin (int): The margin between the text and the rectangle border.
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

    Attributes:
        plot_im (np.ndarray): Processed image with counts, blurred, or other effects from solutions.
        in_count (int): The total number of "in" counts in a video stream.
        out_count (int): The total number of "out" counts in a video stream.
        classwise_count (Dict[str, int]): A dictionary containing counts of objects categorized by class.
        queue_count (int): The count of objects in a queue or waiting area.
        workout_count (int): The count of workout repetitions.
        workout_angle (float): The angle calculated during a workout exercise.
        workout_stage (str): The current stage of the workout.
        pixels_distance (float): The calculated distance in pixels between two points or objects.
        available_slots (int): The number of available slots in a monitored area.
        filled_slots (int): The number of filled slots in a monitored area.
        email_sent (bool): A flag indicating whether an email notification was sent.
        total_tracks (int): The total number of tracked objects.
        region_counts (Dict): The count of objects within a specific region.
        speed_dict (Dict[str, float]): A dictionary containing speed information for tracked objects.
        total_crop_objects (int): Total number of cropped objects using ObjectCropper class.
    """

    def __init__(self, **kwargs):
        """
        Initialize a SolutionResults object with default or user-specified values.

        Args:
            **kwargs (Any): Optional arguments to override default attribute values.
        """
        self.plot_im = None
        self.in_count = 0
        self.out_count = 0
        self.classwise_count = {}
        self.queue_count = 0
        self.workout_count = 0
        self.workout_angle = 0.0
        self.workout_stage = None
        self.pixels_distance = 0.0
        self.available_slots = 0
        self.filled_slots = 0
        self.email_sent = False
        self.total_tracks = 0
        self.region_counts = {}
        self.speed_dict = {}
        self.total_crop_objects = 0

        # Override with user-defined values
        self.__dict__.update(kwargs)

    def __str__(self):
        """
        Return a formatted string representation of the SolutionResults object.

        Returns:
            (str): A string representation listing non-null attributes.
        """
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if k != "plot_im" and v not in [None, {}, 0, 0.0, False]  # Exclude `plot_im` explicitly
        }
        return f"SolutionResults({', '.join(f'{k}={v}' for k, v in attrs.items())})"
