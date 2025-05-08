# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2


@dataclass
class SolutionConfig:
    """
    Manages configuration parameters for Ultralytics Vision AI solutions.

    The SolutionConfig class serves as a centralized configuration container for all the
    Ultralytics solution modules: https://docs.ultralytics.com/solutions/#solutions.
    It leverages Python `dataclass` for clear, type-safe, and maintainable parameter definitions.

    Attributes:
        source (Optional[str]): Path to the input source (video, RTSP, etc.). Only usable with Solutions CLI.
        model (Optional[str]): Path to the Ultralytics YOLO model to be used for inference.
        classes (Optional[List[int]]): List of class indices to filter detections.
        show_conf (bool): Whether to show confidence scores on the visual output.
        show_labels (bool): Whether to display class labels on visual output.
        region (Optional[List[Tuple[int, int]]]): Polygonal region or line for object counting.
        colormap (Optional[int]): OpenCV colormap constant for visual overlays (e.g., cv2.COLORMAP_JET).
        show_in (bool): Whether to display count number for objects entering the region.
        show_out (bool): Whether to display count number for objects leaving the region.
        up_angle (float): Upper angle threshold used in pose-based workouts monitoring.
        down_angle (int): Lower angle threshold used in pose-based workouts monitoring.
        kpts (List[int]): Keypoint indices to monitor, e.g., for pose analytics.
        analytics_type (str): Type of analytics to perform ("line", "area", "bar", "pie", etc.).
        figsize (Optional[Tuple[int, int]]): Size of the matplotlib figure used for analytical plots (width, height).
        blur_ratio (float): Ratio used to blur objects in the video frames (0.0 to 1.0).
        vision_point (Tuple[int, int]): Reference point for directional tracking or perspective drawing.
        crop_dir (str): Directory path to save cropped detection images.
        json_file (str): Path to a JSON file containing data for parking areas.
        line_width (int): Width for visual display i.e. bounding boxes, keypoints, counts.
        records (int): Number of detection records to send email alerts.
        fps (float): Frame rate (Frames Per Second) for speed estimation calculation.
        max_hist (int): Maximum number of historical points or states stored per tracked object for speed estimation.
        meter_per_pixel (float): Scale for real-world measurement, used in speed or distance calculations.
        max_speed (int): Maximum speed limit (e.g., km/h or mph) used in visual alerts or constraints.
        show (bool): Whether to display the visual output on screen.
        iou (float): Intersection-over-Union threshold for detection filtering.
        conf (float): Confidence threshold for keeping predictions.
        device (Optional[str]): Device to run inference on (e.g., 'cpu', '0' for CUDA GPU).
        max_det (int): Maximum number of detections allowed per video frame.
        half (bool): Whether to use FP16 precision (requires a supported CUDA device).
        tracker (str): Path to tracking configuration YAML file (e.g., 'botsort.yaml').
        verbose (bool): Enable verbose logging output for debugging or diagnostics.
        data (str): Path to image directory used for similarity search.

    Methods:
        update: Update the configuration with user-defined keyword arguments and raise error on invalid keys.

    Examples:
        >>> from ultralytics.solutions.config import SolutionConfig
        >>> cfg = SolutionConfig(model="yolo11n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> cfg.update(show=False, conf=0.3)
        >>> print(cfg.model)
    """

    source: Optional[str] = None
    model: Optional[str] = None
    classes: Optional[List[int]] = None
    show_conf: bool = True
    show_labels: bool = True
    region: Optional[List[Tuple[int, int]]] = None
    colormap: Optional[int] = cv2.COLORMAP_DEEPGREEN
    show_in: bool = True
    show_out: bool = True
    up_angle: float = 145.0
    down_angle: int = 90
    kpts: List[int] = field(default_factory=lambda: [6, 8, 10])
    analytics_type: str = "line"
    figsize: Optional[Tuple[int, int]] = (12.8, 7.2)
    blur_ratio: float = 0.5
    vision_point: Tuple[int, int] = (20, 20)
    crop_dir: str = "cropped-detections"
    json_file: str = None
    line_width: int = 2
    records: int = 5
    fps: float = 30.0
    max_hist: int = 5
    meter_per_pixel: float = 0.05
    max_speed: int = 120
    show: bool = False
    iou: float = 0.7
    conf: float = 0.25
    device: Optional[str] = None
    max_det: int = 300
    half: bool = False
    tracker: str = "botsort.yaml"
    verbose: bool = True
    data: str = "images"

    def update(self, **kwargs):
        """Update configuration parameters with new values provided as keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"❌ {key} is not a valid solution argument, available arguments here: https://docs.ultralytics.com/solutions/#solutions-arguments"
                )
        return self
