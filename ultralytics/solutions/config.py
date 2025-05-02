from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SolutionConfig:
    """
    Manages configuration parameters for Ultralytics Vision AI solutions.

    The SolutionConfig class serves as a centralized configuration container for all the
    Ultralytics solution modules: https://docs.ultralytics.com/solutions/#solutions. It leverages
    Python `dataclass` for clear, type-safe, and maintainable parameter definitions.

    Attributes:
        source (Optional[str]): Path to the input source (image, video, RTSP, etc.). Only usable with Solutions CLI.
        model (Optional[str]): Path to the YOLO model to be used for inference.
        classes (Optional[List[int]]): List of class indices to filter detections.
        show_conf (bool): Whether to show confidence scores on the visual output.
        show_labels (bool): Whether to display class labels on detections.
        region (Optional[List[Tuple[int, int]]]): Polygonal region of interest for analytics.
        colormap (Optional[int]): OpenCV colormap constant for visual overlays (e.g., cv2.COLORMAP_JET).
        show_in (bool): Whether to visualize objects entering the region.
        show_out (bool): Whether to visualize objects exiting the region.
        up_angle (float): Upper angle threshold used in direction- or pose-based filtering.
        down_angle (int): Lower angle threshold for filtering purposes.
        kpts (List[int]): Keypoint indices to monitor, e.g., for pose analytics.
        analytics_type (str): Type of analytics to perform ("line", "zone", etc.).
        figsize (Optional[Tuple[int, int]]): Size of the matplotlib figure used for plots.
        blur_ratio (float): Ratio used to blur sensitive areas in the frame (0.0 to 1.0).
        vision_point (Tuple[int, int]): Reference point for directional tracking or perspective.
        crop_dir (str): Directory path to save cropped detection images.
        line_width (int): Width of lines used in region visualizations.
        show (bool): Whether to display the visual output on screen.
        iou (float): Intersection-over-Union threshold for detection filtering.
        conf (float): Confidence threshold for filtering predictions.
        device (Optional[str]): Device to run inference on (e.g., 'cpu', '0' for GPU).
        max_det (int): Maximum number of detections per image.
        half (bool): Whether to use FP16 precision (only applicable to CUDA devices).
        tracker (str): Path to tracker configuration YAML (e.g., 'botsort.yaml').
        verbose (bool): Enable verbose logging for debugging and transparency.

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
    colormap: Optional[int] = None
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
    line_width: int = 2
    show: bool = False
    iou: float = 0.7
    conf: float = 0.25
    device: Optional[str] = None
    max_det: int = 300
    half: bool = False
    tracker: str = "botsort.yaml"
    verbose: bool = True

    def update(self, **kwargs):
        """Update configuration parameters with new values provided as keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Unknown solution argument: {key}, \n"
                    f"Valid parameters: https://docs.ultralytics.com/solutions/#solutions-arguments"
                )
