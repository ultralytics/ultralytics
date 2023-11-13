import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

from supervision.annotators.core import (
    BlurAnnotator,
    BoundingBoxAnnotator,
    BoxCornerAnnotator,
    BoxMaskAnnotator,
    CircleAnnotator,
    DotAnnotator,
    EllipseAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    TraceAnnotator,
)
from supervision.annotators.utils import ColorLookup
from supervision.classification.core import Classifications
from supervision.dataset.core import (
    BaseDataset,
    ClassificationDataset,
    DetectionDataset,
)
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.core import Detections
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator
from supervision.detection.utils import (
    box_iou_batch,
    filter_polygons_by_area,
    mask_to_polygons,
    mask_to_xyxy,
    non_max_suppression,
    polygon_to_mask,
    polygon_to_xyxy,
)
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import (
    draw_filled_rectangle,
    draw_image,
    draw_line,
    draw_polygon,
    draw_rectangle,
    draw_text,
)
from supervision.geometry.core import Point, Position, Rect
from supervision.geometry.utils import get_polygon_center
from supervision.metrics.detection import ConfusionMatrix, MeanAveragePrecision
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.utils.file import list_files_with_extensions
from supervision.utils.image import ImageSink, crop_image
from supervision.utils.notebook import plot_image, plot_images_grid
from supervision.utils.video import (
    FPSMonitor,
    VideoInfo,
    VideoSink,
    get_video_frames_generator,
    process_video,
)
