# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .ai_gym import AIGym
from .analytics import Analytics
from .distance_calculation import DistanceCalculation
from .heatmap import Heatmap
from .instance_segmentation import InstanceSegmentation
from .object_blurrer import ObjectBlurrer
from .object_counter import ObjectCounter
from .object_cropper import ObjectCropper
from .parking_management import ParkingManagement, ParkingPtsSelection
from .queue_management import QueueManager
from .region_counter import RegionCounter
from .security_alarm import SecurityAlarm
from .similarity_search import SearchApp, VisualAISearch
from .speed_estimation import SpeedEstimator
from .streamlit_inference import Inference
from .trackzone import TrackZone
from .vision_eye import VisionEye

__all__ = (
    "AIGym",
    "Analytics",
    "DistanceCalculation",
    "Heatmap",
    "Inference",
    "InstanceSegmentation",
    "ObjectBlurrer",
    "ObjectCounter",
    "ObjectCropper",
    "ParkingManagement",
    "ParkingPtsSelection",
    "QueueManager",
    "RegionCounter",
    "RegionSelector",
    "SearchApp",
    "SecurityAlarm",
    "SpeedEstimator",
    "TrackZone",
    "VisionEye",
    "VisualAISearch",
    "select_region",
)


def __getattr__(name):
    """Lazily import RegionSelector and select_region to avoid cv2 side-effects at import time."""
    if name == "RegionSelector":
        from .region_selector import RegionSelector

        return RegionSelector
    if name == "select_region":
        from .region_selector import select_region

        return select_region
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Ensure lazy attributes appear in dir() and IDE autocomplete."""
    return sorted([*list(globals().keys()), "RegionSelector", "select_region"])
