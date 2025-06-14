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
    "ObjectCounter",
    "ObjectCropper",
    "ObjectBlurrer",
    "AIGym",
    "RegionCounter",
    "SecurityAlarm",
    "Heatmap",
    "InstanceSegmentation",
    "VisionEye",
    "SpeedEstimator",
    "DistanceCalculation",
    "QueueManager",
    "ParkingManagement",
    "ParkingPtsSelection",
    "Analytics",
    "Inference",
    "TrackZone",
    "SearchApp",
    "VisualAISearch",
)


def patch_streamlit_watcher():
    """Patch Streamlit's module watcher to safely skip torch.classes and avoid __path__ crashes."""
    try:
        import streamlit.watcher.local_sources_watcher as lsw

        lsw._orig_get_module_paths = getattr(lsw, "_orig_get_module_paths", lsw.get_module_paths)
        lsw.get_module_paths = (
            lambda m: []
            if getattr(m, "__name__", "").startswith("torch.classes")
            else (lsw._orig_get_module_paths(m) if callable(lsw._orig_get_module_paths) else [])
        )
    except Exception:
        pass


patch_streamlit_watcher()
