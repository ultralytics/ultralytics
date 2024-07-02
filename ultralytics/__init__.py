# Ultralytics YOLO 🚀, AGPL-3.0 license

<<<<<<< HEAD
__version__ = "8.2.48"
=======
__version__ = "8.0.200"
>>>>>>> 2d87fb01604a79af96d1d3778626415fb4b54ac9

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

<<<<<<< HEAD
settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
=======
__all__ = "__version__", "YOLO", "NAS", "SAM", "FastSAM", "RTDETR", "checks", "download", "settings"
>>>>>>> 2d87fb01604a79af96d1d3778626415fb4b54ac9
