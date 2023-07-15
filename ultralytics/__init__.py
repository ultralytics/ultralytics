# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.135'

from ultralytics.engine.model import YOLO
from ultralytics.hub import start
from ultralytics.models import RTDETR, SAM
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'start'  # allow simpler import
