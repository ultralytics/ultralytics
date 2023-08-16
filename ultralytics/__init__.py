# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.155'

from ultralytics.hub import start
from ultralytics.models import RTDETR, SAM, YOLO
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download
from ultralytics.data.annotator import auto_annotate_segment, auto_annotate_detect

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'start', 'settings', 'auto_annotate_segment', 'auto_annotate_detect'  # allow simpler import
