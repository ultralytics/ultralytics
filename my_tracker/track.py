# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial
from pathlib import Path

import torch

from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml


from .byte_track import ByteTrack

TRACKER_MAP = {'bytetrack': ByteTrack}