import importlib
import sys

from ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.utils'] = importlib.import_module('ultralytics.utils')

LOGGER.warning(
    "WARNING ⚠️ 'ultralytics.yolo.utils' is deprecated and will be removed. Please use 'ultralytics.utils' instead.")
