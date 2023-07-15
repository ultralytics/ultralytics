import importlib
import sys

from ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.data'] = importlib.import_module('ultralytics.data')

LOGGER.warning(
    "WARNING ⚠️ 'ultralytics.yolo.data' is deprecated and will be removed. Please use 'ultralytics.data' instead.")
