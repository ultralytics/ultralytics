import importlib
import sys

from ultralytics.cfg import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.cfg'] = importlib.import_module('ultralytics.cfg')

LOGGER.warning(
    "WARNING ⚠️ 'ultralytics.yolo.cfg' is deprecated and will be removed. Please use 'ultralytics.cfg' instead.")
