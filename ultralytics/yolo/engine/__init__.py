import importlib
import sys

from ultralytics.engine import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.engine'] = importlib.import_module('ultralytics.engine')

LOGGER.warning(
    "WARNING ⚠️ 'ultralytics.yolo.engine' is deprecated and will be removed. Please use 'ultralytics.engine' instead.")
