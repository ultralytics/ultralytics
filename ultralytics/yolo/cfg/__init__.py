import importlib
import sys

from ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.cfg'] = importlib.import_module('ultralytics.cfg')

LOGGER.warning(
    "WARNING ⚠️ 'ultralytics.yolo.cfg' is deprecated since '8.0.136' and will be removed. Please use 'ultralytics.cfg' instead."
)
