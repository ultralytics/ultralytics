import importlib
import sys

from ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.v8'] = importlib.import_module('ultralytics.models.yolo')

LOGGER.warning("WARNING ⚠️ 'ultralytics.yolo.v8' is deprecated since '8.0.136' and will be removed in '8.1.0'. "
               "Please use 'ultralytics.models.yolo' instead.")
