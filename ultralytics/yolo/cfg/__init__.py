from ultralytics.cfg import LOGGER
import importlib
import sys

# Set modules in sys.modules under their old name
sys.modules[f'ultralytics.yolo.cfg'] = importlib.import_module(f'ultralytics.cfg')

LOGGER.warning(
    "WARNING ⚠️ 'ultralytics.yolo.cfg' is deprecated and will be removed. Please use 'ultralytics.cfg' instead.")
