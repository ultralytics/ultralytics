import importlib
import sys

from ultralytics.data import LOGGER

# Set modules in sys.modules under their old name
sys.modules[f'ultralytics.yolo.data'] = importlib.import_module(f'ultralytics.data')

LOGGER.warning(
    "WARNING ⚠️ 'ultralytics.yolo.data' is deprecated and will be removed. Please use 'ultralytics.data' instead.")
