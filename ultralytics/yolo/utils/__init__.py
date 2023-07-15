from ultralytics.utils import LOGGER
import importlib
import sys

# Set modules in sys.modules under their old name
sys.modules[f'ultralytics.yolo.utils'] = importlib.import_module(f'ultralytics.utils')

LOGGER.warning(
    "WARNING ⚠️ 'ultralytics.yolo.utils' is deprecated and will be removed. Please use 'ultralytics.utils' instead.")
