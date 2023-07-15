import importlib
import sys

from ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.utils'] = importlib.import_module('ultralytics.utils')

UTILS_WARNING = """WARNING ⚠️ 'ultralytics.yolo.utils' is deprecated since '8.0.136' and will be removed. Please use 'ultralytics.utils' instead.
Or this warning might be related to the pose model you're using, please update your pose model by:
    import torch
    ckpt = torch.load("original-pose.pt")  # it applies to both official and custom pose models.
    torch.save(ckpt, "new-pose.pt")
"""
LOGGER.warning(UTILS_WARNING)
