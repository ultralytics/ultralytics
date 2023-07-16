import importlib
import sys

from ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.data'] = importlib.import_module('ultralytics.data')
# This is for updating old cls models, or the way in following warning won't work.
sys.modules['ultralytics.yolo.data.augment'] = importlib.import_module('ultralytics.data.augment')

DATA_WARNING = """WARNING ⚠️ 'ultralytics.yolo.data' is deprecated since '8.0.136' and will be removed in '8.1.0'. Please use 'ultralytics.data' instead.
Note this warning may be related to loading older models. You can update your model to current structure with:
    import torch
    ckpt = torch.load("model.pt")  # applies to both official and custom models
    torch.save(ckpt, "updated-model.pt")
"""
LOGGER.warning(DATA_WARNING)
