# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.data import *
from ultralytics.utils import LOGGER

DATA_WARNING = """WARNING ⚠️ 'ultralytics.yolo.data' is deprecated and will be removed. Please use 'ultralytics.data' instead.
Or this warning might be related to the classify model you're using, please update your classify model by:
    import torch
    ckpt = torch.load("original-cls.pt")  # it applies to both official and custom classify models.
    torch.save(ckpt, "new-cls.pt")
"""
LOGGER.warning(DATA_WARNING)
