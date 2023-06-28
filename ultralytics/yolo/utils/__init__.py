# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import *

UTILS_WARNING = """WARNING ⚠️ 'ultralytics.yolo.utils' is deprecated and will be removed. Please use 'ultralytics.utils' instead.
Or this warning might be related to the pose model you're using, please update your pose model by:
    import torch
    ckpt = torch.load("original-pose.pt")  # it applies to both official and custom pose models.
    torch.save(ckpt, "new-pose.pt")
"""
LOGGER.warning(UTILS_WARNING)
