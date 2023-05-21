# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Monkey patches to update/extend functionality of existing functions
"""

from pathlib import Path

import cv2
import numpy as np
import torch

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors


def imread(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename, img):
    try:
        cv2.imencode(Path(filename).suffix, img)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(path, im):
    _imshow(path.encode('unicode_escape').decode(), im)


# PyTorch functions ----------------------------------------------------------------------------------------------------
_torch_save = torch.save  # copy to avoid recursion errors


def torch_save(*args, **kwargs):
    # Use dill (if exists) to serialize the lambda functions where pickle does not do this
    try:
        import dill as pickle
    except ImportError:
        import pickle

    if 'pickle_module' not in kwargs:
        kwargs['pickle_module'] = pickle
    return _torch_save(*args, **kwargs)
