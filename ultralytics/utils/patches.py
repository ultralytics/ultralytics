# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Monkey patches to update/extend functionality of existing functions."""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    """
    Read an image from a file.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.

    Returns:
        (np.ndarray): The read image.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename: str, img: np.ndarray, params=None):
    """
    Write an image to a file.

    Args:
        filename (str): Path to the file to write.
        img (np.ndarray): Image to write.
        params (list of ints, optional): Additional parameters. See OpenCV documentation.

    Returns:
        (bool): True if the file was written, False otherwise.
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray):
    """
    Displays an image in the specified window.

    Args:
        winname (str): Name of the window.
        mat (np.ndarray): Image to be shown.
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


# PyTorch functions ----------------------------------------------------------------------------------------------------
_torch_save = torch.save  # copy to avoid recursion errors


def torch_save(*args, **kwargs):
    """
    Use dill (if exists) to serialize the lambda functions where pickle does not do this. Also adds 3 retries with
    exponential standoff in case of save failure to improve robustness to transient issues.

    Args:
        *args (tuple): Positional arguments to pass to torch.save.
        **kwargs (dict): Keyword arguments to pass to torch.save.
    """
    try:
        import dill as pickle  # noqa
    except ImportError:
        import pickle

    if "pickle_module" not in kwargs:
        kwargs["pickle_module"] = pickle  # noqa

    for i in range(4):  # 3 retries
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError:  # unable to save, possibly waiting for device to flush or anti-virus to finish scanning
            if i == 3:
                raise
            time.sleep((2**i) / 2)  # exponential standoff 0.5s, 1.0s, 2.0s
