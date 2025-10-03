# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Monkey patches to update/extend functionality of existing functions."""

from __future__ import annotations

import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors


def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    Read an image from a file with multilanguage filename support.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Controls how the image is read.

    Returns:
        (np.ndarray | None): The read image array, or None if reading fails.

    Examples:
        >>> img = imread("path/to/image.jpg")
        >>> img = imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)
    """
    file_bytes = np.fromfile(filename, np.uint8)
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            # Handle RGB images in tif/tiff format
            return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
        return None
    else:
        im = cv2.imdecode(file_bytes, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im  # Always ensure 3 dimensions


def imwrite(filename: str, img: np.ndarray, params: list[int] | None = None) -> bool:
    """
    Write an image to a file with multilanguage filename support.

    Args:
        filename (str): Path to the file to write.
        img (np.ndarray): Image to write.
        params (list[int], optional): Additional parameters for image encoding.

    Returns:
        (bool): True if the file was written successfully, False otherwise.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)  # Create a black image
        >>> success = imwrite("output.jpg", img)  # Write image to file
        >>> print(success)
        True
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray) -> None:
    """
    Display an image in the specified window with multilanguage window name support.

    This function is a wrapper around OpenCV's imshow function that displays an image in a named window. It handles
    multilanguage window names by encoding them properly for OpenCV compatibility.

    Args:
        winname (str): Name of the window where the image will be displayed. If a window with this name already
            exists, the image will be displayed in that window.
        mat (np.ndarray): Image to be shown. Should be a valid numpy array representing an image.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((300, 300, 3), dtype=np.uint8)  # Create a black image
        >>> img[:100, :100] = [255, 0, 0]  # Add a blue square
        >>> imshow("Example Window", img)  # Display the image
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


# PyTorch functions ----------------------------------------------------------------------------------------------------
_torch_save = torch.save


def torch_load(*args, **kwargs):
    """
    Load a PyTorch model with updated arguments to avoid warnings.

    This function wraps torch.load and adds the 'weights_only' argument for PyTorch 1.13.0+ to prevent warnings.

    Args:
        *args (Any): Variable length argument list to pass to torch.load.
        **kwargs (Any): Arbitrary keyword arguments to pass to torch.load.

    Returns:
        (Any): The loaded PyTorch object.

    Notes:
        For PyTorch versions 2.0 and above, this function automatically sets 'weights_only=False'
        if the argument is not provided, to avoid deprecation warnings.
    """
    from ultralytics.utils.torch_utils import TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return torch.load(*args, **kwargs)


def torch_save(*args, **kwargs):
    """
    Save PyTorch objects with retry mechanism for robustness.

    This function wraps torch.save with 3 retries and exponential backoff in case of save failures, which can occur
    due to device flushing delays or antivirus scanning.

    Args:
        *args (Any): Positional arguments to pass to torch.save.
        **kwargs (Any): Keyword arguments to pass to torch.save.

    Examples:
        >>> model = torch.nn.Linear(10, 1)
        >>> torch_save(model.state_dict(), "model.pt")
    """
    for i in range(4):  # 3 retries
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:  # Unable to save, possibly waiting for device to flush or antivirus scan
            if i == 3:
                raise e
            time.sleep((2**i) / 2)  # Exponential backoff: 0.5s, 1.0s, 2.0s


@contextmanager
def arange_patch(args):
    """
    Workaround for ONNX torch.arange incompatibility with FP16.

    https://github.com/pytorch/pytorch/issues/148041.
    """
    if args.dynamic and args.half and args.format == "onnx":
        func = torch.arange

        def arange(*args, dtype=None, **kwargs):
            """Return a 1-D tensor of size with values from the interval and common difference."""
            return func(*args, **kwargs).to(dtype)  # cast to dtype instead of passing dtype

        torch.arange = arange  # patch
        yield
        torch.arange = func  # unpatch
    else:
        yield


@contextmanager
def override_configs(args, overrides: dict[str, Any] | None = None):
    """
    Context manager to temporarily override configurations in args.

    Args:
        args (IterableSimpleNamespace): Original configuration arguments.
        overrides (dict[str, Any]): Dictionary of overrides to apply.

    Yields:
        (IterableSimpleNamespace): Configuration arguments with overrides applied.
    """
    if overrides:
        original_args = copy(args)
        for key, value in overrides.items():
            setattr(args, key, value)
        try:
            yield args
        finally:
            args.__dict__.update(original_args.__dict__)
    else:
        yield args
