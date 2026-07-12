# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


def read_tflite_metadata(file: str | Path) -> dict | None:
    """Read Ultralytics metadata embedded in a ``.tflite`` file.

    Ultralytics appends metadata to the end of ``.tflite`` flatbuffers as a zip entry (``metadata.json`` for
    litert-torch/single-file exports, or a single literal-dict entry for legacy onnx2tf exports). Returns the parsed
    metadata dict, or ``None`` if the file has no readable embedded metadata.

    Args:
        file (str | Path): Path to the ``.tflite`` model file.

    Returns:
        (dict | None): Parsed metadata dictionary, or ``None`` if absent or unreadable.
    """
    import json
    import zipfile

    try:
        with zipfile.ZipFile(file, "r") as zf:
            names = zf.namelist()
            if "metadata.json" in names:
                return json.loads(zf.read("metadata.json"))
            if names:  # legacy onnx2tf exports store a single Python-literal dict entry
                return ast.literal_eval(zf.read(names[0]).decode("utf-8"))
    except (zipfile.BadZipFile, SyntaxError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return None


class BaseBackend(ABC):
    """Base class for all inference backends.

    This abstract class defines the interface that all inference backends must implement. It provides common
    functionality for model loading, metadata processing, and device management.

    Attributes:
        model: The underlying inference model or runtime session.
        device (torch.device): The device to run inference on.
        fp16 (bool): Whether to use FP16 (half-precision) inference.
        nhwc (bool): Whether the model expects NHWC input format instead of NCHW.
        stride (int): Model stride, typically 32 for YOLO models.
        names (dict): Dictionary mapping class indices to class names.
        task (str | None): The task type (detect, segment, semantic, classify, pose, obb).
        batch (int): Batch size for inference.
        imgsz (tuple): Input image size as (height, width).
        channels (int): Number of input channels, typically 3 for RGB.
        end2end (bool): Whether the model includes end-to-end NMS post-processing.
        dynamic (bool): Whether the model supports dynamic input shapes.
        metadata (dict): Model metadata dictionary containing export configuration.
    """

    def __init__(self, weight: str | torch.nn.Module, device: torch.device | str, fp16: bool = False):
        """Initialize the base backend with common attributes and load the model.

        Args:
            weight (str | torch.nn.Module): Path to the model weights file or a PyTorch module instance.
            device (torch.device | str): Device to run inference on (e.g., 'cpu', 'cuda:0').
            fp16 (bool): Whether to use FP16 half-precision inference.
        """
        self.device = device
        self.fp16 = fp16
        self.nhwc = False
        self.stride = 32
        self.names = {}
        self.task = None
        self.batch = 1
        self.channels = 3
        self.end2end = False
        self.dynamic = False
        self.metadata = {}
        self.model = None
        self.load_model(weight)

    @abstractmethod
    def load_model(self, weight: str | torch.nn.Module) -> None:
        """Load the model from a weights file or module instance.

        Args:
            weight (str | torch.nn.Module): Path to model weights or a PyTorch module.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, im: torch.Tensor) -> Any:
        """Run inference on the input image tensor.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (Any): The raw output from the model's forward pass, which may require post-processing.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        """Allow the backend instance to be called directly to perform inference, forwarding arguments to the `forward`
        method.
        """
        return self.forward(*args, **kwargs)

    def apply_metadata(self, metadata: dict | None) -> None:
        """Process and apply model metadata to backend attributes.

        Handles type conversions for common metadata fields (e.g., stride, batch, names) and sets them as
        instance attributes. Also resolves end-to-end NMS and dynamic shape settings from export args.

        Args:
            metadata (dict | None): Dictionary containing metadata key-value pairs from model export.
        """
        if not metadata:
            return

        # Store raw metadata
        self.metadata = metadata

        # Process type conversions for known fields
        for k, v in metadata.items():
            if k in {"stride", "batch", "channels"}:
                metadata[k] = int(v)
            elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
                metadata[k] = ast.literal_eval(v)

        # Handle models exported with end-to-end NMS
        metadata["end2end"] = metadata.get("end2end", False) or metadata.get("args", {}).get("nms", False)
        metadata["dynamic"] = metadata.get("args", {}).get("dynamic", self.dynamic)

        # Apply all metadata fields as backend attributes
        for k, v in metadata.items():
            setattr(self, k, v)
