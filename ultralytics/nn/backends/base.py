# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import YAML


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


def read_export_metadata(file: str | Path) -> dict:
    """Read Ultralytics metadata from an exported model without loading it or importing its framework.

    Directory-based formats write a `metadata.yaml` sidecar, while single-file formats embed it in a length-prefixed
    JSON header (`.engine`), a zip entry (`.torchscript`, `.tflite`) or protobuf string map entries (`.onnx`,
    `.mlpackage`). MNN keeps it in a flatbuffer `bizCode` field and Triton serves it over HTTP, so neither is read here.

    Args:
        file (str | Path): Path to an exported model file or directory.

    Returns:
        (dict): Parsed metadata, empty if the export holds none or predates metadata embedding.
    """
    import json
    import zipfile

    path = Path(file)
    try:
        if path.suffix == ".engine":
            with open(path, "rb") as f:
                length = int.from_bytes(f.read(4), byteorder="little", signed=True)
                return json.loads(f.read(length)) if length > 0 else {}
        if path.suffix == ".tflite":
            return read_tflite_metadata(path) or {}
        if path.suffix == ".torchscript":
            with zipfile.ZipFile(path) as zf:
                name = next((n for n in zf.namelist() if n.endswith("extra/config.txt")), "")
                return json.loads(zf.read(name)) if name else {}
        if path.suffix == ".onnx":
            return _read_proto_metadata(path, (14,))  # ModelProto.metadata_props
        if path.suffix in {".mlpackage", ".mlmodel"}:  # Model.description.metadata.userDefined
            return _read_proto_metadata(next(path.rglob("*.mlmodel"), path) if path.is_dir() else path, (2, 100, 100))
    except (OSError, ValueError, LookupError, zipfile.BadZipFile):
        return {}  # a third-party or truncated file carries no Ultralytics header

    sidecar = (path if path.is_dir() else path.parent) / "metadata.yaml"  # openvino, paddle, ncnn, saved_model
    if path.suffix == ".pb":  # a frozen graph keeps its metadata in the sibling saved_model directory
        sidecar = next(path.resolve().parent.rglob(f"{path.stem}_saved_model*/metadata.yaml"), sidecar)
    return YAML.load(sidecar) if sidecar.exists() else {}


def _read_proto_metadata(file: Path, field_path: tuple[int, ...]) -> dict:
    """Collect a protobuf `map<string, string>` at a nested field path, stepping over every field off the path.

    Args:
        file (Path): Path to the protobuf file, i.e. ONNX with `metadata_props` at top-level field 14 or CoreML with
            `description.metadata.userDefined` at fields 2, 100, 100.
        field_path (tuple[int, ...]): Field numbers to descend, the last holding the repeated map entries.

    Returns:
        (dict): Map entries as string key-value pairs.
    """
    import mmap

    def varint(buf, i):
        """Decode a base-128 varint at offset `i`, returning the value and the offset after it."""
        value = shift = 0
        while True:
            value |= (buf[i] & 0x7F) << shift
            i, shift = i + 1, shift + 7
            if not buf[i - 1] & 0x80:
                return value, i

    def fields(buf):
        """Yield `(number, payload)` for each length-delimited field in a message, skipping varint fields."""
        i = 0
        while i < len(buf):
            tag, i = varint(buf, i)
            if tag & 7 == 0:  # varint field, i.e. ir_version
                _, i = varint(buf, i)
            elif tag & 7 == 2:  # length-delimited field, i.e. graph, weights or a nested message
                length, i = varint(buf, i)
                yield tag >> 3, buf[i : i + length]
                i += length
            else:
                return  # these protos carry no fixed-width fields

    with open(file, "rb") as f:
        messages = [memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))]  # zero-copy slicing
    for number in field_path:  # the last field holds the StringStringEntryProto entries, 1=key and 2=value
        messages = [payload for m in messages for n, payload in fields(m) if n == number]
    return {bytes(e[1]).decode(): bytes(e.get(2, b"")).decode() for e in map(dict, map(fields, messages))}


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
        base_model (bool): Whether the loaded model is an Ultralytics `BaseModel`, and so implements the `augment` and
            `embed` forward arguments.
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
        self.base_model = False
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
