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


def read_export_metadata(file: str | Path) -> dict:
    """Read Ultralytics metadata from an exported model without loading it or importing its framework.

    Every Ultralytics export embeds the metadata dict built by `Exporter`, but each format stores it differently:
    directory-based formats write a `metadata.yaml` sidecar, while single-file formats embed it in a length-prefixed
    JSON header (`.engine`), a zip entry (`.torchscript`, `.tflite`), or protobuf metadata props (`.onnx`). Reading it
    up front resolves a model's task and head when its filename carries neither, i.e. `best.engine`. MNN keeps its
    metadata in a flatbuffer `bizCode` field and Triton serves it over HTTP, so neither is read here.

    Args:
        file (str | Path): Path to an exported model file or directory.

    Returns:
        (dict): Parsed metadata, empty if the export holds none or predates metadata embedding.
    """
    import json
    import zipfile

    path = Path(file)
    if path.suffix in {".pt", ".yaml", ".yml"}:  # checkpoints and configs are not exports
        return {}
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
            model = path / "Data/com.apple.CoreML/model.mlmodel" if path.suffix == ".mlpackage" else path
            return _read_proto_metadata(model, (2, 100, 100))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return {}  # a third-party or truncated file carries no Ultralytics header

    sidecar = (path if path.is_dir() else path.parent) / "metadata.yaml"  # openvino, paddle, ncnn, saved_model
    if path.suffix == ".pb":  # a frozen graph keeps its metadata in the sibling saved_model directory
        sidecar = next(path.resolve().parent.rglob(f"{path.stem}_saved_model*/metadata.yaml"), sidecar)
    if sidecar.exists():
        from ultralytics.utils import YAML

        return YAML.load(sidecar)
    return {}


def _read_proto_metadata(file: Path, field_path: tuple[int, ...]) -> dict:
    """Collect a protobuf `map<string, string>` at a nested field path, without parsing the fields around it.

    ONNX keeps its metadata props at top-level field 14 while CoreML nests them under
    `description.metadata.userDefined`, and both encode entries the same way, so each is read by stepping over every
    field using its declared wire-format length and descending only into the fields on the path.

    Args:
        file (Path): Path to the protobuf file.
        field_path (tuple[int, ...]): Field numbers to descend, the last holding the repeated map entries.

    Returns:
        (dict): Map entries as string key-value pairs.
    """
    import io

    def varint(read):
        """Decode a base-128 varint from a `read(n)` callable, returning None at end of stream."""
        shift = value = 0
        while chunk := read(1):
            value |= (chunk[0] & 0x7F) << shift
            if not chunk[0] & 0x80:
                return value
            shift += 7

    def walk(stream, fields):
        """Scan one message level, descending into `fields[0]` and collecting map entries at the last field."""
        metadata = {}
        while (tag := varint(stream.read)) is not None:
            wire = tag & 7
            if wire == 0:  # varint field, i.e. ir_version
                varint(stream.read)
                continue
            if wire != 2:  # these protos carry only varint and length-delimited fields
                break
            length = varint(stream.read)
            if tag >> 3 != fields[0]:  # off the path, so step over the payload, i.e. the graph or weights
                stream.seek(length, 1)
                continue
            nested = io.BytesIO(stream.read(length))
            if len(fields) > 1:
                return walk(nested, fields[1:])
            pair = {}
            while (field := varint(nested.read)) is not None:  # map entry, 1=key and 2=value
                pair[field >> 3] = nested.read(varint(nested.read)).decode()
            metadata[pair[1]] = pair.get(2, "")
        return metadata

    with open(file, "rb") as f:
        return walk(f, field_path)


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
