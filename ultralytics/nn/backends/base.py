# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import contextlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import YAML


def _read_proto_map(file: Path, path: tuple[int, ...]) -> dict:
    """Read a protobuf ``map<string, string>`` at a nested field path, without importing the format's framework.

    Args:
        file (Path): Path to the protobuf file, i.e. an ONNX or CoreML model.
        path (tuple[int, ...]): Field numbers to descend, the last holding the repeated ``key``/``value`` entries.

    Returns:
        (dict): Map entries as string key-value pairs.
    """
    import mmap

    def fields(buf):
        """Yield ``(number, payload)`` for each length-delimited field of a message, stepping over varint fields."""
        i = 0

        def varint():
            """Decode the base-128 varint at the current offset."""
            nonlocal i
            v = shift = 0
            while buf[i] & 0x80:
                v, i, shift = v | (buf[i] & 0x7F) << shift, i + 1, shift + 7
            v, i = v | buf[i] << shift, i + 1
            return v

        while i < len(buf):
            tag = varint()
            if tag & 7 == 0:  # varint field, i.e. ONNX ir_version
                varint()
            elif tag & 7 == 2:  # length-delimited field, i.e. a nested message, string or weights blob
                n = varint()
                yield tag >> 3, buf[i : i + n]  # a memoryview slice, so a large payload is never copied
                i += n
            else:
                return  # these protos carry no fixed-width fields

    with open(file, "rb") as f:
        messages = [memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))]
    for number in path:
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

    @staticmethod
    def engine_header(file: str | Path) -> tuple[int, dict]:
        """Read the metadata header an Ultralytics ``.engine`` export writes ahead of its serialized engine.

        Args:
            file (str | Path): Path to the TensorRT engine file.

        Returns:
            (tuple[int, dict]): Byte offset of the engine bytes and the header metadata, ``(0, {})`` without a header.
        """
        with open(file, "rb") as f:
            n = int.from_bytes(f.read(4), byteorder="little")  # 4-byte little-endian JSON length, if a header exists
            if 0 < n <= f.seek(0, 2) - 4:  # a length overrunning the file is not a header
                f.seek(4)
                with contextlib.suppress(ValueError):  # engine bytes are not JSON, so a real header parses
                    return 4 + n, json.loads(f.read(n))
        return 0, {}

    @staticmethod
    def read_metadata(file: str | Path) -> dict:
        """Read Ultralytics metadata from an export without loading it or importing its framework.

        Single-file formats embed metadata in a length-prefixed JSON header (``.engine``), a zip entry
        (``.torchscript``, ``.tflite``) or protobuf string map entries (``.onnx``, ``.mlpackage``), and every other
        format writes a ``metadata.yaml`` sidecar beside or inside the export. Core AI has its own
        ``metadata.json`` inside the ``.aimodel`` asset. MNN keeps it in a flatbuffer
        ``bizCode`` field and Triton serves it over HTTP, so neither is read here.

        Args:
            file (str | Path): Path to an exported model file or directory.

        Returns:
            (dict): Parsed metadata, empty for a third-party export or one predating metadata embedding.
        """
        import zipfile

        p = Path(file)
        try:
            if p.suffix == ".engine":  # 4-byte little-endian length then that many bytes of JSON
                return BaseBackend.engine_header(p)[1]
            if p.suffix in {".tflite", ".torchscript"}:  # metadata appended to or saved inside the model zip
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
                    if "metadata.json" in names:  # litert-torch and single-file tflite exports
                        return json.loads(z.read("metadata.json"))
                    name = next((n for n in names if n.endswith("extra/config.txt")), None)  # torch.jit extra file
                    return json.loads(z.read(name)) if name else ast.literal_eval(z.read(names[0]).decode())
            if p.suffix == ".onnx" or p.name.endswith("_imx_model"):  # IMX packages its ONNX in a directory
                return _read_proto_map(next(p.glob("*.onnx")) if p.is_dir() else p, (14,))  # metadata_props
            if p.suffix in {".mlpackage", ".mlmodel"}:  # description.metadata.userDefined
                return _read_proto_map(next(p.rglob("*.mlmodel")) if p.is_dir() else p, (2, 100, 100))
            if p.suffix == ".aimodel":  # Core AI keeps it in the asset's own metadata.json
                return json.loads((p / "metadata.json").read_text()).get("creatorDefinedMetadata", {})
            sidecar = (p if p.is_dir() else p.parent) / "metadata.yaml"  # openvino, ncnn, paddle, saved_model, ...
            if p.suffix == ".pb":  # a frozen graph keeps its metadata in the sibling saved_model directory
                sidecar = next(p.resolve().parent.rglob(f"{p.stem}_saved_model*/metadata.yaml"), sidecar)
            return YAML.load(sidecar) if sidecar.exists() else {}
        except Exception:  # a third-party, truncated or metadata-less artifact
            return {}

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
        metadata["end2end"] = bool(metadata.get("end2end", False) or metadata.get("args", {}).get("nms", False))
        metadata["dynamic"] = metadata.get("args", {}).get("dynamic", self.dynamic)

        # Apply all metadata fields as backend attributes
        for k, v in metadata.items():
            setattr(self, k, v)
