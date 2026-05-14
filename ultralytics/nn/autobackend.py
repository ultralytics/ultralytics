# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils.checks import check_suffix
from ultralytics.utils.downloads import is_url

from .backends import (
    AxeleraBackend,
    CoreMLBackend,
    DeepXBackend,
    ExecuTorchBackend,
    MNNBackend,
    NCNNBackend,
    ONNXBackend,
    ONNXIMXBackend,
    OpenVINOBackend,
    PaddleBackend,
    PyTorchBackend,
    RKNNBackend,
    TensorFlowBackend,
    TensorRTBackend,
    TorchScriptBackend,
    TritonBackend,
)


def check_class_names(names: list | dict) -> dict[int, str]:
    """Check class names and convert to dict format if needed.

    Args:
        names (list | dict): Class names as list or dict format.

    Returns:
        (dict): Class names in dict format with integer keys and string values.

    Raises:
        KeyError: If class indices are invalid for the dataset size.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
            from ultralytics.utils import ROOT, YAML

            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data: str | Path | None = None) -> dict[int, str]:
    """Load class names from a YAML file or return numerical class names.

    Args:
        data (str | Path, optional): Path to YAML file containing class names.

    Returns:
        (dict): Dictionary mapping class indices to class names.
    """
    if data:
        try:
            from ultralytics.utils import YAML
            from ultralytics.utils.checks import check_yaml

            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


class AutoBackend(nn.Module):
    """Handle dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix       |
            | --------------------- | ----------------- |
            | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Lite       | *.tflite          |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |
            | IMX                   | *_imx_model/      |
            | RKNN                  | *_rknn_model/     |
            | Triton Inference      | triton://model    |
            | ExecuTorch            | *.pte             |
            | Axelera AI            | *_axelera_model/  |
            | DeepX                 | *_deepx_model/    |

    Attributes:
        backend (BaseBackend): The loaded inference backend instance.
        format (str): The model format (e.g., 'pt', 'onnx', 'engine').
        model: The underlying model (nn.Module for PyTorch backends, backend instance otherwise).
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        task (str): The type of task the model performs (detect, segment, classify, pose).
        names (dict): A dictionary of class names that the model can detect.
        stride (int): The model stride, typically 32 for YOLO models.
        fp16 (bool): Whether the model uses half-precision (FP16) inference.
        nhwc (bool): Whether the model expects NHWC input format instead of NCHW.

    Methods:
        forward: Run inference on an input image.
        from_numpy: Convert NumPy arrays to tensors on the model device.
        warmup: Warm up the model with a dummy input.
        _model_type: Determine the model type from file path.

    Examples:
        >>> model = AutoBackend(model="yolo26n.pt", device="cuda")
        >>> results = model(img)
    """

    _BACKEND_MAP = {
        "pt": PyTorchBackend,
        "torchscript": TorchScriptBackend,
        "onnx": ONNXBackend,
        "dnn": ONNXBackend,  # Special case: ONNX with DNN
        "openvino": OpenVINOBackend,
        "engine": TensorRTBackend,
        "coreml": CoreMLBackend,
        "saved_model": TensorFlowBackend,
        "pb": TensorFlowBackend,
        "tflite": TensorFlowBackend,
        "edgetpu": TensorFlowBackend,
        "paddle": PaddleBackend,
        "mnn": MNNBackend,
        "ncnn": NCNNBackend,
        "imx": ONNXIMXBackend,
        "rknn": RKNNBackend,
        "triton": TritonBackend,
        "executorch": ExecuTorchBackend,
        "axelera": AxeleraBackend,
        "deepx": DeepXBackend,
    }

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo26n.pt",
        device: torch.device = torch.device("cpu"),
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """Initialize the AutoBackend for inference.

        Args:
            model (str | torch.nn.Module): Path to the model weights file or a module instance.
            device (torch.device): Device to run the model on.
            dnn (bool): Use OpenCV DNN module for ONNX inference.
            data (str | Path, optional): Path to the additional data.yaml file containing class names.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization.
            verbose (bool): Enable verbose logging.
        """
        super().__init__()
        # Determine model format from path/URL
        format = "pt" if isinstance(model, nn.Module) else self._model_type(model, dnn)

        # Check if format supports FP16
        fp16 &= format in {"pt", "torchscript", "onnx", "openvino", "engine", "triton"}

        # Set device
        if (
            isinstance(device, torch.device)
            and torch.cuda.is_available()
            and device.type != "cpu"
            and format not in {"pt", "torchscript", "engine", "onnx", "paddle"}
        ):
            device = torch.device("cpu")

        # Select and initialize the appropriate backend
        backend_kwargs = {"device": device, "fp16": fp16}

        if format == "tfjs":
            raise NotImplementedError("Ultralytics TF.js inference is not currently supported.")
        if format not in self._BACKEND_MAP:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{model}' is not a supported model format. "
                f"Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )
        if format == "pt":
            backend_kwargs["fuse"] = fuse
            backend_kwargs["verbose"] = verbose
        elif format in {"saved_model", "pb", "tflite", "edgetpu", "dnn"}:
            backend_kwargs["format"] = format
        self.backend = self._BACKEND_MAP[format](model, **backend_kwargs)

        self.nhwc = format in {"coreml", "saved_model", "pb", "tflite", "edgetpu", "rknn"}
        self.format = format

        # Ensure backend has names (fallback to default if not set by metadata)
        if not self.backend.names:
            self.backend.names = default_class_names(data)
        self.backend.names = check_class_names(self.backend.names)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the backend.

        This allows AutoBackend to transparently expose backend attributes
        without explicit copying.

        Args:
            name: Attribute name to look up.

        Returns:
            The attribute value from the backend.

        Raises:
            AttributeError: If the attribute is not found in backend.
        """
        if "backend" in self.__dict__ and hasattr(self.backend, name):
            return getattr(self.backend, name)
        return super().__getattr__(name)

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: list | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on an AutoBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): Whether to perform data augmentation during inference.
            visualize (bool): Whether to visualize the output predictions.
            embed (list, optional): A list of layer indices to return embeddings from.
            **kwargs (Any): Additional keyword arguments for model configuration.

        Returns:
            (torch.Tensor | list[torch.Tensor]): The raw output tensor(s) from the model.
        """
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
        if self.backend.fp16 and im.dtype != torch.float16:
            im = im.half()

        # Build forward kwargs based on backend type
        forward_kwargs = {}
        if self.format == "pt":
            forward_kwargs = {"augment": augment, "visualize": visualize, "embed": embed, **kwargs}

        y = self.backend.forward(im, **forward_kwargs)

        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 116, 8400), (1, 32, 160, 160)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert a NumPy array to a torch tensor on the model device.

        Args:
            x (np.ndarray | torch.Tensor): Input array or tensor.

        Returns:
            (torch.Tensor): Tensor on `self.device`.
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        """Warm up the model by running forward pass(es) with a dummy input.

        Args:
            imgsz (tuple[int, int, int, int]): Dummy input shape in (batch, channels, height, width) format.
        """
        from ultralytics.utils.nms import non_max_suppression

        if self.format in {"pt", "torchscript", "onnx", "engine", "saved_model", "pb", "triton"} and (
            self.device.type != "cpu" or self.format == "triton"
        ):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.format == "torchscript" else 1):
                self.forward(im)  # warmup model
                warmup_boxes = torch.rand(1, 84, 16, device=self.device)  # 16 boxes works best empirically
                warmup_boxes[:, :4] *= imgsz[-1]
                non_max_suppression(warmup_boxes)  # warmup NMS

    @staticmethod
    def _model_type(p: str = "path/to/model.pt", dnn: bool = False) -> str:
        """Take a path to a model file and return the model format string.

        Args:
            p (str): Path to the model file.
            dnn (bool): Whether to use OpenCV DNN module for ONNX inference.

        Returns:
            (str): Model format string (e.g., 'pt', 'onnx', 'engine', 'triton').

        Examples:
            >>> fmt = AutoBackend._model_type("path/to/model.onnx")
            >>> assert fmt == "onnx"
        """
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")
        types[8] &= not types[9]
        format = next((f for i, f in enumerate(export_formats()["Argument"]) if types[i]), None)
        if format == "-":
            format = "pt"
        elif format == "onnx" and dnn:
            format = "dnn"
        elif not any(types):
            from urllib.parse import urlsplit

            url = urlsplit(p)
            if bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}:
                format = "triton"
        return format

    def eval(self) -> AutoBackend:
        """Set the backend model to evaluation mode if supported."""
        if hasattr(self.backend, "model") and hasattr(self.backend.model, "eval"):
            self.backend.model.eval()
        return super().eval()

    def _apply(self, fn) -> AutoBackend:
        """Apply a function to backend.model parameters, buffers, and tensors.

        This method extends the functionality of the parent class's _apply method by additionally resetting the
        predictor and updating the device in the model's overrides. It's typically used for operations like moving the
        model to a different device or changing its precision.

        Args:
            fn (Callable): A function to be applied to the model's tensors. This is typically a method like to(), cpu(),
                cuda(), half(), or float().

        Returns:
            (AutoBackend): The model instance with the function applied and updated attributes.
        """
        self = super()._apply(fn)
        if hasattr(self.backend, "model") and isinstance(self.backend.model, nn.Module):
            self.backend.model._apply(fn)
            self.backend.device = next(self.backend.model.parameters()).device  # update device after move
        return self
