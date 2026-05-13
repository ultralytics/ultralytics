# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics YOLO inference backends.

This package provides modular inference backends for various deep learning frameworks and hardware accelerators.
Each backend implements the `BaseBackend` interface and can be used independently or through the unified
`AutoBackend` dispatcher for automatic format detection and inference routing.
"""

from .axelera import AxeleraBackend
from .base import BaseBackend
from .coreml import CoreMLBackend
from .deepx import DeepXBackend
from .executorch import ExecuTorchBackend
from .mnn import MNNBackend
from .ncnn import NCNNBackend
from .onnx import ONNXBackend, ONNXIMXBackend
from .openvino import OpenVINOBackend
from .paddle import PaddleBackend
from .pytorch import PyTorchBackend, TorchScriptBackend
from .rknn import RKNNBackend
from .tensorflow import TensorFlowBackend
from .tensorrt import TensorRTBackend
from .triton import TritonBackend

__all__ = [
    "AxeleraBackend",
    "BaseBackend",
    "CoreMLBackend",
    "DeepXBackend",
    "ExecuTorchBackend",
    "MNNBackend",
    "NCNNBackend",
    "ONNXBackend",
    "ONNXIMXBackend",
    "OpenVINOBackend",
    "PaddleBackend",
    "PyTorchBackend",
    "RKNNBackend",
    "TensorFlowBackend",
    "TensorRTBackend",
    "TorchScriptBackend",
    "TritonBackend",
]
