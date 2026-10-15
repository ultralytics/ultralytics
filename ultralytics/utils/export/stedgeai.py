# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import types
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import Depth, SemanticSegment
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.export.onnx import onnx_calibration_reader
from ultralytics.utils.export.tensorflow import tf_wrapper


def stedgeai_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """Patch model heads with ST Edge AI export implementations.

    Detect and Pose heads emit box and keypoint coordinates normalized to [0, 1] by the input size so INT8 quantization
    preserves class-score resolution. Semantic heads emit raw [B, nc, H/8, W/8] logits and Depth heads raw
    [B, 1, H/4, W/4] log-depth, leaving the final upsampling and the depth clamp/exp/calibration decode to
    post-processing, which keeps the on-device output small and the INT8 depth range linear in log space.

    Args:
        model (torch.nn.Module): Model whose heads are patched in place.

    Returns:
        (torch.nn.Module): The same model with patched head methods.
    """
    for m in model.modules():
        if isinstance(m, SemanticSegment):
            m.forward = types.MethodType(_semantic_forward, m)
        elif isinstance(m, Depth):
            m.forward = types.MethodType(_depth_forward, m)
    return tf_wrapper(model)


def _semantic_forward(self, x: list[torch.Tensor]) -> torch.Tensor:
    """Return raw semantic logits at 1/8 input resolution."""
    return self.classifier(x[0])


def _depth_forward(self, x: list[torch.Tensor]) -> torch.Tensor:
    """Return raw log-depth at 1/4 input resolution, before the clamp/exp/calibration decode and upsampling."""
    feats = [self.proj[i](x[i]) for i in range(self.nl)]
    out = feats[-1]
    for i in range(self.nl - 2, -1, -1):
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True) + feats[i]
        out = self.refine[i](out)
    return self.head(out)


def onnx2stedgeai(
    onnx_file: str | Path,
    output_file: str | Path,
    dataset: Iterable,
    transform_fn: Callable,
    prefix: str = "",
) -> str:
    """Quantize an ST Edge AI float ONNX model to QDQ INT8 with ONNX Runtime static quantization.

    Matches the STM32 AI Model Zoo quantization recipe: QDQ format, symmetric per-channel INT8 weights, asymmetric
    per-tensor INT8 activations and MinMax calibration. The whole graph is quantized because the exported head emits
    input-normalized coordinates, so boxes, keypoints and class scores share a comparable [0, 1] range.

    Args:
        onnx_file (str | Path): Path to the float ONNX model with input-normalized coordinates.
        output_file (str | Path): Path to save the QDQ INT8 ONNX model.
        dataset (Iterable): Calibration dataloader yielding batch dicts.
        transform_fn (Callable): Function converting a batch dict to a float32 NCHW numpy array.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the QDQ INT8 ONNX model.
    """
    check_requirements([("onnxruntime", "onnxruntime-gpu")])
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    LOGGER.info(f"{prefix} quantizing QDQ INT8 with ONNX Runtime (per-channel weights, MinMax calibration)...")
    quantize_static(
        str(onnx_file),
        str(output_file),
        onnx_calibration_reader(dataset, transform_fn),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
    )
    return str(output_file)
