# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import LOGGER


def torch2openvino(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str | None = None,
    dynamic: bool = False,
    half: bool = False,
    int8: bool = False,
    calibration_dataset: Any | None = None,
    ignored_scope: dict | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to OpenVINO format with optional INT8 quantization.

    Args:
        model (torch.nn.Module): The model to export (may be NMS-wrapped).
        im (torch.Tensor): Example input tensor.
        file (Path | str | None): Source model path used to derive output directory.
        dynamic (bool): Whether to use dynamic input shapes.
        half (bool): Whether to compress to FP16.
        int8 (bool): Whether to apply INT8 quantization.
        calibration_dataset (nn.Dataset): Dataset for nncf.Dataset (required when ``int8=True``).
        ignored_scope (dict | None): Kwargs passed to ``nncf.IgnoredScope`` for head patterns.
        prefix (str): Prefix for log messages.

    Returns:
        (ov.Model): The converted OpenVINO model.
    """
    import openvino as ov

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")

    ov_model = ov.convert_model(model, input=None if dynamic else [im.shape], example_input=im)
    if int8:
        import nncf

        ov_model = nncf.quantize(
            model=ov_model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope,
        )

    if file is not None:
        file = Path(file)
        suffix = f"_{'int8_' if int8 else ''}openvino_model{os.sep}"
        f = str(file).replace(file.suffix, suffix)
        f_ov = str(Path(f) / file.with_suffix(".xml").name)
        ov.save_model(ov_model, f_ov, compress_to_fp16=half)
    return ov_model
