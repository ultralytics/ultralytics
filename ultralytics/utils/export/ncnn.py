# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


def torch2ncnn(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str,
    half: bool = False,
    metadata: dict | None = None,
    device: torch.device | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to NCNN format using PNNX.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model path used to derive the output directory.
        half (bool): Whether to enable FP16 export.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        device (torch.device | None): Device the model lives on.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_ncnn_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements("ncnn", cmds="--no-deps")  # no deps to avoid installing opencv-python
    check_requirements("pnnx")
    import ncnn
    import pnnx

    LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__} and PNNX {pnnx.__version__}...")
    file = Path(file)
    f = Path(str(file).replace(file.suffix, f"_ncnn_model{os.sep}"))

    ncnn_args = dict(
        ncnnparam=(f / "model.ncnn.param").as_posix(),
        ncnnbin=(f / "model.ncnn.bin").as_posix(),
        ncnnpy=(f / "model_ncnn.py").as_posix(),
    )
    pnnx_args = dict(
        ptpath=(f / "model.pt").as_posix(),
        pnnxparam=(f / "model.pnnx.param").as_posix(),
        pnnxbin=(f / "model.pnnx.bin").as_posix(),
        pnnxpy=(f / "model_pnnx.py").as_posix(),
        pnnxonnx=(f / "model.pnnx.onnx").as_posix(),
    )

    f.mkdir(exist_ok=True)  # make ncnn_model directory
    device_type = device.type if device is not None else "cpu"
    pnnx.export(model, inputs=im, **ncnn_args, **pnnx_args, fp16=half, device=device_type)

    for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_args.values()):
        Path(f_debug).unlink(missing_ok=True)

    if metadata:
        YAML.save(f / "metadata.yaml", metadata)  # add metadata.yaml
    return str(f)
