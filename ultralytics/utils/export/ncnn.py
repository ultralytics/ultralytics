# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.files import WorkingDirectory


def torch2ncnn(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    device: torch.device | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to NCNN format using PNNX.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        output_dir (Path | str): Directory to save the exported NCNN model.
        quantize (int | str | None): Precision scheme, e.g. 16 for FP16.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        device (torch.device | None): Device the model lives on.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_ncnn_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements("ncnn", cmds="--no-deps")  # no deps to avoid installing opencv-python
    # Pin until PNNX 20260704 NCNN inference segfault is fixed: https://github.com/pnnx/pnnx/issues/293
    check_requirements("pnnx==20260526")
    import ncnn
    import pnnx

    LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__} and PNNX {pnnx.__version__}...")
    output_dir = Path(output_dir)

    ncnn_args = {
        "ncnnparam": (output_dir.absolute() / "model.ncnn.param").as_posix(),
        "ncnnbin": (output_dir.absolute() / "model.ncnn.bin").as_posix(),
    }
    pnnx_args = {
        "ptpath": "model.pt",
        "pnnxparam": "model.pnnx.param",
        "pnnxbin": "model.pnnx.bin",
        "pnnxpy": "model_pnnx.py",
        "pnnxonnx": "model.pnnx.onnx",
    }

    output_dir.mkdir(parents=True, exist_ok=True)  # make ncnn_model directory
    device_type = device.type if device is not None else "cpu"
    with WorkingDirectory(output_dir):  # PNNX writes pnnx_args unescaped into the Python it generates and imports
        pnnx.export(model, inputs=im, **ncnn_args, **pnnx_args, fp16=quantize == 16, device=device_type)
        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_args.values()):
            Path(f_debug).unlink(missing_ok=True)

    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)  # add metadata.yaml
    return str(output_dir)
