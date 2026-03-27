# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

import torch

from ultralytics.utils import LOGGER, TORCH_VERSION


def torch2torchscript(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str,
    optimize: bool = False,
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """Export a PyTorch model to TorchScript format.

    Args:
        model (torch.nn.Module): The PyTorch model to export (may be NMS-wrapped).
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model file path used to derive output path.
        optimize (bool): Whether to optimize for mobile deployment.
        metadata (dict | None): Optional metadata to embed in the TorchScript archive.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``.torchscript`` file.
    """
    LOGGER.info(f"\n{prefix} starting export with torch {TORCH_VERSION}...")
    file = Path(file)
    f = file.with_suffix(".torchscript")

    ts = torch.jit.trace(model, im, strict=False)
    extra_files = {"config.txt": json.dumps(metadata or {})}  # torch._C.ExtraFilesMap()
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        LOGGER.info(f"{prefix} optimizing for mobile...")
        from torch.utils.mobile_optimizer import optimize_for_mobile

        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f
