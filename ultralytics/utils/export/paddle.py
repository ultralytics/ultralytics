# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path

import torch

from ultralytics.utils import ARM64, IS_JETSON, LOGGER, YAML


def torch2paddle(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to PaddlePaddle format using X2Paddle.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model path used to derive the output directory.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_paddle_model`` directory.
    """
    assert not IS_JETSON, "Jetson Paddle exports not supported yet"
    from ultralytics.utils.checks import check_requirements

    check_requirements(
        (
            "paddlepaddle-gpu>=3.0.0,<3.3.0"  # pin <3.3.0 https://github.com/PaddlePaddle/Paddle/issues/77340
            if torch.cuda.is_available()
            else "paddlepaddle==3.0.0"  # pin 3.0.0 for ARM64
            if ARM64
            else "paddlepaddle>=3.0.0,<3.3.0",  # pin <3.3.0 https://github.com/PaddlePaddle/Paddle/issues/77340
            "x2paddle",
        )
    )

    import x2paddle
    from x2paddle.convert import pytorch2paddle

    LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
    file = Path(file)
    f = str(file).replace(file.suffix, f"_paddle_model{os.sep}")

    pytorch2paddle(module=model, save_dir=f, jit_type="trace", input_examples=[im])  # export
    if metadata:
        YAML.save(Path(f) / "metadata.yaml", metadata)  # add metadata.yaml
    return f
