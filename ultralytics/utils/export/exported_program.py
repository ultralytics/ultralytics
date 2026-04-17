# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

import torch

from ultralytics.utils import LOGGER


def torch2exported_program(
    model: torch.nn.Module,
    file: Path | str,
    sample_input: torch.Tensor,
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """
    Export a PyTorch model to torch.export ExportedProgram (.pt2) format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        file (Path | str): Source model file path used to derive output names.
        sample_input (torch.Tensor): Example input tensor for tracing/export.
        metadata (dict | None): Optional metadata to embed in the PT2 archive.
        prefix (str, optional): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``.pt2`` file.
    """
    LOGGER.info(f"\n{prefix} starting export with ExportedProgram {torch.__version__}...")

    file = Path(file)
    f = file.with_suffix(".pt2")

    exported_program = torch.export.export(model, (sample_input,))
    extra_files = {"config.txt": json.dumps(metadata or {})}
    torch.export.save(exported_program, str(f), extra_files=extra_files)

    return f
