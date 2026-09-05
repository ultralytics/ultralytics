# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

import torch

from ultralytics.utils import LOGGER, TORCH_VERSION


def torch2exported_program(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_file: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to torch.export ExportedProgram (.pt2) format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing/export.
        output_file (Path | str): Path to save the exported ExportedProgram.
        metadata (dict | None): Optional metadata to embed in the PT2 archive.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``.pt2`` file.
    """
    LOGGER.info(f"\n{prefix} starting export with torch {TORCH_VERSION}...")

    output_file = str(output_file)
    exported_program = torch.export.export(model, (im,))
    extra_files = {"config.txt": json.dumps(metadata or {})}
    torch.export.save(exported_program, output_file, extra_files=extra_files)
    return output_file
