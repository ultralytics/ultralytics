# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from ultralytics.nn.modules import Pose, Pose26
from ultralytics.utils import LOGGER, YAML


def executorch_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """Apply ExecuTorch-specific model patches required for export/runtime compatibility."""
    import types

    for m in model.modules():
        if not isinstance(m, Pose):
            continue
        m.kpts_decode = types.MethodType(partial(_executorch_kpts_decode, is_pose26=type(m) is Pose26), m)
    return model


def _executorch_kpts_decode(self, kpts: torch.Tensor, is_pose26: bool = False) -> torch.Tensor:
    """Decode pose keypoints for ExecuTorch export with XNNPACK-safe broadcasting."""
    ndim = self.kpt_shape[1]
    bs = kpts.shape[0]
    y = kpts.view(bs, *self.kpt_shape, -1)

    # XNNPACK requires explicit dim matching for broadcasting, expand 2D tensors to 4D.
    anchors = self.anchors[None, None]
    strides = self.strides[None, None]
    a = ((y[:, :, :2] + anchors) if is_pose26 else (y[:, :, :2] * 2.0 + (anchors - 0.5))) * strides
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)


def torch2executorch(
    model: torch.nn.Module,
    file: Path | str,
    sample_input: torch.Tensor,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to ExecuTorch format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        file (Path | str): Source model file path used to derive output names.
        sample_input (torch.Tensor): Example input tensor for tracing/export.
        metadata (dict | None, optional): Optional metadata to save as YAML.
        prefix (str, optional): Prefix for log messages.

    Returns:
        (str): Path to the exported ExecuTorch model directory.
    """
    from executorch import version as executorch_version
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    LOGGER.info(f"\n{prefix} starting export with ExecuTorch {executorch_version.__version__}...")

    file = Path(file)
    output_dir = Path(str(file).replace(file.suffix, "_executorch_model"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pte_file = output_dir / file.with_suffix(".pte").name
    et_program = to_edge_transform_and_lower(
        torch.export.export(model, (sample_input,)),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    pte_file.write_bytes(et_program.buffer)

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
