# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

import torch

from ultralytics.nn.modules import Detect, Pose, Pose26
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_executorch_requirements, check_version


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
    im: torch.Tensor,
    output_dir: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
    calibration_dataset: torch.utils.data.DataLoader | None = None,
) -> str:
    """Export a PyTorch model to ExecuTorch format, with static INT8 quantization when calibration data is given.

    Args:
        model (torch.nn.Module): The PyTorch model to export, on CPU.
        im (torch.Tensor): Example input tensor for tracing/export, on CPU.
        output_dir (Path | str): Directory to save the exported ExecuTorch model.
        metadata (dict | None, optional): Optional metadata to save as YAML.
        prefix (str, optional): Prefix for log messages.
        calibration_dataset (DataLoader | None, optional): Calibration dataloader for static INT8 (int8 weights + int8
            activations, FP32 head on Detect models), as returned by ``get_int8_calibration_dataloader``. Requires
            executorch>=1.6.0.

    Returns:
        (str): Path to the exported ExecuTorch model directory.
    """
    check_executorch_requirements()
    from executorch import version as executorch_version
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    LOGGER.info(f"\n{prefix} starting export with ExecuTorch {executorch_version.__version__}...")

    if calibration_dataset is not None:
        # Earlier releases quantize the head concat despite the exclusion below, which zeroes the scores
        assert check_version(executorch_version.__version__, "1.6.0"), "INT8 export requires executorch>=1.6.0"
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
        for m in model.modules():
            if isinstance(m, Detect):  # keep the head float: one int8 scale over boxes and scores zeroes the scores
                quantizer.set_module_type(type(m), None)
        model = prepare_pt2e(torch.export.export(model, (im,)).module(), quantizer)
        for batch in calibration_dataset:
            model(batch["img"].float() / 255)
        model = convert_pt2e(model)

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, (im,)),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    output_dir = Path(output_dir).resolve()  # rmtree refuses a symlinked directory
    shutil.rmtree(output_dir, ignore_errors=True)  # a re-export would keep an older export's <stem>.pte
    output_dir.mkdir(parents=True)
    (output_dir / "model.pte").write_bytes(et_program.buffer)

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
