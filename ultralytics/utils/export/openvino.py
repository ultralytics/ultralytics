# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import LOGGER


def torch2openvino(
    model: torch.nn.Module,
    im: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    output_dir: Path | str | None = None,
    dynamic: bool = False,
    quantize: int | str | None = None,
    calibration_dataset: Any | None = None,
    prefix: str = "",
) -> Any:
    """Export a PyTorch model to OpenVINO format with optional INT8 quantization.

    Args:
        model (torch.nn.Module): The model to export (may be NMS-wrapped).
        im (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]): Example input tensor(s) for tracing.
        output_dir (Path | str | None): Directory to save the exported OpenVINO model.
        dynamic (bool): Whether to use dynamic input shapes.
        quantize (int | str | None): Precision scheme, e.g. 16 for FP16 or 8 for INT8.
        calibration_dataset (nncf.Dataset | None): Dataset for INT8 calibration (required when ``quantize=8``).
        prefix (str): Prefix for log messages.

    Returns:
        (ov.Model): The converted OpenVINO model.
    """
    import openvino as ov

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")

    input_shape = [i.shape for i in im] if isinstance(im, (list, tuple)) else im.shape
    # Hand OpenVINO an already-traced ScriptModule (torchscript/coreml exports trace the same way), not a raw
    # nn.Module, so it doesn't re-trace internally with check_trace=True - that re-trace-and-diff sanity check is
    # non-deterministic on NMS models and fails with "Graphs differed across invocations!". check_trace=False skips
    # the same check on our own trace.
    ts = torch.jit.trace(model, im, strict=False, check_trace=False)
    ov_model = ov.convert_model(ts, input=None if dynamic else input_shape, example_input=im)
    if quantize == 8:
        import nncf

        from ultralytics.nn.modules.head import Detect, RTDETRDecoder

        head = model
        while hasattr(head, "model"):  # unwrap the NMS wrapper and the task model to reach the module list
            head = head.model
        head = head[-1] if isinstance(head, torch.nn.Sequential) else head  # non-YOLO models have no module list

        ignored_scope = None
        if isinstance(head, (Detect, RTDETRDecoder)):
            ops = ov_model.get_ordered_ops()
            names = [op.get_friendly_name() for op in ops]
            scope = [n for n, op in zip(names, ops) if op.get_type_name() == "Sigmoid"][-1].split("/", 1)[0]
            if isinstance(head, RTDETRDecoder):
                # A DETR head reads its queries straight off the neck, so quantizing those feature maps alone
                # collapses it. Keep the head and the blocks feeding it in floating point.
                modules = [scope, *(f"{scope.rsplit('.', 1)[0]}.{i}" for i in head.f)]
                prefixes = tuple(f"{m}{sep}" for m in modules for sep in "/.")
                keep = [n for n in names if n in modules or n.startswith(prefixes)]
            else:
                keep = [
                    n
                    for n, op in zip(names, ops)
                    if op.get_type_name() == "Sigmoid" or n.startswith((f"{scope}/", f"{scope}.dfl"))
                ]
            ignored_scope = nncf.IgnoredScope(names=keep)
        ov_model = nncf.quantize(
            model=ov_model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            # Calibrate on the full dataset like other INT8 backends, not nncf's 300-batch default
            subset_size=calibration_dataset.get_length() or 300,
            ignored_scope=ignored_scope,
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "model.xml"
        ov.save_model(ov_model, output_file, compress_to_fp16=quantize == 16)
    return ov_model
