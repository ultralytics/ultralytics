# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


def _remainder_scalar(x: torch.Tensor, s: float) -> torch.Tensor:
    """Decompose aten.remainder.Scalar, which Core AI has no lowering for.

    Floor division, not torch.fmod: fmod takes the sign of the dividend where remainder takes the sign of the divisor,
    so fmod is only equivalent for non-negative inputs.
    """
    if x.dtype in {torch.int64, torch.int32, torch.int16, torch.int8}:
        return x - s * torch.div(x, s, rounding_mode="floor")
    return x - s * torch.floor(x / s)


def torch2coreai(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_file: Path | str,
    quantize: int | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to an Apple Core AI `.aimodel` asset.

    Args:
        model (torch.nn.Module): Model to export.
        im (torch.Tensor): Example input driving `torch.export`.
        output_file (Path | str): Destination `.aimodel` asset directory.
        quantize (int | None): 16 for an FP16 asset, None or 32 for FP32.
        metadata (dict | None): Ultralytics metadata, written into the asset's own metadata.json.
        prefix (str): Log message prefix.

    Returns:
        (str): Path to the exported asset.
    """
    check_requirements("coreai-torch>=0.4.2")
    import coreai_torch
    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter

    LOGGER.info(f"\n{prefix} starting export with coreai-torch {coreai_torch.__version__}...")

    if quantize == 16:
        model, im = model.half(), im.half()

    with torch.no_grad():
        ep = torch.export.export(model, (im,))
    table = coreai_torch.get_decomp_table()
    table[torch.ops.aten.remainder.Scalar] = _remainder_scalar  # (index % nc) in the end2end head
    ep = ep.run_decompositions(table)

    converter = TorchConverter()
    n_outputs = len(ep.graph_signature.user_outputs)
    converter.add_exported_program(
        ep,
        entrypoint_name="main",
        input_names=["images"],
        output_names=[f"output{i}" for i in range(n_outputs)],
    )
    program = converter.to_coreai()
    program.optimize()

    asset_metadata = AIModelAssetMetadata()
    asset_metadata.author = "Ultralytics"
    asset_metadata.license = "AGPL-3.0 License (https://ultralytics.com/license)"
    for k, v in (metadata or {}).items():
        asset_metadata.set_custom(k, str(v))  # matches the CoreML exporter; set_custom rejects nested dicts

    output_file = Path(output_file)
    program.save_asset(output_file, metadata=asset_metadata)
    return str(output_file)
