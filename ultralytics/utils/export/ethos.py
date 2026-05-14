# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_executorch_requirements, check_requirements


def torch2ethos(
    torch_model: torch.nn.Module,
    file: Path | str,
    sample_input: torch.Tensor,
    dataset: Iterable[Any] | None = None,
    target: str = "ethos-u85-256",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export PyTorch model to Arm Ethos-U NPU using ExecuTorch.

    Args:
        torch_model (torch.nn.Module): PyTorch model to export.
        file (Path | str): Source model file path used to derive output names.
        sample_input (torch.Tensor): Example input tensor for tracing/export.
        dataset (Iterable[Any] | None, optional): Representative calibration dataset for PTQ. Each item may be a tensor
            batch or a batch dictionary containing an ``img`` tensor.
        target (str, optional): Ethos target to compile for.
        metadata (dict | None, optional): Optional metadata to save as YAML.
        prefix (str, optional): Prefix for log messages.

    Returns:
        (str): Path to the exported ExecuTorch with ARM Ethos-U Backend model directory.
    """
    check_executorch_requirements()
    check_requirements("tosa-tools")
    check_requirements("ethos-u-vela")

    from executorch import version as executorch_version
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from executorch.extension.export_util.utils import save_pte_program
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    LOGGER.info(f"\n{prefix} starting export with ExecuTorch {executorch_version.__version__}...")

    file = Path(file)
    output_dir = Path(str(file).replace(file.suffix, "_ethos_model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    pte_file = output_dir / file.with_suffix(".pte").name

    exported_program = torch.export.export(torch_model, (sample_input,))
    graph_module = exported_program.module(check_guards=False)

    compile_spec = EthosUCompileSpec(
        target=target,
        memory_mode="Shared_Sram",
    )

    quantizer = EthosUQuantizer(compile_spec)
    operator_config = get_symmetric_quantization_config()
    quantizer.set_global(operator_config)

    # Post training quantization
    quantized_graph_module = prepare_pt2e(graph_module, quantizer)

    calibration_batches = 0
    for batch in dataset:
        tensor = batch["img"] if isinstance(batch, dict) else batch
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected calibration batch to be a torch.Tensor, but got {type(tensor).__name__}.")
        quantized_graph_module(tensor.to(device=sample_input.device, dtype=sample_input.dtype))
        calibration_batches += 1

    quantized_graph_module = convert_pt2e(quantized_graph_module)

    _ = quantized_graph_module.print_readable()

    quantized_exported_program = torch.export.export(quantized_graph_module, (sample_input,))

    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import ReplaceQuantNodesPass
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )

    # Create partitioner from compile spec
    partitioner = EthosUPartitioner(compile_spec)

    # Lower the exported program to the Ethos-U backend
    edge_program_manager = to_edge_transform_and_lower(
        quantized_exported_program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    # Rewrite remaining CPU-side quantized_decomposed::{quantize,dequantize}_per_tensor.out
    # ops at the Ethos-U delegate boundaries into cortex_m::* ops the Zephyr/Cortex-M
    # runtime registers. Without this pass, the .pte fails at runtime with
    # "Missing operator: quantized_decomposed::quantize_per_tensor.out".
    edge_program_manager = edge_program_manager.transform([ReplaceQuantNodesPass()])

    # Convert edge program to executorch
    executorch_program_manager = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    _ = executorch_program_manager.exported_program().module(check_guards=False).print_readable()

    # Save pte file
    save_pte_program(executorch_program_manager, str(pte_file))

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
