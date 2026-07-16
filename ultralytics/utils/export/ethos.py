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
    dataset: Iterable[Any],
    target: str = "ethos-u85-256",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export PyTorch model to Arm Ethos-U NPU using ExecuTorch.

    Args:
        torch_model (torch.nn.Module): PyTorch model to export.
        file (Path | str): Source model file path used to derive output names.
        sample_input (torch.Tensor): Example input tensor for tracing/export.
        dataset (Iterable[Any]): Representative calibration dataset for PTQ. Each item may be a tensor batch or a batch
            dictionary containing an ``img`` tensor.
        target (str, optional): Ethos target to compile for.
        metadata (dict | None, optional): Optional metadata to save as YAML.
        prefix (str, optional): Prefix for log messages.

    Returns:
        (str): Path to the exported ExecuTorch with ARM Ethos-U Backend model directory.
    """
    check_executorch_requirements()
    check_requirements(["tosa-tools", "ethos-u-vela"])

    from executorch import version as executorch_version
    from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.extension.export_util.utils import save_pte_program
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    try:
        from torchao.quantization.pt2e import move_exported_model_to_eval
    except ImportError:
        from torchao.quantization.pt2e.utils import move_exported_model_to_eval

    LOGGER.info(f"\n{prefix} starting export with ExecuTorch {executorch_version.__version__} for '{target}'...")

    file = Path(file)
    output_dir = Path(str(file).replace(file.suffix, "_ethos_model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    pte_file = output_dir / file.with_suffix(".pte").name

    exported_program = torch.export.export(torch_model, (sample_input,))
    graph_module = exported_program.module(check_guards=False)

    # Shared_Sram permits system DRAM; the U65/U85 default Sram_Only requires the whole model to fit in
    # on-chip SRAM and fails for YOLO. config_ini is left at ExecuTorch's built-in Arm/vela.ini default.
    compile_spec = EthosUCompileSpec(target=target, memory_mode="Shared_Sram")

    # Keep final YOLO cat and model IO in FP32 (mixed precision). set_node_name(None) requires the
    # composable quantizer and leaves the cat outside the Ethos-U delegate on the CPU, matching Arm's
    # reference flow for keeping parts of a model unquantized.
    quantizer = EthosUQuantizer(compile_spec, use_composable_quantizer=True)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    cat_nodes = [n for n in graph_module.graph.nodes if n.op == "call_function" and "aten.cat" in str(n.target)]
    if cat_nodes:
        LOGGER.info(f"{prefix} keeping final cat node '{cat_nodes[-1].name}' in FP32")
        quantizer.set_node_name(cat_nodes[-1].name, None)
    quantizer.set_io(None)

    # Post training quantization
    quantized_graph_module = prepare_pt2e(graph_module, quantizer)

    for batch in dataset:
        tensor = batch["img"] if isinstance(batch, dict) else batch
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected calibration batch to be a torch.Tensor, but got {type(tensor).__name__}.")
        if tensor.dtype == torch.uint8:
            tensor = tensor / 255.0  # normalize to 0-1 to match inference preprocessing
        quantized_graph_module(tensor.to(device=sample_input.device, dtype=sample_input.dtype))

    quantized_graph_module = convert_pt2e(quantized_graph_module)
    move_exported_model_to_eval(quantized_graph_module)

    quantized_exported_program = torch.export.export(quantized_graph_module, (sample_input,))

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

    # Report how much of the model runs on the Ethos-U NPU vs falls back to the CPU. The CPU-fallback ops
    # are exactly the ones whose kernels the runtime must register, so listing them here points users at
    # the runtime kernel requirements before they hit a missing-operator failure on device.
    try:
        from executorch.devtools.backend_debug import get_delegation_info

        info = get_delegation_info(edge_program_manager.exported_program().graph_module)
        state = "fully" if not info.num_non_delegated_nodes else "not" if not info.num_delegated_nodes else "partially"
        LOGGER.info(
            f"{prefix} model {state} delegated to Ethos-U: "
            f"{info.num_delegated_nodes} NPU nodes, {info.num_non_delegated_nodes} CPU nodes"
        )
        cpu_ops = sorted(
            ((b.op_type, b.non_delegated) for b in info.delegation_by_operator.values() if b.non_delegated > 0),
            key=lambda x: (-x[1], x[0]),
        )
        if cpu_ops:
            LOGGER.info(f"{prefix} CPU fallback ops: {', '.join(f'{op} ({n})' for op, n in cpu_ops)}")
    except Exception as e:
        LOGGER.debug(f"{prefix} delegation info unavailable: {e}")

    # Convert edge program to executorch. The .pte keeps the standard quantized_decomposed q/dq ops at
    # the delegate boundaries and an FP32 cat on the CPU, so the runtime must link the ExecuTorch
    # quantized_ops_lib (and portable ops for aten::cat.out), as Arm's reference executor_runner does.
    # We deliberately do not apply ReplaceQuantNodesPass here: rewriting to cortex_m::* ops would pin the
    # .pte to runtimes linking cortex_m_ops_lib without removing the quantized_ops_lib requirement.
    executorch_program_manager = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    # Save pte file
    save_pte_program(executorch_program_manager, str(pte_file))

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
