# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_executorch_requirements, check_requirements

# Fixed INT8 ranges of the class logits and scores: calibration fits the logits to where nearly all of them are, far
# below 0, and every score then quantizes to sigmoid(0) = 0.5 at most.
LOGIT_RANGE = (-8.0, 8.0)  # sigmoid 0.0003 .. 0.9997
SCORE_RANGE = (0.0, 1.0)


def ethos_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """Cut the one-to-one YOLO26 Detect head before its top-k for Arm Ethos-U export.

    TOSA has no top-k, so the end-to-end postprocess would run on the CPU. The head instead returns the box distances
    (B, N, 4) in anchor strides and the class scores (B, N, nc), anchor-major: the NPU's own NHWC layout, so neither
    output needs a transpose. The device runtime keeps the anchors above its threshold, no NMS.
    """
    import types

    from ultralytics.nn.modules import Detect

    for m in model.modules():
        if type(m) is Detect and m.end2end:
            m.forward = types.MethodType(_ethos_detect_forward, m)
    return model


def _ethos_detect_forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the raw one-to-one box distances and class scores, anchor-major."""
    head = self.one2one
    box = torch.cat([head["box_head"][i](x[i]).permute(0, 2, 3, 1).flatten(1, 2) for i in range(self.nl)], 1)
    cls = torch.cat([head["cls_head"][i](x[i]).permute(0, 2, 3, 1).flatten(1, 2) for i in range(self.nl)], 1)
    return box, cls.sigmoid()


class _NHWCInput(torch.nn.Module):
    """Take the image channels-last (B, H, W, C), as a camera delivers it, and hand the model BCHW."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        """Run the wrapped model on the BCHW view of the NHWC input."""
        return self.model(x.permute(0, 3, 1, 2))


def _nhwc_batches(dataset: Iterable[Any]):
    """Yield the calibration batches channels-last."""
    for batch in dataset:
        yield (batch["img"] if isinstance(batch, dict) else batch).permute(0, 2, 3, 1)


def _fixed_qspec(low: float, high: float):
    """INT8 quantization spec with a fixed range instead of a calibrated one."""
    from torchao.quantization.pt2e.quantizer import FixedQParamsQuantizationSpec

    scale = (high - low) / 256
    return FixedQParamsQuantizationSpec(
        dtype=torch.int8,
        scale=scale,
        zero_point=-128 - round(low / scale),
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
    )


def _fix_score_ranges(model: torch.fx.GraphModule) -> None:
    """Give the class logits LOGIT_RANGE and the score output SCORE_RANGE, from the sigmoid back to the class convs."""
    from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec

    logits, scores = _fixed_qspec(*LOGIT_RANGE), _fixed_qspec(*SCORE_RANGE)
    output = model.graph.output_node()
    sigmoid = output.args[0][-1]
    annotation = lambda node: node.meta["quantization_annotation"]
    annotation(sigmoid).output_qspec = scores
    annotation(output).input_qspec_map[sigmoid] = scores
    todo = [sigmoid]
    while todo:
        node = todo.pop()
        qspecs = annotation(node).input_qspec_map
        for arg in node.all_input_nodes:
            if arg not in qspecs:
                continue
            if not isinstance(qspecs[arg], SharedQuantizationSpec):
                qspecs[arg] = logits
            if arg.target == torch.ops.aten.conv2d.default:
                annotation(arg).output_qspec = logits  # a class conv: the start of the logits
            else:
                todo.append(arg)


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
        (str): Path to the exported Arm Ethos-U ExecuTorch model directory.
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

    raw_head = any(getattr(m.forward, "__func__", None) is _ethos_detect_forward for m in torch_model.modules())
    if raw_head:
        # NHWC input: the NPU's own layout, so the BCHW permute cancels in TOSA. An NCHW input is a transpose of the
        # full-resolution image as the first NPU operation, which takes 2.7x the NPU scratch (YOLO26n at 320).
        torch_model = _NHWCInput(torch_model).eval()
        sample_input = sample_input.permute(0, 2, 3, 1).contiguous()
        dataset = _nhwc_batches(dataset)
        if metadata is not None:
            metadata["input_layout"] = "NHWC"

    exported_program = torch.export.export(torch_model, (sample_input,))
    graph_module = exported_program.module(check_guards=False)

    # Shared_Sram permits system DRAM; the U65/U85 default Sram_Only requires the whole model to fit in
    # on-chip SRAM and fails for YOLO. config_ini is left at ExecuTorch's built-in Arm/vela.ini default.
    compile_spec = EthosUCompileSpec(target=target, memory_mode="Shared_Sram")

    quantizer = EthosUQuantizer(compile_spec, use_composable_quantizer=True)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    if raw_head:
        # The whole graph in INT8 on the NPU, model IO included: fixed logit and score ranges keep the scores.
        LOGGER.info(f"{prefix} fixed INT8 ranges: class logits {LOGIT_RANGE}, scores {SCORE_RANGE}")
        annotate = quantizer.annotate

        def annotate_with_score_ranges(model):
            model = annotate(model)
            _fix_score_ranges(model)
            return model

        quantizer.annotate = annotate_with_score_ranges
    else:
        # Keep the score sigmoid, final cat, and model IO in FP32: the near-zero sigmoid scores lose too much
        # resolution under INT8 and hurt recall. Match nodes by aten op, not fragile FX node names.
        for label, op in (("score sigmoid", "aten.sigmoid"), ("final cat", "aten.cat")):
            nodes = [n for n in graph_module.graph.nodes if n.op == "call_function" and op in str(n.target)]
            if nodes:
                LOGGER.info(f"{prefix} keeping {label} node '{nodes[-1].name}' in FP32")
                quantizer.set_node_name(nodes[-1].name, None)
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

    config = ExecutorchBackendConfig(extract_delegate_segments=False)
    if raw_head:
        # INT8 IO: the q/dq at the delegate boundary leave the program, the runtime feeds and reads the INT8 tensors
        # with the scales recorded in metadata.yaml. The input is not memory-planned: the delegate copies it from the
        # caller's buffer into the NPU scratch.
        from executorch.exir.passes import MemoryPlanningPass
        from executorch.exir.passes.quantize_io_pass import quantize_input, quantize_output

        program = edge_program_manager.exported_program()
        io = {"inputs": [quantize_input(program, 0)]}
        io["outputs"] = [quantize_output(program, i) for i in range(len(program.graph_signature.user_outputs))]
        if metadata is not None:
            metadata["quantization"] = {
                k: [{"scale": float(q[0]), "zero_point": int(q[1])} for q in v] for k, v in io.items()
            }
        config = ExecutorchBackendConfig(
            extract_delegate_segments=False, memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
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

    # Convert edge program to executorch. Without the raw head, the .pte keeps the standard quantized_decomposed q/dq
    # ops at the delegate boundaries and an FP32 cat on the CPU, so the runtime must link the ExecuTorch
    # quantized_ops_lib (and portable ops for aten::cat.out), as Arm's reference executor_runner does.
    # We deliberately do not apply ReplaceQuantNodesPass here: rewriting to cortex_m::* ops would pin the
    # .pte to runtimes linking cortex_m_ops_lib without removing the quantized_ops_lib requirement.
    executorch_program_manager = edge_program_manager.to_executorch(config=config)

    # Save pte file
    save_pte_program(executorch_program_manager, str(pte_file))

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
