# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER, WINDOWS
from ultralytics.utils.checks import check_requirements


def qnn_library_paths() -> tuple[str | None, str]:
    """Resolve the QNN Execution Provider and HTP backend library paths for the installed onnxruntime-qnn build.

    onnxruntime-qnn ships two ways: plugin builds expose an `onnxruntime_qnn` helper module, while monolithic builds
    expose `QNNExecutionProvider` directly and bundle the QNN backend libraries in `onnxruntime/capi`.

    Returns:
        (tuple[str | None, str]): `(ep_library_path, htp_backend_path)`. `ep_library_path` is `None` when QNN is already
            built into ONNX Runtime and does not need `register_execution_provider_library`.
    """
    try:
        import onnxruntime_qnn as qnn_ep

        return qnn_ep.get_library_path(), qnn_ep.get_qnn_htp_path()
    except ImportError:
        import onnxruntime

        capi = Path(onnxruntime.__file__).parent / "capi"
        if "QNNExecutionProvider" in onnxruntime.get_available_providers():
            ep_lib = None
        else:
            ep_lib = capi / ("onnxruntime_providers_qnn.dll" if WINDOWS else "libonnxruntime_providers_qnn.so")
        htp_lib = "QnnHtp.dll" if WINDOWS else "libQnnHtp.so"
        return str(ep_lib) if ep_lib else None, str(capi / htp_lib)


def _prepare_qnn_graph(onnx_file: Path, task: str | None) -> tuple[Path, bool]:
    """Rewrite the ONNX graph for Hexagon-friendly I/O before quantization.

    Two rewrites, both following Qualcomm's published deployment recipes:

    1. Channel-last image input: the HTP's native layout is NHWC, so an NCHW input costs a boundary transpose on
    every inference (and a matching CPU-side permute in apps). The input is converted to NHWC with an in-graph Transpose
    that ONNX Runtime's layout transformer folds away during context generation.
    2. Semantic class-map output: semantic models otherwise emit full float logits (~20M values at 1024px) that
    must be dequantized and argmax-decoded on the CPU every frame. An in-graph ArgMax + uint8 Cast reduces the output to
    a compact `[N, H, W]` class map computed on the NPU.

    Returns:
        (tuple[Path, bool]): Path to the rewritten model and whether the input was converted to NHWC.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_file))
    graph = model.graph

    nhwc = False
    inp = graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) == 4 and dims[1] in {1, 3}:
        n, c, h, w = dims
        nchw_name = f"{inp.name}_nchw"
        for node in graph.node:
            node.input[:] = [nchw_name if name == inp.name else name for name in node.input]
        graph.node.insert(
            0, helper.make_node("Transpose", [inp.name], [nchw_name], perm=[0, 3, 1, 2], name="qnn_nhwc_input")
        )
        del inp.type.tensor_type.shape.dim[:]
        for d in (n, h, w, c):
            inp.type.tensor_type.shape.dim.add().dim_value = d
        nhwc = True

    if task == "semantic":
        out = graph.output[0]
        odims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        graph.node.extend(
            [
                helper.make_node("ArgMax", [out.name], [f"{out.name}_argmax"], axis=1, keepdims=0, name="qnn_argmax"),
                helper.make_node(
                    "Cast", [f"{out.name}_argmax"], ["class_map"], to=TensorProto.UINT8, name="qnn_classmap"
                ),
            ]
        )
        del graph.output[:]
        graph.output.extend([helper.make_tensor_value_info("class_map", TensorProto.UINT8, [odims[0], *odims[2:]])])

    prepared = onnx_file.with_name(f"{onnx_file.stem}_qnn_prepared.onnx")
    onnx.save(model, str(prepared))
    return prepared, nhwc


def onnx2qnn(
    onnx_file: str | Path,
    output_file: Path | str,
    dataset,
    transform_fn,
    name: str = "73",
    metadata: dict | None = None,
    batch: int = 0,
    task: str | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to a Qualcomm QNN context binary using the ONNX Runtime QNN Execution Provider.

    The conversion runs entirely on the host with no Qualcomm account or cloud upload. The model is quantized with ONNX
    Runtime's QNN QDQ flow to 16-bit activations and 8-bit weights (the recommended accuracy/performance balance for the
    Hexagon NPU), then the `onnxruntime-qnn` Execution Provider — which bundles the Qualcomm AI Runtime (QAIRT)
    libraries — compiles the quantized graph into a QNN context binary embedded in `<stem>_qnn.onnx`. No inference is
    run.

    Args:
        onnx_file (str | Path): Path to the source ONNX file (already exported).
        output_file (Path | str): Path to save the exported QNN ONNX context-binary model.
        dataset (DataLoader): Calibration dataloader (from `Exporter.get_int8_calibration_dataloader`) used for INT8
            quantization.
        transform_fn (Callable): Preprocessing transform (`Exporter._transform_fn`) converting a calibration item to a
            normalized `float32` NCHW array.
        name (str): Target Hexagon Tensor Processor (HTP) architecture version, e.g. `"73"` (Snapdragon 8 Gen 2), `"75"`
            (8 Gen 3), `"79"` (8 Elite). Finalizes the graph for the target chip when exporting on a host without a
            Snapdragon NPU.
        metadata (dict | None): Ultralytics model metadata ensured present in the context model's `metadata_props` (ONNX
            Runtime normally carries the source model's metadata through, but this is not a documented guarantee).
        batch (int): Static batch dimension of the ONNX graph used to tile undersized calibration batches, or 0 for
            dynamic-batch models.
        task (str | None): Model task; semantic models additionally get an in-graph ArgMax producing a compact uint8
            class-map output instead of float logits.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported `*_qnn.onnx` file.

    Notes:
        `onnxruntime-qnn` wheels may expose QNN either as a plugin library or as a built-in ONNX Runtime provider.
    """
    check_requirements("onnxruntime-qnn")
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from ultralytics.utils.export.onnx import onnx_calibration_reader

    ep_library, htp_backend = qnn_library_paths()

    onnx_file = Path(onnx_file)
    ctx_file = Path(output_file)
    ctx_file.parent.mkdir(parents=True, exist_ok=True)
    pre_file = ctx_file.with_name(f"{onnx_file.stem}_qnn_preprocessed.onnx")
    qdq_file = ctx_file.with_name(f"{onnx_file.stem}_qnn_qdq.onnx")

    LOGGER.info(f"\n{prefix} starting A16W8 quantization and export with ONNX Runtime QNN (HTP arch {name})...")
    prepared_file, nhwc = _prepare_qnn_graph(onnx_file, task)
    if nhwc:  # the graph now takes channel-last input, so calibration batches must be fed NHWC as well
        nchw_transform = transform_fn
        transform_fn = lambda data_item: nchw_transform(data_item).transpose(0, 2, 3, 1)  # noqa: E731
    try:
        quant_pre_process(str(prepared_file), str(pre_file))
        # 16-bit activations + 8-bit weights is the ORT-recommended accuracy/perf balance for the HTP backend
        qdq_config = get_qnn_qdq_config(
            str(pre_file),
            onnx_calibration_reader(dataset, transform_fn, batch=batch),
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QUInt8,
        )
        quantize(str(pre_file), str(qdq_file), qdq_config)

        # Register the QNN EP, then compile the quantized graph to a context binary during session init (no inference
        # run). htp_arch targets the chip so the graph finalizes offline on a host without an NPU, and the
        # shared-memory allocator is disabled (no device present). ONNX Runtime's htp_arch parser accepts
        # 68/69/73/75/81 but not 79 (invalid values only log a warning, leaving the graph untargeted), so v79
        # (Snapdragon 8 Elite) is targeted via its SoC model instead.
        ep_name = "QNNExecutionProvider"
        ep_options = {
            "backend_path": htp_backend,
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_shared_memory_allocator": "0",
        }
        if name == "79":
            ep_options["soc_model"] = "69"  # SM8750 (Snapdragon 8 Elite) -> HTP v79
        else:
            ep_options["htp_arch"] = name
        options = ort.SessionOptions()
        options.add_session_config_entry("ep.context_enable", "1")
        options.add_session_config_entry("ep.context_file_path", str(ctx_file))
        options.add_session_config_entry("ep.context_embed_mode", "1")
        if ep_library:
            ort.register_execution_provider_library(ep_name, ep_library)
        try:
            if ep_library:
                devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
                if not devices:
                    raise RuntimeError("QNN EP registered but no QNN devices were found by ONNX Runtime.")
                options.add_provider_for_devices(devices, ep_options)
                ort.InferenceSession(str(qdq_file), sess_options=options)
            else:
                ort.InferenceSession(
                    str(qdq_file), sess_options=options, providers=[ep_name], provider_options=[ep_options]
                )
        finally:
            if ep_library:
                ort.unregister_execution_provider_library(ep_name)
    finally:
        for f in (prepared_file, pre_file, qdq_file):  # remove intermediates; the context binary is self-contained
            f.unlink(missing_ok=True)

    if not ctx_file.exists():
        raise RuntimeError(f"QNN context binary was not generated at {ctx_file}. See {prefix} logs for details.")

    if metadata:  # ensure Ultralytics metadata is present in the context model (usually preserved by ONNX Runtime)
        import onnx

        ctx_model = onnx.load(str(ctx_file))
        existing = {p.key for p in ctx_model.metadata_props}
        if missing := {k: v for k, v in metadata.items() if str(k) not in existing}:
            for k, v in missing.items():
                entry = ctx_model.metadata_props.add()
                entry.key, entry.value = str(k), str(v)
            onnx.save(ctx_model, str(ctx_file))
    return str(ctx_file)
