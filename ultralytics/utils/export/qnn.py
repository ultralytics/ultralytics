# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER, WINDOWS, YAML
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


def onnx2qnn(
    onnx_file: str | Path,
    output_dir: Path | str,
    dataset,
    transform_fn,
    name: str = "73",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to an INT8 Qualcomm QNN context binary using the ONNX Runtime QNN Execution Provider.

    The conversion runs entirely on the host with no Qualcomm account or cloud upload. The model is INT8-quantized with
    ONNX Runtime's QNN QDQ flow (the Hexagon NPU is an int8 accelerator), then the `onnxruntime-qnn` Execution Provider
    — which bundles the Qualcomm AI Runtime (QAIRT) libraries — compiles the quantized graph into a QNN context binary
    embedded in `<stem>_qnn.onnx`. No inference is run.

    Args:
        onnx_file (str | Path): Path to the source ONNX file (already exported).
        output_dir (Path | str): Directory to save the exported QNN model.
        dataset (DataLoader): Calibration dataloader (from `Exporter.get_int8_calibration_dataloader`) used for INT8
            quantization.
        transform_fn (Callable): Preprocessing transform (`Exporter._transform_fn`) converting a calibration item to a
            normalized `float32` NCHW array.
        name (str): Target Hexagon Tensor Processor (HTP) architecture version, e.g. `"73"` (Snapdragon 8 Gen 2), `"75"`
            (8 Gen 3), `"79"` (8 Elite). Finalizes the graph for the target chip when exporting on a host without a
            Snapdragon NPU.
        metadata (dict | None): Metadata saved as `metadata.yaml`.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported `_qnn_model` directory.

    Notes:
        `onnxruntime-qnn` wheels may expose QNN either as a plugin library or as a built-in ONNX Runtime provider.
    """
    check_requirements("onnxruntime-qnn")
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
    from onnxruntime.quantization.shape_inference import quant_pre_process

    ep_library, htp_backend = qnn_library_paths()

    onnx_file = Path(onnx_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx_file = output_dir / f"{onnx_file.stem}_qnn.onnx"
    pre_file = output_dir / "preprocessed.onnx"
    qdq_file = output_dir / "qdq.onnx"

    # INT8 QDQ quantization (HTP is an int8 accelerator). Reuses the shared calibration dataloader + transform_fn.
    class _CalibrationReader(CalibrationDataReader):
        def __init__(self):
            """Materialize calibration inputs as `{input_name: float32_NCHW}` dicts."""
            self.samples = [{"images": transform_fn(batch)} for batch in dataset]
            self.iterator = iter(self.samples)

        def get_next(self):
            """Return the next calibration sample, or None when exhausted."""
            return next(self.iterator, None)

        def rewind(self):
            """Reset the iterator for an additional calibration pass."""
            self.iterator = iter(self.samples)

    LOGGER.info(f"\n{prefix} starting INT8 quantization and export with ONNX Runtime QNN (HTP arch {name})...")
    quant_pre_process(str(onnx_file), str(pre_file))
    qdq_config = get_qnn_qdq_config(str(pre_file), _CalibrationReader())
    quantize(str(pre_file), str(qdq_file), qdq_config)

    # Register the QNN EP, then compile the quantized graph to a context binary during session init (no inference run).
    # htp_arch targets the chip so the graph finalizes offline on a host without an NPU, and the shared-memory
    # allocator is disabled (no device present).
    ep_name = "QNNExecutionProvider"
    ep_options = {
        "backend_path": htp_backend,
        "htp_arch": name,
        "htp_graph_finalization_optimization_mode": "3",
        "enable_htp_shared_memory_allocator": "0",
    }
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

    if not ctx_file.exists():
        raise RuntimeError(f"QNN context binary was not generated at {ctx_file}. See {prefix} logs for details.")

    for f in (pre_file, qdq_file):  # remove quantization intermediates; the context binary is self-contained
        f.unlink(missing_ok=True)
    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
