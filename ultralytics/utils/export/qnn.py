# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import ARM64, LINUX, LOGGER, WINDOWS, YAML
from ultralytics.utils.checks import check_requirements


def onnx2qnn(
    onnx_file: str | Path,
    output_dir: Path | str,
    backend: str = "htp",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to a Qualcomm QNN context binary locally using the ONNX Runtime QNN Execution Provider.

    The conversion runs entirely on the host with no Qualcomm account or cloud upload — the ``onnxruntime-qnn`` package
    bundles the Qualcomm AI Runtime (QAIRT) libraries. Initializing an ONNX Runtime session with context-binary caching
    enabled compiles the ONNX graph into a QNN context binary embedded in ``<stem>_qnn.onnx``; no inference is run.

    Args:
        onnx_file (str | Path): Path to the source ONNX file (already exported).
        output_dir (Path | str): Directory to save the exported QNN model.
        backend (str): QNN backend to target, one of ``"htp"`` (Hexagon NPU), ``"gpu"`` (Adreno), or ``"cpu"``. The
            HTP backend runs the float model at fp16 precision (via ``enable_htp_fp16_precision``); int8 calibration is
            not performed.
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_qnn_model`` directory.

    Notes:
        ``onnxruntime-qnn`` ships prebuilt wheels for Windows (x64/ARM64) and Linux ARM64 (aarch64) only. There is no
        Linux x86-64 or macOS wheel — on those hosts build ONNX Runtime from source with ``--use_qnn``, or generate the
        context binary on a supported platform.
    """
    assert WINDOWS or (LINUX and ARM64), (
        "QNN export requires 'onnxruntime-qnn', which ships prebuilt wheels only for Windows (x64/ARM64) and Linux "
        "ARM64 (aarch64). No wheel exists for Linux x86-64 or macOS — build ONNX Runtime from source with '--use_qnn' "
        "or run the export on a supported platform."
    )
    backends = {"htp": "QnnHtp", "gpu": "QnnGpu", "cpu": "QnnCpu"}
    assert backend in backends, f"Invalid QNN backend '{backend}', use one of {list(backends)}."

    check_requirements("onnxruntime-qnn")
    import onnxruntime as ort

    backend_lib = f"{backends[backend]}.dll" if WINDOWS else f"lib{backends[backend]}.so"
    LOGGER.info(f"\n{prefix} starting export with ONNX Runtime QNN ({backend_lib})...")

    onnx_file = Path(onnx_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx_file = output_dir / f"{onnx_file.stem}_qnn.onnx"

    provider_options = {"backend_path": backend_lib}
    if backend == "htp":
        provider_options["enable_htp_fp16_precision"] = "1"  # run the float model on HTP at fp16 (no int8 calibration)

    # Enable QNN context-binary caching, then initialize the session to compile and write the binary (no run needed)
    options = ort.SessionOptions()
    options.add_session_config_entry("ep.context_enable", "1")
    options.add_session_config_entry("ep.context_file_path", str(ctx_file))
    options.add_session_config_entry("ep.context_embed_mode", "1")
    ort.InferenceSession(
        str(onnx_file),
        sess_options=options,
        providers=["QNNExecutionProvider"],
        provider_options=[provider_options],
    )

    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
