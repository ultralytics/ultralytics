---
title: utils.export.qnn API Reference
description: Reference for the Ultralytics Qualcomm QNN export utility. Learn how to compile ONNX models to the QNN format locally with the ONNX Runtime QNN Execution Provider.
keywords: onnx2qnn, QNN export, Qualcomm export, ONNX Runtime QNN, onnxruntime-qnn, Qualcomm AI Engine Direct, Qualcomm AI Hub, QAIRT, SNPE, ONNX, model conversion, Ultralytics, Snapdragon, Hexagon NPU, Hexagon HTP
---

# Reference for `ultralytics/utils/export/qnn.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/qnn.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/qnn.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`qnn_library_paths`](#ultralytics.utils.export.qnn.qnn_library_paths)
        - [`onnx2qnn`](#ultralytics.utils.export.qnn.onnx2qnn)


## Function `ultralytics.utils.export.qnn.qnn_library_paths` {#ultralytics.utils.export.qnn.qnn\_library\_paths}

```python
def qnn_library_paths() -> tuple[str | None, str]
```

Resolve the QNN Execution Provider and HTP backend library paths for the installed onnxruntime-qnn build.

onnxruntime-qnn ships two ways: plugin builds expose an `onnxruntime_qnn` helper module, while monolithic builds expose `QNNExecutionProvider` directly and bundle the QNN backend libraries in `onnxruntime/capi`.

**Returns**

| Type | Description |
| --- | --- |
| `tuple[str \| None, str]` | `(ep_library_path, htp_backend_path)`. `ep_library_path` is `None` when QNN is already built into ONNX Runtime and does not need `register_execution_provider_library`. |

<details>
<summary>Source code in <code>ultralytics/utils/export/qnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/qnn.py#L11-L34">View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.export.qnn.onnx2qnn` {#ultralytics.utils.export.qnn.onnx2qnn}

```python
def onnx2qnn(
    onnx_file: str | Path,
    output_file: Path | str,
    dataset,
    transform_fn,
    name: str = "73",
    metadata: dict | None = None,
    batch: int = 0,
    prefix: str = "",
) -> str
```

Convert an ONNX model to a Qualcomm QNN context binary using the ONNX Runtime QNN Execution Provider.

The conversion runs entirely on the host with no Qualcomm account or cloud upload. The model is quantized with ONNX Runtime's QNN QDQ flow to 16-bit activations and 8-bit weights (the recommended accuracy/performance balance for the Hexagon NPU), then the `onnxruntime-qnn` Execution Provider — which bundles the Qualcomm AI Runtime (QAIRT) libraries — compiles the quantized graph into a QNN context binary embedded in `<stem>_qnn.onnx`. No inference is run.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `onnx_file` | `str \| Path` | Path to the source ONNX file (already exported). | *required* |
| `output_file` | `Path \| str` | Path to save the exported QNN ONNX context-binary model. | *required* |
| `dataset` | `DataLoader` | Calibration dataloader (from `Exporter.get_int8_calibration_dataloader`) used for INT8 quantization. | *required* |
| `transform_fn` | `Callable` | Preprocessing transform (`Exporter._transform_fn`) converting a calibration item to a normalized `float32` NCHW array. | *required* |
| `name` | `str` | Target Hexagon Tensor Processor (HTP) architecture version, e.g. `"73"` (Snapdragon 8 Gen 2), `"75"` (8 Gen 3), `"79"` (8 Elite), or supported SoC name such as `"iq-8275"`. Finalizes the graph for the target chip when exporting on a host without a Snapdragon NPU. | `"73"` |
| `metadata` | `dict \| None` | Ultralytics model metadata ensured present in the context model's `metadata_props` (ONNX Runtime normally carries the source model's metadata through, but this is not a documented guarantee). | `None` |
| `batch` | `int` | Static batch dimension of the ONNX graph used to tile undersized calibration batches, or 0 for dynamic-batch models. | `0` |
| `prefix` | `str` | Prefix for log messages. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Path to the exported `*_qnn.onnx` file. |

!!! note "Notes"

    `onnxruntime-qnn` wheels may expose QNN either as a plugin library or as a built-in ONNX Runtime provider.

<details>
<summary>Source code in <code>ultralytics/utils/export/qnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/qnn.py#L37-L164">View on GitHub</a>
```python
def onnx2qnn(
    onnx_file: str | Path,
    output_file: Path | str,
    dataset,
    transform_fn,
    name: str = "73",
    metadata: dict | None = None,
    batch: int = 0,
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
            (8 Gen 3), `"79"` (8 Elite), or supported SoC name such as `"iq-8275"`. Finalizes the graph for the target
            chip when exporting on a host without a Snapdragon NPU.
        metadata (dict | None): Ultralytics model metadata ensured present in the context model's `metadata_props` (ONNX
            Runtime normally carries the source model's metadata through, but this is not a documented guarantee).
        batch (int): Static batch dimension of the ONNX graph used to tile undersized calibration batches, or 0 for
            dynamic-batch models.
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

    LOGGER.info(f"\n{prefix} starting W8A16 quantization and export with ONNX Runtime QNN (HTP target {name})...")
    import onnx

    dims = [d.dim_value for d in onnx.load(str(onnx_file)).graph.input[0].type.tensor_type.shape.dim]
    if len(dims) == 4 and dims[3] in {1, 3} and dims[1] not in {1, 3}:  # channel-last graph (QNNModel export)
        nchw_transform = transform_fn

        def transform_fn(data_item):
            """Transform calibration data from NCHW to NHWC."""
            return nchw_transform(data_item).transpose(0, 2, 3, 1)

    try:
        quant_pre_process(str(onnx_file), str(pre_file))
        # 16-bit activations + 8-bit weights is the ORT-recommended accuracy/perf balance for the HTP backend
        qdq_config = get_qnn_qdq_config(
            str(pre_file),
            onnx_calibration_reader(dataset, transform_fn, batch=batch),
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QUInt8,
        )
        quantize(str(pre_file), str(qdq_file), qdq_config)

        # Register the QNN EP, then compile the quantized graph to a context binary during session init (no inference
        # run). The provider target finalizes the graph offline on a host without an NPU, and the shared-memory allocator
        # is disabled (no device present). Targets not exposed by ONNX Runtime's htp_arch parser are finalized through
        # their QNN SoC model instead.
        ep_name = "QNNExecutionProvider"
        ep_options = {
            "backend_path": htp_backend,
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_shared_memory_allocator": "0",
        }
        option, value = QNN_HTP_TARGETS[name]
        ep_options[option] = value
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
        for f in (pre_file, qdq_file):  # remove quantization intermediates; the context binary is self-contained
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
```
</details>

<br><br>
