---
title: utils.export.axelera API Reference
description: Axelera export utilities for converting PyTorch models to Axelera format for deployment on Metis AI processors. Supports INT8 quantization and optimized inference for computer vision workloads on Axelera hardware.
keywords: Ultralytics, Axelera, model export, PyTorch to Axelera, Metis AI processor, INT8 quantization, computer vision, AI inference, edge AI, hardware acceleration
---

# Reference for `ultralytics/utils/export/axelera.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/axelera.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/axelera.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`torch2axelera`](#ultralytics.utils.export.axelera.torch2axelera)


## Function `ultralytics.utils.export.axelera.torch2axelera` {#ultralytics.utils.export.axelera.torch2axelera}

```python
def torch2axelera(
    model: torch.nn.Module,
    output_dir: Path | str,
    calibration_dataset: torch.utils.data.DataLoader,
    transform_fn: Callable[[Any], np.ndarray],
    model_name: str = "model",
    metadata: dict | None = None,
    prefix: str = "",
) -> str
```

Convert a YOLO model to Axelera format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | Source YOLO model for quantization. | *required* |
| `output_dir` | `Path \| str` | Directory to save the exported Axelera model. | *required* |
| `calibration_dataset` | `torch.utils.data.DataLoader` | Calibration dataloader for quantization. | *required* |
| `transform_fn` | `Callable[[Any], np.ndarray]` | Calibration preprocessing transform function. | *required* |
| `model_name` | `str, optional` | Name for the compiled model. Defaults to "model". | `"model"` |
| `metadata` | `dict \| None, optional` | Optional metadata to save as YAML. Defaults to None. | `None` |
| `prefix` | `str, optional` | Prefix for log messages. Defaults to "". | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Path to exported Axelera model directory. |

<details>
<summary>Source code in <code>ultralytics/utils/export/axelera.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/axelera.py#L25-L122">View on GitHub</a>
```python
def torch2axelera(
    model: torch.nn.Module,
    output_dir: Path | str,
    calibration_dataset: torch.utils.data.DataLoader,
    transform_fn: Callable[[Any], np.ndarray],
    model_name: str = "model",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert a YOLO model to Axelera format.

    Args:
        model (torch.nn.Module): Source YOLO model for quantization.
        output_dir (Path | str): Directory to save the exported Axelera model.
        calibration_dataset (torch.utils.data.DataLoader): Calibration dataloader for quantization.
        transform_fn (Callable[[Any], np.ndarray]): Calibration preprocessing transform function.
        model_name (str, optional): Name for the compiled model. Defaults to "model".
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Prefix for log messages. Defaults to "".

    Returns:
        (str): Path to exported Axelera model directory.
    """
    # Serialize within the process: the steps below mutate process-global state (the protobuf env var and any
    # working-directory files the compiler writes), so concurrent in-process exports must not overlap.
    with _AXELERA_EXPORT_LOCK:
        prev_protobuf = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        try:
            try:
                from axelera import compiler
            except ImportError:
                check_requirements(
                    ["axelera-devkit==1.7.0", "numpy<=2.3.5"],
                    cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
                )
                from axelera import compiler

            check_requirements("omnimalloc==0.5.0")
            from axelera.compiler import CompilerConfig
            from axelera.compiler.config.model_specific import extract_ultralytics_metadata

            LOGGER.info(f"\n{prefix} starting export with Axelera compiler...")

            # Resolve to an absolute path so the relative compile dir below can never alias it.
            output_dir = Path(output_dir).resolve()
            if output_dir.exists():
                shutil.rmtree(output_dir)

            axelera_model_metadata = extract_ultralytics_metadata(model)
            config = CompilerConfig(
                model_metadata=axelera_model_metadata,
                model_name=model_name,
                resources_used=0.25,
                aipu_cores_used=1,
                multicore_mode="batch",
                output_axm_format=True,
            )
            qmodel = compiler.quantize(
                model=model,
                calibration_dataset=calibration_dataset,
                config=config,
                transform_fn=transform_fn,
            )

            # The Axelera compiler emits invalid artifacts for absolute output paths, so compile into a local
            # relative directory. TemporaryDirectory gives it a unique name (so back-to-back exports of identically
            # named models never collide) in the current working directory, and removes it on exit even if
            # compilation raises; passing its relative basename keeps it from aliasing the absolute output_dir, so
            # cleanup can never delete the result.
            with tempfile.TemporaryDirectory(prefix="axelera_compile_", dir=".") as compile_root:
                compile_dir = Path(Path(compile_root).name)
                compiler.compile(model=qmodel, config=config, output_dir=compile_dir)

                output_dir.mkdir(parents=True, exist_ok=True)
                for artifact in [f"{model_name}.axm", "compiler_config_final.toml"]:
                    for artifact_path in [compile_dir / artifact, Path(artifact)]:
                        if artifact_path.exists():
                            artifact_path.replace(output_dir / artifact_path.name)
                            break

                # Remove intermediate compiler artifacts, keeping only the compiled model and config.
                keep_suffixes = {".axm"}
                keep_names = {"compiler_config_final.toml", "metadata.yaml"}
                for f in output_dir.iterdir():
                    if f.is_file() and f.suffix not in keep_suffixes and f.name not in keep_names:
                        f.unlink()

                if metadata is not None:
                    YAML.save(output_dir / "metadata.yaml", metadata)

            return str(output_dir)
        finally:
            # Restore original PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION value
            if prev_protobuf is None:
                os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
            else:
                os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = prev_protobuf
```
</details>

<br><br>
