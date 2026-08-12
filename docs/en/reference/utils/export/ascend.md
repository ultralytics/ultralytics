---
title: utils.export.ascend API Reference
description: Reference for `ultralytics.utils.export.ascend` in the Ultralytics package.
keywords: Ultralytics, ultralytics.utils.export.ascend, API reference, YOLO, Python
---

# Reference for `ultralytics/utils/export/ascend.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/ascend.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/ascend.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`_check_atc`](#ultralytics.utils.export.ascend._check_atc)
        - [`onnx2ascend`](#ultralytics.utils.export.ascend.onnx2ascend)


## Function `ultralytics.utils.export.ascend._check_atc` {#ultralytics.utils.export.ascend.\_check\_atc}

```python
def _check_atc() -> None
```

Raise if the CANN ATC compiler is not on PATH.

<details>
<summary>Source code in <code>ultralytics/utils/export/ascend.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/ascend.py#L14-L21">View on GitHub</a>
```python
def _check_atc() -> None:
    """Raise if the CANN ATC compiler is not on PATH."""
    if not shutil.which("atc"):
        raise FileNotFoundError(
            "Ascend export requires the CANN toolkit 'atc' compiler, which was not found on PATH. Install CANN and "
            "source its environment, e.g. `source /usr/local/Ascend/ascend-toolkit/set_env.sh`. "
            "See https://docs.ultralytics.com/integrations/ascend/"
        )
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.export.ascend.onnx2ascend` {#ultralytics.utils.export.ascend.onnx2ascend}

```python
def onnx2ascend(
    onnx_file: str | Path,
    output_dir: str | Path,
    name: str,
    imgsz: tuple[int, int],
    batch: int = 1,
    channels: int = 3,
    metadata: dict | None = None,
    prefix: str = "",
) -> str
```

Convert an ONNX model to a Huawei Ascend offline model (.om) with the CANN ATC compiler.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `onnx_file` | `str \| Path` | Input ONNX model path. | *required* |
| `output_dir` | `str \| Path` | Directory to write the compiled .om model into. | *required* |
| `name` | `str` | Target Ascend SoC passed to ATC as ``--soc_version``, e.g. ``"Ascend310B4"``. | *required* |
| `imgsz` | `tuple[int, int]` | Export image size as ``(height, width)``. | *required* |
| `batch` | `int, optional` | Static batch size baked into the offline model. Defaults to 1. | `1` |
| `channels` | `int, optional` | Input channel count, matching the traced ONNX graph. Defaults to 3. | `3` |
| `metadata` | `dict \| None, optional` | Optional metadata to save as YAML. Defaults to None. | `None` |
| `prefix` | `str, optional` | Logging prefix. Defaults to "". | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Path to the exported Ascend model directory. |

<details>
<summary>Source code in <code>ultralytics/utils/export/ascend.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/ascend.py#L24-L71">View on GitHub</a>
```python
def onnx2ascend(
    onnx_file: str | Path,
    output_dir: str | Path,
    name: str,
    imgsz: tuple[int, int],
    batch: int = 1,
    channels: int = 3,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to a Huawei Ascend offline model (.om) with the CANN ATC compiler.

    Args:
        onnx_file (str | Path): Input ONNX model path.
        output_dir (str | Path): Directory to write the compiled .om model into.
        name (str): Target Ascend SoC passed to ATC as ``--soc_version``, e.g. ``"Ascend310B4"``.
        imgsz (tuple[int, int]): Export image size as ``(height, width)``.
        batch (int, optional): Static batch size baked into the offline model. Defaults to 1.
        channels (int, optional): Input channel count, matching the traced ONNX graph. Defaults to 3.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (str): Path to the exported Ascend model directory.
    """
    _check_atc()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "atc",
        f"--model={Path(onnx_file).resolve()}",  # absolute: ATC runs with cwd=output_dir
        "--framework=5",  # 5 = ONNX
        f"--output={output_dir / f'{Path(onnx_file).stem}_{name}'}",  # ATC appends the .om suffix
        "--input_format=NCHW",
        f"--input_shape=images:{batch},{channels},{imgsz[0]},{imgsz[1]}",
        f"--soc_version={name}",
        "--precision_mode=force_fp16",  # Ascend AI Core convolutions reject FP32 inputs
    ]  # argv list avoids shell metacharacter issues in onnx_file/output_dir paths
    LOGGER.info(f"\n{prefix} starting export with ATC for {name}...")
    LOGGER.info(f"{prefix} running '{shlex.join(cmd)}'")
    with tempfile.TemporaryDirectory() as scratch:  # ATC drops kernel_meta/ and fusion_result.json in its CWD
        subprocess.run(cmd, check=True, cwd=scratch)

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
```
</details>

<br><br>
