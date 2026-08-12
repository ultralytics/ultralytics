---
title: utils.export.deepx API Reference
description: Reference for the Ultralytics DEEPX export utility. Learn how to convert ONNX models to DEEPX format using the DX-Compiler.
keywords: onnx2deepx, DEEPX export, dx_com, ONNX, model conversion, Ultralytics, INT8 quantization
---

# Reference for `ultralytics/utils/export/deepx.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/deepx.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/deepx.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`onnx2deepx`](#ultralytics.utils.export.deepx.onnx2deepx)


## Function `ultralytics.utils.export.deepx.onnx2deepx` {#ultralytics.utils.export.deepx.onnx2deepx}

```python
def onnx2deepx(
    onnx_file: str | Path,
    imgsz: tuple[int, int],
    dataset,
    metadata: dict | None = None,
    optimize: bool = False,
    prefix: str = "",
) -> Path
```

Convert an ONNX model to DEEPX format using the DEEPX DX-Compiler.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `onnx_file` | `str \| Path` | Input ONNX model path. | *required* |
| `imgsz` | `tuple[int, int]` | Export image size as ``(height, width)``. | *required* |
| `dataset` | `DataLoader` | Calibration dataloader used to build the DEEPX config. | *required* |
| `metadata` | `dict \| None, optional` | Optional metadata to save as YAML. Defaults to None. | `None` |
| `optimize` | `bool, optional` | If True, enables higher compiler optimization which reduces inference latency and increases compilation time. Defaults to False. | `False` |
| `prefix` | `str, optional` | Logging prefix. Defaults to "". | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `Path` | Path to the exported DEEPX model directory. |

<details>
<summary>Source code in <code>ultralytics/utils/export/deepx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/deepx.py#L12-L73">View on GitHub</a>
```python
def onnx2deepx(
    onnx_file: str | Path,
    imgsz: tuple[int, int],
    dataset,
    metadata: dict | None = None,
    optimize: bool = False,
    prefix: str = "",
) -> Path:
    """Convert an ONNX model to DEEPX format using the DEEPX DX-Compiler.

    Args:
        onnx_file (str | Path): Input ONNX model path.
        imgsz (tuple[int, int]): Export image size as ``(height, width)``.
        dataset (DataLoader): Calibration dataloader used to build the DEEPX config.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        optimize (bool, optional): If True, enables higher compiler optimization which reduces inference latency and
            increases compilation time. Defaults to False.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (Path): Path to the exported DEEPX model directory.
    """
    try:
        import dx_com
    except ImportError:
        check_requirements("dx_com", cmds="-f https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html")
        import dx_com

    LOGGER.info(f"\n{prefix} starting export with DEEPX...")

    onnx_file = Path(onnx_file)
    export_path = onnx_file.parent / f"{onnx_file.stem}_deepx_model"
    export_path.mkdir(exist_ok=True)
    config_path = export_path / "config.json"

    config = {
        "inputs": {"images": [1, 3, imgsz[0], imgsz[1]]},
        "calibration_num": 100,  # number of steps used during calibration
        "calibration_method": "ema",  # calibration method used during quantization
        "default_loader": {
            # JSON needs str; ClassificationDataset stores its image directory as 'root' rather than 'img_path'
            "dataset_path": str(getattr(dataset.dataset, "img_path", None) or dataset.dataset.root),
            "file_extensions": [val for x in ["jpeg", "jpg", "png"] for val in (x.lower(), x.upper())],
            "preprocessings": [
                {"resize": {"mode": "pad", "size": imgsz[0], "pad_location": "edge", "pad_value": [114, 114, 114]}},
                {"div": {"x": 255.0}},
                {"convertColor": {"form": "BGR2RGB"}},
                {"transpose": {"axis": [2, 0, 1]}},
                {"expandDim": {"axis": 0}},
            ],
        },
    }

    with open(config_path, "w") as file:
        json.dump(config, file)

    dx_com.compile(model=str(onnx_file), output_dir=str(export_path), config=str(config_path), opt_level=int(optimize))

    if metadata is not None:
        YAML.save(export_path / "metadata.yaml", metadata)

    return export_path
```
</details>

<br><br>
