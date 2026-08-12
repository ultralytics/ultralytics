---
title: nn.backends.rknn API Reference
description: Explore RKNNBackend for Rockchip RKNN inference, enabling optimized YOLO deployment on Rockchip NPU-equipped edge devices.
keywords: Ultralytics, RKNNBackend, RKNN inference, Rockchip RKNN, NPU inference, edge AI, embedded deployment, deep learning
---

# Reference for `ultralytics/nn/backends/rknn.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/rknn.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/rknn.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`RKNNBackend`](#ultralytics.nn.backends.rknn.RKNNBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`RKNNBackend.load_model`](#ultralytics.nn.backends.rknn.RKNNBackend.load_model)
        - [`RKNNBackend.forward`](#ultralytics.nn.backends.rknn.RKNNBackend.forward)


## Class `ultralytics.nn.backends.rknn.RKNNBackend` {#ultralytics.nn.backends.rknn.RKNNBackend}

```python
RKNNBackend()
```

**Bases:** `BaseBackend`

Rockchip RKNN inference backend for Rockchip NPU hardware.

Loads and runs inference with RKNN models (.rknn files) using the RKNN-Toolkit-Lite2 runtime. Only supported on Rockchip devices with NPU hardware (e.g., RK3588, RK3566).

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.rknn.RKNNBackend.forward) | Run inference on the Rockchip NPU. |
| [`load_model`](#ultralytics.nn.backends.rknn.RKNNBackend.load_model) | Load a Rockchip RKNN model from a .rknn file or model directory. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/rknn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/rknn.py#L15-L86">View on GitHub</a>
```python
class RKNNBackend(BaseBackend):
    """Rockchip RKNN inference backend for Rockchip NPU hardware.

    Loads and runs inference with RKNN models (.rknn files) using the RKNN-Toolkit-Lite2 runtime. Only supported on
    Rockchip devices with NPU hardware (e.g., RK3588, RK3566).
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.rknn.RKNNBackend.forward` {#ultralytics.nn.backends.rknn.RKNNBackend.forward}

```python
def forward(self, im: torch.Tensor) -> list
```

Run inference on the Rockchip NPU.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BHWC format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | Model predictions as a list of output arrays. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/rknn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/rknn.py#L59-L86">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> list:
    """Run inference on the Rockchip NPU.

    Args:
        im (torch.Tensor): Input image tensor in BHWC format, normalized to [0, 1].

    Returns:
        (list): Model predictions as a list of output arrays.
    """
    h, w = im.shape[1:3]
    im = (im.cpu().numpy() * 255).astype("uint8")
    im = im if isinstance(im, (list, tuple)) else [im]
    y = self.model.inference(inputs=im)
    # INT8 exports use input-relative coordinates so a single per-tensor scale preserves class scores.
    if (
        self.metadata.get("args", {}).get("quantize") == 8
        and self.task in {"detect", "segment", "pose", "obb"}
        and not self.end2end
    ):
        kpt_start = 4 + len(self.names)  # pose keypoints follow the box (4) and class-score (nc) channels
        for x in y:
            if x.ndim == 3:
                x[:, [0, 2]] *= w
                x[:, [1, 3]] *= h
                if self.task == "pose":
                    x[:, kpt_start::3] *= w
                    x[:, kpt_start + 1 :: 3] *= h
    return y
```
</details>

<br>

### Method `ultralytics.nn.backends.rknn.RKNNBackend.load_model` {#ultralytics.nn.backends.rknn.RKNNBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load a Rockchip RKNN model from a .rknn file or model directory.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .rknn file or directory containing the model. | *required* |

**Raises**

| Type | Description |
| --- | --- |
| `OSError` | If not running on a Rockchip device. |
| `RuntimeError` | If model loading or runtime initialization fails. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/rknn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/rknn.py#L22-L57">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load a Rockchip RKNN model from a .rknn file or model directory.

    Args:
        weight (str | Path): Path to the .rknn file or directory containing the model.

    Raises:
        OSError: If not running on a Rockchip device.
        RuntimeError: If model loading or runtime initialization fails.
    """
    if not is_rockchip():
        raise OSError("RKNN inference is only supported on Rockchip devices.")

    LOGGER.info(f"Loading {weight} for RKNN inference...")
    check_requirements("rknn-toolkit-lite2")
    from rknnlite.api import RKNNLite

    w = Path(weight)
    if not w.is_file():
        w = next(w.rglob("*.rknn"))

    self.model = RKNNLite()
    ret = self.model.load_rknn(str(w))
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {ret}")

    ret = self.model.init_runtime()
    if ret != 0:
        raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

    # Load metadata
    metadata_file = w.parent / "metadata.yaml"
    if metadata_file.exists():
        from ultralytics.utils import YAML

        self.apply_metadata(YAML.load(metadata_file))
```
</details>

<br><br>
