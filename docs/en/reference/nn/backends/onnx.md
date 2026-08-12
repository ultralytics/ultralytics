---
title: nn.backends.onnx API Reference
description: Explore ONNXBackend and ONNXIMXBackend for Microsoft ONNX Runtime inference, supporting standard ONNX models and NXP IMX-optimized variants.
keywords: Ultralytics, ONNXBackend, ONNXIMXBackend, Microsoft ONNX Runtime, Sony IMX, ONNX inference, edge deployment, deep learning
---

# Reference for `ultralytics/nn/backends/onnx.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/onnx.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/onnx.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ONNXBackend`](#ultralytics.nn.backends.onnx.ONNXBackend)
        - [`ONNXIMXBackend`](#ultralytics.nn.backends.onnx.ONNXIMXBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ONNXBackend.load_model`](#ultralytics.nn.backends.onnx.ONNXBackend.load_model)
        - [`ONNXBackend.forward`](#ultralytics.nn.backends.onnx.ONNXBackend.forward)
        - [`ONNXIMXBackend.load_model`](#ultralytics.nn.backends.onnx.ONNXIMXBackend.load_model)
        - [`ONNXIMXBackend.forward`](#ultralytics.nn.backends.onnx.ONNXIMXBackend.forward)


## Class `ultralytics.nn.backends.onnx.ONNXBackend` {#ultralytics.nn.backends.onnx.ONNXBackend}

```python
ONNXBackend(weight: str | Path, device: torch.device, fp16: bool = False, format: str = "onnx")
```

**Bases:** `BaseBackend`

Microsoft ONNX Runtime inference backend with optional OpenCV DNN support.

Loads and runs inference with ONNX models (.onnx files) using either Microsoft ONNX Runtime with CUDA/CoreML execution providers, or OpenCV DNN for lightweight CPU inference. Supports IO binding for optimized GPU inference with static input shapes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .onnx model file. | *required* |
| `device` | `torch.device` | Device to run inference on. | *required* |
| `fp16` | `bool` | Whether to use FP16 half-precision inference. | `False` |
| `format` | `str` | Inference engine, either "onnx" for ONNX Runtime or "dnn" for OpenCV DNN. | `"onnx"` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.onnx.ONNXBackend.forward) | Run ONNX inference using IO binding (CUDA) or standard session execution. |
| [`load_model`](#ultralytics.nn.backends.onnx.ONNXBackend.load_model) | Load an ONNX model using ONNX Runtime or OpenCV DNN. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/onnx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/onnx.py#L27-L155">View on GitHub</a>
```python
class ONNXBackend(BaseBackend):
    """Microsoft ONNX Runtime inference backend with optional OpenCV DNN support.

    Loads and runs inference with ONNX models (.onnx files) using either Microsoft ONNX Runtime with CUDA/CoreML
    execution providers, or OpenCV DNN for lightweight CPU inference. Supports IO binding for optimized GPU inference
    with static input shapes.
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False, format: str = "onnx"):
        """Initialize the ONNX backend.

        Args:
            weight (str | Path): Path to the .onnx model file.
            device (torch.device): Device to run inference on.
            fp16 (bool): Whether to use FP16 half-precision inference.
            format (str): Inference engine, either "onnx" for ONNX Runtime or "dnn" for OpenCV DNN.
        """
        assert format in {"onnx", "dnn"}, f"Unsupported ONNX format: {format}."
        self.format = format
        super().__init__(weight, device, fp16)
```
</details>

<br>

### Method `ultralytics.nn.backends.onnx.ONNXBackend.forward` {#ultralytics.nn.backends.onnx.ONNXBackend.forward}

```python
def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor] | np.ndarray
```

Run ONNX inference using IO binding (CUDA) or standard session execution.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor \| list[torch.Tensor] \| np.ndarray` | Model predictions as tensor(s) or numpy array(s). |

<details>
<summary>Source code in <code>ultralytics/nn/backends/onnx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/onnx.py#L126-L155">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor] | np.ndarray:
    """Run ONNX inference using IO binding (CUDA) or standard session execution.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (torch.Tensor | list[torch.Tensor] | np.ndarray): Model predictions as tensor(s) or numpy array(s).
    """
    if self.format == "dnn":
        # OpenCV DNN
        self.net.setInput(im.cpu().numpy())
        return self.net.forward()

    # ONNX Runtime
    if self.use_io_binding:
        if self.device.type == "cpu":
            im = im.cpu()
        self.io.bind_input(
            name="images",
            device_type=im.device.type,
            device_id=im.device.index if im.device.type == "cuda" else 0,
            element_type=np.float16 if self.fp16 else np.float32,
            shape=tuple(im.shape),
            buffer_ptr=im.data_ptr(),
        )
        self.session.run_with_iobinding(self.io)
        return self.bindings
    else:
        return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})
```
</details>

<br>

### Method `ultralytics.nn.backends.onnx.ONNXBackend.load_model` {#ultralytics.nn.backends.onnx.ONNXBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load an ONNX model using ONNX Runtime or OpenCV DNN.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .onnx model file. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/onnx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/onnx.py#L48-L124">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load an ONNX model using ONNX Runtime or OpenCV DNN.

    Args:
        weight (str | Path): Path to the .onnx model file.
    """
    cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"

    if self.format == "dnn":
        # OpenCV DNN
        LOGGER.info(f"Loading {weight} for ONNX OpenCV DNN inference...")
        import cv2

        self.net = cv2.dnn.readNetFromONNX(weight)
    else:
        # ONNX Runtime
        LOGGER.info(f"Loading {weight} for ONNX Runtime inference...")
        check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
        import onnxruntime

        # Select execution provider
        available = onnxruntime.get_available_providers()
        if cuda and "CUDAExecutionProvider" in available:
            providers = [("CUDAExecutionProvider", {"device_id": self.device.index}), "CPUExecutionProvider"]
        elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            if cuda:
                LOGGER.warning("CUDA requested but CUDAExecutionProvider not available. Using CPU...")
                self.device = torch.device("cpu")
                cuda = False

        LOGGER.info(
            f"Using ONNX Runtime {onnxruntime.__version__} with "
            f"{providers[0] if isinstance(providers[0], str) else providers[0][0]}"
        )

        try:
            self.session = onnxruntime.InferenceSession(weight, providers=providers)
        except onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf as e:
            # ONNX Runtime reports an unparsable graph as a raw protobuf error naming neither the problem
            # nor a remedy. Only this one type is caught: other load failures are execution-provider or
            # model-support issues, where the runtime's own message is the useful one.
            raise TypeError(
                f"ERROR ❌️ {weight} is not a loadable ONNX model — the file is empty, truncated or corrupted "
                f"({type(e).__name__}: {e}).\nRecommend fixes are to re-export it with "
                f"'yolo export model=yolo26n.pt format=onnx', or to re-download the file."
            ) from e
        self.output_names = [x.name for x in self.session.get_outputs()]

        # Get metadata
        metadata_map = self.session.get_modelmeta().custom_metadata_map
        if metadata_map:
            self.apply_metadata(dict(metadata_map))

        # Check if dynamic shapes
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.session.get_inputs()[0].type

        # Setup IO binding for CUDA
        self.use_io_binding = not self.dynamic and cuda
        if self.use_io_binding:
            self.io = self.session.io_binding()
            self.bindings = []
            for output in self.session.get_outputs():
                torch_dtype, np_dtype = _ORT_DTYPES.get(output.type, (torch.float32, np.float32))
                y_tensor = torch.empty(output.shape, dtype=torch_dtype).to(self.device)
                self.io.bind_output(
                    name=output.name,
                    device_type=self.device.type,
                    device_id=self.device.index if cuda else 0,
                    element_type=np_dtype,
                    shape=tuple(y_tensor.shape),
                    buffer_ptr=y_tensor.data_ptr(),
                )
                self.bindings.append(y_tensor)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.backends.onnx.ONNXIMXBackend` {#ultralytics.nn.backends.onnx.ONNXIMXBackend}

```python
ONNXIMXBackend(weight: str | Path, device: torch.device, fp16: bool = False, format: str = "onnx")
```

**Bases:** `ONNXBackend`

ONNX IMX inference backend for NXP i.MX processors.

Extends `ONNXBackend` with support for quantized models targeting NXP i.MX edge devices. Uses MCT (Model Compression Toolkit) quantizers and custom NMS operations for optimized inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .onnx model file. | *required* |
| `device` | `torch.device` | Device to run inference on. | *required* |
| `fp16` | `bool` | Whether to use FP16 half-precision inference. | `False` |
| `format` | `str` | Inference engine, either "onnx" for ONNX Runtime or "dnn" for OpenCV DNN. | `"onnx"` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.onnx.ONNXIMXBackend.forward) | Run IMX inference with task-specific output concatenation for detect, pose, and segment tasks. |
| [`load_model`](#ultralytics.nn.backends.onnx.ONNXIMXBackend.load_model) | Load a quantized ONNX model from an IMX model directory. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/onnx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/onnx.py#L158-L214">View on GitHub</a>
```python
class ONNXIMXBackend(ONNXBackend):
    """ONNX IMX inference backend for NXP i.MX processors.

    Extends `ONNXBackend` with support for quantized models targeting NXP i.MX edge devices. Uses MCT (Model Compression
    Toolkit) quantizers and custom NMS operations for optimized inference.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.onnx.ONNXIMXBackend.forward` {#ultralytics.nn.backends.onnx.ONNXIMXBackend.forward}

```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]
```

Run IMX inference with task-specific output concatenation for detect, pose, and segment tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray \| list[np.ndarray] \| tuple[np.ndarray, ...]` | Task-formatted model predictions. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/onnx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/onnx.py#L192-L214">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]:
    """Run IMX inference with task-specific output concatenation for detect, pose, and segment tasks.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]): Task-formatted model predictions.
    """
    y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})

    if self.task == "detect":
        # boxes, conf, cls
        return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
    elif self.task == "pose":
        # boxes, conf, kpts
        return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype)
    elif self.task == "segment":
        return (
            np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype),
            y[4],
        )
    return y
```
</details>

<br>

### Method `ultralytics.nn.backends.onnx.ONNXIMXBackend.load_model` {#ultralytics.nn.backends.onnx.ONNXIMXBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load a quantized ONNX model from an IMX model directory.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the IMX model directory containing the .onnx file. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/onnx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/onnx.py#L165-L190">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load a quantized ONNX model from an IMX model directory.

    Args:
        weight (str | Path): Path to the IMX model directory containing the .onnx file.
    """
    check_requirements(("model-compression-toolkit>=2.4.1", "edge-mdt-cl<1.1.0", "onnxruntime-extensions"))
    check_requirements(("onnx", "onnxruntime"))
    import mct_quantizers as mctq
    import onnxruntime
    from edgemdt_cl.pytorch.nms import nms_ort  # noqa - register custom NMS ops

    w = Path(weight)
    onnx_file = next(w.glob("*.onnx"))
    LOGGER.info(f"Loading {onnx_file} for ONNX IMX inference...")

    session_options = mctq.get_ort_session_options()
    session_options.enable_mem_reuse = False

    self.session = onnxruntime.InferenceSession(onnx_file, session_options, providers=["CPUExecutionProvider"])
    self.output_names = [x.name for x in self.session.get_outputs()]
    self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
    self.fp16 = "float16" in self.session.get_inputs()[0].type
    metadata_map = self.session.get_modelmeta().custom_metadata_map
    if metadata_map:
        self.apply_metadata(dict(metadata_map))
```
</details>

<br><br>
