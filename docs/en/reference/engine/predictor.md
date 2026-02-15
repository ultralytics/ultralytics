---
description: Discover how to use the Base Predictor class in the Ultralytics YOLO engine for efficient image and video inference.
keywords: Ultralytics, YOLO, Base Predictor, image inference, video inference, machine learning, Python
---

# Reference for `ultralytics/engine/predictor.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`BasePredictor`](#ultralytics.engine.predictor.BasePredictor)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`BasePredictor.preprocess`](#ultralytics.engine.predictor.BasePredictor.preprocess)
        - [`BasePredictor.inference`](#ultralytics.engine.predictor.BasePredictor.inference)
        - [`BasePredictor.pre_transform`](#ultralytics.engine.predictor.BasePredictor.pre_transform)
        - [`BasePredictor.postprocess`](#ultralytics.engine.predictor.BasePredictor.postprocess)
        - [`BasePredictor.__call__`](#ultralytics.engine.predictor.BasePredictor.__call__)
        - [`BasePredictor.predict_cli`](#ultralytics.engine.predictor.BasePredictor.predict_cli)
        - [`BasePredictor.setup_source`](#ultralytics.engine.predictor.BasePredictor.setup_source)
        - [`BasePredictor.stream_inference`](#ultralytics.engine.predictor.BasePredictor.stream_inference)
        - [`BasePredictor.setup_model`](#ultralytics.engine.predictor.BasePredictor.setup_model)
        - [`BasePredictor.write_results`](#ultralytics.engine.predictor.BasePredictor.write_results)
        - [`BasePredictor.save_predicted_images`](#ultralytics.engine.predictor.BasePredictor.save_predicted_images)
        - [`BasePredictor.show`](#ultralytics.engine.predictor.BasePredictor.show)
        - [`BasePredictor.run_callbacks`](#ultralytics.engine.predictor.BasePredictor.run_callbacks)
        - [`BasePredictor.add_callback`](#ultralytics.engine.predictor.BasePredictor.add_callback)


## Class `ultralytics.engine.predictor.BasePredictor` {#ultralytics.engine.predictor.BasePredictor}

```python
def __init__(
    self,
    cfg=DEFAULT_CFG,
    overrides: dict[str, Any] | None = None,
    _callbacks: dict[str, list[callable]] | None = None,
)
```

A base class for creating predictors.

This class provides the foundation for prediction functionality, handling model setup, inference, and result processing across various input sources.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | Path | dict | SimpleNamespace` | Path to a configuration file or a configuration dictionary. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Configuration overrides. | `None` |
| `_callbacks` | `dict, optional` | Dictionary of callback functions. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `args` | `SimpleNamespace` | Configuration for the predictor. |
| `save_dir` | `Path` | Directory to save results. |
| `done_warmup` | `bool` | Whether the predictor has finished setup. |
| `model` | `torch.nn.Module` | Model used for prediction. |
| `data` | `str` | Data configuration. |
| `device` | `torch.device` | Device used for prediction. |
| `dataset` | `Dataset` | Dataset used for prediction. |
| `vid_writer` | `dict[Path, cv2.VideoWriter]` | Dictionary of {save_path: video_writer} for saving video output. |
| `plotted_img` | `np.ndarray` | Last plotted image. |
| `source_type` | `SimpleNamespace` | Type of input source. |
| `seen` | `int` | Number of images processed. |
| `windows` | `list[str]` | List of window names for visualization. |
| `batch` | `tuple` | Current batch data. |
| `results` | `list[Any]` | Current batch results. |
| `transforms` | `callable` | Image transforms for classification. |
| `callbacks` | `dict[str, list[callable]]` | Callback functions for different events. |
| `txt_path` | `Path` | Path to save text results. |
| `_lock` | `threading.Lock` | Lock for thread-safe inference. |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.engine.predictor.BasePredictor.__call__) | Perform inference on an image or stream. |
| [`add_callback`](#ultralytics.engine.predictor.BasePredictor.add_callback) | Add a callback function for a specific event. |
| [`inference`](#ultralytics.engine.predictor.BasePredictor.inference) | Run inference on a given image using the specified model and arguments. |
| [`postprocess`](#ultralytics.engine.predictor.BasePredictor.postprocess) | Post-process predictions for an image and return them. |
| [`pre_transform`](#ultralytics.engine.predictor.BasePredictor.pre_transform) | Pre-transform input image before inference. |
| [`predict_cli`](#ultralytics.engine.predictor.BasePredictor.predict_cli) | Method used for Command Line Interface (CLI) prediction. |
| [`preprocess`](#ultralytics.engine.predictor.BasePredictor.preprocess) | Prepare input image before inference. |
| [`run_callbacks`](#ultralytics.engine.predictor.BasePredictor.run_callbacks) | Run all registered callbacks for a specific event. |
| [`save_predicted_images`](#ultralytics.engine.predictor.BasePredictor.save_predicted_images) | Save video predictions as mp4/avi or images as jpg at specified path. |
| [`setup_model`](#ultralytics.engine.predictor.BasePredictor.setup_model) | Initialize YOLO model with given parameters and set it to evaluation mode. |
| [`setup_source`](#ultralytics.engine.predictor.BasePredictor.setup_source) | Set up source and inference mode. |
| [`show`](#ultralytics.engine.predictor.BasePredictor.show) | Display an image in a window. |
| [`stream_inference`](#ultralytics.engine.predictor.BasePredictor.stream_inference) | Stream inference on input source and save results to file. |
| [`write_results`](#ultralytics.engine.predictor.BasePredictor.write_results) | Write inference results to a file or directory. |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L70-L513"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BasePredictor:
    """A base class for creating predictors.

    This class provides the foundation for prediction functionality, handling model setup, inference, and result
    processing across various input sources.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (torch.nn.Module): Model used for prediction.
        data (str): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict[Path, cv2.VideoWriter]): Dictionary of {save_path: video_writer} for saving video output.
        plotted_img (np.ndarray): Last plotted image.
        source_type (SimpleNamespace): Type of input source.
        seen (int): Number of images processed.
        windows (list[str]): List of window names for visualization.
        batch (tuple): Current batch data.
        results (list[Any]): Current batch results.
        transforms (callable): Image transforms for classification.
        callbacks (dict[str, list[callable]]): Callback functions for different events.
        txt_path (Path): Path to save text results.
        _lock (threading.Lock): Lock for thread-safe inference.

    Methods:
        preprocess: Prepare input image before inference.
        inference: Run inference on a given image.
        postprocess: Process raw predictions into structured results.
        predict_cli: Run prediction for command line interface.
        setup_source: Set up input source and inference mode.
        stream_inference: Stream inference on input source.
        setup_model: Initialize and configure the model.
        write_results: Write inference results to files.
        save_predicted_images: Save prediction visualizations.
        show: Display results in a window.
        run_callbacks: Execute registered callbacks for an event.
        add_callback: Register a new callback function.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict[str, list[callable]] | None = None,
    ):
        """Initialize the BasePredictor class.

        Args:
            cfg (str | Path | dict | SimpleNamespace): Path to a configuration file or a configuration dictionary.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.__call__` {#ultralytics.engine.predictor.BasePredictor.\_\_call\_\_}

```python
def __call__(self, source = None, model = None, stream: bool = False, *args, **kwargs)
```

Perform inference on an image or stream.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional` | Source for inference. | `None` |
| `model` | `str | Path | torch.nn.Module, optional` | Model for inference. | `None` |
| `stream` | `bool` | Whether to stream the inference results. If True, returns a generator. | `False` |
| `*args` | `Any` | Additional arguments for the inference method. | *required* |
| `**kwargs` | `Any` | Additional keyword arguments for the inference method. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[ultralytics.engine.results.Results] | generator` | Results objects or generator of Results objects. |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L207-L225"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
    """Perform inference on an image or stream.

    Args:
        source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
            Source for inference.
        model (str | Path | torch.nn.Module, optional): Model for inference.
        stream (bool): Whether to stream the inference results. If True, returns a generator.
        *args (Any): Additional arguments for the inference method.
        **kwargs (Any): Additional keyword arguments for the inference method.

    Returns:
        (list[ultralytics.engine.results.Results] | generator): Results objects or generator of Results objects.
    """
    self.stream = stream
    if stream:
        return self.stream_inference(source, model, *args, **kwargs)
    else:
        return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Results into one
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.add_callback` {#ultralytics.engine.predictor.BasePredictor.add\_callback}

```python
def add_callback(self, event: str, func: callable)
```

Add a callback function for a specific event.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `event` | `str` |  | *required* |
| `func` | `callable` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L511-L513"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def add_callback(self, event: str, func: callable):
    """Add a callback function for a specific event."""
    self.callbacks[event].append(func)
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.inference` {#ultralytics.engine.predictor.BasePredictor.inference}

```python
def inference(self, im: torch.Tensor, *args, **kwargs)
```

Run inference on a given image using the specified model and arguments.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` |  | *required* |
| `*args` |  |  | *required* |
| `**kwargs` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L175-L182"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def inference(self, im: torch.Tensor, *args, **kwargs):
    """Run inference on a given image using the specified model and arguments."""
    visualize = (
        increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
        if self.args.visualize and (not self.source_type.tensor)
        else False
    )
    return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.postprocess` {#ultralytics.engine.predictor.BasePredictor.postprocess}

```python
def postprocess(self, preds, img, orig_imgs)
```

Post-process predictions for an image and return them.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` |  |  | *required* |
| `img` |  |  | *required* |
| `orig_imgs` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L203-L205"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def postprocess(self, preds, img, orig_imgs):
    """Post-process predictions for an image and return them."""
    return preds
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.pre_transform` {#ultralytics.engine.predictor.BasePredictor.pre\_transform}

```python
def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]
```

Pre-transform input image before inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `list[np.ndarray]` | List of images with shape [(H, W, 3) x N]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[np.ndarray]` | List of transformed images. |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L184-L201"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
    """Pre-transform input image before inference.

    Args:
        im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].

    Returns:
        (list[np.ndarray]): List of transformed images.
    """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(
        self.imgsz,
        auto=same_shapes
        and self.args.rect
        and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
        stride=self.model.stride,
    )
    return [letterbox(image=x) for x in im]
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.predict_cli` {#ultralytics.engine.predictor.BasePredictor.predict\_cli}

```python
def predict_cli(self, source = None, model = None)
```

Method used for Command Line Interface (CLI) prediction.

This function is designed to run predictions using the CLI. It sets up the source and model, then processes the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the generator without storing results.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional` | Source for inference. | `None` |
| `model` | `str | Path | torch.nn.Module, optional` | Model for inference. | `None` |

!!! note "Notes"

    Do not modify this function or remove the generator. The generator ensures that no outputs are
    accumulated in memory, which is critical for preventing memory issues during long-running predictions.

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L227-L245"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict_cli(self, source=None, model=None):
    """Method used for Command Line Interface (CLI) prediction.

    This function is designed to run predictions using the CLI. It sets up the source and model, then processes the
    inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
    generator without storing results.

    Args:
        source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
            Source for inference.
        model (str | Path | torch.nn.Module, optional): Model for inference.

    Notes:
        Do not modify this function or remove the generator. The generator ensures that no outputs are
        accumulated in memory, which is critical for preventing memory issues during long-running predictions.
    """
    gen = self.stream_inference(source, model)
    for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
        pass
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.preprocess` {#ultralytics.engine.predictor.BasePredictor.preprocess}

```python
def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor
```

Prepare input image before inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor | list[np.ndarray]` | Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Preprocessed image tensor of shape (N, 3, H, W). |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L151-L173"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
    """Prepare input image before inference.

    Args:
        im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

    Returns:
        (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(self.pre_transform(im))
        if im.shape[-1] == 3:
            im = im[..., ::-1]  # BGR to RGB
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(self.device)
    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.run_callbacks` {#ultralytics.engine.predictor.BasePredictor.run\_callbacks}

```python
def run_callbacks(self, event: str)
```

Run all registered callbacks for a specific event.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `event` | `str` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L506-L509"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def run_callbacks(self, event: str):
    """Run all registered callbacks for a specific event."""
    for callback in self.callbacks.get(event, []):
        callback(self)
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.save_predicted_images` {#ultralytics.engine.predictor.BasePredictor.save\_predicted\_images}

```python
def save_predicted_images(self, save_path: Path, frame: int = 0)
```

Save video predictions as mp4/avi or images as jpg at specified path.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `save_path` | `Path` | Path to save the results. | *required* |
| `frame` | `int` | Frame number for video mode. | `0` |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L462-L493"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def save_predicted_images(self, save_path: Path, frame: int = 0):
    """Save video predictions as mp4/avi or images as jpg at specified path.

    Args:
        save_path (Path): Path to save the results.
        frame (int): Frame number for video mode.
    """
    im = self.plotted_img

    # Save videos and streams
    if self.dataset.mode in {"stream", "video"}:
        fps = self.dataset.fps if self.dataset.mode == "video" else 30
        frames_path = self.save_dir / f"{save_path.stem}_frames"  # save frames to a separate directory
        if save_path not in self.vid_writer:  # new video
            if self.args.save_frames:
                Path(frames_path).mkdir(parents=True, exist_ok=True)
            suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
            self.vid_writer[save_path] = cv2.VideoWriter(
                filename=str(Path(save_path).with_suffix(suffix)),
                fourcc=cv2.VideoWriter_fourcc(*fourcc),
                fps=fps,  # integer required, floats produce error in MP4 codec
                frameSize=(im.shape[1], im.shape[0]),  # (width, height)
            )

        # Save video
        self.vid_writer[save_path].write(im)
        if self.args.save_frames:
            cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

    # Save images
    else:
        cv2.imwrite(str(save_path.with_suffix(".jpg")), im)  # save to JPG for best support
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.setup_model` {#ultralytics.engine.predictor.BasePredictor.setup\_model}

```python
def setup_model(self, model, verbose: bool = True)
```

Initialize YOLO model with given parameters and set it to evaluation mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str | Path | torch.nn.Module` | Model to load or use. | *required* |
| `verbose` | `bool` | Whether to print verbose output. | `True` |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L383-L410"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def setup_model(self, model, verbose: bool = True):
    """Initialize YOLO model with given parameters and set it to evaluation mode.

    Args:
        model (str | Path | torch.nn.Module): Model to load or use.
        verbose (bool): Whether to print verbose output.
    """
    if hasattr(model, "end2end"):
        if self.args.end2end is not None:
            model.end2end = self.args.end2end
        if model.end2end:
            model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
    self.model = AutoBackend(
        model=model or self.args.model,
        device=select_device(self.args.device, verbose=verbose),
        dnn=self.args.dnn,
        data=self.args.data,
        fp16=self.args.half,
        fuse=True,
        verbose=verbose,
    )

    self.device = self.model.device  # update device
    self.args.half = self.model.fp16  # update half
    if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
        self.args.imgsz = self.model.imgsz  # reuse imgsz from export metadata
    self.model.eval()
    self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.setup_source` {#ultralytics.engine.predictor.BasePredictor.setup\_source}

```python
def setup_source(self, source, stride: int | None = None)
```

Set up source and inference mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor` | Source for<br>    inference. | *required* |
| `stride` | `int, optional` | Model stride for image size checking. | `None` |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L247-L274"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def setup_source(self, source, stride: int | None = None):
    """Set up source and inference mode.

    Args:
        source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor): Source for
            inference.
        stride (int, optional): Model stride for image size checking.
    """
    self.imgsz = check_imgsz(self.args.imgsz, stride=stride or self.model.stride, min_dim=2)  # check image size
    self.dataset = load_inference_source(
        source=source,
        batch=self.args.batch,
        vid_stride=self.args.vid_stride,
        buffer=self.args.stream_buffer,
        channels=getattr(self.model, "ch", 3),
    )
    self.source_type = self.dataset.source_type
    if (
        self.source_type.stream
        or self.source_type.screenshot
        or len(self.dataset) > 1000  # many images
        or any(getattr(self.dataset, "video_flag", [False]))
    ):  # long sequence
        import torchvision  # noqa (import here triggers torchvision NMS use in nms.py)

        if not getattr(self, "stream", True):  # videos
            LOGGER.warning(STREAM_WARNING)
    self.vid_writer = {}
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.show` {#ultralytics.engine.predictor.BasePredictor.show}

```python
def show(self, p: str = "")
```

Display an image in a window.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `p` | `str` |  | `""` |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L495-L504"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def show(self, p: str = ""):
    """Display an image in a window."""
    im = self.plotted_img
    if platform.system() == "Linux" and p not in self.windows:
        self.windows.append(p)
        cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (width, height)
    cv2.imshow(p, im)
    if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 0xFF == ord("q"):  # 300ms if image; else 1ms
        raise StopIteration
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.stream_inference` {#ultralytics.engine.predictor.BasePredictor.stream\_inference}

```python
def stream_inference(self, source = None, model = None, *args, **kwargs)
```

Stream inference on input source and save results to file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional` | Source for inference. | `None` |
| `model` | `str | Path | torch.nn.Module, optional` | Model for inference. | `None` |
| `*args` | `Any` | Additional arguments for the inference method. | *required* |
| `**kwargs` | `Any` | Additional keyword arguments for the inference method. | *required* |

**Yields**

| Type | Description |
| --- | --- |
| `ultralytics.engine.results.Results` | Results objects. |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L277-L381"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def stream_inference(self, source=None, model=None, *args, **kwargs):
    """Stream inference on input source and save results to file.

    Args:
        source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
            Source for inference.
        model (str | Path | torch.nn.Module, optional): Model for inference.
        *args (Any): Additional arguments for the inference method.
        **kwargs (Any): Additional keyword arguments for the inference method.

    Yields:
        (ultralytics.engine.results.Results): Results objects.
    """
    if self.args.verbose:
        LOGGER.info("")

    # Setup model
    if not self.model:
        self.setup_model(model)

    with self._lock:  # for thread-safe inference
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(
                imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, self.model.ch, *self.imgsz)
            )
            self.done_warmup = True

        self.seen, self.windows, self.batch = 0, [], None
        profilers = (
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
        )
        self.run_callbacks("on_predict_start")
        for batch in self.dataset:
            self.batch = batch
            self.run_callbacks("on_predict_batch_start")
            paths, im0s, s = self.batch

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.inference(im, *args, **kwargs)
                if self.args.embed:
                    yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                    continue

            # Postprocess
            with profilers[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks("on_predict_postprocess_end")

            # Visualize, save, write results
            n = len(im0s)
            try:
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)
            except StopIteration:
                break

            # Print batch results
            if self.args.verbose:
                LOGGER.info("\n".join(s))

            self.run_callbacks("on_predict_batch_end")
            yield from self.results

    # Release assets
    for v in self.vid_writer.values():
        if isinstance(v, cv2.VideoWriter):
            v.release()

    if self.args.show:
        cv2.destroyAllWindows()  # close any open windows

    # Print final results
    if self.args.verbose and self.seen:
        t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
        LOGGER.info(
            f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
            f"{(min(self.args.batch, self.seen), getattr(self.model, 'ch', 3), *im.shape[2:])}" % t
        )
    if self.args.save or self.args.save_txt or self.args.save_crop:
        nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
    self.run_callbacks("on_predict_end")
```
</details>

<br>

### Method `ultralytics.engine.predictor.BasePredictor.write_results` {#ultralytics.engine.predictor.BasePredictor.write\_results}

```python
def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str
```

Write inference results to a file or directory.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `i` | `int` | Index of the current image in the batch. | *required* |
| `p` | `Path` | Path to the current image. | *required* |
| `im` | `torch.Tensor` | Preprocessed image tensor. | *required* |
| `s` | `list[str]` | List of result strings. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | String with result information. |

<details>
<summary>Source code in <code>ultralytics/engine/predictor.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L412-L460"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
    """Write inference results to a file or directory.

    Args:
        i (int): Index of the current image in the batch.
        p (Path): Path to the current image.
        im (torch.Tensor): Preprocessed image tensor.
        s (list[str]): List of result strings.

    Returns:
        (str): String with result information.
    """
    string = ""  # print string
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
        string += f"{i}: "
        frame = self.dataset.count
    else:
        match = re.search(r"frame (\d+)/", s[i])
        frame = int(match[1]) if match else None  # 0 if frame undetermined

    self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
    string += "{:g}x{:g} ".format(*im.shape[2:])
    result = self.results[i]
    result.save_dir = self.save_dir.__str__()  # used in other locations
    string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

    # Add predictions to image
    if self.args.save or self.args.show:
        self.plotted_img = result.plot(
            line_width=self.args.line_width,
            boxes=self.args.show_boxes,
            conf=self.args.show_conf,
            labels=self.args.show_labels,
            im_gpu=None if self.args.retina_masks else im[i],
        )

    # Save results
    if self.args.save_txt:
        result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
    if self.args.save_crop:
        result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
    if self.args.show:
        self.show(str(p))
    if self.args.save:
        self.save_predicted_images(self.save_dir / p.name, frame)

    return string
```
</details>

<br><br>
