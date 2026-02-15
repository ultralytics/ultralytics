---
description: Explore the ultralytics.models.yolo.model module for YOLO object detection. Learn initialization, model mapping, and more.
keywords: YOLO, object detection, Ultralytics, YOLO model, machine learning, Python, model initialization
---

# Reference for `ultralytics/models/yolo/model.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`YOLO`](#ultralytics.models.yolo.model.YOLO)
        - [`YOLOWorld`](#ultralytics.models.yolo.model.YOLOWorld)
        - [`YOLOE`](#ultralytics.models.yolo.model.YOLOE)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`YOLO.task_map`](#ultralytics.models.yolo.model.YOLO.task_map)
        - [`YOLOWorld.task_map`](#ultralytics.models.yolo.model.YOLOWorld.task_map)
        - [`YOLOE.task_map`](#ultralytics.models.yolo.model.YOLOE.task_map)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`YOLOWorld.set_classes`](#ultralytics.models.yolo.model.YOLOWorld.set_classes)
        - [`YOLOE.get_text_pe`](#ultralytics.models.yolo.model.YOLOE.get_text_pe)
        - [`YOLOE.get_visual_pe`](#ultralytics.models.yolo.model.YOLOE.get_visual_pe)
        - [`YOLOE.set_vocab`](#ultralytics.models.yolo.model.YOLOE.set_vocab)
        - [`YOLOE.get_vocab`](#ultralytics.models.yolo.model.YOLOE.get_vocab)
        - [`YOLOE.set_classes`](#ultralytics.models.yolo.model.YOLOE.set_classes)
        - [`YOLOE.val`](#ultralytics.models.yolo.model.YOLOE.val)
        - [`YOLOE.predict`](#ultralytics.models.yolo.model.YOLOE.predict)


## Class `ultralytics.models.yolo.model.YOLO` {#ultralytics.models.yolo.model.YOLO}

```python
YOLO(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False)
```

**Bases:** `Model`

YOLO (You Only Look Once) object detection model.

This class provides a unified interface for YOLO models, automatically switching to specialized model types (YOLOWorld or YOLOE) based on the model filename. It supports various computer vision tasks including object detection, segmentation, classification, pose estimation, and oriented bounding box detection.

This constructor initializes a YOLO model, automatically switching to specialized model types (YOLOWorld or YOLOE) based on the model filename.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str | Path` | Model name or path to model file, i.e. 'yolo26n.pt', 'yolo26n.yaml'. | `"yolo26n.pt"` |
| `task` | `str, optional` | YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'. Defaults<br>    to auto-detection based on model. | `None` |
| `verbose` | `bool` | Display model info on load. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` |  | The loaded YOLO model instance. |
| `task` |  | The task type (detect, segment, classify, pose, obb). |
| `overrides` |  | Configuration overrides for the model. |

**Methods**

| Name | Description |
| --- | --- |
| [`task_map`](#ultralytics.models.yolo.model.YOLO.task_map) | Map head to model, trainer, validator, and predictor classes. |

**Examples**

```python
Load a pretrained YOLO26n detection model
>>> model = YOLO("yolo26n.pt")

Load a pretrained YOLO26n segmentation model
>>> model = YOLO("yolo26n-seg.pt")

Initialize from a YAML configuration
>>> model = YOLO("yolo26n.yaml")
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L26-L118"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLO(Model):
    """YOLO (You Only Look Once) object detection model.

    This class provides a unified interface for YOLO models, automatically switching to specialized model types
    (YOLOWorld or YOLOE) based on the model filename. It supports various computer vision tasks including object
    detection, segmentation, classification, pose estimation, and oriented bounding box detection.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect, segment, classify, pose, obb).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a YOLO model with automatic type detection.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained YOLO26n detection model
        >>> model = YOLO("yolo26n.pt")

        Load a pretrained YOLO26n segmentation model
        >>> model = YOLO("yolo26n-seg.pt")

        Initialize from a YAML configuration
        >>> model = YOLO("yolo26n.yaml")
    """

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO model.

        This constructor initializes a YOLO model, automatically switching to specialized model types (YOLOWorld or
        YOLOE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo26n.pt', 'yolo26n.yaml'.
            task (str, optional): YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'. Defaults
                to auto-detection based on model.
            verbose (bool): Display model info on load.
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # if RTDETR head
                from ultralytics import RTDETR

                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__
```
</details>

<br>

### Property `ultralytics.models.yolo.model.YOLO.task_map` {#ultralytics.models.yolo.model.YOLO.task\_map}

```python
def task_map(self) -> dict[str, dict[str, Any]]
```

Map head to model, trainer, validator, and predictor classes.

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L85-L118"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def task_map(self) -> dict[str, dict[str, Any]]:
    """Map head to model, trainer, validator, and predictor classes."""
    return {
        "classify": {
            "model": ClassificationModel,
            "trainer": yolo.classify.ClassificationTrainer,
            "validator": yolo.classify.ClassificationValidator,
            "predictor": yolo.classify.ClassificationPredictor,
        },
        "detect": {
            "model": DetectionModel,
            "trainer": yolo.detect.DetectionTrainer,
            "validator": yolo.detect.DetectionValidator,
            "predictor": yolo.detect.DetectionPredictor,
        },
        "segment": {
            "model": SegmentationModel,
            "trainer": yolo.segment.SegmentationTrainer,
            "validator": yolo.segment.SegmentationValidator,
            "predictor": yolo.segment.SegmentationPredictor,
        },
        "pose": {
            "model": PoseModel,
            "trainer": yolo.pose.PoseTrainer,
            "validator": yolo.pose.PoseValidator,
            "predictor": yolo.pose.PosePredictor,
        },
        "obb": {
            "model": OBBModel,
            "trainer": yolo.obb.OBBTrainer,
            "validator": yolo.obb.OBBValidator,
            "predictor": yolo.obb.OBBPredictor,
        },
    }
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.model.YOLOWorld` {#ultralytics.models.yolo.model.YOLOWorld}

```python
YOLOWorld(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None
```

**Bases:** `Model`

YOLO-World object detection model.

YOLO-World is an open-vocabulary object detection model that can detect objects based on text descriptions without requiring training on specific classes. It extends the YOLO architecture to support real-time open-vocabulary detection.

Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default COCO class names.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str | Path` | Path to the pre-trained model file. Supports *.pt and *.yaml formats. | `"yolov8s-world.pt"` |
| `verbose` | `bool` | If True, prints additional information during initialization. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` |  | The loaded YOLO-World model instance. |
| `task` |  | Always set to 'detect' for object detection. |
| `overrides` |  | Configuration overrides for the model. |

**Methods**

| Name | Description |
| --- | --- |
| [`task_map`](#ultralytics.models.yolo.model.YOLOWorld.task_map) | Map head to model, trainer, validator, and predictor classes. |
| [`set_classes`](#ultralytics.models.yolo.model.YOLOWorld.set_classes) | Set the model's class names for detection. |

**Examples**

```python
Load a YOLOv8-World model
>>> model = YOLOWorld("yolov8s-world.pt")

Set custom classes for detection
>>> model.set_classes(["person", "car", "bicycle"])
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L121-L189"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOWorld(Model):
    """YOLO-World object detection model.

    YOLO-World is an open-vocabulary object detection model that can detect objects based on text descriptions without
    requiring training on specific classes. It extends the YOLO architecture to support real-time open-vocabulary
    detection.

    Attributes:
        model: The loaded YOLO-World model instance.
        task: Always set to 'detect' for object detection.
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOv8-World model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        set_classes: Set the model's class names for detection.

    Examples:
        Load a YOLOv8-World model
        >>> model = YOLOWorld("yolov8s-world.pt")

        Set custom classes for detection
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None:
        """Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default COCO
        class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")
```
</details>

<br>

### Property `ultralytics.models.yolo.model.YOLOWorld.task_map` {#ultralytics.models.yolo.model.YOLOWorld.task\_map}

```python
def task_map(self) -> dict[str, dict[str, Any]]
```

Map head to model, trainer, validator, and predictor classes.

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L163-L172"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def task_map(self) -> dict[str, dict[str, Any]]:
    """Map head to model, trainer, validator, and predictor classes."""
    return {
        "detect": {
            "model": WorldModel,
            "validator": yolo.detect.DetectionValidator,
            "predictor": yolo.detect.DetectionPredictor,
            "trainer": yolo.world.WorldTrainer,
        }
    }
```
</details>

<br>

### Method `ultralytics.models.yolo.model.YOLOWorld.set_classes` {#ultralytics.models.yolo.model.YOLOWorld.set\_classes}

```python
def set_classes(self, classes: list[str]) -> None
```

Set the model's class names for detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `classes` | `list[str]` | A list of categories i.e. ["person"]. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L174-L189"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_classes(self, classes: list[str]) -> None:
    """Set the model's class names for detection.

    Args:
        classes (list[str]): A list of categories i.e. ["person"].
    """
    self.model.set_classes(classes)
    # Remove background if it's given
    background = " "
    if background in classes:
        classes.remove(background)
    self.model.names = classes

    # Reset method class names
    if self.predictor:
        self.predictor.model.names = classes
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.model.YOLOE` {#ultralytics.models.yolo.model.YOLOE}

```python
YOLOE(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None
```

**Bases:** `Model`

YOLOE object detection and segmentation model.

YOLOE is an enhanced YOLO model that supports both object detection and instance segmentation tasks with improved performance and additional features like visual and text positional embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str | Path` | Path to the pre-trained model file. Supports *.pt and *.yaml formats. | `"yoloe-11s-seg.pt"` |
| `task` | `str, optional` | Task type for the model. Auto-detected if None. | `None` |
| `verbose` | `bool` | If True, prints additional information during initialization. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` |  | The loaded YOLOE model instance. |
| `task` |  | The task type (detect or segment). |
| `overrides` |  | Configuration overrides for the model. |

**Methods**

| Name | Description |
| --- | --- |
| [`task_map`](#ultralytics.models.yolo.model.YOLOE.task_map) | Map head to model, trainer, validator, and predictor classes. |
| [`get_text_pe`](#ultralytics.models.yolo.model.YOLOE.get_text_pe) | Get text positional embeddings for the given texts. |
| [`get_visual_pe`](#ultralytics.models.yolo.model.YOLOE.get_visual_pe) | Get visual positional embeddings for the given image and visual features. |
| [`get_vocab`](#ultralytics.models.yolo.model.YOLOE.get_vocab) | Get vocabulary for the given class names. |
| [`predict`](#ultralytics.models.yolo.model.YOLOE.predict) | Run prediction on images, videos, directories, streams, etc. |
| [`set_classes`](#ultralytics.models.yolo.model.YOLOE.set_classes) | Set the model's class names and embeddings for detection. |
| [`set_vocab`](#ultralytics.models.yolo.model.YOLOE.set_vocab) | Set vocabulary and class names for the YOLOE model. |
| [`val`](#ultralytics.models.yolo.model.YOLOE.val) | Validate the model using text or visual prompts. |

**Examples**

```python
Load a YOLOE detection model
>>> model = YOLOE("yoloe-11s-seg.pt")

Set vocabulary and class names
>>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

Predict with visual prompts
>>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
>>> results = model.predict("image.jpg", visual_prompts=prompts)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L192-L431"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOE(Model):
    """YOLOE object detection and segmentation model.

    YOLOE is an enhanced YOLO model that supports both object detection and instance segmentation tasks with improved
    performance and additional features like visual and text positional embeddings.

    Attributes:
        model: The loaded YOLOE model instance.
        task: The task type (detect or segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOE model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        get_text_pe: Get text positional embeddings for the given texts.
        get_visual_pe: Get visual positional embeddings for the given image and visual features.
        set_vocab: Set vocabulary and class names for the YOLOE model.
        get_vocab: Get vocabulary for the given class names.
        set_classes: Set the model's class names and embeddings for detection.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a YOLOE detection model
        >>> model = YOLOE("yoloe-11s-seg.pt")

        Set vocabulary and class names
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

        Predict with visual prompts
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("image.jpg", visual_prompts=prompts)
    """

    def __init__(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        """Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)
```
</details>

<br>

### Property `ultralytics.models.yolo.model.YOLOE.task_map` {#ultralytics.models.yolo.model.YOLOE.task\_map}

```python
def task_map(self) -> dict[str, dict[str, Any]]
```

Map head to model, trainer, validator, and predictor classes.

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L237-L252"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def task_map(self) -> dict[str, dict[str, Any]]:
    """Map head to model, trainer, validator, and predictor classes."""
    return {
        "detect": {
            "model": YOLOEModel,
            "validator": yolo.yoloe.YOLOEDetectValidator,
            "predictor": yolo.detect.DetectionPredictor,
            "trainer": yolo.yoloe.YOLOETrainer,
        },
        "segment": {
            "model": YOLOESegModel,
            "validator": yolo.yoloe.YOLOESegValidator,
            "predictor": yolo.segment.SegmentationPredictor,
            "trainer": yolo.yoloe.YOLOESegTrainer,
        },
    }
```
</details>

<br>

### Method `ultralytics.models.yolo.model.YOLOE.get_text_pe` {#ultralytics.models.yolo.model.YOLOE.get\_text\_pe}

```python
def get_text_pe(self, texts)
```

Get text positional embeddings for the given texts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L254-L257"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_text_pe(self, texts):
    """Get text positional embeddings for the given texts."""
    assert isinstance(self.model, YOLOEModel)
    return self.model.get_text_pe(texts)
```
</details>

<br>

### Method `ultralytics.models.yolo.model.YOLOE.get_visual_pe` {#ultralytics.models.yolo.model.YOLOE.get\_visual\_pe}

```python
def get_visual_pe(self, img, visual)
```

Get visual positional embeddings for the given image and visual features.

This method extracts positional embeddings from visual features based on the input image. It requires that the model is an instance of YOLOEModel.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img` | `torch.Tensor` | Input image tensor. | *required* |
| `visual` | `torch.Tensor` | Visual features extracted from the image. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Visual positional embeddings. |

**Examples**

```python
>>> model = YOLOE("yoloe-11s-seg.pt")
>>> img = torch.rand(1, 3, 640, 640)
>>> visual_features = torch.rand(1, 1, 80, 80)
>>> pe = model.get_visual_pe(img, visual_features)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L259-L279"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_visual_pe(self, img, visual):
    """Get visual positional embeddings for the given image and visual features.

    This method extracts positional embeddings from visual features based on the input image. It requires that the
    model is an instance of YOLOEModel.

    Args:
        img (torch.Tensor): Input image tensor.
        visual (torch.Tensor): Visual features extracted from the image.

    Returns:
        (torch.Tensor): Visual positional embeddings.

    Examples:
        >>> model = YOLOE("yoloe-11s-seg.pt")
        >>> img = torch.rand(1, 3, 640, 640)
        >>> visual_features = torch.rand(1, 1, 80, 80)
        >>> pe = model.get_visual_pe(img, visual_features)
    """
    assert isinstance(self.model, YOLOEModel)
    return self.model.get_visual_pe(img, visual)
```
</details>

<br>

### Method `ultralytics.models.yolo.model.YOLOE.get_vocab` {#ultralytics.models.yolo.model.YOLOE.get\_vocab}

```python
def get_vocab(self, names)
```

Get vocabulary for the given class names.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L301-L304"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_vocab(self, names):
    """Get vocabulary for the given class names."""
    assert isinstance(self.model, YOLOEModel)
    return self.model.get_vocab(names)
```
</details>

<br>

### Method `ultralytics.models.yolo.model.YOLOE.predict` {#ultralytics.models.yolo.model.YOLOE.predict}

```python
def predict(
    self,
    source=None,
    stream: bool = False,
    visual_prompts: dict[str, list] = {},
    refer_image=None,
    predictor=yolo.yoloe.YOLOEVPDetectPredictor,
    **kwargs,
)
```

Run prediction on images, videos, directories, streams, etc.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `str | int | PIL.Image | np.ndarray, optional` | Source for prediction. Accepts image paths, directory<br>    paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices. | `None` |
| `stream` | `bool` | Whether to stream the prediction results. If True, results are yielded as a generator as they<br>    are computed. | `False` |
| `visual_prompts` | `dict[str, list]` | Dictionary containing visual prompts for the model. Must include 'bboxes'<br>    and 'cls' keys when non-empty. | `{}` |
| `refer_image` | `str | PIL.Image | np.ndarray, optional` | Reference image for visual prompts. | `None` |
| `predictor` | `callable` | Custom predictor class for visual prompt predictions. Defaults to<br>    YOLOEVPDetectPredictor. | `yolo.yoloe.YOLOEVPDetectPredictor` |
| `**kwargs` | `Any` | Additional keyword arguments passed to the predictor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list | generator` | List of Results objects or generator of Results objects if stream=True. |

**Examples**

```python
>>> model = YOLOE("yoloe-11s-seg.pt")
>>> results = model.predict("path/to/image.jpg")
>>> # With visual prompts
>>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
>>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L351-L431"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(
    self,
    source=None,
    stream: bool = False,
    visual_prompts: dict[str, list] = {},
    refer_image=None,
    predictor=yolo.yoloe.YOLOEVPDetectPredictor,
    **kwargs,
):
    """Run prediction on images, videos, directories, streams, etc.

    Args:
        source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths, directory
            paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
        stream (bool): Whether to stream the prediction results. If True, results are yielded as a generator as they
            are computed.
        visual_prompts (dict[str, list]): Dictionary containing visual prompts for the model. Must include 'bboxes'
            and 'cls' keys when non-empty.
        refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
        predictor (callable): Custom predictor class for visual prompt predictions. Defaults to
            YOLOEVPDetectPredictor.
        **kwargs (Any): Additional keyword arguments passed to the predictor.

    Returns:
        (list | generator): List of Results objects or generator of Results objects if stream=True.

    Examples:
        >>> model = YOLOE("yoloe-11s-seg.pt")
        >>> results = model.predict("path/to/image.jpg")
        >>> # With visual prompts
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
    """
    if len(visual_prompts):
        assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
            f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
        )
        assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
            f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
            f"{len(visual_prompts['cls'])} respectively"
        )
        if type(self.predictor) is not predictor:
            self.predictor = predictor(
                overrides={
                    "task": self.model.task,
                    "mode": "predict",
                    "save": False,
                    "verbose": refer_image is None,
                    "batch": 1,
                    "device": kwargs.get("device", None),
                    "half": kwargs.get("half", False),
                    "imgsz": kwargs.get("imgsz", self.overrides.get("imgsz", 640)),
                },
                _callbacks=self.callbacks,
            )

        num_cls = (
            max(len(set(c)) for c in visual_prompts["cls"])
            if isinstance(source, list) and refer_image is None  # means multiple images
            else len(set(visual_prompts["cls"]))
        )
        self.model.model[-1].nc = num_cls
        self.model.names = [f"object{i}" for i in range(num_cls)]
        self.predictor.set_prompts(visual_prompts.copy())
        self.predictor.setup_model(model=self.model)

        if refer_image is None and source is not None:
            dataset = load_inference_source(source)
            if dataset.mode in {"video", "stream"}:
                # NOTE: set the first frame as refer image for videos/streams inference
                refer_image = next(iter(dataset))[1][0]
        if refer_image is not None:
            vpe = self.predictor.get_vpe(refer_image)
            self.model.set_classes(self.model.names, vpe)
            self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
            self.predictor = None  # reset predictor
    elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
        self.predictor = None  # reset predictor if no visual prompts
    self.overrides["agnostic_nms"] = True  # use agnostic nms for YOLOE default

    return super().predict(source, stream, **kwargs)
```
</details>

<br>

### Method `ultralytics.models.yolo.model.YOLOE.set_classes` {#ultralytics.models.yolo.model.YOLOE.set\_classes}

```python
def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None
```

Set the model's class names and embeddings for detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `classes` | `list[str]` | A list of categories i.e. ["person"]. | *required* |
| `embeddings` | `torch.Tensor, optional` | Embeddings corresponding to the classes. | `None` |

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L306-L323"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
    """Set the model's class names and embeddings for detection.

    Args:
        classes (list[str]): A list of categories i.e. ["person"].
        embeddings (torch.Tensor, optional): Embeddings corresponding to the classes.
    """
    # Verify no background class is present
    assert " " not in classes
    assert isinstance(self.model, YOLOEModel)
    if sorted(list(self.model.names.values())) != sorted(classes):
        if embeddings is None:
            embeddings = self.get_text_pe(classes)  # generate text embeddings if not provided
        self.model.set_classes(classes, embeddings)

    # Reset method class names
    if self.predictor:
        self.predictor.model.names = self.model.names
```
</details>

<br>

### Method `ultralytics.models.yolo.model.YOLOE.set_vocab` {#ultralytics.models.yolo.model.YOLOE.set\_vocab}

```python
def set_vocab(self, vocab: list[str], names: list[str]) -> None
```

Set vocabulary and class names for the YOLOE model.

This method configures the vocabulary and class names used by the model for text processing and classification tasks. The model must be an instance of YOLOEModel.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `vocab` | `list[str]` | Vocabulary list containing tokens or words used by the model for text processing. | *required* |
| `names` | `list[str]` | List of class names that the model can detect or classify. | *required* |

**Examples**

```python
>>> model = YOLOE("yoloe-11s-seg.pt")
>>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
```

**Raises**

| Type | Description |
| --- | --- |
| `AssertionError` | If the model is not an instance of YOLOEModel. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L281-L299"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_vocab(self, vocab: list[str], names: list[str]) -> None:
    """Set vocabulary and class names for the YOLOE model.

    This method configures the vocabulary and class names used by the model for text processing and classification
    tasks. The model must be an instance of YOLOEModel.

    Args:
        vocab (list[str]): Vocabulary list containing tokens or words used by the model for text processing.
        names (list[str]): List of class names that the model can detect or classify.

    Raises:
        AssertionError: If the model is not an instance of YOLOEModel.

    Examples:
        >>> model = YOLOE("yoloe-11s-seg.pt")
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
    """
    assert isinstance(self.model, YOLOEModel)
    self.model.set_vocab(vocab, names=names)
```
</details>

<br>

### Method `ultralytics.models.yolo.model.YOLOE.val` {#ultralytics.models.yolo.model.YOLOE.val}

```python
def val(self, validator = None, load_vp: bool = False, refer_data: str | None = None, **kwargs)
```

Validate the model using text or visual prompts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `validator` | `callable, optional` | A callable validator function. If None, a default validator is loaded. | `None` |
| `load_vp` | `bool` | Whether to load visual prompts. If False, text prompts are used. | `False` |
| `refer_data` | `str, optional` | Path to the reference data for visual prompts. | `None` |
| `**kwargs` | `Any` | Additional keyword arguments to override default settings. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Validation statistics containing metrics computed during validation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py#L325-L349"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def val(
    self,
    validator=None,
    load_vp: bool = False,
    refer_data: str | None = None,
    **kwargs,
):
    """Validate the model using text or visual prompts.

    Args:
        validator (callable, optional): A callable validator function. If None, a default validator is loaded.
        load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
        refer_data (str, optional): Path to the reference data for visual prompts.
        **kwargs (Any): Additional keyword arguments to override default settings.

    Returns:
        (dict): Validation statistics containing metrics computed during validation.
    """
    custom = {"rect": not load_vp}  # method defaults
    args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

    validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
    validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
    self.metrics = validator.metrics
    return validator.metrics
```
</details>

<br><br>
