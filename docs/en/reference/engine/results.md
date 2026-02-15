---
description: Explore the details of Ultralytics engine results including classes like BaseTensor, Results, Boxes, Masks, Keypoints, Probs, and OBB to handle inference results efficiently.
keywords: Ultralytics, engine results, BaseTensor, Results class, Boxes, Masks, Keypoints, Probs, OBB, inference results, machine learning, PyTorch
---

# Reference for `ultralytics/engine/results.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`BaseTensor`](#ultralytics.engine.results.BaseTensor)
        - [`Results`](#ultralytics.engine.results.Results)
        - [`Boxes`](#ultralytics.engine.results.Boxes)
        - [`Masks`](#ultralytics.engine.results.Masks)
        - [`Keypoints`](#ultralytics.engine.results.Keypoints)
        - [`Probs`](#ultralytics.engine.results.Probs)
        - [`OBB`](#ultralytics.engine.results.OBB)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`BaseTensor.shape`](#ultralytics.engine.results.BaseTensor.shape)
        - [`Boxes.xyxy`](#ultralytics.engine.results.Boxes.xyxy)
        - [`Boxes.conf`](#ultralytics.engine.results.Boxes.conf)
        - [`Boxes.cls`](#ultralytics.engine.results.Boxes.cls)
        - [`Boxes.id`](#ultralytics.engine.results.Boxes.id)
        - [`Boxes.xywh`](#ultralytics.engine.results.Boxes.xywh)
        - [`Boxes.xyxyn`](#ultralytics.engine.results.Boxes.xyxyn)
        - [`Boxes.xywhn`](#ultralytics.engine.results.Boxes.xywhn)
        - [`Masks.xyn`](#ultralytics.engine.results.Masks.xyn)
        - [`Masks.xy`](#ultralytics.engine.results.Masks.xy)
        - [`Keypoints.xy`](#ultralytics.engine.results.Keypoints.xy)
        - [`Keypoints.xyn`](#ultralytics.engine.results.Keypoints.xyn)
        - [`Keypoints.conf`](#ultralytics.engine.results.Keypoints.conf)
        - [`Probs.top1`](#ultralytics.engine.results.Probs.top1)
        - [`Probs.top5`](#ultralytics.engine.results.Probs.top5)
        - [`Probs.top1conf`](#ultralytics.engine.results.Probs.top1conf)
        - [`Probs.top5conf`](#ultralytics.engine.results.Probs.top5conf)
        - [`OBB.xywhr`](#ultralytics.engine.results.OBB.xywhr)
        - [`OBB.conf`](#ultralytics.engine.results.OBB.conf)
        - [`OBB.cls`](#ultralytics.engine.results.OBB.cls)
        - [`OBB.id`](#ultralytics.engine.results.OBB.id)
        - [`OBB.xyxyxyxy`](#ultralytics.engine.results.OBB.xyxyxyxy)
        - [`OBB.xyxyxyxyn`](#ultralytics.engine.results.OBB.xyxyxyxyn)
        - [`OBB.xyxy`](#ultralytics.engine.results.OBB.xyxy)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`BaseTensor.cpu`](#ultralytics.engine.results.BaseTensor.cpu)
        - [`BaseTensor.numpy`](#ultralytics.engine.results.BaseTensor.numpy)
        - [`BaseTensor.cuda`](#ultralytics.engine.results.BaseTensor.cuda)
        - [`BaseTensor.to`](#ultralytics.engine.results.BaseTensor.to)
        - [`BaseTensor.__len__`](#ultralytics.engine.results.BaseTensor.__len__)
        - [`BaseTensor.__getitem__`](#ultralytics.engine.results.BaseTensor.__getitem__)
        - [`Results.__getitem__`](#ultralytics.engine.results.Results.__getitem__)
        - [`Results.__len__`](#ultralytics.engine.results.Results.__len__)
        - [`Results.update`](#ultralytics.engine.results.Results.update)
        - [`Results._apply`](#ultralytics.engine.results.Results._apply)
        - [`Results.cpu`](#ultralytics.engine.results.Results.cpu)
        - [`Results.numpy`](#ultralytics.engine.results.Results.numpy)
        - [`Results.cuda`](#ultralytics.engine.results.Results.cuda)
        - [`Results.to`](#ultralytics.engine.results.Results.to)
        - [`Results.new`](#ultralytics.engine.results.Results.new)
        - [`Results.plot`](#ultralytics.engine.results.Results.plot)
        - [`Results.show`](#ultralytics.engine.results.Results.show)
        - [`Results.save`](#ultralytics.engine.results.Results.save)
        - [`Results.verbose`](#ultralytics.engine.results.Results.verbose)
        - [`Results.save_txt`](#ultralytics.engine.results.Results.save_txt)
        - [`Results.save_crop`](#ultralytics.engine.results.Results.save_crop)
        - [`Results.summary`](#ultralytics.engine.results.Results.summary)


## Class `ultralytics.engine.results.BaseTensor` {#ultralytics.engine.results.BaseTensor}

```python
BaseTensor(self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None
```

**Bases:** `SimpleClass`

Base tensor class with additional methods for easy manipulation and device handling.

This class provides a foundation for tensor-like objects with device management capabilities, supporting both PyTorch tensors and NumPy arrays. It includes methods for moving data between devices and converting between tensor types.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data` | `torch.Tensor | np.ndarray` | Prediction data such as bounding boxes, masks, or keypoints. | *required* |
| `orig_shape` | `tuple[int, int]` | Original shape of the image in (height, width) format. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `torch.Tensor | np.ndarray` | Prediction data such as bounding boxes, masks, or keypoints. |
| `orig_shape` | `tuple[int, int]` | Original shape of the image, typically in the format (height, width). |

**Methods**

| Name | Description |
| --- | --- |
| [`shape`](#ultralytics.engine.results.BaseTensor.shape) | Return the shape of the underlying data tensor. |
| [`__getitem__`](#ultralytics.engine.results.BaseTensor.__getitem__) | Return a new BaseTensor instance containing the specified indexed elements of the data tensor. |
| [`__len__`](#ultralytics.engine.results.BaseTensor.__len__) | Return the length of the underlying data tensor. |
| [`cpu`](#ultralytics.engine.results.BaseTensor.cpu) | Return a copy of the tensor stored in CPU memory. |
| [`cuda`](#ultralytics.engine.results.BaseTensor.cuda) | Move the tensor to GPU memory. |
| [`numpy`](#ultralytics.engine.results.BaseTensor.numpy) | Return a copy of this object with its data converted to a NumPy array. |
| [`to`](#ultralytics.engine.results.BaseTensor.to) | Return a copy of the tensor with the specified device and dtype. |

**Examples**

```python
>>> import torch
>>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> orig_shape = (720, 1280)
>>> base_tensor = BaseTensor(data, orig_shape)
>>> cpu_tensor = base_tensor.cpu()
>>> numpy_array = base_tensor.numpy()
>>> gpu_tensor = base_tensor.cuda()
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L23-L173"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BaseTensor(SimpleClass):
    """Base tensor class with additional methods for easy manipulation and device handling.

    This class provides a foundation for tensor-like objects with device management capabilities, supporting both
    PyTorch tensors and NumPy arrays. It includes methods for moving data between devices and converting between tensor
    types.

    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
        orig_shape (tuple[int, int]): Original shape of the image, typically in the format (height, width).

    Methods:
        cpu: Return a copy of the tensor stored in CPU memory.
        numpy: Return a copy of the tensor as a numpy array.
        cuda: Move the tensor to GPU memory, returning a new instance if necessary.
        to: Return a copy of the tensor with the specified device and dtype.

    Examples:
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize BaseTensor with prediction data and the original shape of the image.

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (tuple[int, int]): Original shape of the image in (height, width) format.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape
```
</details>

<br>

### Property `ultralytics.engine.results.BaseTensor.shape` {#ultralytics.engine.results.BaseTensor.shape}

```python
def shape(self) -> tuple[int, ...]
```

Return the shape of the underlying data tensor.

**Returns**

| Type | Description |
| --- | --- |
| `tuple[int, ...]` | The shape of the data tensor. |

**Examples**

```python
>>> data = torch.rand(100, 4)
>>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
>>> print(base_tensor.shape)
(100, 4)
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L62-L74"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def shape(self) -> tuple[int, ...]:
    """Return the shape of the underlying data tensor.

    Returns:
        (tuple[int, ...]): The shape of the data tensor.

    Examples:
        >>> data = torch.rand(100, 4)
        >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
        >>> print(base_tensor.shape)
        (100, 4)
    """
    return self.data.shape
```
</details>

<br>

### Method `ultralytics.engine.results.BaseTensor.__getitem__` {#ultralytics.engine.results.BaseTensor.\_\_getitem\_\_}

```python
def __getitem__(self, idx)
```

Return a new BaseTensor instance containing the specified indexed elements of the data tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `idx` | `int | list[int] | torch.Tensor` | Index or indices to select from the data tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `BaseTensor` | A new BaseTensor instance containing the indexed data. |

**Examples**

```python
>>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
>>> result = base_tensor[0]  # Select the first row
>>> print(result.data)
tensor([1, 2, 3])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L157-L173"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __getitem__(self, idx):
    """Return a new BaseTensor instance containing the specified indexed elements of the data tensor.

    Args:
        idx (int | list[int] | torch.Tensor): Index or indices to select from the data tensor.

    Returns:
        (BaseTensor): A new BaseTensor instance containing the indexed data.

    Examples:
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
        >>> result = base_tensor[0]  # Select the first row
        >>> print(result.data)
        tensor([1, 2, 3])
    """
    return self.__class__(self.data[idx], self.orig_shape)
```
</details>

<br>

### Method `ultralytics.engine.results.BaseTensor.__len__` {#ultralytics.engine.results.BaseTensor.\_\_len\_\_}

```python
def __len__(self) -> int
```

Return the length of the underlying data tensor.

**Returns**

| Type | Description |
| --- | --- |
| `int` | The number of elements in the first dimension of the data tensor. |

**Examples**

```python
>>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
>>> len(base_tensor)
2
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L143-L155"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __len__(self) -> int:
    """Return the length of the underlying data tensor.

    Returns:
        (int): The number of elements in the first dimension of the data tensor.

    Examples:
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
        >>> len(base_tensor)
        2
    """
    return len(self.data)
```
</details>

<br>

### Method `ultralytics.engine.results.BaseTensor.cpu` {#ultralytics.engine.results.BaseTensor.cpu}

```python
def cpu(self)
```

Return a copy of the tensor stored in CPU memory.

**Returns**

| Type | Description |
| --- | --- |
| `BaseTensor` | A new BaseTensor object with the data tensor moved to CPU memory. |

**Examples**

```python
>>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
>>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
>>> cpu_tensor = base_tensor.cpu()
>>> isinstance(cpu_tensor, BaseTensor)
True
>>> cpu_tensor.data.device
device(type='cpu')
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L76-L91"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def cpu(self):
    """Return a copy of the tensor stored in CPU memory.

    Returns:
        (BaseTensor): A new BaseTensor object with the data tensor moved to CPU memory.

    Examples:
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
        >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
        >>> cpu_tensor = base_tensor.cpu()
        >>> isinstance(cpu_tensor, BaseTensor)
        True
        >>> cpu_tensor.data.device
        device(type='cpu')
    """
    return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)
```
</details>

<br>

### Method `ultralytics.engine.results.BaseTensor.cuda` {#ultralytics.engine.results.BaseTensor.cuda}

```python
def cuda(self)
```

Move the tensor to GPU memory.

**Returns**

| Type | Description |
| --- | --- |
| `BaseTensor` | A new BaseTensor instance with the data moved to GPU memory. |

**Examples**

```python
>>> import torch
>>> from ultralytics.engine.results import BaseTensor
>>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
>>> gpu_tensor = base_tensor.cuda()
>>> print(gpu_tensor.data.device)
cuda:0
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L109-L124"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def cuda(self):
    """Move the tensor to GPU memory.

    Returns:
        (BaseTensor): A new BaseTensor instance with the data moved to GPU memory.

    Examples:
        >>> import torch
        >>> from ultralytics.engine.results import BaseTensor
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
        >>> gpu_tensor = base_tensor.cuda()
        >>> print(gpu_tensor.data.device)
        cuda:0
    """
    return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)
```
</details>

<br>

### Method `ultralytics.engine.results.BaseTensor.numpy` {#ultralytics.engine.results.BaseTensor.numpy}

```python
def numpy(self)
```

Return a copy of this object with its data converted to a NumPy array.

**Returns**

| Type | Description |
| --- | --- |
| `BaseTensor` | A new instance with `data` as a NumPy array. |

**Examples**

```python
>>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> orig_shape = (720, 1280)
>>> base_tensor = BaseTensor(data, orig_shape)
>>> numpy_tensor = base_tensor.numpy()
>>> print(type(numpy_tensor.data))
<class 'numpy.ndarray'>
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L93-L107"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def numpy(self):
    """Return a copy of this object with its data converted to a NumPy array.

    Returns:
        (BaseTensor): A new instance with `data` as a NumPy array.

    Examples:
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> numpy_tensor = base_tensor.numpy()
        >>> print(type(numpy_tensor.data))
        <class 'numpy.ndarray'>
    """
    return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)
```
</details>

<br>

### Method `ultralytics.engine.results.BaseTensor.to` {#ultralytics.engine.results.BaseTensor.to}

```python
def to(self, *args, **kwargs)
```

Return a copy of the tensor with the specified device and dtype.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` | Variable length argument list to be passed to torch.Tensor.to(). | *required* |
| `**kwargs` | `Any` | Arbitrary keyword arguments to be passed to torch.Tensor.to(). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `BaseTensor` | A new BaseTensor instance with the data moved to the specified device and/or dtype. |

**Examples**

```python
>>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
>>> cuda_tensor = base_tensor.to("cuda")
>>> float16_tensor = base_tensor.to(dtype=torch.float16)
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L126-L141"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def to(self, *args, **kwargs):
    """Return a copy of the tensor with the specified device and dtype.

    Args:
        *args (Any): Variable length argument list to be passed to torch.Tensor.to().
        **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

    Returns:
        (BaseTensor): A new BaseTensor instance with the data moved to the specified device and/or dtype.

    Examples:
        >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
        >>> cuda_tensor = base_tensor.to("cuda")
        >>> float16_tensor = base_tensor.to(dtype=torch.float16)
    """
    return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)
```
</details>


<br><br><hr><br>

## Class `ultralytics.engine.results.Results` {#ultralytics.engine.results.Results}

```python
def __init__(
    self,
    orig_img: np.ndarray,
    path: str,
    names: dict[int, str],
    boxes: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
    probs: torch.Tensor | None = None,
    keypoints: torch.Tensor | None = None,
    obb: torch.Tensor | None = None,
    speed: dict[str, float] | None = None,
) -> None
```

**Bases:** `SimpleClass`, `DataExportMixin`

A class for storing and manipulating inference results.

This class provides comprehensive functionality for handling inference results from various Ultralytics models, including detection, segmentation, classification, and pose estimation. It supports visualization, data export, and various coordinate transformations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `orig_img` | `np.ndarray` | The original image as a numpy array. | *required* |
| `path` | `str` | The path to the image file. | *required* |
| `names` | `dict` | A dictionary of class names. | *required* |
| `boxes` | `torch.Tensor | None` | A 2D tensor of bounding box coordinates for each detection. | `None` |
| `masks` | `torch.Tensor | None` | A 3D tensor of detection masks, where each mask is a binary image. | `None` |
| `probs` | `torch.Tensor | None` | A 1D tensor of probabilities of each class for classification task. | `None` |
| `keypoints` | `torch.Tensor | None` | A 2D tensor of keypoint coordinates for each detection. | `None` |
| `obb` | `torch.Tensor | None` | A 2D tensor of oriented bounding box coordinates for each detection. | `None` |
| `speed` | `dict | None` | A dictionary containing preprocess, inference, and postprocess speeds (ms/image). | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `orig_img` | `np.ndarray` | The original image as a numpy array. |
| `orig_shape` | `tuple[int, int]` | Original image shape in (height, width) format. |
| `boxes` | `Boxes | None` | Detected bounding boxes. |
| `masks` | `Masks | None` | Segmentation masks. |
| `probs` | `Probs | None` | Classification probabilities. |
| `keypoints` | `Keypoints | None` | Detected keypoints. |
| `obb` | `OBB | None` | Oriented bounding boxes. |
| `speed` | `dict` | Dictionary containing inference speed information. |
| `names` | `dict` | Dictionary mapping class indices to class names. |
| `path` | `str` | Path to the input image file. |
| `save_dir` | `str | None` | Directory to save results. |

**Methods**

| Name | Description |
| --- | --- |
| [`__getitem__`](#ultralytics.engine.results.Results.__getitem__) | Return a Results object for a specific index of inference results. |
| [`__len__`](#ultralytics.engine.results.Results.__len__) | Return the number of detections in the Results object. |
| [`_apply`](#ultralytics.engine.results.Results._apply) | Apply a function to all non-empty attributes and return a new Results object with modified attributes. |
| [`cpu`](#ultralytics.engine.results.Results.cpu) | Return a copy of the Results object with all its tensors moved to CPU memory. |
| [`cuda`](#ultralytics.engine.results.Results.cuda) | Move all tensors in the Results object to GPU memory. |
| [`new`](#ultralytics.engine.results.Results.new) | Create a new Results object with the same image, path, names, and speed attributes. |
| [`numpy`](#ultralytics.engine.results.Results.numpy) | Convert all tensors in the Results object to numpy arrays. |
| [`plot`](#ultralytics.engine.results.Results.plot) | Plot detection results on an input BGR image. |
| [`save`](#ultralytics.engine.results.Results.save) | Save annotated inference results image to file. |
| [`save_crop`](#ultralytics.engine.results.Results.save_crop) | Save cropped detection images to specified directory. |
| [`save_txt`](#ultralytics.engine.results.Results.save_txt) | Save detection results to a text file. |
| [`show`](#ultralytics.engine.results.Results.show) | Display the image with annotated inference results. |
| [`summary`](#ultralytics.engine.results.Results.summary) | Convert inference results to a summarized dictionary with optional normalization for box coordinates. |
| [`to`](#ultralytics.engine.results.Results.to) | Move all tensors in the Results object to the specified device and dtype. |
| [`update`](#ultralytics.engine.results.Results.update) | Update the Results object with new detection data. |
| [`verbose`](#ultralytics.engine.results.Results.verbose) | Return a log string for each task in the results, detailing detection and classification outcomes. |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> result = results[0]  # Get the first result
>>> boxes = result.boxes  # Get the boxes for the first result
>>> masks = result.masks  # Get the masks for the first result
>>> for result in results:
...     result.plot()  # Plot detection results
```

!!! note "Notes"

    For the default pose model, keypoint indices for human body pose estimation are:
    0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
    5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
    9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
    13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L176-L822"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Results(SimpleClass, DataExportMixin):
    """A class for storing and manipulating inference results.

    This class provides comprehensive functionality for handling inference results from various Ultralytics models,
    including detection, segmentation, classification, and pose estimation. It supports visualization, data export, and
    various coordinate transformations.

    Attributes:
        orig_img (np.ndarray): The original image as a numpy array.
        orig_shape (tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Detected bounding boxes.
        masks (Masks | None): Segmentation masks.
        probs (Probs | None): Classification probabilities.
        keypoints (Keypoints | None): Detected keypoints.
        obb (OBB | None): Oriented bounding boxes.
        speed (dict): Dictionary containing inference speed information.
        names (dict): Dictionary mapping class indices to class names.
        path (str): Path to the input image file.
        save_dir (str | None): Directory to save results.

    Methods:
        update: Update the Results object with new detection data.
        cpu: Return a copy of the Results object with all tensors moved to CPU memory.
        numpy: Convert all tensors in the Results object to numpy arrays.
        cuda: Move all tensors in the Results object to GPU memory.
        to: Move all tensors to the specified device and dtype.
        new: Create a new Results object with the same image, path, names, and speed attributes.
        plot: Plot detection results on an input BGR image.
        show: Display the image with annotated inference results.
        save: Save annotated inference results image to file.
        verbose: Return a log string for each task in the results.
        save_txt: Save detection results to a text file.
        save_crop: Save cropped detection images to specified directory.
        summary: Convert inference results to a summarized dictionary.
        to_df: Convert detection results to a Polars DataFrame.
        to_json: Convert detection results to JSON format.
        to_csv: Convert detection results to a CSV format.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # Get the first result
        >>> boxes = result.boxes  # Get the boxes for the first result
        >>> masks = result.masks  # Get the masks for the first result
        >>> for result in results:
        ...     result.plot()  # Plot detection results
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
        """Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (np.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection.
            speed (dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).

        Notes:
            For the default pose model, keypoint indices for human body pose estimation are:
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"
```
</details>

<br>

### Method `ultralytics.engine.results.Results.__getitem__` {#ultralytics.engine.results.Results.\_\_getitem\_\_}

```python
def __getitem__(self, idx)
```

Return a Results object for a specific index of inference results.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `idx` | `int | slice` | Index or slice to retrieve from the Results object. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Results` | A new Results object containing the specified subset of inference results. |

**Examples**

```python
>>> results = model("path/to/image.jpg")  # Perform inference
>>> single_result = results[0]  # Get the first result
>>> subset_results = results[1:4]  # Get a slice of results
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L268-L282"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __getitem__(self, idx):
    """Return a Results object for a specific index of inference results.

    Args:
        idx (int | slice): Index or slice to retrieve from the Results object.

    Returns:
        (Results): A new Results object containing the specified subset of inference results.

    Examples:
        >>> results = model("path/to/image.jpg")  # Perform inference
        >>> single_result = results[0]  # Get the first result
        >>> subset_results = results[1:4]  # Get a slice of results
    """
    return self._apply("__getitem__", idx)
```
</details>

<br>

### Method `ultralytics.engine.results.Results.__len__` {#ultralytics.engine.results.Results.\_\_len\_\_}

```python
def __len__(self) -> int
```

Return the number of detections in the Results object.

**Returns**

| Type | Description |
| --- | --- |
| `int` | The number of detections, determined by the length of the first non-empty attribute in (boxes, masks, |

**Examples**

```python
>>> results = Results(orig_img, path, names, boxes=torch.rand(5, 6))
>>> len(results)
5
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L284-L299"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __len__(self) -> int:
    """Return the number of detections in the Results object.

    Returns:
        (int): The number of detections, determined by the length of the first non-empty attribute in (boxes, masks,
            probs, keypoints, or obb).

    Examples:
        >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 6))
        >>> len(results)
        5
    """
    for k in self._keys:
        v = getattr(self, k)
        if v is not None:
            return len(v)
```
</details>

<br>

### Method `ultralytics.engine.results.Results._apply` {#ultralytics.engine.results.Results.\_apply}

```python
def _apply(self, fn: str, *args, **kwargs)
```

Apply a function to all non-empty attributes and return a new Results object with modified attributes.

This method is internally called by methods like .to(), .cuda(), .cpu(), etc.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `fn` | `str` | The name of the function to apply. | *required* |
| `*args` | `Any` | Variable length argument list to pass to the function. | *required* |
| `**kwargs` | `Any` | Arbitrary keyword arguments to pass to the function. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Results` | A new Results object with attributes modified by the applied function. |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> for result in results:
...     result_cuda = result.cuda()
...     result_cpu = result.cpu()
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L338-L362"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _apply(self, fn: str, *args, **kwargs):
    """Apply a function to all non-empty attributes and return a new Results object with modified attributes.

    This method is internally called by methods like .to(), .cuda(), .cpu(), etc.

    Args:
        fn (str): The name of the function to apply.
        *args (Any): Variable length argument list to pass to the function.
        **kwargs (Any): Arbitrary keyword arguments to pass to the function.

    Returns:
        (Results): A new Results object with attributes modified by the applied function.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     result_cuda = result.cuda()
        ...     result_cpu = result.cpu()
    """
    r = self.new()
    for k in self._keys:
        v = getattr(self, k)
        if v is not None:
            setattr(r, k, getattr(v, fn)(*args, **kwargs))
    return r
```
</details>

<br>

### Method `ultralytics.engine.results.Results.cpu` {#ultralytics.engine.results.Results.cpu}

```python
def cpu(self)
```

Return a copy of the Results object with all its tensors moved to CPU memory.

This method creates a new Results object with all tensor attributes (boxes, masks, probs, keypoints, obb) transferred to CPU memory. It's useful for moving data from GPU to CPU for further processing or saving.

**Returns**

| Type | Description |
| --- | --- |
| `Results` | A new Results object with all tensor attributes on CPU memory. |

**Examples**

```python
>>> results = model("path/to/image.jpg")  # Perform inference
>>> cpu_result = results[0].cpu()  # Move the first result to CPU
>>> print(cpu_result.boxes.device)  # Output: cpu
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L364-L378"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def cpu(self):
    """Return a copy of the Results object with all its tensors moved to CPU memory.

    This method creates a new Results object with all tensor attributes (boxes, masks, probs, keypoints, obb)
    transferred to CPU memory. It's useful for moving data from GPU to CPU for further processing or saving.

    Returns:
        (Results): A new Results object with all tensor attributes on CPU memory.

    Examples:
        >>> results = model("path/to/image.jpg")  # Perform inference
        >>> cpu_result = results[0].cpu()  # Move the first result to CPU
        >>> print(cpu_result.boxes.device)  # Output: cpu
    """
    return self._apply("cpu")
```
</details>

<br>

### Method `ultralytics.engine.results.Results.cuda` {#ultralytics.engine.results.Results.cuda}

```python
def cuda(self)
```

Move all tensors in the Results object to GPU memory.

**Returns**

| Type | Description |
| --- | --- |
| `Results` | A new Results object with all tensors moved to CUDA device. |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> cuda_results = results[0].cuda()  # Move first result to GPU
>>> for result in results:
...     result_cuda = result.cuda()  # Move each result to GPU
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L398-L410"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def cuda(self):
    """Move all tensors in the Results object to GPU memory.

    Returns:
        (Results): A new Results object with all tensors moved to CUDA device.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> cuda_results = results[0].cuda()  # Move first result to GPU
        >>> for result in results:
        ...     result_cuda = result.cuda()  # Move each result to GPU
    """
    return self._apply("cuda")
```
</details>

<br>

### Method `ultralytics.engine.results.Results.new` {#ultralytics.engine.results.Results.new}

```python
def new(self)
```

Create a new Results object with the same image, path, names, and speed attributes.

**Returns**

| Type | Description |
| --- | --- |
| `Results` | A new Results object with copied attributes from the original instance. |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> new_result = results[0].new()
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L430-L440"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def new(self):
    """Create a new Results object with the same image, path, names, and speed attributes.

    Returns:
        (Results): A new Results object with copied attributes from the original instance.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> new_result = results[0].new()
    """
    return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)
```
</details>

<br>

### Method `ultralytics.engine.results.Results.numpy` {#ultralytics.engine.results.Results.numpy}

```python
def numpy(self)
```

Convert all tensors in the Results object to numpy arrays.

**Returns**

| Type | Description |
| --- | --- |
| `Results` | A new Results object with all tensors converted to numpy arrays. |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> numpy_result = results[0].numpy()
>>> type(numpy_result.boxes.data)
<class 'numpy.ndarray'>
```

!!! note "Notes"

    This method creates a new Results object, leaving the original unchanged. It's useful for
    interoperability with numpy-based libraries or when CPU-based operations are required.

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L380-L396"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def numpy(self):
    """Convert all tensors in the Results object to numpy arrays.

    Returns:
        (Results): A new Results object with all tensors converted to numpy arrays.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> numpy_result = results[0].numpy()
        >>> type(numpy_result.boxes.data)
        <class 'numpy.ndarray'>

    Notes:
        This method creates a new Results object, leaving the original unchanged. It's useful for
        interoperability with numpy-based libraries or when CPU-based operations are required.
    """
    return self._apply("numpy")
```
</details>

<br>

### Method `ultralytics.engine.results.Results.plot` {#ultralytics.engine.results.Results.plot}

```python
def plot(
    self,
    conf: bool = True,
    line_width: float | None = None,
    font_size: float | None = None,
    font: str = "Arial.ttf",
    pil: bool = False,
    img: np.ndarray | None = None,
    im_gpu: torch.Tensor | None = None,
    kpt_radius: int = 5,
    kpt_line: bool = True,
    labels: bool = True,
    boxes: bool = True,
    masks: bool = True,
    probs: bool = True,
    show: bool = False,
    save: bool = False,
    filename: str | None = None,
    color_mode: str = "class",
    txt_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray
```

Plot detection results on an input BGR image.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `conf` | `bool` | Whether to plot detection confidence scores. | `True` |
| `line_width` | `float | None` | Line width of bounding boxes. If None, scaled to image size. | `None` |
| `font_size` | `float | None` | Font size for text. If None, scaled to image size. | `None` |
| `font` | `str` | Font to use for text. | `"Arial.ttf"` |
| `pil` | `bool` | Whether to return the image as a PIL Image. | `False` |
| `img` | `np.ndarray | None` | Image to plot on. If None, uses original image. | `None` |
| `im_gpu` | `torch.Tensor | None` | Normalized image on GPU for faster mask plotting. | `None` |
| `kpt_radius` | `int` | Radius of drawn keypoints. | `5` |
| `kpt_line` | `bool` | Whether to draw lines connecting keypoints. | `True` |
| `labels` | `bool` | Whether to plot labels of bounding boxes. | `True` |
| `boxes` | `bool` | Whether to plot bounding boxes. | `True` |
| `masks` | `bool` | Whether to plot masks. | `True` |
| `probs` | `bool` | Whether to plot classification probabilities. | `True` |
| `show` | `bool` | Whether to display the annotated image. | `False` |
| `save` | `bool` | Whether to save the annotated image. | `False` |
| `filename` | `str | None` | Filename to save image if save is True. | `None` |
| `color_mode` | `str` | Specify the color mode, e.g., 'instance' or 'class'. | `"class"` |
| `txt_color` | `tuple[int, int, int]` | Text color in BGR format for classification output. | `(255, 255, 255)` |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | PIL.Image.Image` | Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`. |

**Examples**

```python
>>> results = model("image.jpg")
>>> for result in results:
...     im = result.plot()
...     im.show()
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L442-L579"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot(
    self,
    conf: bool = True,
    line_width: float | None = None,
    font_size: float | None = None,
    font: str = "Arial.ttf",
    pil: bool = False,
    img: np.ndarray | None = None,
    im_gpu: torch.Tensor | None = None,
    kpt_radius: int = 5,
    kpt_line: bool = True,
    labels: bool = True,
    boxes: bool = True,
    masks: bool = True,
    probs: bool = True,
    show: bool = False,
    save: bool = False,
    filename: str | None = None,
    color_mode: str = "class",
    txt_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Plot detection results on an input BGR image.

    Args:
        conf (bool): Whether to plot detection confidence scores.
        line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
        font_size (float | None): Font size for text. If None, scaled to image size.
        font (str): Font to use for text.
        pil (bool): Whether to return the image as a PIL Image.
        img (np.ndarray | None): Image to plot on. If None, uses original image.
        im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.
        kpt_radius (int): Radius of drawn keypoints.
        kpt_line (bool): Whether to draw lines connecting keypoints.
        labels (bool): Whether to plot labels of bounding boxes.
        boxes (bool): Whether to plot bounding boxes.
        masks (bool): Whether to plot masks.
        probs (bool): Whether to plot classification probabilities.
        show (bool): Whether to display the annotated image.
        save (bool): Whether to save the annotated image.
        filename (str | None): Filename to save image if save is True.
        color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.
        txt_color (tuple[int, int, int]): Text color in BGR format for classification output.

    Returns:
        (np.ndarray | PIL.Image.Image): Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`.

    Examples:
        >>> results = model("image.jpg")
        >>> for result in results:
        ...     im = result.plot()
        ...     im.show()
    """
    assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
    if img is None and isinstance(self.orig_img, torch.Tensor):
        img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

    names = self.names
    is_obb = self.obb is not None
    pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
    pred_masks, show_masks = self.masks, masks
    pred_probs, show_probs = self.probs, probs
    annotator = Annotator(
        deepcopy(self.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )

    # Plot Segment results
    if pred_masks and show_masks:
        if im_gpu is None:
            img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
            im_gpu = (
                torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                .permute(2, 0, 1)
                .flip(0)
                .contiguous()
                / 255
            )
        idx = (
            pred_boxes.id
            if pred_boxes.is_track and color_mode == "instance"
            else pred_boxes.cls
            if pred_boxes and color_mode == "class"
            else reversed(range(len(pred_masks)))
        )
        annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for i, d in enumerate(reversed(pred_boxes)):
            c, d_conf, id = int(d.cls), float(d.conf) if conf else None, int(d.id.item()) if d.is_track else None
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None
            box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(
                box,
                label,
                color=colors(
                    c
                    if color_mode == "class"
                    else id
                    if id is not None
                    else i
                    if color_mode == "instance"
                    else None,
                    True,
                ),
            )

    # Plot Classify results
    if pred_probs is not None and show_probs:
        text = "\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
        x = round(self.orig_shape[0] * 0.03)
        annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))  # RGBA box

    # Plot Pose results
    if self.keypoints is not None:
        for i, k in enumerate(reversed(self.keypoints.data)):
            annotator.kpts(
                k,
                self.orig_shape,
                radius=kpt_radius,
                kpt_line=kpt_line,
                kpt_color=colors(i, True) if color_mode == "instance" else None,
            )

    # Show results
    if show:
        annotator.show(self.path)

    # Save results
    if save:
        annotator.save(filename or f"results_{Path(self.path).name}")

    return annotator.result(pil)
```
</details>

<br>

### Method `ultralytics.engine.results.Results.save` {#ultralytics.engine.results.Results.save}

```python
def save(self, filename: str | None = None, *args, **kwargs) -> str
```

Save annotated inference results image to file.

This method plots the detection results on the original image and saves the annotated image to a file. It utilizes the `plot` method to generate the annotated image and then saves it to the specified filename.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `filename` | `str | None` | The filename to save the annotated image. If None, a default filename is generated<br>    based on the original image path. | `None` |
| `*args` | `Any` | Variable length argument list to be passed to the `plot` method. | *required* |
| `**kwargs` | `Any` | Arbitrary keyword arguments to be passed to the `plot` method. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | The filename where the image was saved. |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> for result in results:
...     result.save("annotated_image.jpg")
>>> # Or with custom plot arguments
>>> for result in results:
...     result.save("annotated_image.jpg", conf=False, line_width=2)
>>> # Directory will be created automatically if it does not exist
>>> result.save("path/to/annotated_image.jpg")
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L599-L628"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def save(self, filename: str | None = None, *args, **kwargs) -> str:
    """Save annotated inference results image to file.

    This method plots the detection results on the original image and saves the annotated image to a file. It
    utilizes the `plot` method to generate the annotated image and then saves it to the specified filename.

    Args:
        filename (str | None): The filename to save the annotated image. If None, a default filename is generated
            based on the original image path.
        *args (Any): Variable length argument list to be passed to the `plot` method.
        **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot` method.

    Returns:
        (str): The filename where the image was saved.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     result.save("annotated_image.jpg")
        >>> # Or with custom plot arguments
        >>> for result in results:
        ...     result.save("annotated_image.jpg", conf=False, line_width=2)
        >>> # Directory will be created automatically if it does not exist
        >>> result.save("path/to/annotated_image.jpg")
    """
    if not filename:
        filename = f"results_{Path(self.path).name}"
    Path(filename).absolute().parent.mkdir(parents=True, exist_ok=True)
    self.plot(save=True, filename=filename, *args, **kwargs)
    return filename
```
</details>

<br>

### Method `ultralytics.engine.results.Results.save_crop` {#ultralytics.engine.results.Results.save\_crop}

```python
def save_crop(self, save_dir: str | Path, file_name: str | Path = Path("im.jpg"))
```

Save cropped detection images to specified directory.

This method saves cropped images of detected objects to a specified directory. Each crop is saved in a subdirectory named after the object's class, with the filename based on the input file_name.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `save_dir` | `str | Path` | Directory path where cropped images will be saved. | *required* |
| `file_name` | `str | Path` | Base filename for the saved cropped images. | `Path("im.jpg")` |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> for result in results:
...     result.save_crop(save_dir="path/to/crops", file_name="detection")
```

!!! note "Notes"

    - This method does not support Classify or Oriented Bounding Box (OBB) tasks.
    - Crops are saved as 'save_dir/class_name/file_name.jpg'.
    - The method will create necessary subdirectories if they don't exist.
    - Original image is copied before cropping to avoid modifying the original.

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L717-L750"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def save_crop(self, save_dir: str | Path, file_name: str | Path = Path("im.jpg")):
    """Save cropped detection images to specified directory.

    This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
    subdirectory named after the object's class, with the filename based on the input file_name.

    Args:
        save_dir (str | Path): Directory path where cropped images will be saved.
        file_name (str | Path): Base filename for the saved cropped images.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     result.save_crop(save_dir="path/to/crops", file_name="detection")

    Notes:
        - This method does not support Classify or Oriented Bounding Box (OBB) tasks.
        - Crops are saved as 'save_dir/class_name/file_name.jpg'.
        - The method will create necessary subdirectories if they don't exist.
        - Original image is copied before cropping to avoid modifying the original.
    """
    if self.probs is not None:
        LOGGER.warning("Classify task does not support `save_crop`.")
        return
    if self.obb is not None:
        LOGGER.warning("OBB task does not support `save_crop`.")
        return
    for d in self.boxes:
        save_one_box(
            d.xyxy,
            self.orig_img.copy(),
            file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),
            BGR=True,
        )
```
</details>

<br>

### Method `ultralytics.engine.results.Results.save_txt` {#ultralytics.engine.results.Results.save\_txt}

```python
def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str
```

Save detection results to a text file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `txt_file` | `str | Path` | Path to the output text file. | *required* |
| `save_conf` | `bool` | Whether to include confidence scores in the output. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Path to the saved text file. |

**Examples**

```python
>>> from ultralytics import YOLO
>>> model = YOLO("yolo26n.pt")
>>> results = model("path/to/image.jpg")
>>> for result in results:
...     result.save_txt("output.txt")
```

!!! note "Notes"

    - The file will contain one line per detection or classification with the following structure:
      - For detections: `class x_center y_center width height [confidence] [track_id]`
      - For classifications: `confidence class_name`
      - For masks and keypoints, the specific formats will vary accordingly.
    - The function will create the output directory if it does not exist.
    - If save_conf is False, the confidence scores will be excluded from the output.
    - Existing contents of the file will not be overwritten; new results will be appended.

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L661-L715"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str:
    """Save detection results to a text file.

    Args:
        txt_file (str | Path): Path to the output text file.
        save_conf (bool): Whether to include confidence scores in the output.

    Returns:
        (str): Path to the saved text file.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     result.save_txt("output.txt")

    Notes:
        - The file will contain one line per detection or classification with the following structure:
          - For detections: `class x_center y_center width height [confidence] [track_id]`
          - For classifications: `confidence class_name`
          - For masks and keypoints, the specific formats will vary accordingly.
        - The function will create the output directory if it does not exist.
        - If save_conf is False, the confidence scores will be excluded from the output.
        - Existing contents of the file will not be overwritten; new results will be appended.
    """
    is_obb = self.obb is not None
    boxes = self.obb if is_obb else self.boxes
    masks = self.masks
    probs = self.probs
    kpts = self.keypoints
    texts = []
    if probs is not None:
        # Classify
        [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
    elif boxes:
        # Detect/segment/pose
        for j, d in enumerate(boxes):
            c, conf, id = int(d.cls), float(d.conf), int(d.id.item()) if d.is_track else None
            line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
            if masks:
                seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                line = (c, *seg)
            if kpts is not None:
                kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                line += (*kpt.reshape(-1).tolist(),)
            line += (conf,) * save_conf + (() if id is None else (id,))
            texts.append(("%g " * len(line)).rstrip() % line)

    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, "a", encoding="utf-8") as f:
            f.writelines(text + "\n" for text in texts)

    return str(txt_file)
```
</details>

<br>

### Method `ultralytics.engine.results.Results.show` {#ultralytics.engine.results.Results.show}

```python
def show(self, *args, **kwargs)
```

Display the image with annotated inference results.

This method plots the detection results on the original image and displays it. It's a convenient way to visualize the model's predictions directly.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` | Variable length argument list to be passed to the `plot()` method. | *required* |
| `**kwargs` | `Any` | Arbitrary keyword arguments to be passed to the `plot()` method. | *required* |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> results[0].show()  # Display the first result
>>> for result in results:
...     result.show()  # Display all results
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L581-L597"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def show(self, *args, **kwargs):
    """Display the image with annotated inference results.

    This method plots the detection results on the original image and displays it. It's a convenient way to
    visualize the model's predictions directly.

    Args:
        *args (Any): Variable length argument list to be passed to the `plot()` method.
        **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot()` method.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> results[0].show()  # Display the first result
        >>> for result in results:
        ...     result.show()  # Display all results
    """
    self.plot(show=True, *args, **kwargs)
```
</details>

<br>

### Method `ultralytics.engine.results.Results.summary` {#ultralytics.engine.results.Results.summary}

```python
def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, Any]]
```

Convert inference results to a summarized dictionary with optional normalization for box coordinates.

This method creates a list of detection dictionaries, each containing information about a single detection or classification result. For classification tasks, it returns the top 5 classes and their confidences. For detection tasks, it includes class information, bounding box coordinates, and optionally mask segments and keypoints.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool` | Whether to normalize bounding box coordinates by image dimensions. | `False` |
| `decimals` | `int` | Number of decimal places to round the output values to. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, Any]]` | A list of dictionaries, each containing summarized information for a single |

**Examples**

```python
>>> results = model("image.jpg")
>>> for result in results:
...     summary = result.summary()
...     print(summary)
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L752-L822"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, Any]]:
    """Convert inference results to a summarized dictionary with optional normalization for box coordinates.

    This method creates a list of detection dictionaries, each containing information about a single detection or
    classification result. For classification tasks, it returns the top 5 classes and their
    confidences. For detection tasks, it includes class information, bounding box coordinates, and
    optionally mask segments and keypoints.

    Args:
        normalize (bool): Whether to normalize bounding box coordinates by image dimensions.
        decimals (int): Number of decimal places to round the output values to.

    Returns:
        (list[dict[str, Any]]): A list of dictionaries, each containing summarized information for a single
            detection or classification result. The structure of each dictionary varies based on the task type
            (classification or detection) and available information (boxes, masks, keypoints).

    Examples:
        >>> results = model("image.jpg")
        >>> for result in results:
        ...     summary = result.summary()
        ...     print(summary)
    """
    # Create list of detection dictionaries
    results = []
    if self.probs is not None:
        # Return top 5 classification results
        for class_id, conf in zip(self.probs.top5, self.probs.top5conf.tolist()):
            class_id = int(class_id)
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(conf, decimals),
                }
            )
        return results

    is_obb = self.obb is not None
    data = self.obb if is_obb else self.boxes
    h, w = self.orig_shape if normalize else (1, 1)
    for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
        class_id, conf = int(row.cls), round(row.conf.item(), decimals)
        box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
        xy = {}
        for j, b in enumerate(box):
            xy[f"x{j + 1}"] = round(b[0] / w, decimals)
            xy[f"y{j + 1}"] = round(b[1] / h, decimals)
        result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
        if data.is_track:
            result["track_id"] = int(row.id.item())  # track ID
        if self.masks:
            result["segments"] = {
                "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
            }
        if self.keypoints is not None:
            kpt = self.keypoints[i]
            if kpt.has_visible:
                x, y, visible = kpt.data[0].cpu().unbind(dim=1)
            else:
                x, y = kpt.data[0].cpu().unbind(dim=1)
            result["keypoints"] = {
                "x": (x / w).numpy().round(decimals).tolist(),
                "y": (y / h).numpy().round(decimals).tolist(),
            }
            if kpt.has_visible:
                result["keypoints"]["visible"] = visible.numpy().round(decimals).tolist()
        results.append(result)

    return results
```
</details>

<br>

### Method `ultralytics.engine.results.Results.to` {#ultralytics.engine.results.Results.to}

```python
def to(self, *args, **kwargs)
```

Move all tensors in the Results object to the specified device and dtype.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` | Variable length argument list to be passed to torch.Tensor.to(). | *required* |
| `**kwargs` | `Any` | Arbitrary keyword arguments to be passed to torch.Tensor.to(). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Results` | A new Results object with all tensors moved to the specified device and dtype. |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> result_cuda = results[0].to("cuda")  # Move first result to GPU
>>> result_cpu = results[0].to("cpu")  # Move first result to CPU
>>> result_half = results[0].to(dtype=torch.float16)  # Convert first result to half precision
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L412-L428"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def to(self, *args, **kwargs):
    """Move all tensors in the Results object to the specified device and dtype.

    Args:
        *args (Any): Variable length argument list to be passed to torch.Tensor.to().
        **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

    Returns:
        (Results): A new Results object with all tensors moved to the specified device and dtype.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> result_cuda = results[0].to("cuda")  # Move first result to GPU
        >>> result_cpu = results[0].to("cpu")  # Move first result to CPU
        >>> result_half = results[0].to(dtype=torch.float16)  # Convert first result to half precision
    """
    return self._apply("to", *args, **kwargs)
```
</details>

<br>

### Method `ultralytics.engine.results.Results.update` {#ultralytics.engine.results.Results.update}

```python
def update(
    self,
    boxes: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
    probs: torch.Tensor | None = None,
    obb: torch.Tensor | None = None,
    keypoints: torch.Tensor | None = None,
)
```

Update the Results object with new detection data.

This method allows updating the boxes, masks, keypoints, probabilities, and oriented bounding boxes (OBB) of the Results object. It ensures that boxes are clipped to the original image shape.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `boxes` | `torch.Tensor | None` | A tensor of shape (N, 6) containing bounding box coordinates and confidence<br>    scores. The format is (x1, y1, x2, y2, conf, class). | `None` |
| `masks` | `torch.Tensor | None` | A tensor of shape (N, H, W) containing segmentation masks. | `None` |
| `probs` | `torch.Tensor | None` | A tensor of shape (num_classes,) containing class probabilities. | `None` |
| `obb` | `torch.Tensor | None` | A tensor of shape (N, 7) or (N, 8) containing oriented bounding box coordinates. | `None` |
| `keypoints` | `torch.Tensor | None` | A tensor of shape (N, K, 3) containing keypoints, were K=17 for persons. | `None` |

**Examples**

```python
>>> results = model("image.jpg")
>>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
>>> results[0].update(boxes=new_boxes)
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L301-L336"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(
    self,
    boxes: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
    probs: torch.Tensor | None = None,
    obb: torch.Tensor | None = None,
    keypoints: torch.Tensor | None = None,
):
    """Update the Results object with new detection data.

    This method allows updating the boxes, masks, keypoints, probabilities, and oriented bounding boxes (OBB) of
    the Results object. It ensures that boxes are clipped to the original image shape.

    Args:
        boxes (torch.Tensor | None): A tensor of shape (N, 6) containing bounding box coordinates and confidence
            scores. The format is (x1, y1, x2, y2, conf, class).
        masks (torch.Tensor | None): A tensor of shape (N, H, W) containing segmentation masks.
        probs (torch.Tensor | None): A tensor of shape (num_classes,) containing class probabilities.
        obb (torch.Tensor | None): A tensor of shape (N, 7) or (N, 8) containing oriented bounding box coordinates.
        keypoints (torch.Tensor | None): A tensor of shape (N, K, 3) containing keypoints, were K=17 for persons.

    Examples:
        >>> results = model("image.jpg")
        >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
        >>> results[0].update(boxes=new_boxes)
    """
    if boxes is not None:
        self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
    if masks is not None:
        self.masks = Masks(masks, self.orig_shape)
    if probs is not None:
        self.probs = probs
    if obb is not None:
        self.obb = OBB(obb, self.orig_shape)
    if keypoints is not None:
        self.keypoints = Keypoints(keypoints, self.orig_shape)
```
</details>

<br>

### Method `ultralytics.engine.results.Results.verbose` {#ultralytics.engine.results.Results.verbose}

```python
def verbose(self) -> str
```

Return a log string for each task in the results, detailing detection and classification outcomes.

This method generates a human-readable string summarizing the detection and classification results. It includes the number of detections for each class and the top probabilities for classification tasks.

**Returns**

| Type | Description |
| --- | --- |
| `str` | A formatted string containing a summary of the results. For detection tasks, it includes the number |

**Examples**

```python
>>> results = model("path/to/image.jpg")
>>> for result in results:
...     print(result.verbose())
2 persons, 1 car, 3 traffic lights,
dog 0.92, cat 0.78, horse 0.64,
```

!!! note "Notes"

    - If there are no detections, the method returns "(no detections), " for detection tasks.
    - For classification tasks, it returns the top 5 class probabilities and their corresponding class names.
    - The returned string is comma-separated and ends with a comma and a space.

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L630-L659"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def verbose(self) -> str:
    """Return a log string for each task in the results, detailing detection and classification outcomes.

    This method generates a human-readable string summarizing the detection and classification results. It includes
    the number of detections for each class and the top probabilities for classification tasks.

    Returns:
        (str): A formatted string containing a summary of the results. For detection tasks, it includes the number
            of detections per class. For classification tasks, it includes the top 5 class probabilities.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     print(result.verbose())
        2 persons, 1 car, 3 traffic lights,
        dog 0.92, cat 0.78, horse 0.64,

    Notes:
        - If there are no detections, the method returns "(no detections), " for detection tasks.
        - For classification tasks, it returns the top 5 class probabilities and their corresponding class names.
        - The returned string is comma-separated and ends with a comma and a space.
    """
    boxes = self.obb if self.obb is not None else self.boxes
    if len(self) == 0:
        return "" if self.probs is not None else "(no detections), "
    if self.probs is not None:
        return f"{', '.join(f'{self.names[j]} {self.probs.data[j]:.2f}' for j in self.probs.top5)}, "
    if boxes:
        counts = boxes.cls.int().bincount()
        return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts) if n > 0)
```
</details>


<br><br><hr><br>

## Class `ultralytics.engine.results.Boxes` {#ultralytics.engine.results.Boxes}

```python
Boxes(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None
```

**Bases:** `BaseTensor`

A class for managing and manipulating detection boxes.

This class provides comprehensive functionality for handling detection boxes, including their coordinates, confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipulation and conversion between different coordinate systems.

This class manages detection boxes, providing easy access and manipulation of box coordinates, confidence scores, class identifiers, and optional tracking IDs. It supports multiple formats for box coordinates, including both absolute and normalized forms.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `boxes` | `torch.Tensor | np.ndarray` | A tensor or numpy array with detection boxes of shape (num_boxes, 6) or<br>    (num_boxes, 7). Columns should contain [x1, y1, x2, y2, (optional) track_id, confidence, class]. | *required* |
| `orig_shape` | `tuple[int, int]` | The original image shape as (height, width). Used for normalization. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `torch.Tensor | np.ndarray` | The raw tensor containing detection boxes and associated data. |
| `orig_shape` | `tuple[int, int]` | The original image dimensions (height, width). |
| `is_track` | `bool` | Indicates whether tracking IDs are included in the box data. |
| `xyxy` | `torch.Tensor | np.ndarray` | Boxes in [x1, y1, x2, y2] format. |
| `conf` | `torch.Tensor | np.ndarray` | Confidence scores for each box. |
| `cls` | `torch.Tensor | np.ndarray` | Class labels for each box. |
| `id` | `torch.Tensor | None` | Tracking IDs for each box (if available). |
| `xywh` | `torch.Tensor | np.ndarray` | Boxes in [x, y, width, height] format. |
| `xyxyn` | `torch.Tensor | np.ndarray` | Normalized [x1, y1, x2, y2] boxes relative to orig_shape. |
| `xywhn` | `torch.Tensor | np.ndarray` | Normalized [x, y, width, height] boxes relative to orig_shape. |

**Methods**

| Name | Description |
| --- | --- |
| [`xyxy`](#ultralytics.engine.results.Boxes.xyxy) | Return bounding boxes in [x1, y1, x2, y2] format. |
| [`conf`](#ultralytics.engine.results.Boxes.conf) | Return the confidence scores for each detection box. |
| [`cls`](#ultralytics.engine.results.Boxes.cls) | Return the class ID tensor representing category predictions for each bounding box. |
| [`id`](#ultralytics.engine.results.Boxes.id) | Return the tracking IDs for each detection box if available. |
| [`xywh`](#ultralytics.engine.results.Boxes.xywh) | Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format. |
| [`xyxyn`](#ultralytics.engine.results.Boxes.xyxyn) | Return normalized bounding box coordinates relative to the original image size. |
| [`xywhn`](#ultralytics.engine.results.Boxes.xywhn) | Return normalized bounding boxes in [x, y, width, height] format. |

**Examples**

```python
>>> import torch
>>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
>>> orig_shape = (480, 640)  # height, width
>>> boxes = Boxes(boxes_data, orig_shape)
>>> print(boxes.xyxy)
>>> print(boxes.conf)
>>> print(boxes.cls)
>>> print(boxes.xywhn)
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L825-L1019"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Boxes(BaseTensor):
    """A class for managing and manipulating detection boxes.

    This class provides comprehensive functionality for handling detection boxes, including their coordinates,
    confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for
    easy manipulation and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | np.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | np.ndarray): Confidence scores for each box.
        cls (torch.Tensor | np.ndarray): Class labels for each box.
        id (torch.Tensor | None): Tracking IDs for each box (if available).
        xywh (torch.Tensor | np.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | np.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | np.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu: Return a copy of the object with all tensors on CPU memory.
        numpy: Return a copy of the object with all tensors as numpy arrays.
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates, confidence
        scores, class identifiers, and optional tracking IDs. It supports multiple formats for box coordinates,
        including both absolute and normalized forms.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6) or
                (num_boxes, 7). Columns should contain [x1, y1, x2, y2, (optional) track_id, confidence, class].
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape
```
</details>

<br>

### Property `ultralytics.engine.results.Boxes.xyxy` {#ultralytics.engine.results.Boxes.xyxy}

```python
def xyxy(self) -> torch.Tensor | np.ndarray
```

Return bounding boxes in [x1, y1, x2, y2] format.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor or numpy array of shape (n, 4) containing bounding box coordinates in |

**Examples**

```python
>>> results = model("image.jpg")
>>> boxes = results[0].boxes
>>> xyxy = boxes.xyxy
>>> print(xyxy)
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L882-L895"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def xyxy(self) -> torch.Tensor | np.ndarray:
    """Return bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or numpy array of shape (n, 4) containing bounding box coordinates in
            [x1, y1, x2, y2] format, where n is the number of boxes.

    Examples:
        >>> results = model("image.jpg")
        >>> boxes = results[0].boxes
        >>> xyxy = boxes.xyxy
        >>> print(xyxy)
    """
    return self.data[:, :4]
```
</details>

<br>

### Property `ultralytics.engine.results.Boxes.conf` {#ultralytics.engine.results.Boxes.conf}

```python
def conf(self) -> torch.Tensor | np.ndarray
```

Return the confidence scores for each detection box.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A 1D tensor or array containing confidence scores for each detection, with |

**Examples**

```python
>>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
>>> conf_scores = boxes.conf
>>> print(conf_scores)
tensor([0.9000])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L898-L911"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def conf(self) -> torch.Tensor | np.ndarray:
    """Return the confidence scores for each detection box.

    Returns:
        (torch.Tensor | np.ndarray): A 1D tensor or array containing confidence scores for each detection, with
            shape (N,) where N is the number of detections.

    Examples:
        >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
        >>> conf_scores = boxes.conf
        >>> print(conf_scores)
        tensor([0.9000])
    """
    return self.data[:, -2]
```
</details>

<br>

### Property `ultralytics.engine.results.Boxes.cls` {#ultralytics.engine.results.Boxes.cls}

```python
def cls(self) -> torch.Tensor | np.ndarray
```

Return the class ID tensor representing category predictions for each bounding box.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor or numpy array containing the class IDs for each detection box. The |

**Examples**

```python
>>> results = model("image.jpg")
>>> boxes = results[0].boxes
>>> class_ids = boxes.cls
>>> print(class_ids)  # tensor([0., 2., 1.])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L914-L927"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def cls(self) -> torch.Tensor | np.ndarray:
    """Return the class ID tensor representing category predictions for each bounding box.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or numpy array containing the class IDs for each detection box. The
            shape is (N,), where N is the number of boxes.

    Examples:
        >>> results = model("image.jpg")
        >>> boxes = results[0].boxes
        >>> class_ids = boxes.cls
        >>> print(class_ids)  # tensor([0., 2., 1.])
    """
    return self.data[:, -1]
```
</details>

<br>

### Property `ultralytics.engine.results.Boxes.id` {#ultralytics.engine.results.Boxes.id}

```python
def id(self) -> torch.Tensor | np.ndarray | None
```

Return the tracking IDs for each detection box if available.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray | None` | A tensor or array containing tracking IDs for each box if tracking is |

**Examples**

```python
>>> results = model.track("path/to/video.mp4")
>>> for result in results:
...     boxes = result.boxes
...     if boxes.is_track:
...         track_ids = boxes.id
...         print(f"Tracking IDs: {track_ids}")
...     else:
...         print("Tracking is not enabled for these boxes.")
```

!!! note "Notes"

    - This property is only available when tracking is enabled (i.e., when `is_track` is True).
    - The tracking IDs are typically used to associate detections across multiple frames in video analysis.

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L930-L951"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def id(self) -> torch.Tensor | np.ndarray | None:
    """Return the tracking IDs for each detection box if available.

    Returns:
        (torch.Tensor | np.ndarray | None): A tensor or array containing tracking IDs for each box if tracking is
            enabled, otherwise None. Shape is (N,) where N is the number of boxes.

    Examples:
        >>> results = model.track("path/to/video.mp4")
        >>> for result in results:
        ...     boxes = result.boxes
        ...     if boxes.is_track:
        ...         track_ids = boxes.id
        ...         print(f"Tracking IDs: {track_ids}")
        ...     else:
        ...         print("Tracking is not enabled for these boxes.")

    Notes:
        - This property is only available when tracking is enabled (i.e., when `is_track` is True).
        - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
    """
    return self.data[:, -3] if self.is_track else None
```
</details>

<br>

### Property `ultralytics.engine.results.Boxes.xywh` {#ultralytics.engine.results.Boxes.xywh}

```python
def xywh(self) -> torch.Tensor | np.ndarray
```

Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | Boxes in [x_center, y_center, width, height] format, where x_center, y_center |

**Examples**

```python
>>> boxes = Boxes(
...     torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]]), orig_shape=(480, 640)
... )
>>> xywh = boxes.xywh
>>> print(xywh)
tensor([[125.0000,  75.0000,  50.0000,  50.0000],
        [250.0000, 200.0000, 100.0000, 100.0000]])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L955-L972"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=2)
def xywh(self) -> torch.Tensor | np.ndarray:
    """Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

    Returns:
        (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center
            are the coordinates of the center point of the bounding box, width, height are the dimensions of the
            bounding box and the shape of the returned tensor is (N, 4), where N is the number of boxes.

    Examples:
        >>> boxes = Boxes(
        ...     torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]]), orig_shape=(480, 640)
        ... )
        >>> xywh = boxes.xywh
        >>> print(xywh)
        tensor([[125.0000,  75.0000,  50.0000,  50.0000],
                [250.0000, 200.0000, 100.0000, 100.0000]])
    """
    return ops.xyxy2xywh(self.xyxy)
```
</details>

<br>

### Property `ultralytics.engine.results.Boxes.xyxyn` {#ultralytics.engine.results.Boxes.xyxyn}

```python
def xyxyn(self) -> torch.Tensor | np.ndarray
```

Return normalized bounding box coordinates relative to the original image size.

This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format, normalized to the range [0, 1] based on the original image dimensions.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | Normalized bounding box coordinates with shape (N, 4), where N is the number of |

**Examples**

```python
>>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
>>> normalized = boxes.xyxyn
>>> print(normalized)
tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L976-L995"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=2)
def xyxyn(self) -> torch.Tensor | np.ndarray:
    """Return normalized bounding box coordinates relative to the original image size.

    This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format, normalized to the
    range [0, 1] based on the original image dimensions.

    Returns:
        (torch.Tensor | np.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is the number of
            boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].

    Examples:
        >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
        >>> normalized = boxes.xyxyn
        >>> print(normalized)
        tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
    """
    xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
    xyxy[..., [0, 2]] /= self.orig_shape[1]
    xyxy[..., [1, 3]] /= self.orig_shape[0]
    return xyxy
```
</details>

<br>

### Property `ultralytics.engine.results.Boxes.xywhn` {#ultralytics.engine.results.Boxes.xywhn}

```python
def xywhn(self) -> torch.Tensor | np.ndarray
```

Return normalized bounding boxes in [x, y, width, height] format.

This property calculates and returns the normalized bounding box coordinates in the format [x_center, y_center, width, height], where all values are relative to the original image dimensions.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | Normalized bounding boxes with shape (N, 4), where N is the number of boxes. |

**Examples**

```python
>>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
>>> normalized = boxes.xywhn
>>> print(normalized)
tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L999-L1019"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=2)
def xywhn(self) -> torch.Tensor | np.ndarray:
    """Return normalized bounding boxes in [x, y, width, height] format.

    This property calculates and returns the normalized bounding box coordinates in the format [x_center, y_center,
    width, height], where all values are relative to the original image dimensions.

    Returns:
        (torch.Tensor | np.ndarray): Normalized bounding boxes with shape (N, 4), where N is the number of boxes.
            Each row contains [x_center, y_center, width, height] values normalized to [0, 1] based on the original
            image dimensions.

    Examples:
        >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
        >>> normalized = boxes.xywhn
        >>> print(normalized)
        tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
    """
    xywh = ops.xyxy2xywh(self.xyxy)
    xywh[..., [0, 2]] /= self.orig_shape[1]
    xywh[..., [1, 3]] /= self.orig_shape[0]
    return xywh
```
</details>


<br><br><hr><br>

## Class `ultralytics.engine.results.Masks` {#ultralytics.engine.results.Masks}

```python
Masks(self, masks: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None
```

**Bases:** `BaseTensor`

A class for storing and manipulating detection masks.

This class extends BaseTensor and provides functionality for handling segmentation masks, including methods for converting between pixel and normalized coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `masks` | `torch.Tensor | np.ndarray` | Detection masks with shape (num_masks, height, width). | *required* |
| `orig_shape` | `tuple[int, int]` | The original image shape as (height, width). Used for normalization. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `torch.Tensor | np.ndarray` | The raw tensor or array containing mask data. |
| `orig_shape` | `tuple[int, int]` | Original image shape in (height, width) format. |
| `xy` | `list[np.ndarray]` | A list of segments in pixel coordinates. |
| `xyn` | `list[np.ndarray]` | A list of normalized segments. |

**Methods**

| Name | Description |
| --- | --- |
| [`xyn`](#ultralytics.engine.results.Masks.xyn) | Return normalized xy-coordinates of the segmentation masks. |
| [`xy`](#ultralytics.engine.results.Masks.xy) | Return the [x, y] pixel coordinates for each segment in the mask tensor. |

**Examples**

```python
>>> masks_data = torch.rand(1, 160, 160)
>>> orig_shape = (720, 1280)
>>> masks = Masks(masks_data, orig_shape)
>>> pixel_coords = masks.xy
>>> normalized_coords = masks.xyn
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1022-L1105"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Masks(BaseTensor):
    """A class for storing and manipulating detection masks.

    This class extends BaseTensor and provides functionality for handling segmentation masks, including methods for
    converting between pixel and normalized coordinates.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor or array containing mask data.
        orig_shape (tuple[int, int]): Original image shape in (height, width) format.
        xy (list[np.ndarray]): A list of segments in pixel coordinates.
        xyn (list[np.ndarray]): A list of normalized segments.

    Methods:
        cpu: Return a copy of the Masks object with the mask tensor on CPU memory.
        numpy: Return a copy of the Masks object with the mask tensor as a numpy array.
        cuda: Return a copy of the Masks object with the mask tensor on GPU memory.
        to: Return a copy of the Masks object with the mask tensor on specified device and dtype.

    Examples:
        >>> masks_data = torch.rand(1, 160, 160)
        >>> orig_shape = (720, 1280)
        >>> masks = Masks(masks_data, orig_shape)
        >>> pixel_coords = masks.xy
        >>> normalized_coords = masks.xyn
    """

    def __init__(self, masks: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize the Masks class with detection mask data and the original image shape.

        Args:
            masks (torch.Tensor | np.ndarray): Detection masks with shape (num_masks, height, width).
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization.
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)
```
</details>

<br>

### Property `ultralytics.engine.results.Masks.xyn` {#ultralytics.engine.results.Masks.xyn}

```python
def xyn(self) -> list[np.ndarray]
```

Return normalized xy-coordinates of the segmentation masks.

This property calculates and caches the normalized xy-coordinates of the segmentation masks. The coordinates are normalized relative to the original image shape.

**Returns**

| Type | Description |
| --- | --- |
| `list[np.ndarray]` | A list of numpy arrays, where each array contains the normalized xy-coordinates of a |

**Examples**

```python
>>> results = model("image.jpg")
>>> masks = results[0].masks
>>> normalized_coords = masks.xyn
>>> print(normalized_coords[0])  # Normalized coordinates of the first mask
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1061-L1081"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def xyn(self) -> list[np.ndarray]:
    """Return normalized xy-coordinates of the segmentation masks.

    This property calculates and caches the normalized xy-coordinates of the segmentation masks. The coordinates are
    normalized relative to the original image shape.

    Returns:
        (list[np.ndarray]): A list of numpy arrays, where each array contains the normalized xy-coordinates of a
            single segmentation mask. Each array has shape (N, 2), where N is the number of points in the
            mask contour.

    Examples:
        >>> results = model("image.jpg")
        >>> masks = results[0].masks
        >>> normalized_coords = masks.xyn
        >>> print(normalized_coords[0])  # Normalized coordinates of the first mask
    """
    return [
        ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
        for x in ops.masks2segments(self.data)
    ]
```
</details>

<br>

### Property `ultralytics.engine.results.Masks.xy` {#ultralytics.engine.results.Masks.xy}

```python
def xy(self) -> list[np.ndarray]
```

Return the [x, y] pixel coordinates for each segment in the mask tensor.

This property calculates and returns a list of pixel coordinates for each segmentation mask in the Masks object. The coordinates are scaled to match the original image dimensions.

**Returns**

| Type | Description |
| --- | --- |
| `list[np.ndarray]` | A list of numpy arrays, where each array contains the [x, y] pixel coordinates for a |

**Examples**

```python
>>> results = model("image.jpg")
>>> masks = results[0].masks
>>> xy_coords = masks.xy
>>> print(len(xy_coords))  # Number of masks
>>> print(xy_coords[0].shape)  # Shape of first mask's coordinates
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1085-L1105"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def xy(self) -> list[np.ndarray]:
    """Return the [x, y] pixel coordinates for each segment in the mask tensor.

    This property calculates and returns a list of pixel coordinates for each segmentation mask in the Masks object.
    The coordinates are scaled to match the original image dimensions.

    Returns:
        (list[np.ndarray]): A list of numpy arrays, where each array contains the [x, y] pixel coordinates for a
            single segmentation mask. Each array has shape (N, 2), where N is the number of points in the segment.

    Examples:
        >>> results = model("image.jpg")
        >>> masks = results[0].masks
        >>> xy_coords = masks.xy
        >>> print(len(xy_coords))  # Number of masks
        >>> print(xy_coords[0].shape)  # Shape of first mask's coordinates
    """
    return [
        ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
        for x in ops.masks2segments(self.data)
    ]
```
</details>


<br><br><hr><br>

## Class `ultralytics.engine.results.Keypoints` {#ultralytics.engine.results.Keypoints}

```python
Keypoints(self, keypoints: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None
```

**Bases:** `BaseTensor`

A class for storing and manipulating detection keypoints.

This class encapsulates functionality for handling keypoint data, including coordinate manipulation, normalization, and confidence values. It supports keypoint detection results with optional visibility information.

This method processes the input keypoints tensor, handling both 2D and 3D formats.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `keypoints` | `torch.Tensor | np.ndarray` | A tensor or array containing keypoint data. Shape can be either:<br>    - (num_objects, num_keypoints, 2) for x, y coordinates only<br>    - (num_objects, num_keypoints, 3) for x, y coordinates and confidence scores | *required* |
| `orig_shape` | `tuple[int, int]` | The original image dimensions (height, width). | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `torch.Tensor` | The raw tensor containing keypoint data. |
| `orig_shape` | `tuple[int, int]` | The original image dimensions (height, width). |
| `has_visible` | `bool` | Indicates whether visibility information is available for keypoints. |
| `xy` | `torch.Tensor` | Keypoint coordinates in [x, y] format. |
| `xyn` | `torch.Tensor` | Normalized keypoint coordinates in [x, y] format, relative to orig_shape. |
| `conf` | `torch.Tensor | None` | Confidence values for each keypoint, if available. |

**Methods**

| Name | Description |
| --- | --- |
| [`xy`](#ultralytics.engine.results.Keypoints.xy) | Return x, y coordinates of keypoints. |
| [`xyn`](#ultralytics.engine.results.Keypoints.xyn) | Return normalized coordinates (x, y) of keypoints relative to the original image size. |
| [`conf`](#ultralytics.engine.results.Keypoints.conf) | Return confidence values for each keypoint. |

**Examples**

```python
>>> import torch
>>> from ultralytics.engine.results import Keypoints
>>> keypoints_data = torch.rand(1, 17, 3)  # 1 detection, 17 keypoints, (x, y, conf)
>>> orig_shape = (480, 640)  # Original image shape (height, width)
>>> keypoints = Keypoints(keypoints_data, orig_shape)
>>> print(keypoints.xy.shape)  # Access xy coordinates
>>> print(keypoints.conf)  # Access confidence values
>>> keypoints_cpu = keypoints.cpu()  # Move keypoints to CPU
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1108-L1213"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Keypoints(BaseTensor):
    """A class for storing and manipulating detection keypoints.

    This class encapsulates functionality for handling keypoint data, including coordinate manipulation, normalization,
    and confidence values. It supports keypoint detection results with optional visibility information.

    Attributes:
        data (torch.Tensor): The raw tensor containing keypoint data.
        orig_shape (tuple[int, int]): The original image dimensions (height, width).
        has_visible (bool): Indicates whether visibility information is available for keypoints.
        xy (torch.Tensor): Keypoint coordinates in [x, y] format.
        xyn (torch.Tensor): Normalized keypoint coordinates in [x, y] format, relative to orig_shape.
        conf (torch.Tensor | None): Confidence values for each keypoint, if available.

    Methods:
        cpu: Return a copy of the keypoints tensor on CPU memory.
        numpy: Return a copy of the keypoints tensor as a numpy array.
        cuda: Return a copy of the keypoints tensor on GPU memory.
        to: Return a copy of the keypoints tensor with specified device and dtype.

    Examples:
        >>> import torch
        >>> from ultralytics.engine.results import Keypoints
        >>> keypoints_data = torch.rand(1, 17, 3)  # 1 detection, 17 keypoints, (x, y, conf)
        >>> orig_shape = (480, 640)  # Original image shape (height, width)
        >>> keypoints = Keypoints(keypoints_data, orig_shape)
        >>> print(keypoints.xy.shape)  # Access xy coordinates
        >>> print(keypoints.conf)  # Access confidence values
        >>> keypoints_cpu = keypoints.cpu()  # Move keypoints to CPU
    """

    def __init__(self, keypoints: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize the Keypoints object with detection keypoints and original image dimensions.

        This method processes the input keypoints tensor, handling both 2D and 3D formats.

        Args:
            keypoints (torch.Tensor | np.ndarray): A tensor or array containing keypoint data. Shape can be either:
                - (num_objects, num_keypoints, 2) for x, y coordinates only
                - (num_objects, num_keypoints, 3) for x, y coordinates and confidence scores
            orig_shape (tuple[int, int]): The original image dimensions (height, width).
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3
```
</details>

<br>

### Property `ultralytics.engine.results.Keypoints.xy` {#ultralytics.engine.results.Keypoints.xy}

```python
def xy(self) -> torch.Tensor | np.ndarray
```

Return x, y coordinates of keypoints.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor or array containing the x, y coordinates of keypoints with shape (N, |

**Examples**

```python
>>> results = model("image.jpg")
>>> keypoints = results[0].keypoints
>>> xy = keypoints.xy
>>> print(xy.shape)  # (N, K, 2)
>>> print(xy[0])  # x, y coordinates of keypoints for first detection
```

!!! note "Notes"

    - The returned coordinates are in pixel units relative to the original image dimensions.
    - This property uses LRU caching to improve performance on repeated access.

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1157-L1175"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def xy(self) -> torch.Tensor | np.ndarray:
    """Return x, y coordinates of keypoints.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or array containing the x, y coordinates of keypoints with shape (N,
            K, 2), where N is the number of detections and K is the number of keypoints per detection.

    Examples:
        >>> results = model("image.jpg")
        >>> keypoints = results[0].keypoints
        >>> xy = keypoints.xy
        >>> print(xy.shape)  # (N, K, 2)
        >>> print(xy[0])  # x, y coordinates of keypoints for first detection

    Notes:
        - The returned coordinates are in pixel units relative to the original image dimensions.
        - This property uses LRU caching to improve performance on repeated access.
    """
    return self.data[..., :2]
```
</details>

<br>

### Property `ultralytics.engine.results.Keypoints.xyn` {#ultralytics.engine.results.Keypoints.xyn}

```python
def xyn(self) -> torch.Tensor | np.ndarray
```

Return normalized coordinates (x, y) of keypoints relative to the original image size.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor or array of shape (N, K, 2) containing normalized keypoint |

**Examples**

```python
>>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
>>> normalized_kpts = keypoints.xyn
>>> print(normalized_kpts.shape)
torch.Size([1, 17, 2])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1179-L1196"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def xyn(self) -> torch.Tensor | np.ndarray:
    """Return normalized coordinates (x, y) of keypoints relative to the original image size.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or array of shape (N, K, 2) containing normalized keypoint
            coordinates, where N is the number of instances, K is the number of keypoints, and the last dimension
            contains [x, y] values in the range [0, 1].

    Examples:
        >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
        >>> normalized_kpts = keypoints.xyn
        >>> print(normalized_kpts.shape)
        torch.Size([1, 17, 2])
    """
    xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
    xy[..., 0] /= self.orig_shape[1]
    xy[..., 1] /= self.orig_shape[0]
    return xy
```
</details>

<br>

### Property `ultralytics.engine.results.Keypoints.conf` {#ultralytics.engine.results.Keypoints.conf}

```python
def conf(self) -> torch.Tensor | np.ndarray | None
```

Return confidence values for each keypoint.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray | None` | A tensor or array containing confidence scores for each keypoint if |

**Examples**

```python
>>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 detection, 17 keypoints
>>> conf = keypoints.conf
>>> print(conf.shape)  # torch.Size([1, 17])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1200-L1213"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def conf(self) -> torch.Tensor | np.ndarray | None:
    """Return confidence values for each keypoint.

    Returns:
        (torch.Tensor | np.ndarray | None): A tensor or array containing confidence scores for each keypoint if
            available, otherwise None. Shape is (num_detections, num_keypoints) for batched data or (num_keypoints,)
            for single detection.

    Examples:
        >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 detection, 17 keypoints
        >>> conf = keypoints.conf
        >>> print(conf.shape)  # torch.Size([1, 17])
    """
    return self.data[..., 2] if self.has_visible else None
```
</details>


<br><br><hr><br>

## Class `ultralytics.engine.results.Probs` {#ultralytics.engine.results.Probs}

```python
Probs(self, probs: torch.Tensor | np.ndarray, orig_shape: tuple[int, int] | None = None) -> None
```

**Bases:** `BaseTensor`

A class for storing and manipulating classification probabilities.

This class extends BaseTensor and provides methods for accessing and manipulating classification probabilities, including top-1 and top-5 predictions.

This class stores and manages classification probabilities, providing easy access to top predictions and their confidences.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `probs` | `torch.Tensor | np.ndarray` | A 1D tensor or array of classification probabilities. | *required* |
| `orig_shape` | `tuple[int, int] | None` | The original image shape as (height, width). Not used in this class but<br>    kept for consistency with other result classes. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `torch.Tensor | np.ndarray` | The raw tensor or array containing classification probabilities. |
| `orig_shape` | `tuple[int, int] | None` | The original image shape as (height, width). Not used in this class. |
| `top1` | `int` | Index of the class with the highest probability. |
| `top5` | `list[int]` | Indices of the top 5 classes by probability. |
| `top1conf` | `torch.Tensor | np.ndarray` | Confidence score of the top 1 class. |
| `top5conf` | `torch.Tensor | np.ndarray` | Confidence scores of the top 5 classes. |

**Methods**

| Name | Description |
| --- | --- |
| [`top1`](#ultralytics.engine.results.Probs.top1) | Return the index of the class with the highest probability. |
| [`top5`](#ultralytics.engine.results.Probs.top5) | Return the indices of the top 5 class probabilities. |
| [`top1conf`](#ultralytics.engine.results.Probs.top1conf) | Return the confidence score of the highest probability class. |
| [`top5conf`](#ultralytics.engine.results.Probs.top5conf) | Return confidence scores for the top 5 classification predictions. |

**Examples**

```python
>>> probs = torch.tensor([0.1, 0.3, 0.6])
>>> p = Probs(probs)
>>> print(p.top1)
2
>>> print(p.top5)
[2, 1, 0]
>>> print(p.top1conf)
tensor(0.6000)
>>> print(p.top5conf)
tensor([0.6000, 0.3000, 0.1000])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1216-L1330"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Probs(BaseTensor):
    """A class for storing and manipulating classification probabilities.

    This class extends BaseTensor and provides methods for accessing and manipulating classification probabilities,
    including top-1 and top-5 predictions.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor or array containing classification probabilities.
        orig_shape (tuple[int, int] | None): The original image shape as (height, width). Not used in this class.
        top1 (int): Index of the class with the highest probability.
        top5 (list[int]): Indices of the top 5 classes by probability.
        top1conf (torch.Tensor | np.ndarray): Confidence score of the top 1 class.
        top5conf (torch.Tensor | np.ndarray): Confidence scores of the top 5 classes.

    Methods:
        cpu: Return a copy of the probabilities tensor on CPU memory.
        numpy: Return a copy of the probabilities tensor as a numpy array.
        cuda: Return a copy of the probabilities tensor on GPU memory.
        to: Return a copy of the probabilities tensor with specified device and dtype.

    Examples:
        >>> probs = torch.tensor([0.1, 0.3, 0.6])
        >>> p = Probs(probs)
        >>> print(p.top1)
        2
        >>> print(p.top5)
        [2, 1, 0]
        >>> print(p.top1conf)
        tensor(0.6000)
        >>> print(p.top5conf)
        tensor([0.6000, 0.3000, 0.1000])
    """

    def __init__(self, probs: torch.Tensor | np.ndarray, orig_shape: tuple[int, int] | None = None) -> None:
        """Initialize the Probs class with classification probabilities.

        This class stores and manages classification probabilities, providing easy access to top predictions and their
        confidences.

        Args:
            probs (torch.Tensor | np.ndarray): A 1D tensor or array of classification probabilities.
            orig_shape (tuple[int, int] | None): The original image shape as (height, width). Not used in this class but
                kept for consistency with other result classes.
        """
        super().__init__(probs, orig_shape)
```
</details>

<br>

### Property `ultralytics.engine.results.Probs.top1` {#ultralytics.engine.results.Probs.top1}

```python
def top1(self) -> int
```

Return the index of the class with the highest probability.

**Returns**

| Type | Description |
| --- | --- |
| `int` | Index of the class with the highest probability. |

**Examples**

```python
>>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
>>> probs.top1
2
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1264-L1275"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def top1(self) -> int:
    """Return the index of the class with the highest probability.

    Returns:
        (int): Index of the class with the highest probability.

    Examples:
        >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
        >>> probs.top1
        2
    """
    return int(self.data.argmax())
```
</details>

<br>

### Property `ultralytics.engine.results.Probs.top5` {#ultralytics.engine.results.Probs.top5}

```python
def top5(self) -> list[int]
```

Return the indices of the top 5 class probabilities.

**Returns**

| Type | Description |
| --- | --- |
| `list[int]` | A list containing the indices of the top 5 class probabilities, sorted in descending order. |

**Examples**

```python
>>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
>>> print(probs.top5)
[4, 3, 2, 1, 0]
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1279-L1290"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def top5(self) -> list[int]:
    """Return the indices of the top 5 class probabilities.

    Returns:
        (list[int]): A list containing the indices of the top 5 class probabilities, sorted in descending order.

    Examples:
        >>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
        >>> print(probs.top5)
        [4, 3, 2, 1, 0]
    """
    return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.
```
</details>

<br>

### Property `ultralytics.engine.results.Probs.top1conf` {#ultralytics.engine.results.Probs.top1conf}

```python
def top1conf(self) -> torch.Tensor | np.ndarray
```

Return the confidence score of the highest probability class.

This property retrieves the confidence score (probability) of the class with the highest predicted probability from the classification results.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor containing the confidence score of the top 1 class. |

**Examples**

```python
>>> results = model("image.jpg")  # classify an image
>>> probs = results[0].probs  # get classification probabilities
>>> top1_confidence = probs.top1conf  # get confidence of top 1 class
>>> print(f"Top 1 class confidence: {top1_confidence.item():.4f}")
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1294-L1309"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def top1conf(self) -> torch.Tensor | np.ndarray:
    """Return the confidence score of the highest probability class.

    This property retrieves the confidence score (probability) of the class with the highest predicted probability
    from the classification results.

    Returns:
        (torch.Tensor | np.ndarray): A tensor containing the confidence score of the top 1 class.

    Examples:
        >>> results = model("image.jpg")  # classify an image
        >>> probs = results[0].probs  # get classification probabilities
        >>> top1_confidence = probs.top1conf  # get confidence of top 1 class
        >>> print(f"Top 1 class confidence: {top1_confidence.item():.4f}")
    """
    return self.data[self.top1]
```
</details>

<br>

### Property `ultralytics.engine.results.Probs.top5conf` {#ultralytics.engine.results.Probs.top5conf}

```python
def top5conf(self) -> torch.Tensor | np.ndarray
```

Return confidence scores for the top 5 classification predictions.

This property retrieves the confidence scores corresponding to the top 5 class probabilities predicted by the model. It provides a quick way to access the most likely class predictions along with their associated confidence levels.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor or array containing the confidence scores for the top 5 predicted |

**Examples**

```python
>>> results = model("image.jpg")
>>> probs = results[0].probs
>>> top5_conf = probs.top5conf
>>> print(top5_conf)  # Prints confidence scores for top 5 classes
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1313-L1330"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=1)
def top5conf(self) -> torch.Tensor | np.ndarray:
    """Return confidence scores for the top 5 classification predictions.

    This property retrieves the confidence scores corresponding to the top 5 class probabilities predicted by the
    model. It provides a quick way to access the most likely class predictions along with their associated
    confidence levels.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or array containing the confidence scores for the top 5 predicted
            classes, sorted in descending order of probability.

    Examples:
        >>> results = model("image.jpg")
        >>> probs = results[0].probs
        >>> top5_conf = probs.top5conf
        >>> print(top5_conf)  # Prints confidence scores for top 5 classes
    """
    return self.data[self.top5]
```
</details>


<br><br><hr><br>

## Class `ultralytics.engine.results.OBB` {#ultralytics.engine.results.OBB}

```python
OBB(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None
```

**Bases:** `BaseTensor`

A class for storing and manipulating Oriented Bounding Boxes (OBB).

This class provides functionality to handle oriented bounding boxes, including conversion between different formats, normalization, and access to various properties of the boxes. It supports both tracking and non-tracking scenarios.

This class stores and manipulates Oriented Bounding Boxes (OBB) for object detection tasks. It provides various properties and methods to access and transform the OBB data.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `boxes` | `torch.Tensor | np.ndarray` | A tensor or numpy array containing the detection boxes, with shape<br>    (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values. If present,<br>    the third last column contains track IDs, and the fifth column contains rotation. | *required* |
| `orig_shape` | `tuple[int, int]` | Original image size, in the format (height, width). | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `torch.Tensor` | The raw OBB tensor containing box coordinates and associated data. |
| `orig_shape` | `tuple[int, int]` | Original image size as (height, width). |
| `is_track` | `bool` | Indicates whether tracking IDs are included in the box data. |
| `xywhr` | `torch.Tensor | np.ndarray` | Boxes in [x_center, y_center, width, height, rotation] format. |
| `conf` | `torch.Tensor | np.ndarray` | Confidence scores for each box. |
| `cls` | `torch.Tensor | np.ndarray` | Class labels for each box. |
| `id` | `torch.Tensor | np.ndarray` | Tracking IDs for each box, if available. |
| `xyxyxyxy` | `torch.Tensor | np.ndarray` | Boxes in 8-point [x1, y1, x2, y2, x3, y3, x4, y4] format. |
| `xyxyxyxyn` | `torch.Tensor | np.ndarray` | Normalized 8-point coordinates relative to orig_shape. |
| `xyxy` | `torch.Tensor | np.ndarray` | Axis-aligned bounding boxes in [x1, y1, x2, y2] format. |

**Methods**

| Name | Description |
| --- | --- |
| [`xywhr`](#ultralytics.engine.results.OBB.xywhr) | Return boxes in [x_center, y_center, width, height, rotation] format. |
| [`conf`](#ultralytics.engine.results.OBB.conf) | Return the confidence scores for Oriented Bounding Boxes (OBBs). |
| [`cls`](#ultralytics.engine.results.OBB.cls) | Return the class values of the oriented bounding boxes. |
| [`id`](#ultralytics.engine.results.OBB.id) | Return the tracking IDs of the oriented bounding boxes (if available). |
| [`xyxyxyxy`](#ultralytics.engine.results.OBB.xyxyxyxy) | Convert OBB format to 8-point (xyxyxyxy) coordinate format for rotated bounding boxes. |
| [`xyxyxyxyn`](#ultralytics.engine.results.OBB.xyxyxyxyn) | Convert rotated bounding boxes to normalized xyxyxyxy format. |
| [`xyxy`](#ultralytics.engine.results.OBB.xyxy) | Convert oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format. |

**Examples**

```python
>>> boxes = torch.tensor([[100, 50, 150, 100, 30, 0.9, 0]])  # xywhr, conf, cls
>>> obb = OBB(boxes, orig_shape=(480, 640))
>>> print(obb.xyxyxyxy)
>>> print(obb.conf)
>>> print(obb.cls)
```

**Raises**

| Type | Description |
| --- | --- |
| `AssertionError` | If the number of values per box is not 7 or 8. |

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1333-L1533"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class OBB(BaseTensor):
    """A class for storing and manipulating Oriented Bounding Boxes (OBB).

    This class provides functionality to handle oriented bounding boxes, including conversion between different formats,
    normalization, and access to various properties of the boxes. It supports both tracking and non-tracking scenarios.

    Attributes:
        data (torch.Tensor): The raw OBB tensor containing box coordinates and associated data.
        orig_shape (tuple[int, int]): Original image size as (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xywhr (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height, rotation] format.
        conf (torch.Tensor | np.ndarray): Confidence scores for each box.
        cls (torch.Tensor | np.ndarray): Class labels for each box.
        id (torch.Tensor | np.ndarray): Tracking IDs for each box, if available.
        xyxyxyxy (torch.Tensor | np.ndarray): Boxes in 8-point [x1, y1, x2, y2, x3, y3, x4, y4] format.
        xyxyxyxyn (torch.Tensor | np.ndarray): Normalized 8-point coordinates relative to orig_shape.
        xyxy (torch.Tensor | np.ndarray): Axis-aligned bounding boxes in [x1, y1, x2, y2] format.

    Methods:
        cpu: Return a copy of the OBB object with all tensors on CPU memory.
        numpy: Return a copy of the OBB object with all tensors as numpy arrays.
        cuda: Return a copy of the OBB object with all tensors on GPU memory.
        to: Return a copy of the OBB object with tensors on specified device and dtype.

    Examples:
        >>> boxes = torch.tensor([[100, 50, 150, 100, 30, 0.9, 0]])  # xywhr, conf, cls
        >>> obb = OBB(boxes, orig_shape=(480, 640))
        >>> print(obb.xyxyxyxy)
        >>> print(obb.conf)
        >>> print(obb.cls)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize an OBB (Oriented Bounding Box) instance with oriented bounding box data and original image shape.

        This class stores and manipulates Oriented Bounding Boxes (OBB) for object detection tasks. It provides various
        properties and methods to access and transform the OBB data.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array containing the detection boxes, with shape
                (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values. If present,
                the third last column contains track IDs, and the fifth column contains rotation.
            orig_shape (tuple[int, int]): Original image size, in the format (height, width).

        Raises:
            AssertionError: If the number of values per box is not 7 or 8.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape
```
</details>

<br>

### Property `ultralytics.engine.results.OBB.xywhr` {#ultralytics.engine.results.OBB.xywhr}

```python
def xywhr(self) -> torch.Tensor | np.ndarray
```

Return boxes in [x_center, y_center, width, height, rotation] format.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor or numpy array containing the oriented bounding boxes with format |

**Examples**

```python
>>> results = model("image.jpg")
>>> obb = results[0].obb
>>> xywhr = obb.xywhr
>>> print(xywhr.shape)
torch.Size([3, 5])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1389-L1403"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def xywhr(self) -> torch.Tensor | np.ndarray:
    """Return boxes in [x_center, y_center, width, height, rotation] format.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or numpy array containing the oriented bounding boxes with format
            [x_center, y_center, width, height, rotation]. The shape is (N, 5) where N is the number of boxes.

    Examples:
        >>> results = model("image.jpg")
        >>> obb = results[0].obb
        >>> xywhr = obb.xywhr
        >>> print(xywhr.shape)
        torch.Size([3, 5])
    """
    return self.data[:, :5]
```
</details>

<br>

### Property `ultralytics.engine.results.OBB.conf` {#ultralytics.engine.results.OBB.conf}

```python
def conf(self) -> torch.Tensor | np.ndarray
```

Return the confidence scores for Oriented Bounding Boxes (OBBs).

This property retrieves the confidence values associated with each OBB detection. The confidence score represents the model's certainty in the detection.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor or numpy array of shape (N,) containing confidence scores for N |

**Examples**

```python
>>> results = model("image.jpg")
>>> obb_result = results[0].obb
>>> confidence_scores = obb_result.conf
>>> print(confidence_scores)
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1406-L1422"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def conf(self) -> torch.Tensor | np.ndarray:
    """Return the confidence scores for Oriented Bounding Boxes (OBBs).

    This property retrieves the confidence values associated with each OBB detection. The confidence score
    represents the model's certainty in the detection.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or numpy array of shape (N,) containing confidence scores for N
            detections, where each score is in the range [0, 1].

    Examples:
        >>> results = model("image.jpg")
        >>> obb_result = results[0].obb
        >>> confidence_scores = obb_result.conf
        >>> print(confidence_scores)
    """
    return self.data[:, -2]
```
</details>

<br>

### Property `ultralytics.engine.results.OBB.cls` {#ultralytics.engine.results.OBB.cls}

```python
def cls(self) -> torch.Tensor | np.ndarray
```

Return the class values of the oriented bounding boxes.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | A tensor or numpy array containing the class values for each oriented bounding |

**Examples**

```python
>>> results = model("image.jpg")
>>> result = results[0]
>>> obb = result.obb
>>> class_values = obb.cls
>>> print(class_values)
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1425-L1439"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def cls(self) -> torch.Tensor | np.ndarray:
    """Return the class values of the oriented bounding boxes.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or numpy array containing the class values for each oriented bounding
            box. The shape is (N,), where N is the number of boxes.

    Examples:
        >>> results = model("image.jpg")
        >>> result = results[0]
        >>> obb = result.obb
        >>> class_values = obb.cls
        >>> print(class_values)
    """
    return self.data[:, -1]
```
</details>

<br>

### Property `ultralytics.engine.results.OBB.id` {#ultralytics.engine.results.OBB.id}

```python
def id(self) -> torch.Tensor | np.ndarray | None
```

Return the tracking IDs of the oriented bounding boxes (if available).

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray | None` | A tensor or numpy array containing the tracking IDs for each oriented |

**Examples**

```python
>>> results = model("image.jpg", tracker=True)  # Run inference with tracking
>>> for result in results:
...     if result.obb is not None:
...         track_ids = result.obb.id
...         if track_ids is not None:
...             print(f"Tracking IDs: {track_ids}")
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1442-L1457"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def id(self) -> torch.Tensor | np.ndarray | None:
    """Return the tracking IDs of the oriented bounding boxes (if available).

    Returns:
        (torch.Tensor | np.ndarray | None): A tensor or numpy array containing the tracking IDs for each oriented
            bounding box. Returns None if tracking IDs are not available.

    Examples:
        >>> results = model("image.jpg", tracker=True)  # Run inference with tracking
        >>> for result in results:
        ...     if result.obb is not None:
        ...         track_ids = result.obb.id
        ...         if track_ids is not None:
        ...             print(f"Tracking IDs: {track_ids}")
    """
    return self.data[:, -3] if self.is_track else None
```
</details>

<br>

### Property `ultralytics.engine.results.OBB.xyxyxyxy` {#ultralytics.engine.results.OBB.xyxyxyxy}

```python
def xyxyxyxy(self) -> torch.Tensor | np.ndarray
```

Convert OBB format to 8-point (xyxyxyxy) coordinate format for rotated bounding boxes.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | Rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2), where N is the |

**Examples**

```python
>>> obb = OBB(torch.tensor([[100, 100, 50, 30, 0.5, 0.9, 0]]), orig_shape=(640, 640))
>>> xyxyxyxy = obb.xyxyxyxy
>>> print(xyxyxyxy.shape)
torch.Size([1, 4, 2])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1461-L1475"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=2)
def xyxyxyxy(self) -> torch.Tensor | np.ndarray:
    """Convert OBB format to 8-point (xyxyxyxy) coordinate format for rotated bounding boxes.

    Returns:
        (torch.Tensor | np.ndarray): Rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2), where N is the
            number of boxes. Each box is represented by 4 points (x, y), starting from the top-left corner and
            moving clockwise.

    Examples:
        >>> obb = OBB(torch.tensor([[100, 100, 50, 30, 0.5, 0.9, 0]]), orig_shape=(640, 640))
        >>> xyxyxyxy = obb.xyxyxyxy
        >>> print(xyxyxyxy.shape)
        torch.Size([1, 4, 2])
    """
    return ops.xywhr2xyxyxyxy(self.xywhr)
```
</details>

<br>

### Property `ultralytics.engine.results.OBB.xyxyxyxyn` {#ultralytics.engine.results.OBB.xyxyxyxyn}

```python
def xyxyxyxyn(self) -> torch.Tensor | np.ndarray
```

Convert rotated bounding boxes to normalized xyxyxyxy format.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | Normalized rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2), |

**Examples**

```python
>>> obb = OBB(torch.rand(10, 7), orig_shape=(640, 480))  # 10 random OBBs
>>> normalized_boxes = obb.xyxyxyxyn
>>> print(normalized_boxes.shape)
torch.Size([10, 4, 2])
```

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1479-L1496"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=2)
def xyxyxyxyn(self) -> torch.Tensor | np.ndarray:
    """Convert rotated bounding boxes to normalized xyxyxyxy format.

    Returns:
        (torch.Tensor | np.ndarray): Normalized rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2),
            where N is the number of boxes. Each box is represented by 4 points (x, y), normalized relative to the
            original image dimensions.

    Examples:
        >>> obb = OBB(torch.rand(10, 7), orig_shape=(640, 480))  # 10 random OBBs
        >>> normalized_boxes = obb.xyxyxyxyn
        >>> print(normalized_boxes.shape)
        torch.Size([10, 4, 2])
    """
    xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
    xyxyxyxyn[..., 0] /= self.orig_shape[1]
    xyxyxyxyn[..., 1] /= self.orig_shape[0]
    return xyxyxyxyn
```
</details>

<br>

### Property `ultralytics.engine.results.OBB.xyxy` {#ultralytics.engine.results.OBB.xyxy}

```python
def xyxy(self) -> torch.Tensor | np.ndarray
```

Convert oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format.

This property calculates the minimal enclosing rectangle for each oriented bounding box and returns it in xyxy format (x1, y1, x2, y2). This is useful for operations that require axis-aligned bounding boxes, such as IoU calculation with non-rotated boxes.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | Axis-aligned bounding boxes in xyxy format with shape (N, 4), where N is the |

**Examples**

```python
>>> import torch
>>> from ultralytics import YOLO
>>> model = YOLO("yolo26n-obb.pt")
>>> results = model("path/to/image.jpg")
>>> for result in results:
...     obb = result.obb
...     if obb is not None:
...         xyxy_boxes = obb.xyxy
...         print(xyxy_boxes.shape)  # (N, 4)
```

!!! note "Notes"

    - This method approximates the OBB by its minimal enclosing rectangle.
    - The returned format is compatible with standard object detection metrics and visualization tools.
    - The property uses caching to improve performance for repeated access.

<details>
<summary>Source code in <code>ultralytics/engine/results.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L1500-L1533"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
@lru_cache(maxsize=2)
def xyxy(self) -> torch.Tensor | np.ndarray:
    """Convert oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format.

    This property calculates the minimal enclosing rectangle for each oriented bounding box and returns it in xyxy
    format (x1, y1, x2, y2). This is useful for operations that require axis-aligned bounding boxes, such as IoU
    calculation with non-rotated boxes.

    Returns:
        (torch.Tensor | np.ndarray): Axis-aligned bounding boxes in xyxy format with shape (N, 4), where N is the
            number of boxes. Each row contains [x1, y1, x2, y2] coordinates.

    Examples:
        >>> import torch
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n-obb.pt")
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     obb = result.obb
        ...     if obb is not None:
        ...         xyxy_boxes = obb.xyxy
        ...         print(xyxy_boxes.shape)  # (N, 4)

    Notes:
        - This method approximates the OBB by its minimal enclosing rectangle.
        - The returned format is compatible with standard object detection metrics and visualization tools.
        - The property uses caching to improve performance for repeated access.
    """
    x = self.xyxyxyxy[..., 0]
    y = self.xyxyxyxy[..., 1]
    return (
        torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
        if isinstance(x, torch.Tensor)
        else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
    )
```
</details>

<br><br>
