---
description: Explore Ultralytics utilities for bounding boxes and instances, providing detailed documentation on handling bbox formats, conversions, and more.
keywords: Ultralytics, bounding boxes, Instances, bbox formats, conversions, AI, deep learning, YOLO, xyxy, xywh, ltwh
---

# Reference for `ultralytics/utils/instance.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`Bboxes`](#ultralytics.utils.instance.Bboxes)
        - [`Instances`](#ultralytics.utils.instance.Instances)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`Instances.bbox_areas`](#ultralytics.utils.instance.Instances.bbox_areas)
        - [`Instances.bboxes`](#ultralytics.utils.instance.Instances.bboxes)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`Bboxes.convert`](#ultralytics.utils.instance.Bboxes.convert)
        - [`Bboxes.areas`](#ultralytics.utils.instance.Bboxes.areas)
        - [`Bboxes.mul`](#ultralytics.utils.instance.Bboxes.mul)
        - [`Bboxes.add`](#ultralytics.utils.instance.Bboxes.add)
        - [`Bboxes.__len__`](#ultralytics.utils.instance.Bboxes.__len__)
        - [`Bboxes.concatenate`](#ultralytics.utils.instance.Bboxes.concatenate)
        - [`Bboxes.__getitem__`](#ultralytics.utils.instance.Bboxes.__getitem__)
        - [`Instances.convert_bbox`](#ultralytics.utils.instance.Instances.convert_bbox)
        - [`Instances.scale`](#ultralytics.utils.instance.Instances.scale)
        - [`Instances.denormalize`](#ultralytics.utils.instance.Instances.denormalize)
        - [`Instances.normalize`](#ultralytics.utils.instance.Instances.normalize)
        - [`Instances.add_padding`](#ultralytics.utils.instance.Instances.add_padding)
        - [`Instances.__getitem__`](#ultralytics.utils.instance.Instances.__getitem__)
        - [`Instances.flipud`](#ultralytics.utils.instance.Instances.flipud)
        - [`Instances.fliplr`](#ultralytics.utils.instance.Instances.fliplr)
        - [`Instances.clip`](#ultralytics.utils.instance.Instances.clip)
        - [`Instances.remove_zero_area_boxes`](#ultralytics.utils.instance.Instances.remove_zero_area_boxes)
        - [`Instances.update`](#ultralytics.utils.instance.Instances.update)
        - [`Instances.__len__`](#ultralytics.utils.instance.Instances.__len__)
        - [`Instances.concatenate`](#ultralytics.utils.instance.Instances.concatenate)
        - [`Instances.__repr__`](#ultralytics.utils.instance.Instances.__repr__)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`_ntuple`](#ultralytics.utils.instance._ntuple)


## Class `ultralytics.utils.instance.Bboxes` {#ultralytics.utils.instance.Bboxes}

```python
Bboxes(self, bboxes: np.ndarray, format: str = "xyxy") -> None
```

A class for handling bounding boxes in multiple formats.

The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh' and provides methods for format conversion, scaling, and area calculation. Bounding box data should be provided as numpy arrays.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `bboxes` | `np.ndarray` | Array of bounding boxes with shape (N, 4) or (4,). | *required* |
| `format` | `str` | Format of the bounding boxes, one of 'xyxy', 'xywh', or 'ltwh'. | `"xyxy"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `bboxes` | `np.ndarray` | The bounding boxes stored in a 2D numpy array with shape (N, 4). |
| `format` | `str` | The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh'). |

**Methods**

| Name | Description |
| --- | --- |
| [`__getitem__`](#ultralytics.utils.instance.Bboxes.__getitem__) | Retrieve a specific bounding box or a set of bounding boxes using indexing. |
| [`__len__`](#ultralytics.utils.instance.Bboxes.__len__) | Return the number of bounding boxes. |
| [`add`](#ultralytics.utils.instance.Bboxes.add) | Add offset to bounding box coordinates. |
| [`areas`](#ultralytics.utils.instance.Bboxes.areas) | Calculate the area of bounding boxes. |
| [`concatenate`](#ultralytics.utils.instance.Bboxes.concatenate) | Concatenate a list of Bboxes objects into a single Bboxes object. |
| [`convert`](#ultralytics.utils.instance.Bboxes.convert) | Convert bounding box format from one type to another. |
| [`mul`](#ultralytics.utils.instance.Bboxes.mul) | Multiply bounding box coordinates by scale factor(s). |

**Examples**

```python
Create bounding boxes in YOLO format
>>> bboxes = Bboxes(np.array([[100, 50, 150, 100]]), format="xywh")
>>> bboxes.convert("xyxy")
>>> print(bboxes.areas())
```

!!! note "Notes"

    This class does not handle normalization or denormalization of bounding boxes.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L35-L178"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Bboxes:
    """A class for handling bounding boxes in multiple formats.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh' and provides methods for format
    conversion, scaling, and area calculation. Bounding box data should be provided as numpy arrays.

    Attributes:
        bboxes (np.ndarray): The bounding boxes stored in a 2D numpy array with shape (N, 4).
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Methods:
        convert: Convert bounding box format from one type to another.
        areas: Calculate the area of bounding boxes.
        mul: Multiply bounding box coordinates by scale factor(s).
        add: Add offset to bounding box coordinates.
        concatenate: Concatenate multiple Bboxes objects.

    Examples:
        Create bounding boxes in YOLO format
        >>> bboxes = Bboxes(np.array([[100, 50, 150, 100]]), format="xywh")
        >>> bboxes.convert("xyxy")
        >>> print(bboxes.areas())

    Notes:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes: np.ndarray, format: str = "xyxy") -> None:
        """Initialize the Bboxes class with bounding box data in a specified format.

        Args:
            bboxes (np.ndarray): Array of bounding boxes with shape (N, 4) or (4,).
            format (str): Format of the bounding boxes, one of 'xyxy', 'xywh', or 'ltwh'.
        """
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
```
</details>

<br>

### Method `ultralytics.utils.instance.Bboxes.__getitem__` {#ultralytics.utils.instance.Bboxes.\_\_getitem\_\_}

```python
def __getitem__(self, index: int | np.ndarray | slice) -> Bboxes
```

Retrieve a specific bounding box or a set of bounding boxes using indexing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `index` | `int | slice | np.ndarray` | The index, slice, or boolean array to select the desired bounding boxes. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Bboxes` | A new Bboxes object containing the selected bounding boxes. |

!!! note "Notes"

    When using boolean indexing, make sure to provide a boolean array with the same length as the number of
    bounding boxes.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L161-L178"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __getitem__(self, index: int | np.ndarray | slice) -> Bboxes:
    """Retrieve a specific bounding box or a set of bounding boxes using indexing.

    Args:
        index (int | slice | np.ndarray): The index, slice, or boolean array to select the desired bounding boxes.

    Returns:
        (Bboxes): A new Bboxes object containing the selected bounding boxes.

    Notes:
        When using boolean indexing, make sure to provide a boolean array with the same length as the number of
        bounding boxes.
    """
    if isinstance(index, int):
        return Bboxes(self.bboxes[index].reshape(1, -1))
    b = self.bboxes[index]
    assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
    return Bboxes(b)
```
</details>

<br>

### Method `ultralytics.utils.instance.Bboxes.__len__` {#ultralytics.utils.instance.Bboxes.\_\_len\_\_}

```python
def __len__(self) -> int
```

Return the number of bounding boxes.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L134-L136"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __len__(self) -> int:
    """Return the number of bounding boxes."""
    return len(self.bboxes)
```
</details>

<br>

### Method `ultralytics.utils.instance.Bboxes.add` {#ultralytics.utils.instance.Bboxes.add}

```python
def add(self, offset: int | tuple | list) -> None
```

Add offset to bounding box coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `offset` | `int | tuple | list` | Offset(s) for four coordinates. If int, the same offset is applied to all<br>    coordinates. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L118-L132"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def add(self, offset: int | tuple | list) -> None:
    """Add offset to bounding box coordinates.

    Args:
        offset (int | tuple | list): Offset(s) for four coordinates. If int, the same offset is applied to all
            coordinates.
    """
    if isinstance(offset, Number):
        offset = to_4tuple(offset)
    assert isinstance(offset, (tuple, list))
    assert len(offset) == 4
    self.bboxes[:, 0] += offset[0]
    self.bboxes[:, 1] += offset[1]
    self.bboxes[:, 2] += offset[2]
    self.bboxes[:, 3] += offset[3]
```
</details>

<br>

### Method `ultralytics.utils.instance.Bboxes.areas` {#ultralytics.utils.instance.Bboxes.areas}

```python
def areas(self) -> np.ndarray
```

Calculate the area of bounding boxes.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L94-L100"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def areas(self) -> np.ndarray:
    """Calculate the area of bounding boxes."""
    return (
        (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
        if self.format == "xyxy"
        else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
    )
```
</details>

<br>

### Method `ultralytics.utils.instance.Bboxes.concatenate` {#ultralytics.utils.instance.Bboxes.concatenate}

```python
def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes
```

Concatenate a list of Bboxes objects into a single Bboxes object.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `boxes_list` | `list[Bboxes]` | A list of Bboxes objects to concatenate. | *required* |
| `axis` | `int, optional` | The axis along which to concatenate the bounding boxes. | `0` |

**Returns**

| Type | Description |
| --- | --- |
| `Bboxes` | A new Bboxes object containing the concatenated bounding boxes. |

!!! note "Notes"

    The input should be a list or tuple of Bboxes objects.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L139-L159"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@classmethod
def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes:
    """Concatenate a list of Bboxes objects into a single Bboxes object.

    Args:
        boxes_list (list[Bboxes]): A list of Bboxes objects to concatenate.
        axis (int, optional): The axis along which to concatenate the bounding boxes.

    Returns:
        (Bboxes): A new Bboxes object containing the concatenated bounding boxes.

    Notes:
        The input should be a list or tuple of Bboxes objects.
    """
    assert isinstance(boxes_list, (list, tuple))
    if not boxes_list:
        return cls(np.empty(0))
    assert all(isinstance(box, Bboxes) for box in boxes_list)

    if len(boxes_list) == 1:
        return boxes_list[0]
    return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))
```
</details>

<br>

### Method `ultralytics.utils.instance.Bboxes.convert` {#ultralytics.utils.instance.Bboxes.convert}

```python
def convert(self, format: str) -> None
```

Convert bounding box format from one type to another.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `format` | `str` | Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L76-L92"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def convert(self, format: str) -> None:
    """Convert bounding box format from one type to another.

    Args:
        format (str): Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'.
    """
    assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
    if self.format == format:
        return
    elif self.format == "xyxy":
        func = xyxy2xywh if format == "xywh" else xyxy2ltwh
    elif self.format == "xywh":
        func = xywh2xyxy if format == "xyxy" else xywh2ltwh
    else:
        func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
    self.bboxes = func(self.bboxes)
    self.format = format
```
</details>

<br>

### Method `ultralytics.utils.instance.Bboxes.mul` {#ultralytics.utils.instance.Bboxes.mul}

```python
def mul(self, scale: int | tuple | list) -> None
```

Multiply bounding box coordinates by scale factor(s).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `scale` | `int | tuple | list` | Scale factor(s) for four coordinates. If int, the same scale is applied to all<br>    coordinates. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L102-L116"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def mul(self, scale: int | tuple | list) -> None:
    """Multiply bounding box coordinates by scale factor(s).

    Args:
        scale (int | tuple | list): Scale factor(s) for four coordinates. If int, the same scale is applied to all
            coordinates.
    """
    if isinstance(scale, Number):
        scale = to_4tuple(scale)
    assert isinstance(scale, (tuple, list))
    assert len(scale) == 4
    self.bboxes[:, 0] *= scale[0]
    self.bboxes[:, 1] *= scale[1]
    self.bboxes[:, 2] *= scale[2]
    self.bboxes[:, 3] *= scale[3]
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.instance.Instances` {#ultralytics.utils.instance.Instances}

```python
def __init__(
    self,
    bboxes: np.ndarray,
    segments: np.ndarray = None,
    keypoints: np.ndarray = None,
    bbox_format: str = "xywh",
    normalized: bool = True,
) -> None
```

Container for bounding boxes, segments, and keypoints of detected objects in an image.

This class provides a unified interface for handling different types of object annotations including bounding boxes, segmentation masks, and keypoints. It supports various operations like scaling, normalization, clipping, and format conversion.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `bboxes` | `np.ndarray` | Bounding boxes with shape (N, 4). | *required* |
| `segments` | `np.ndarray, optional` | Segmentation masks. | `None` |
| `keypoints` | `np.ndarray, optional` | Keypoints with shape (N, 17, 3) in format (x, y, visible). | `None` |
| `bbox_format` | `str` | Format of bboxes. | `"xywh"` |
| `normalized` | `bool` | Whether the coordinates are normalized. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `_bboxes` | `Bboxes` | Internal object for handling bounding box operations. |
| `keypoints` | `np.ndarray` | Keypoints with shape (N, 17, 3) in format (x, y, visible). |
| `normalized` | `bool` | Flag indicating whether the bounding box coordinates are normalized. |
| `segments` | `np.ndarray` | Segments array with shape (N, M, 2) after resampling. |

**Methods**

| Name | Description |
| --- | --- |
| [`bbox_areas`](#ultralytics.utils.instance.Instances.bbox_areas) | Calculate the area of bounding boxes. |
| [`bboxes`](#ultralytics.utils.instance.Instances.bboxes) | Return bounding boxes. |
| [`__getitem__`](#ultralytics.utils.instance.Instances.__getitem__) | Retrieve a specific instance or a set of instances using indexing. |
| [`__len__`](#ultralytics.utils.instance.Instances.__len__) | Return the number of instances. |
| [`__repr__`](#ultralytics.utils.instance.Instances.__repr__) | Return a string representation of the Instances object. |
| [`add_padding`](#ultralytics.utils.instance.Instances.add_padding) | Add padding to coordinates. |
| [`clip`](#ultralytics.utils.instance.Instances.clip) | Clip coordinates to stay within image boundaries. |
| [`concatenate`](#ultralytics.utils.instance.Instances.concatenate) | Concatenate a list of Instances objects into a single Instances object. |
| [`convert_bbox`](#ultralytics.utils.instance.Instances.convert_bbox) | Convert bounding box format. |
| [`denormalize`](#ultralytics.utils.instance.Instances.denormalize) | Convert normalized coordinates to absolute coordinates. |
| [`fliplr`](#ultralytics.utils.instance.Instances.fliplr) | Flip coordinates horizontally. |
| [`flipud`](#ultralytics.utils.instance.Instances.flipud) | Flip coordinates vertically. |
| [`normalize`](#ultralytics.utils.instance.Instances.normalize) | Convert absolute coordinates to normalized coordinates. |
| [`remove_zero_area_boxes`](#ultralytics.utils.instance.Instances.remove_zero_area_boxes) | Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height. |
| [`scale`](#ultralytics.utils.instance.Instances.scale) | Scale coordinates by given factors. |
| [`update`](#ultralytics.utils.instance.Instances.update) | Update instance variables. |

**Examples**

```python
Create instances with bounding boxes and segments
>>> instances = Instances(
...     bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
...     segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
...     keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]]),
... )
```

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L181-L497"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Instances:
    """Container for bounding boxes, segments, and keypoints of detected objects in an image.

    This class provides a unified interface for handling different types of object annotations including bounding boxes,
    segmentation masks, and keypoints. It supports various operations like scaling, normalization, clipping, and format
    conversion.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        keypoints (np.ndarray): Keypoints with shape (N, 17, 3) in format (x, y, visible).
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.
        segments (np.ndarray): Segments array with shape (N, M, 2) after resampling.

    Methods:
        convert_bbox: Convert bounding box format.
        scale: Scale coordinates by given factors.
        denormalize: Convert normalized coordinates to absolute coordinates.
        normalize: Convert absolute coordinates to normalized coordinates.
        add_padding: Add padding to coordinates.
        flipud: Flip coordinates vertically.
        fliplr: Flip coordinates horizontally.
        clip: Clip coordinates to stay within image boundaries.
        remove_zero_area_boxes: Remove boxes with zero area.
        update: Update instance variables.
        concatenate: Concatenate multiple Instances objects.

    Examples:
        Create instances with bounding boxes and segments
        >>> instances = Instances(
        ...     bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
        ...     segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
        ...     keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]]),
        ... )
    """

    def __init__(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray = None,
        keypoints: np.ndarray = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        """Initialize the Instances object with bounding boxes, segments, and keypoints.

        Args:
            bboxes (np.ndarray): Bounding boxes with shape (N, 4).
            segments (np.ndarray, optional): Segmentation masks.
            keypoints (np.ndarray, optional): Keypoints with shape (N, 17, 3) in format (x, y, visible).
            bbox_format (str): Format of bboxes.
            normalized (bool): Whether the coordinates are normalized.
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments
```
</details>

<br>

### Property `ultralytics.utils.instance.Instances.bbox_areas` {#ultralytics.utils.instance.Instances.bbox\_areas}

```python
def bbox_areas(self) -> np.ndarray
```

Calculate the area of bounding boxes.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L247-L249"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def bbox_areas(self) -> np.ndarray:
    """Calculate the area of bounding boxes."""
    return self._bboxes.areas()
```
</details>

<br>

### Property `ultralytics.utils.instance.Instances.bboxes` {#ultralytics.utils.instance.Instances.bboxes}

```python
def bboxes(self) -> np.ndarray
```

Return bounding boxes.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L482-L484"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def bboxes(self) -> np.ndarray:
    """Return bounding boxes."""
    return self._bboxes.bboxes
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.__getitem__` {#ultralytics.utils.instance.Instances.\_\_getitem\_\_}

```python
def __getitem__(self, index: int | np.ndarray | slice) -> Instances
```

Retrieve a specific instance or a set of instances using indexing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `index` | `int | slice | np.ndarray` | The index, slice, or boolean array to select the desired instances. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Instances` | A new Instances object containing the selected boxes, segments, and keypoints if present. |

!!! note "Notes"

    When using boolean indexing, make sure to provide a boolean array with the same length as the number of
    instances.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L317-L340"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __getitem__(self, index: int | np.ndarray | slice) -> Instances:
    """Retrieve a specific instance or a set of instances using indexing.

    Args:
        index (int | slice | np.ndarray): The index, slice, or boolean array to select the desired instances.

    Returns:
        (Instances): A new Instances object containing the selected boxes, segments, and keypoints if present.

    Notes:
        When using boolean indexing, make sure to provide a boolean array with the same length as the number of
        instances.
    """
    segments = self.segments[index] if len(self.segments) else self.segments
    keypoints = self.keypoints[index] if self.keypoints is not None else None
    bboxes = self.bboxes[index]
    bbox_format = self._bboxes.format
    return Instances(
        bboxes=bboxes,
        segments=segments,
        keypoints=keypoints,
        bbox_format=bbox_format,
        normalized=self.normalized,
    )
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.__len__` {#ultralytics.utils.instance.Instances.\_\_len\_\_}

```python
def __len__(self) -> int
```

Return the number of instances.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L431-L433"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __len__(self) -> int:
    """Return the number of instances."""
    return len(self.bboxes)
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.__repr__` {#ultralytics.utils.instance.Instances.\_\_repr\_\_}

```python
def __repr__(self) -> str
```

Return a string representation of the Instances object.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L486-L497"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __repr__(self) -> str:
    """Return a string representation of the Instances object."""
    # Map private to public names and include direct attributes
    attr_map = {"_bboxes": "bboxes"}
    parts = []
    for key, value in self.__dict__.items():
        name = attr_map.get(key, key)
        if name == "bboxes":
            value = self.bboxes  # Use the property
        if value is not None:
            parts.append(f"{name}={value!r}")
    return "Instances({})".format("\n".join(parts))
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.add_padding` {#ultralytics.utils.instance.Instances.add\_padding}

```python
def add_padding(self, padw: int, padh: int) -> None
```

Add padding to coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `padw` | `int` | Padding width. | *required* |
| `padh` | `int` | Padding height. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L302-L315"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def add_padding(self, padw: int, padh: int) -> None:
    """Add padding to coordinates.

    Args:
        padw (int): Padding width.
        padh (int): Padding height.
    """
    assert not self.normalized, "you should add padding with absolute coordinates."
    self._bboxes.add(offset=(padw, padh, padw, padh))
    self.segments[..., 0] += padw
    self.segments[..., 1] += padh
    if self.keypoints is not None:
        self.keypoints[..., 0] += padw
        self.keypoints[..., 1] += padh
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.clip` {#ultralytics.utils.instance.Instances.clip}

```python
def clip(self, w: int, h: int) -> None
```

Clip coordinates to stay within image boundaries.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `w` | `int` | Image width. | *required* |
| `h` | `int` | Image height. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L376-L400"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def clip(self, w: int, h: int) -> None:
    """Clip coordinates to stay within image boundaries.

    Args:
        w (int): Image width.
        h (int): Image height.
    """
    ori_format = self._bboxes.format
    self.convert_bbox(format="xyxy")
    self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
    self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
    if ori_format != "xyxy":
        self.convert_bbox(format=ori_format)
    self.segments[..., 0] = self.segments[..., 0].clip(0, w)
    self.segments[..., 1] = self.segments[..., 1].clip(0, h)
    if self.keypoints is not None:
        # Set out of bounds visibility to zero
        self.keypoints[..., 2][
            (self.keypoints[..., 0] < 0)
            | (self.keypoints[..., 0] > w)
            | (self.keypoints[..., 1] < 0)
            | (self.keypoints[..., 1] > h)
        ] = 0.0
        self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
        self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.concatenate` {#ultralytics.utils.instance.Instances.concatenate}

```python
def concatenate(cls, instances_list: list[Instances], axis = 0) -> Instances
```

Concatenate a list of Instances objects into a single Instances object.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `instances_list` | `list[Instances]` | A list of Instances objects to concatenate. | *required* |
| `axis` | `int, optional` | The axis along which the arrays will be concatenated. | `0` |

**Returns**

| Type | Description |
| --- | --- |
| `Instances` | A new Instances object containing the concatenated bounding boxes, segments, and keypoints if |

!!! note "Notes"

    The `Instances` objects in the list should have the same properties, such as the format of the bounding
    boxes, whether keypoints are present, and if the coordinates are normalized.

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L436-L479"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@classmethod
def concatenate(cls, instances_list: list[Instances], axis=0) -> Instances:
    """Concatenate a list of Instances objects into a single Instances object.

    Args:
        instances_list (list[Instances]): A list of Instances objects to concatenate.
        axis (int, optional): The axis along which the arrays will be concatenated.

    Returns:
        (Instances): A new Instances object containing the concatenated bounding boxes, segments, and keypoints if
            present.

    Notes:
        The `Instances` objects in the list should have the same properties, such as the format of the bounding
        boxes, whether keypoints are present, and if the coordinates are normalized.
    """
    assert isinstance(instances_list, (list, tuple))
    if not instances_list:
        return cls(np.empty(0))
    assert all(isinstance(instance, Instances) for instance in instances_list)

    if len(instances_list) == 1:
        return instances_list[0]

    use_keypoint = instances_list[0].keypoints is not None
    bbox_format = instances_list[0]._bboxes.format
    normalized = instances_list[0].normalized

    cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
    seg_len = [b.segments.shape[1] for b in instances_list]
    if len(frozenset(seg_len)) > 1:  # resample segments if there's different length
        max_len = max(seg_len)
        cat_segments = np.concatenate(
            [
                resample_segments(list(b.segments), max_len)
                if len(b.segments)
                else np.zeros((0, max_len, 2), dtype=np.float32)  # re-generating empty segments
                for b in instances_list
            ],
            axis=axis,
        )
    else:
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
    cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
    return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.convert_bbox` {#ultralytics.utils.instance.Instances.convert\_bbox}

```python
def convert_bbox(self, format: str) -> None
```

Convert bounding box format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `format` | `str` | Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L238-L244"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def convert_bbox(self, format: str) -> None:
    """Convert bounding box format.

    Args:
        format (str): Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'.
    """
    self._bboxes.convert(format=format)
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.denormalize` {#ultralytics.utils.instance.Instances.denormalize}

```python
def denormalize(self, w: int, h: int) -> None
```

Convert normalized coordinates to absolute coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `w` | `int` | Image width. | *required* |
| `h` | `int` | Image height. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L268-L283"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def denormalize(self, w: int, h: int) -> None:
    """Convert normalized coordinates to absolute coordinates.

    Args:
        w (int): Image width.
        h (int): Image height.
    """
    if not self.normalized:
        return
    self._bboxes.mul(scale=(w, h, w, h))
    self.segments[..., 0] *= w
    self.segments[..., 1] *= h
    if self.keypoints is not None:
        self.keypoints[..., 0] *= w
        self.keypoints[..., 1] *= h
    self.normalized = False
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.fliplr` {#ultralytics.utils.instance.Instances.fliplr}

```python
def fliplr(self, w: int) -> None
```

Flip coordinates horizontally.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `w` | `int` | Image width. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L359-L374"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fliplr(self, w: int) -> None:
    """Flip coordinates horizontally.

    Args:
        w (int): Image width.
    """
    if self._bboxes.format == "xyxy":
        x1 = self.bboxes[:, 0].copy()
        x2 = self.bboxes[:, 2].copy()
        self.bboxes[:, 0] = w - x2
        self.bboxes[:, 2] = w - x1
    else:
        self.bboxes[:, 0] = w - self.bboxes[:, 0]
    self.segments[..., 0] = w - self.segments[..., 0]
    if self.keypoints is not None:
        self.keypoints[..., 0] = w - self.keypoints[..., 0]
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.flipud` {#ultralytics.utils.instance.Instances.flipud}

```python
def flipud(self, h: int) -> None
```

Flip coordinates vertically.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `h` | `int` | Image height. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L342-L357"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def flipud(self, h: int) -> None:
    """Flip coordinates vertically.

    Args:
        h (int): Image height.
    """
    if self._bboxes.format == "xyxy":
        y1 = self.bboxes[:, 1].copy()
        y2 = self.bboxes[:, 3].copy()
        self.bboxes[:, 1] = h - y2
        self.bboxes[:, 3] = h - y1
    else:
        self.bboxes[:, 1] = h - self.bboxes[:, 1]
    self.segments[..., 1] = h - self.segments[..., 1]
    if self.keypoints is not None:
        self.keypoints[..., 1] = h - self.keypoints[..., 1]
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.normalize` {#ultralytics.utils.instance.Instances.normalize}

```python
def normalize(self, w: int, h: int) -> None
```

Convert absolute coordinates to normalized coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `w` | `int` | Image width. | *required* |
| `h` | `int` | Image height. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L285-L300"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def normalize(self, w: int, h: int) -> None:
    """Convert absolute coordinates to normalized coordinates.

    Args:
        w (int): Image width.
        h (int): Image height.
    """
    if self.normalized:
        return
    self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
    self.segments[..., 0] /= w
    self.segments[..., 1] /= h
    if self.keypoints is not None:
        self.keypoints[..., 0] /= w
        self.keypoints[..., 1] /= h
    self.normalized = True
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.remove_zero_area_boxes` {#ultralytics.utils.instance.Instances.remove\_zero\_area\_boxes}

```python
def remove_zero_area_boxes(self) -> np.ndarray
```

Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Boolean array indicating which boxes were kept. |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L402-L415"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def remove_zero_area_boxes(self) -> np.ndarray:
    """Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.

    Returns:
        (np.ndarray): Boolean array indicating which boxes were kept.
    """
    good = self.bbox_areas > 0
    if not all(good):
        self._bboxes = self._bboxes[good]
        if self.segments is not None and len(self.segments):
            self.segments = self.segments[good]
        if self.keypoints is not None:
            self.keypoints = self.keypoints[good]
    return good
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.scale` {#ultralytics.utils.instance.Instances.scale}

```python
def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False)
```

Scale coordinates by given factors.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `scale_w` | `float` | Scale factor for width. | *required* |
| `scale_h` | `float` | Scale factor for height. | *required* |
| `bbox_only` | `bool, optional` | Whether to scale only bounding boxes. | `False` |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L251-L266"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
    """Scale coordinates by given factors.

    Args:
        scale_w (float): Scale factor for width.
        scale_h (float): Scale factor for height.
        bbox_only (bool, optional): Whether to scale only bounding boxes.
    """
    self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
    if bbox_only:
        return
    self.segments[..., 0] *= scale_w
    self.segments[..., 1] *= scale_h
    if self.keypoints is not None:
        self.keypoints[..., 0] *= scale_w
        self.keypoints[..., 1] *= scale_h
```
</details>

<br>

### Method `ultralytics.utils.instance.Instances.update` {#ultralytics.utils.instance.Instances.update}

```python
def update(self, bboxes: np.ndarray, segments: np.ndarray = None, keypoints: np.ndarray = None)
```

Update instance variables.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `bboxes` | `np.ndarray` | New bounding boxes. | *required* |
| `segments` | `np.ndarray, optional` | New segments. | `None` |
| `keypoints` | `np.ndarray, optional` | New keypoints. | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L417-L429"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, bboxes: np.ndarray, segments: np.ndarray = None, keypoints: np.ndarray = None):
    """Update instance variables.

    Args:
        bboxes (np.ndarray): New bounding boxes.
        segments (np.ndarray, optional): New segments.
        keypoints (np.ndarray, optional): New keypoints.
    """
    self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
    if segments is not None:
        self.segments = segments
    if keypoints is not None:
        self.keypoints = keypoints
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.instance._ntuple` {#ultralytics.utils.instance.\_ntuple}

```python
def _ntuple(n)
```

Create a function that converts input to n-tuple by repeating singleton values.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `n` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/instance.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/instance.py#L14-L21"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _ntuple(n):
    """Create a function that converts input to n-tuple by repeating singleton values."""

    def parse(x):
        """Parse input to return n-tuple by repeating singleton values n times."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse
```
</details>

<br><br>
