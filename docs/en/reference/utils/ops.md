---
description: Explore detailed documentation on utility operations in Ultralytics including non-max suppression, bounding box transformations, and more.
keywords: Ultralytics, utility operations, non-max suppression, bounding box transformations, YOLOv8, machine learning
---

# Reference for `ultralytics/utils/ops.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`Profile`](#ultralytics.utils.ops.Profile)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`Profile.__enter__`](#ultralytics.utils.ops.Profile.__enter__)
        - [`Profile.__exit__`](#ultralytics.utils.ops.Profile.__exit__)
        - [`Profile.__str__`](#ultralytics.utils.ops.Profile.__str__)
        - [`Profile.time`](#ultralytics.utils.ops.Profile.time)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`segment2box`](#ultralytics.utils.ops.segment2box)
        - [`scale_boxes`](#ultralytics.utils.ops.scale_boxes)
        - [`make_divisible`](#ultralytics.utils.ops.make_divisible)
        - [`clip_boxes`](#ultralytics.utils.ops.clip_boxes)
        - [`clip_coords`](#ultralytics.utils.ops.clip_coords)
        - [`xyxy2xywh`](#ultralytics.utils.ops.xyxy2xywh)
        - [`xywh2xyxy`](#ultralytics.utils.ops.xywh2xyxy)
        - [`xywhn2xyxy`](#ultralytics.utils.ops.xywhn2xyxy)
        - [`xyxy2xywhn`](#ultralytics.utils.ops.xyxy2xywhn)
        - [`xywh2ltwh`](#ultralytics.utils.ops.xywh2ltwh)
        - [`xyxy2ltwh`](#ultralytics.utils.ops.xyxy2ltwh)
        - [`ltwh2xywh`](#ultralytics.utils.ops.ltwh2xywh)
        - [`xyxyxyxy2xywhr`](#ultralytics.utils.ops.xyxyxyxy2xywhr)
        - [`xywhr2xyxyxyxy`](#ultralytics.utils.ops.xywhr2xyxyxyxy)
        - [`ltwh2xyxy`](#ultralytics.utils.ops.ltwh2xyxy)
        - [`segments2boxes`](#ultralytics.utils.ops.segments2boxes)
        - [`resample_segments`](#ultralytics.utils.ops.resample_segments)
        - [`crop_mask`](#ultralytics.utils.ops.crop_mask)
        - [`process_mask`](#ultralytics.utils.ops.process_mask)
        - [`process_mask_native`](#ultralytics.utils.ops.process_mask_native)
        - [`scale_masks`](#ultralytics.utils.ops.scale_masks)
        - [`scale_coords`](#ultralytics.utils.ops.scale_coords)
        - [`regularize_rboxes`](#ultralytics.utils.ops.regularize_rboxes)
        - [`masks2segments`](#ultralytics.utils.ops.masks2segments)
        - [`convert_torch2numpy_batch`](#ultralytics.utils.ops.convert_torch2numpy_batch)
        - [`clean_str`](#ultralytics.utils.ops.clean_str)
        - [`empty_like`](#ultralytics.utils.ops.empty_like)


## Class `ultralytics.utils.ops.Profile` {#ultralytics.utils.ops.Profile}

```python
Profile(self, t: float = 0.0, device: torch.device | None = None)
```

**Bases:** `contextlib.ContextDecorator`

Ultralytics Profile class for timing code execution.

Use as a decorator with @Profile() or as a context manager with 'with Profile():'. Provides accurate timing measurements with CUDA synchronization support for GPU operations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `t` | `float` | Initial accumulated time in seconds. | `0.0` |
| `device` | `torch.device, optional` | Device used for model inference to enable CUDA synchronization. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `t` | `float` | Accumulated time in seconds. |
| `device` | `torch.device` | Device used for model inference. |
| `cuda` | `bool` | Whether CUDA is being used for timing synchronization. |

**Methods**

| Name | Description |
| --- | --- |
| [`__enter__`](#ultralytics.utils.ops.Profile.__enter__) | Start timing. |
| [`__exit__`](#ultralytics.utils.ops.Profile.__exit__) | Stop timing. |
| [`__str__`](#ultralytics.utils.ops.Profile.__str__) | Return a human-readable string representing the accumulated elapsed time. |
| [`time`](#ultralytics.utils.ops.Profile.time) | Get current time with CUDA synchronization if applicable. |

**Examples**

```python
Use as a context manager to time code execution
>>> with Profile(device=device) as dt:
...     pass  # slow operation here
>>> print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"

Use as a decorator to time function execution
>>> @Profile()
... def slow_function():
...     time.sleep(0.1)
```

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L18-L70"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Profile(contextlib.ContextDecorator):
    """Ultralytics Profile class for timing code execution.

    Use as a decorator with @Profile() or as a context manager with 'with Profile():'. Provides accurate timing
    measurements with CUDA synchronization support for GPU operations.

    Attributes:
        t (float): Accumulated time in seconds.
        device (torch.device): Device used for model inference.
        cuda (bool): Whether CUDA is being used for timing synchronization.

    Examples:
        Use as a context manager to time code execution
        >>> with Profile(device=device) as dt:
        ...     pass  # slow operation here
        >>> print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"

        Use as a decorator to time function execution
        >>> @Profile()
        ... def slow_function():
        ...     time.sleep(0.1)
    """

    def __init__(self, t: float = 0.0, device: torch.device | None = None):
        """Initialize the Profile class.

        Args:
            t (float): Initial accumulated time in seconds.
            device (torch.device, optional): Device used for model inference to enable CUDA synchronization.
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))
```
</details>

<br>

### Method `ultralytics.utils.ops.Profile.__enter__` {#ultralytics.utils.ops.Profile.\_\_enter\_\_}

```python
def __enter__(self)
```

Start timing.

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L52-L55"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __enter__(self):
    """Start timing."""
    self.start = self.time()
    return self
```
</details>

<br>

### Method `ultralytics.utils.ops.Profile.__exit__` {#ultralytics.utils.ops.Profile.\_\_exit\_\_}

```python
def __exit__(self, type, value, traceback)
```

Stop timing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `type` |  |  | *required* |
| `value` |  |  | *required* |
| `traceback` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L57-L60"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __exit__(self, type, value, traceback):
    """Stop timing."""
    self.dt = self.time() - self.start  # delta-time
    self.t += self.dt  # accumulate dt
```
</details>

<br>

### Method `ultralytics.utils.ops.Profile.__str__` {#ultralytics.utils.ops.Profile.\_\_str\_\_}

```python
def __str__(self)
```

Return a human-readable string representing the accumulated elapsed time.

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L62-L64"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __str__(self):
    """Return a human-readable string representing the accumulated elapsed time."""
    return f"Elapsed time is {self.t} s"
```
</details>

<br>

### Method `ultralytics.utils.ops.Profile.time` {#ultralytics.utils.ops.Profile.time}

```python
def time(self)
```

Get current time with CUDA synchronization if applicable.

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L66-L70"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def time(self):
    """Get current time with CUDA synchronization if applicable."""
    if self.cuda:
        torch.cuda.synchronize(self.device)
    return time.perf_counter()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.segment2box` {#ultralytics.utils.ops.segment2box}

```python
def segment2box(segment, width: int = 640, height: int = 640)
```

Convert segment coordinates to bounding box coordinates.

Converts a single segment label to a box label by finding the minimum and maximum x and y coordinates. Applies inside-image constraint and clips coordinates when necessary.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `segment` | `np.ndarray` | Segment coordinates in format (N, 2) where N is number of points. | *required* |
| `width` | `int` | Width of the image in pixels. | `640` |
| `height` | `int` | Height of the image in pixels. | `640` |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Bounding box coordinates in xyxy format [x1, y1, x2, y2]. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L73-L99"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def segment2box(segment, width: int = 640, height: int = 640):
    """Convert segment coordinates to bounding box coordinates.

    Converts a single segment label to a box label by finding the minimum and maximum x and y coordinates. Applies
    inside-image constraint and clips coordinates when necessary.

    Args:
        segment (np.ndarray): Segment coordinates in format (N, 2) where N is number of points.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.

    Returns:
        (np.ndarray): Bounding box coordinates in xyxy format [x1, y1, x2, y2].
    """
    x, y = segment.T  # segment xy
    # Clip coordinates if 3 out of 4 sides are outside the image
    if np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum() >= 3:
        x = x.clip(0, width)
        y = y.clip(0, height)
    inside = (x > 0) & (y > 0) & (x < width) & (y < height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.scale_boxes` {#ultralytics.utils.ops.scale\_boxes}

```python
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad = None, padding: bool = True, xywh: bool = False)
```

Rescale bounding boxes from one image shape to another.

Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes. Supports both xyxy and xywh box formats.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img1_shape` | `tuple` | Shape of the source image (height, width). | *required* |
| `boxes` | `torch.Tensor` | Bounding boxes to rescale in format (N, 4). | *required* |
| `img0_shape` | `tuple` | Shape of the target image (height, width). | *required* |
| `ratio_pad` | `tuple, optional` | Tuple of (ratio, pad) for scaling. If None, calculated from image shapes. | `None` |
| `padding` | `bool` | Whether boxes are based on YOLO-style augmented images with padding. | `True` |
| `xywh` | `bool` | Whether box format is xywh (True) or xyxy (False). | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Rescaled bounding boxes in the same format as input. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L102-L134"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """Rescale bounding boxes from one image shape to another.

    Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes. Supports
    both xyxy and xywh box formats.

    Args:
        img1_shape (tuple): Shape of the source image (height, width).
        boxes (torch.Tensor): Bounding boxes to rescale in format (N, 4).
        img0_shape (tuple): Shape of the target image (height, width).
        ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling. If None, calculated from image shapes.
        padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
        xywh (bool): Whether box format is xywh (True) or xyxy (False).

    Returns:
        (torch.Tensor): Rescaled bounding boxes in the same format as input.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        if not xywh:
            boxes[..., 2] -= pad_x  # x padding
            boxes[..., 3] -= pad_y  # y padding
    boxes[..., :4] /= gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.make_divisible` {#ultralytics.utils.ops.make\_divisible}

```python
def make_divisible(x: int, divisor)
```

Return the nearest number that is divisible by the given divisor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `int` | The number to make divisible. | *required* |
| `divisor` | `int | torch.Tensor` | The divisor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `int` | The nearest number divisible by the divisor. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L137-L149"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def make_divisible(x: int, divisor):
    """Return the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.clip_boxes` {#ultralytics.utils.ops.clip\_boxes}

```python
def clip_boxes(boxes, shape)
```

Clip bounding boxes to image boundaries.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `boxes` | `torch.Tensor | np.ndarray` | Bounding boxes to clip. | *required* |
| `shape` | `tuple` | Image shape as HWC or HW (supports both). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | Clipped bounding boxes. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L152-L177"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def clip_boxes(boxes, shape):
    """Clip bounding boxes to image boundaries.

    Args:
        boxes (torch.Tensor | np.ndarray): Bounding boxes to clip.
        shape (tuple): Image shape as HWC or HW (supports both).

    Returns:
        (torch.Tensor | np.ndarray): Clipped bounding boxes.
    """
    h, w = shape[:2]  # supports both HWC or HW shapes
    if isinstance(boxes, torch.Tensor):  # faster individually
        if NOT_MACOS14:
            boxes[..., 0].clamp_(0, w)  # x1
            boxes[..., 1].clamp_(0, h)  # y1
            boxes[..., 2].clamp_(0, w)  # x2
            boxes[..., 3].clamp_(0, h)  # y2
        else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(0, w)
            boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
    return boxes
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.clip_coords` {#ultralytics.utils.ops.clip\_coords}

```python
def clip_coords(coords, shape)
```

Clip line coordinates to image boundaries.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `coords` | `torch.Tensor | np.ndarray` | Line coordinates to clip. | *required* |
| `shape` | `tuple` | Image shape as HWC or HW (supports both). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | np.ndarray` | Clipped coordinates. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L180-L201"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def clip_coords(coords, shape):
    """Clip line coordinates to image boundaries.

    Args:
        coords (torch.Tensor | np.ndarray): Line coordinates to clip.
        shape (tuple): Image shape as HWC or HW (supports both).

    Returns:
        (torch.Tensor | np.ndarray): Clipped coordinates.
    """
    h, w = shape[:2]  # supports both HWC or HW shapes
    if isinstance(coords, torch.Tensor):
        if NOT_MACOS14:
            coords[..., 0].clamp_(0, w)  # x
            coords[..., 1].clamp_(0, h)  # y
        else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
            coords[..., 0] = coords[..., 0].clamp(0, w)
            coords[..., 1] = coords[..., 1].clamp(0, h)
    else:  # np.array
        coords[..., 0] = coords[..., 0].clip(0, w)  # x
        coords[..., 1] = coords[..., 1].clip(0, h)  # y
    return coords
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.xyxy2xywh` {#ultralytics.utils.ops.xyxy2xywh}

```python
def xyxy2xywh(x)
```

Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is

the top-left corner and (x2, y2) is the bottom-right corner.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Input bounding box coordinates in (x1, y1, x2, y2) format. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Bounding box coordinates in (x, y, width, height) format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L204-L221"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def xyxy2xywh(x):
    """Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.xywh2xyxy` {#ultralytics.utils.ops.xywh2xyxy}

```python
def xywh2xyxy(x)
```

Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is

the top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Input bounding box coordinates in (x, y, width, height) format. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Bounding box coordinates in (x1, y1, x2, y2) format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L224-L240"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def xywh2xyxy(x):
    """Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x, y, width, height) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.xywhn2xyxy` {#ultralytics.utils.ops.xywhn2xyxy}

```python
def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0)
```

Convert normalized bounding box coordinates to pixel coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Normalized bounding box coordinates in (x, y, w, h) format. | *required* |
| `w` | `int` | Image width in pixels. | `640` |
| `h` | `int` | Image height in pixels. | `640` |
| `padw` | `int` | Padding width in pixels. | `0` |
| `padh` | `int` | Padding height in pixels. | `0` |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Bounding box coordinates in (x1, y1, x2, y2) format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L243-L264"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0):
    """Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, w, h) format.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        padw (int): Padding width in pixels.
        padh (int): Padding height in pixels.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xc, yc, xw, xh = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    half_w, half_h = xw / 2, xh / 2
    y[..., 0] = w * (xc - half_w) + padw  # top left x
    y[..., 1] = h * (yc - half_h) + padh  # top left y
    y[..., 2] = w * (xc + half_w) + padw  # bottom right x
    y[..., 3] = h * (yc + half_h) + padh  # bottom right y
    return y
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.xyxy2xywhn` {#ultralytics.utils.ops.xyxy2xywhn}

```python
def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0)
```

Convert bounding box coordinates from (x1, y1, x2, y2) format to normalized (x, y, width, height) format. x, y,

width and height are normalized to image dimensions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Input bounding box coordinates in (x1, y1, x2, y2) format. | *required* |
| `w` | `int` | Image width in pixels. | `640` |
| `h` | `int` | Image height in pixels. | `640` |
| `clip` | `bool` | Whether to clip boxes to image boundaries. | `False` |
| `eps` | `float` | Minimum value for box width and height. | `0.0` |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Normalized bounding box coordinates in (x, y, width, height) format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L267-L290"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    """Convert bounding box coordinates from (x1, y1, x2, y2) format to normalized (x, y, width, height) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        clip (bool): Whether to clip boxes to image boundaries.
        eps (float): Minimum value for box width and height.

    Returns:
        (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, width, height) format.
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = ((x1 + x2) / 2) / w  # x center
    y[..., 1] = ((y1 + y2) / 2) / h  # y center
    y[..., 2] = (x2 - x1) / w  # width
    y[..., 3] = (y2 - y1) / h  # height
    return y
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.xywh2ltwh` {#ultralytics.utils.ops.xywh2ltwh}

```python
def xywh2ltwh(x)
```

Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Input bounding box coordinates in xywh format. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Bounding box coordinates in ltwh format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L293-L305"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def xywh2ltwh(x):
    """Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xywh format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in ltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.xyxy2ltwh` {#ultralytics.utils.ops.xyxy2ltwh}

```python
def xyxy2ltwh(x)
```

Convert bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h] format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Input bounding box coordinates in xyxy format. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Bounding box coordinates in ltwh format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L308-L320"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def xyxy2ltwh(x):
    """Convert bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h] format.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xyxy format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in ltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.ltwh2xywh` {#ultralytics.utils.ops.ltwh2xywh}

```python
def ltwh2xywh(x)
```

Convert bounding boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Input bounding box coordinates. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Bounding box coordinates in xywh format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L323-L335"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def ltwh2xywh(x):
    """Convert bounding boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xywh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.xyxyxyxy2xywhr` {#ultralytics.utils.ops.xyxyxyxy2xywhr}

```python
def xyxyxyxy2xywhr(x)
```

Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation] format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Input box corners with shape (N, 8) in [xy1, xy2, xy3, xy4] format. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Converted data in [cx, cy, w, h, rotation] format with shape (N, 5). Rotation |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L338-L366"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def xyxyxyxy2xywhr(x):
    """Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation] format.

    Args:
        x (np.ndarray | torch.Tensor): Input box corners with shape (N, 8) in [xy1, xy2, xy3, xy4] format.

    Returns:
        (np.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format with shape (N, 5). Rotation
            values are in radians from [-pi/4, 3pi/4).
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        # convert angle to radian and normalize to [-pi/4, 3pi/4)
        theta = angle / 180 * np.pi
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rboxes.append([cx, cy, w, h, theta])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.xywhr2xyxyxyxy` {#ultralytics.utils.ops.xywhr2xyxyxyxy}

```python
def xywhr2xyxyxyxy(x)
```

Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4] format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Boxes in [cx, cy, w, h, rotation] format with shape (N, 5) or (B, N, 5). Rotation<br>    values should be in radians from [-pi/4, 3pi/4). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Converted corner points with shape (N, 4, 2) or (B, N, 4, 2). |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L369-L396"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def xywhr2xyxyxyxy(x):
    """Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4] format.

    Args:
        x (np.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format with shape (N, 5) or (B, N, 5). Rotation
            values should be in radians from [-pi/4, 3pi/4).

    Returns:
        (np.ndarray | torch.Tensor): Converted corner points with shape (N, 4, 2) or (B, N, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.ltwh2xyxy` {#ultralytics.utils.ops.ltwh2xyxy}

```python
def ltwh2xyxy(x)
```

Convert bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `np.ndarray | torch.Tensor` | Input bounding box coordinates. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray | torch.Tensor` | Bounding box coordinates in xyxy format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L399-L411"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def ltwh2xyxy(x):
    """Convert bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyxy format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # x2
    y[..., 3] = x[..., 3] + x[..., 1]  # y2
    return y
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.segments2boxes` {#ultralytics.utils.ops.segments2boxes}

```python
def segments2boxes(segments)
```

Convert segment coordinates to bounding box labels in xywh format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `segments` | `list` | List of segments where each segment is a list of points, each point is [x, y] coordinates. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Bounding box coordinates in xywh format. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L414-L427"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def segments2boxes(segments):
    """Convert segment coordinates to bounding box labels in xywh format.

    Args:
        segments (list): List of segments where each segment is a list of points, each point is [x, y] coordinates.

    Returns:
        (np.ndarray): Bounding box coordinates in xywh format.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.resample_segments` {#ultralytics.utils.ops.resample\_segments}

```python
def resample_segments(segments, n: int = 1000)
```

Resample segments to n points each using linear interpolation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `segments` | `list` | List of (N, 2) arrays where N is the number of points in each segment. | *required* |
| `n` | `int` | Number of points to resample each segment to. | `1000` |

**Returns**

| Type | Description |
| --- | --- |
| `list` | Resampled segments with n points each. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L430-L450"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def resample_segments(segments, n: int = 1000):
    """Resample segments to n points each using linear interpolation.

    Args:
        segments (list): List of (N, 2) arrays where N is the number of points in each segment.
        n (int): Number of points to resample each segment to.

    Returns:
        (list): Resampled segments with n points each.
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.crop_mask` {#ultralytics.utils.ops.crop\_mask}

```python
def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor
```

Crop masks to bounding box regions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `masks` | `torch.Tensor` | Masks with shape (N, H, W). | *required* |
| `boxes` | `torch.Tensor` | Bounding box coordinates with shape (N, 4) in xyxy pixel format. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Cropped masks. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L453-L477"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Crop masks to bounding box regions.

    Args:
        masks (torch.Tensor): Masks with shape (N, H, W).
        boxes (torch.Tensor): Bounding box coordinates with shape (N, 4) in xyxy pixel format.

    Returns:
        (torch.Tensor): Cropped masks.
    """
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    n, h, w = masks.shape
    if n < 50 and not masks.is_cuda:  # faster for fewer masks (predict)
        for i, (x1, y1, x2, y2) in enumerate(boxes.round().int()):
            masks[i, :y1] = 0
            masks[i, y2:] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks
    else:  # faster for more masks (val)
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.process_mask` {#ultralytics.utils.ops.process\_mask}

```python
def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False)
```

Apply masks to bounding boxes using mask head output.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `protos` | `torch.Tensor` | Mask prototypes with shape (mask_dim, mask_h, mask_w). | *required* |
| `masks_in` | `torch.Tensor` | Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS. | *required* |
| `bboxes` | `torch.Tensor` | Bounding boxes with shape (N, 4) where N is number of masks after NMS. | *required* |
| `shape` | `tuple` | Input image size as (height, width). | *required* |
| `upsample` | `bool` | Whether to upsample masks to original image size. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L480-L504"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """Apply masks to bounding boxes using mask head output.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).
        upsample (bool): Whether to upsample masks to original image size.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # NHW

    width_ratio = mw / shape[1]
    height_ratio = mh / shape[0]
    ratios = torch.tensor([[width_ratio, height_ratio, width_ratio, height_ratio]], device=bboxes.device)

    masks = crop_mask(masks, boxes=bboxes * ratios)  # NHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]  # NHW
    return masks.gt_(0.0).byte()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.process_mask_native` {#ultralytics.utils.ops.process\_mask\_native}

```python
def process_mask_native(protos, masks_in, bboxes, shape)
```

Apply masks to bounding boxes using mask head output with native upsampling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `protos` | `torch.Tensor` | Mask prototypes with shape (mask_dim, mask_h, mask_w). | *required* |
| `masks_in` | `torch.Tensor` | Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS. | *required* |
| `bboxes` | `torch.Tensor` | Bounding boxes with shape (N, 4) where N is number of masks after NMS. | *required* |
| `shape` | `tuple` | Input image size as (height, width). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Binary mask tensor with shape (N, H, W). |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L507-L523"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def process_mask_native(protos, masks_in, bboxes, shape):
    """Apply masks to bounding boxes using mask head output with native upsampling.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).

    Returns:
        (torch.Tensor): Binary mask tensor with shape (N, H, W).
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # NHW
    masks = crop_mask(masks, bboxes)  # NHW
    return masks.gt_(0.0).byte()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.scale_masks` {#ultralytics.utils.ops.scale\_masks}

```python
def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[int, int], tuple[int, int]] | None = None,
    padding: bool = True,
) -> torch.Tensor
```

Rescale segment masks to target shape.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `masks` | `torch.Tensor` | Masks with shape (N, C, H, W). | *required* |
| `shape` | `tuple[int, int]` | Target height and width as (height, width). | *required* |
| `ratio_pad` | `tuple, optional` | Ratio and padding values as ((ratio_h, ratio_w), (pad_w, pad_h)). | `None` |
| `padding` | `bool` | Whether masks are based on YOLO-style augmented images with padding. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Rescaled masks. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L526-L559"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[int, int], tuple[int, int]] | None = None,
    padding: bool = True,
) -> torch.Tensor:
    """Rescale segment masks to target shape.

    Args:
        masks (torch.Tensor): Masks with shape (N, C, H, W).
        shape (tuple[int, int]): Target height and width as (height, width).
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_w, pad_h)).
        padding (bool): Whether masks are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Rescaled masks.
    """
    im1_h, im1_w = masks.shape[2:]
    im0_h, im0_w = shape[:2]
    if im1_h == im0_h and im1_w == im0_w:
        return masks

    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_h / im0_h, im1_w / im0_w)  # gain  = old / new
        pad_w, pad_h = (im1_w - im0_w * gain), (im1_h - im0_h * gain)  # wh padding
        if padding:
            pad_w /= 2
            pad_h /= 2
    else:
        pad_w, pad_h = ratio_pad[1]
    top, left = (round(pad_h - 0.1), round(pad_w - 0.1)) if padding else (0, 0)
    bottom = im1_h - round(pad_h + 0.1)
    right = im1_w - round(pad_w + 0.1)
    return F.interpolate(masks[..., top:bottom, left:right].float(), shape, mode="bilinear")  # NCHW masks
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.scale_coords` {#ultralytics.utils.ops.scale\_coords}

```python
def scale_coords(img1_shape, coords, img0_shape, ratio_pad = None, normalize: bool = False, padding: bool = True)
```

Rescale segment coordinates from img1_shape to img0_shape.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img1_shape` | `tuple` | Source image shape as HWC or HW (supports both). | *required* |
| `coords` | `torch.Tensor` | Coordinates to scale with shape (N, 2). | *required* |
| `img0_shape` | `tuple` | Image 0 shape as HWC or HW (supports both). | *required* |
| `ratio_pad` | `tuple, optional` | Ratio and padding values as ((ratio_h, ratio_w), (pad_w, pad_h)). | `None` |
| `normalize` | `bool` | Whether to normalize coordinates to range [0, 1]. | `False` |
| `padding` | `bool` | Whether coordinates are based on YOLO-style augmented images with padding. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Scaled coordinates. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L562-L594"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize: bool = False, padding: bool = True):
    """Rescale segment coordinates from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): Source image shape as HWC or HW (supports both).
        coords (torch.Tensor): Coordinates to scale with shape (N, 2).
        img0_shape (tuple): Image 0 shape as HWC or HW (supports both).
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_w, pad_h)).
        normalize (bool): Whether to normalize coordinates to range [0, 1].
        padding (bool): Whether coordinates are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Scaled coordinates.
    """
    img0_h, img0_w = img0_shape[:2]  # supports both HWC or HW shapes
    if ratio_pad is None:  # calculate from img0_shape
        img1_h, img1_w = img1_shape[:2]  # supports both HWC or HW shapes
        gain = min(img1_h / img0_h, img1_w / img0_w)  # gain  = old / new
        pad = (img1_w - img0_w * gain) / 2, (img1_h - img0_h * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_w  # width
        coords[..., 1] /= img0_h  # height
    return coords
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.regularize_rboxes` {#ultralytics.utils.ops.regularize\_rboxes}

```python
def regularize_rboxes(rboxes)
```

Regularize rotated bounding boxes to range [0, pi/2).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `rboxes` | `torch.Tensor` | Input rotated boxes with shape (N, 5) in xywhr format. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Regularized rotated boxes. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L597-L612"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def regularize_rboxes(rboxes):
    """Regularize rotated bounding boxes to range [0, pi/2).

    Args:
        rboxes (torch.Tensor): Input rotated boxes with shape (N, 5) in xywhr format.

    Returns:
        (torch.Tensor): Regularized rotated boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge if t >= pi/2 while not being symmetrically opposite
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.masks2segments` {#ultralytics.utils.ops.masks2segments}

```python
def masks2segments(masks: np.ndarray | torch.Tensor, strategy: str = "all") -> list[np.ndarray]
```

Convert masks to segments using contour detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `masks` | `np.ndarray | torch.Tensor` | Binary masks with shape (N, H, W). | *required* |
| `strategy` | `str` | Segmentation strategy, either 'all' or 'largest'. | `"all"` |

**Returns**

| Type | Description |
| --- | --- |
| `list` | List of segment masks as float32 arrays. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L615-L643"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def masks2segments(masks: np.ndarray | torch.Tensor, strategy: str = "all") -> list[np.ndarray]:
    """Convert masks to segments using contour detection.

    Args:
        masks (np.ndarray | torch.Tensor): Binary masks with shape (N, H, W).
        strategy (str): Segmentation strategy, either 'all' or 'largest'.

    Returns:
        (list): List of segment masks as float32 arrays.
    """
    from ultralytics.data.converter import merge_multi_segment

    masks = masks.astype("uint8") if isinstance(masks, np.ndarray) else masks.byte().cpu().numpy()
    segments = []
    for x in np.ascontiguousarray(masks):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # merge and concatenate all segments
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.convert_torch2numpy_batch` {#ultralytics.utils.ops.convert\_torch2numpy\_batch}

```python
def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray
```

Convert a batch of FP32 torch tensors to NumPy uint8 arrays, changing from BCHW to BHWC layout.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `torch.Tensor` | Input tensor batch with shape (Batch, Channels, Height, Width) and dtype torch.float32. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Output NumPy array batch with shape (Batch, Height, Width, Channels) and dtype uint8. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L646-L655"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """Convert a batch of FP32 torch tensors to NumPy uint8 arrays, changing from BCHW to BHWC layout.

    Args:
        batch (torch.Tensor): Input tensor batch with shape (Batch, Channels, Height, Width) and dtype torch.float32.

    Returns:
        (np.ndarray): Output NumPy array batch with shape (Batch, Height, Width, Channels) and dtype uint8.
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).byte().cpu().numpy()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.clean_str` {#ultralytics.utils.ops.clean\_str}

```python
def clean_str(s)
```

Clean a string by replacing special characters with '_' character.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `s` | `str` | A string needing special characters replaced. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | A string with special characters replaced by an underscore _. |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L658-L667"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def clean_str(s):
    """Clean a string by replacing special characters with '_' character.

    Args:
        s (str): A string needing special characters replaced.

    Returns:
        (str): A string with special characters replaced by an underscore _.
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨`><+]", repl="_", string=s)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.ops.empty_like` {#ultralytics.utils.ops.empty\_like}

```python
def empty_like(x)
```

Create empty torch.Tensor or np.ndarray with same shape and dtype as input.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/ops.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L670-L672"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def empty_like(x):
    """Create empty torch.Tensor or np.ndarray with same shape and dtype as input."""
    return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)
```
</details>

<br><br>
