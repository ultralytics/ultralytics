---
description: Explore the utility functions in FastSAM for adjusting bounding boxes and calculating IoU, benefiting computer vision projects.
keywords: FastSAM, bounding boxes, IoU, Ultralytics, image processing, computer vision
---

# Reference for `ultralytics/models/fastsam/utils.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/utils.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`adjust_bboxes_to_image_border`](#ultralytics.models.fastsam.utils.adjust_bboxes_to_image_border)


## Function `ultralytics.models.fastsam.utils.adjust_bboxes_to_image_border` {#ultralytics.models.fastsam.utils.adjust\_bboxes\_to\_image\_border}

```python
def adjust_bboxes_to_image_border(boxes, image_shape, threshold = 20)
```

Adjust bounding boxes to stick to image border if they are within a certain threshold.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `boxes` | `torch.Tensor` | Bounding boxes with shape (N, 4) in xyxy format. | *required* |
| `image_shape` | `tuple` | Image dimensions as (height, width). | *required* |
| `threshold` | `int` | Pixel threshold for considering a box close to the border. | `20` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Adjusted bounding boxes with shape (N, 4). |

<details>
<summary>Source code in <code>ultralytics/models/fastsam/utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/utils.py#L4-L23"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    """Adjust bounding boxes to stick to image border if they are within a certain threshold.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
        image_shape (tuple): Image dimensions as (height, width).
        threshold (int): Pixel threshold for considering a box close to the border.

    Returns:
        (torch.Tensor): Adjusted bounding boxes with shape (N, 4).
    """
    # Image dimensions
    h, w = image_shape

    # Adjust boxes that are close to image borders
    boxes[boxes[:, 0] < threshold, 0] = 0  # x1
    boxes[boxes[:, 1] < threshold, 1] = 0  # y1
    boxes[boxes[:, 2] > w - threshold, 2] = w  # x2
    boxes[boxes[:, 3] > h - threshold, 3] = h  # y2
    return boxes
```
</details>

<br><br>
