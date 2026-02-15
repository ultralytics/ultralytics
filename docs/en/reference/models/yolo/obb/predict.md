---
description: Learn how to use the Ultralytics YOLO OBBPredictor for oriented bounding box predictions. Enhance your object detection models with ease.
keywords: Ultralytics, YOLO, OBBPredictor, oriented bounding box, object detection, AI, machine learning, PyTorch
---

# Reference for `ultralytics/models/yolo/obb/predict.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/predict.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/predict.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`OBBPredictor`](#ultralytics.models.yolo.obb.predict.OBBPredictor)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`OBBPredictor.construct_result`](#ultralytics.models.yolo.obb.predict.OBBPredictor.construct_result)


## Class `ultralytics.models.yolo.obb.predict.OBBPredictor` {#ultralytics.models.yolo.obb.predict.OBBPredictor}

```python
OBBPredictor(self, cfg = DEFAULT_CFG, overrides = None, _callbacks = None)
```

**Bases:** `DetectionPredictor`

A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.

This predictor handles oriented bounding box detection tasks, processing images and returning results with rotated bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict, optional` | Default configuration for the predictor. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Configuration overrides that take precedence over the default config. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be invoked during prediction. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `args` | `namespace` | Configuration arguments for the predictor. |
| `model` | `torch.nn.Module` | The loaded YOLO OBB model. |

**Methods**

| Name | Description |
| --- | --- |
| [`construct_result`](#ultralytics.models.yolo.obb.predict.OBBPredictor.construct_result) | Construct the result object from the prediction. |

**Examples**

```python
>>> from ultralytics.utils import ASSETS
>>> from ultralytics.models.yolo.obb import OBBPredictor
>>> args = dict(model="yolo26n-obb.pt", source=ASSETS)
>>> predictor = OBBPredictor(overrides=args)
>>> predictor.predict_cli()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/predict.py#L10-L56"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class OBBPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.

    This predictor handles oriented bounding box detection tasks, processing images and returning results with rotated
    bounding boxes.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO OBB model.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.obb import OBBPredictor
        >>> args = dict(model="yolo26n-obb.pt", source=ASSETS)
        >>> predictor = OBBPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize OBBPredictor with optional model and data configuration overrides.

        Args:
            cfg (dict, optional): Default configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over the default config.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.predict.OBBPredictor.construct_result` {#ultralytics.models.yolo.obb.predict.OBBPredictor.construct\_result}

```python
def construct_result(self, pred, img, orig_img, img_path)
```

Construct the result object from the prediction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred` | `torch.Tensor` | The predicted bounding boxes, scores, and rotation angles with shape (N, 7) where the<br>    last dimension contains [x, y, w, h, confidence, class_id, angle]. | *required* |
| `img` | `torch.Tensor` | The image after preprocessing with shape (B, C, H, W). | *required* |
| `orig_img` | `np.ndarray` | The original image before preprocessing. | *required* |
| `img_path` | `str` | The path to the original image. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Results` | The result object containing the original image, image path, class names, and oriented bounding |

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/predict.py#L39-L56"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def construct_result(self, pred, img, orig_img, img_path):
    """Construct the result object from the prediction.

    Args:
        pred (torch.Tensor): The predicted bounding boxes, scores, and rotation angles with shape (N, 7) where the
            last dimension contains [x, y, w, h, confidence, class_id, angle].
        img (torch.Tensor): The image after preprocessing with shape (B, C, H, W).
        orig_img (np.ndarray): The original image before preprocessing.
        img_path (str): The path to the original image.

    Returns:
        (Results): The result object containing the original image, image path, class names, and oriented bounding
            boxes.
    """
    rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
    rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
    obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
    return Results(orig_img, path=img_path, names=self.model.names, obb=obb)
```
</details>

<br><br>
