---
description: Learn about the PosePredictor class for YOLO model predictions on pose data. Get setup instructions, example usage, and implementation details.
keywords: YOLO, Pose Prediction, Ultralytics, PosePredictor, YOLOv8, Machine Learning, Deep Learning, Python, AI Models
---

# Reference for `ultralytics/models/yolo/pose/predict.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/predict.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/predict.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`PosePredictor`](#ultralytics.models.yolo.pose.predict.PosePredictor)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`PosePredictor.construct_result`](#ultralytics.models.yolo.pose.predict.PosePredictor.construct_result)


## Class `ultralytics.models.yolo.pose.predict.PosePredictor` {#ultralytics.models.yolo.pose.predict.PosePredictor}

```python
PosePredictor(self, cfg = DEFAULT_CFG, overrides = None, _callbacks = None)
```

**Bases:** `DetectionPredictor`

A class extending the DetectionPredictor class for prediction based on a pose model.

This class specializes in pose estimation, handling keypoints detection alongside standard object detection capabilities inherited from DetectionPredictor.

Sets up a PosePredictor instance, configuring it for pose detection tasks and handling device-specific warnings for Apple MPS.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `Any` | Configuration for the predictor. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Configuration overrides that take precedence over cfg. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be invoked during prediction. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `args` | `namespace` | Configuration arguments for the predictor. |
| `model` | `torch.nn.Module` | The loaded YOLO pose model with keypoint detection capabilities. |

**Methods**

| Name | Description |
| --- | --- |
| [`construct_result`](#ultralytics.models.yolo.pose.predict.PosePredictor.construct_result) | Construct the result object from the prediction, including keypoints. |

**Examples**

```python
>>> from ultralytics.utils import ASSETS
>>> from ultralytics.models.yolo.pose import PosePredictor
>>> args = dict(model="yolo26n-pose.pt", source=ASSETS)
>>> predictor = PosePredictor(overrides=args)
>>> predictor.predict_cli()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/predict.py#L7-L65"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PosePredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a pose model.

    This class specializes in pose estimation, handling keypoints detection alongside standard object detection
    capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO pose model with keypoint detection capabilities.

    Methods:
        construct_result: Construct the result object from the prediction, including keypoints.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo26n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize PosePredictor for pose estimation tasks.

        Sets up a PosePredictor instance, configuring it for pose detection tasks and handling device-specific warnings
        for Apple MPS.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.predict.PosePredictor.construct_result` {#ultralytics.models.yolo.pose.predict.PosePredictor.construct\_result}

```python
def construct_result(self, pred, img, orig_img, img_path)
```

Construct the result object from the prediction, including keypoints.

Extends the parent class implementation by extracting keypoint data from predictions and adding them to the result object.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred` | `torch.Tensor` | The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is<br>    the number of detections, K is the number of keypoints, and D is the keypoint dimension. | *required* |
| `img` | `torch.Tensor` | The processed input image tensor with shape (B, C, H, W). | *required* |
| `orig_img` | `np.ndarray` | The original unprocessed image as a numpy array. | *required* |
| `img_path` | `str` | The path to the original image file. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Results` | The result object containing the original image, image path, class names, bounding boxes, and |

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/predict.py#L42-L65"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def construct_result(self, pred, img, orig_img, img_path):
    """Construct the result object from the prediction, including keypoints.

    Extends the parent class implementation by extracting keypoint data from predictions and adding them to the
    result object.

    Args:
        pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is
            the number of detections, K is the number of keypoints, and D is the keypoint dimension.
        img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
        orig_img (np.ndarray): The original unprocessed image as a numpy array.
        img_path (str): The path to the original image file.

    Returns:
        (Results): The result object containing the original image, image path, class names, bounding boxes, and
            keypoints.
    """
    result = super().construct_result(pred, img, orig_img, img_path)
    # Extract keypoints from prediction and reshape according to model's keypoint shape
    pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
    # Scale keypoints coordinates to match the original image dimensions
    pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
    result.update(keypoints=pred_kpts)
    return result
```
</details>

<br><br>
