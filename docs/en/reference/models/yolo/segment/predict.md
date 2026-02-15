---
description: Understand the SegmentationPredictor class for segmentation-based predictions using YOLO. Learn more about its implementation and example usage.
keywords: YOLO, SegmentationPredictor, machine learning, computer vision, object detection, Ultralytics, prediction, model, non-max suppression
---

# Reference for `ultralytics/models/yolo/segment/predict.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/predict.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/predict.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`SegmentationPredictor`](#ultralytics.models.yolo.segment.predict.SegmentationPredictor)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`SegmentationPredictor.postprocess`](#ultralytics.models.yolo.segment.predict.SegmentationPredictor.postprocess)
        - [`SegmentationPredictor.construct_results`](#ultralytics.models.yolo.segment.predict.SegmentationPredictor.construct_results)
        - [`SegmentationPredictor.construct_result`](#ultralytics.models.yolo.segment.predict.SegmentationPredictor.construct_result)


## Class `ultralytics.models.yolo.segment.predict.SegmentationPredictor` {#ultralytics.models.yolo.segment.predict.SegmentationPredictor}

```python
SegmentationPredictor(self, cfg = DEFAULT_CFG, overrides = None, _callbacks = None)
```

**Bases:** `DetectionPredictor`

A class extending the DetectionPredictor class for prediction based on a segmentation model.

This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the prediction results.

This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the prediction results.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict` | Configuration for the predictor. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Configuration overrides that take precedence over cfg. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be invoked during prediction. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `args` | `dict` | Configuration arguments for the predictor. |
| `model` | `torch.nn.Module` | The loaded YOLO segmentation model. |
| `batch` | `list` | Current batch of images being processed. |

**Methods**

| Name | Description |
| --- | --- |
| [`construct_result`](#ultralytics.models.yolo.segment.predict.SegmentationPredictor.construct_result) | Construct a single result object from the prediction. |
| [`construct_results`](#ultralytics.models.yolo.segment.predict.SegmentationPredictor.construct_results) | Construct a list of result objects from the predictions. |
| [`postprocess`](#ultralytics.models.yolo.segment.predict.SegmentationPredictor.postprocess) | Apply non-max suppression and process segmentation detections for each image in the input batch. |

**Examples**

```python
>>> from ultralytics.utils import ASSETS
>>> from ultralytics.models.yolo.segment import SegmentationPredictor
>>> args = dict(model="yolo26n-seg.pt", source=ASSETS)
>>> predictor = SegmentationPredictor(overrides=args)
>>> predictor.predict_cli()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/predict.py#L8-L109"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SegmentationPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process segmentation detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo26n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.predict.SegmentationPredictor.construct_result` {#ultralytics.models.yolo.segment.predict.SegmentationPredictor.construct\_result}

```python
def construct_result(self, pred, img, orig_img, img_path, proto)
```

Construct a single result object from the prediction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred` | `torch.Tensor` | The predicted bounding boxes, scores, and masks. | *required* |
| `img` | `torch.Tensor` | The image after preprocessing. | *required* |
| `orig_img` | `np.ndarray` | The original image before preprocessing. | *required* |
| `img_path` | `str` | The path to the original image. | *required* |
| `proto` | `torch.Tensor` | The prototype masks. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Results` | Result object containing the original image, image path, class names, bounding boxes, and masks. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/predict.py#L84-L109"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def construct_result(self, pred, img, orig_img, img_path, proto):
    """Construct a single result object from the prediction.

    Args:
        pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
        img (torch.Tensor): The image after preprocessing.
        orig_img (np.ndarray): The original image before preprocessing.
        img_path (str): The path to the original image.
        proto (torch.Tensor): The prototype masks.

    Returns:
        (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
    """
    if pred.shape[0] == 0:  # save empty boxes
        masks = None
    elif self.args.retina_masks:
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
    else:
        masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    if masks is not None:
        keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
        if not all(keep):  # most predictions have masks
            pred, masks = pred[keep], masks[keep]  # indexing is slow
    return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.predict.SegmentationPredictor.construct_results` {#ultralytics.models.yolo.segment.predict.SegmentationPredictor.construct\_results}

```python
def construct_results(self, preds, img, orig_imgs, protos)
```

Construct a list of result objects from the predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `list[torch.Tensor]` | List of predicted bounding boxes, scores, and masks. | *required* |
| `img` | `torch.Tensor` | The image after preprocessing. | *required* |
| `orig_imgs` | `list[np.ndarray]` | List of original images before preprocessing. | *required* |
| `protos` | `torch.Tensor` | Prototype masks tensor with shape (B, C, H, W). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[Results]` | List of result objects containing the original images, image paths, class names, bounding |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/predict.py#L66-L82"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def construct_results(self, preds, img, orig_imgs, protos):
    """Construct a list of result objects from the predictions.

    Args:
        preds (list[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
        img (torch.Tensor): The image after preprocessing.
        orig_imgs (list[np.ndarray]): List of original images before preprocessing.
        protos (torch.Tensor): Prototype masks tensor with shape (B, C, H, W).

    Returns:
        (list[Results]): List of result objects containing the original images, image paths, class names, bounding
            boxes, and masks.
    """
    return [
        self.construct_result(pred, img, orig_img, img_path, proto)
        for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
    ]
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.predict.SegmentationPredictor.postprocess` {#ultralytics.models.yolo.segment.predict.SegmentationPredictor.postprocess}

```python
def postprocess(self, preds, img, orig_imgs)
```

Apply non-max suppression and process segmentation detections for each image in the input batch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `tuple` | Model predictions, containing bounding boxes, scores, classes, and mask coefficients. | *required* |
| `img` | `torch.Tensor` | Input image tensor in model format, with shape (B, C, H, W). | *required* |
| `orig_imgs` | `list | torch.Tensor | np.ndarray` | Original image or batch of images. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | List of Results objects containing the segmentation predictions for each image in the batch. Each |

**Examples**

```python
>>> predictor = SegmentationPredictor(overrides=dict(model="yolo26n-seg.pt"))
>>> results = predictor.postprocess(preds, img, orig_img)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/predict.py#L46-L64"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def postprocess(self, preds, img, orig_imgs):
    """Apply non-max suppression and process segmentation detections for each image in the input batch.

    Args:
        preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
        img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
        orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

    Returns:
        (list): List of Results objects containing the segmentation predictions for each image in the batch. Each
            Results object includes both bounding boxes and segmentation masks.

    Examples:
        >>> predictor = SegmentationPredictor(overrides=dict(model="yolo26n-seg.pt"))
        >>> results = predictor.postprocess(preds, img, orig_img)
    """
    # Extract protos - tuple if PyTorch model or array if exported
    protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
    return super().postprocess(preds[0], img, orig_imgs, protos=protos)
```
</details>

<br><br>
