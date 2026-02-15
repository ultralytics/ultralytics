---
description: Explore the Ultralytics YOLO Detection Predictor. Learn how to implement and use the DetectionPredictor class for object detection in Python.
keywords: YOLO, Ultralytics, DetectionPredictor, object detection, Python, machine learning, AI, non_max_suppression
---

# Reference for `ultralytics/models/yolo/detect/predict.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/predict.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/predict.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DetectionPredictor`](#ultralytics.models.yolo.detect.predict.DetectionPredictor)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DetectionPredictor.postprocess`](#ultralytics.models.yolo.detect.predict.DetectionPredictor.postprocess)
        - [`DetectionPredictor.get_obj_feats`](#ultralytics.models.yolo.detect.predict.DetectionPredictor.get_obj_feats)
        - [`DetectionPredictor.construct_results`](#ultralytics.models.yolo.detect.predict.DetectionPredictor.construct_results)
        - [`DetectionPredictor.construct_result`](#ultralytics.models.yolo.detect.predict.DetectionPredictor.construct_result)


## Class `ultralytics.models.yolo.detect.predict.DetectionPredictor` {#ultralytics.models.yolo.detect.predict.DetectionPredictor}

```python
DetectionPredictor()
```

**Bases:** `BasePredictor`

A class extending the BasePredictor class for prediction based on a detection model.

This predictor specializes in object detection tasks, processing model outputs into meaningful detection results with bounding boxes and class predictions.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `args` | `namespace` | Configuration arguments for the predictor. |
| `model` | `nn.Module` | The detection model used for inference. |
| `batch` | `list` | Batch of images and metadata for processing. |

**Methods**

| Name | Description |
| --- | --- |
| [`construct_result`](#ultralytics.models.yolo.detect.predict.DetectionPredictor.construct_result) | Construct a single Results object from one image prediction. |
| [`construct_results`](#ultralytics.models.yolo.detect.predict.DetectionPredictor.construct_results) | Construct a list of Results objects from model predictions. |
| [`get_obj_feats`](#ultralytics.models.yolo.detect.predict.DetectionPredictor.get_obj_feats) | Extract object features from the feature maps. |
| [`postprocess`](#ultralytics.models.yolo.detect.predict.DetectionPredictor.postprocess) | Post-process predictions and return a list of Results objects. |

**Examples**

```python
>>> from ultralytics.utils import ASSETS
>>> from ultralytics.models.yolo.detect import DetectionPredictor
>>> args = dict(model="yolo26n.pt", source=ASSETS)
>>> predictor = DetectionPredictor(overrides=args)
>>> predictor.predict_cli()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/predict.py#L8-L122"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DetectionPredictor(BasePredictor):
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.predict.DetectionPredictor.construct_result` {#ultralytics.models.yolo.detect.predict.DetectionPredictor.construct\_result}

```python
def construct_result(self, pred, img, orig_img, img_path)
```

Construct a single Results object from one image prediction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred` | `torch.Tensor` | Predicted boxes and scores with shape (N, 6) where N is the number of detections. | *required* |
| `img` | `torch.Tensor` | Preprocessed image tensor used for inference. | *required* |
| `orig_img` | `np.ndarray` | Original image before preprocessing. | *required* |
| `img_path` | `str` | Path to the original image file. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Results` | Results object containing the original image, image path, class names, and scaled bounding boxes. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/predict.py#L109-L122"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def construct_result(self, pred, img, orig_img, img_path):
    """Construct a single Results object from one image prediction.

    Args:
        pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
        img (torch.Tensor): Preprocessed image tensor used for inference.
        orig_img (np.ndarray): Original image before preprocessing.
        img_path (str): Path to the original image file.

    Returns:
        (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
    """
    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.predict.DetectionPredictor.construct_results` {#ultralytics.models.yolo.detect.predict.DetectionPredictor.construct\_results}

```python
def construct_results(self, preds, img, orig_imgs)
```

Construct a list of Results objects from model predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `list[torch.Tensor]` | List of predicted bounding boxes and scores for each image. | *required* |
| `img` | `torch.Tensor` | Batch of preprocessed images used for inference. | *required* |
| `orig_imgs` | `list[np.ndarray]` | List of original images before preprocessing. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[Results]` | List of Results objects containing detection information for each image. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/predict.py#L93-L107"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def construct_results(self, preds, img, orig_imgs):
    """Construct a list of Results objects from model predictions.

    Args:
        preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
        img (torch.Tensor): Batch of preprocessed images used for inference.
        orig_imgs (list[np.ndarray]): List of original images before preprocessing.

    Returns:
        (list[Results]): List of Results objects containing detection information for each image.
    """
    return [
        self.construct_result(pred, img, orig_img, img_path)
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
    ]
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.predict.DetectionPredictor.get_obj_feats` {#ultralytics.models.yolo.detect.predict.DetectionPredictor.get\_obj\_feats}

```python
def get_obj_feats(feat_maps, idxs)
```

Extract object features from the feature maps.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `feat_maps` |  |  | *required* |
| `idxs` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/predict.py#L83-L91"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def get_obj_feats(feat_maps, idxs):
    """Extract object features from the feature maps."""
    import torch

    s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
    obj_feats = torch.cat(
        [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
    )  # mean reduce all vectors to same length
    return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.predict.DetectionPredictor.postprocess` {#ultralytics.models.yolo.detect.predict.DetectionPredictor.postprocess}

```python
def postprocess(self, preds, img, orig_imgs, **kwargs)
```

Post-process predictions and return a list of Results objects.

This method applies non-maximum suppression to raw model predictions and prepares them for visualization and further analysis.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw predictions from the model. | *required* |
| `img` | `torch.Tensor` | Processed input image tensor in model input format. | *required* |
| `orig_imgs` | `torch.Tensor | list` | Original input images before preprocessing. | *required* |
| `**kwargs` | `Any` | Additional keyword arguments. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | List of Results objects containing the post-processed predictions. |

**Examples**

```python
>>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
>>> results = predictor.predict("path/to/image.jpg")
>>> processed_results = predictor.postprocess(preds, img, orig_imgs)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/predict.py#L33-L80"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def postprocess(self, preds, img, orig_imgs, **kwargs):
    """Post-process predictions and return a list of Results objects.

    This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
    further analysis.

    Args:
        preds (torch.Tensor): Raw predictions from the model.
        img (torch.Tensor): Processed input image tensor in model input format.
        orig_imgs (torch.Tensor | list): Original input images before preprocessing.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        (list): List of Results objects containing the post-processed predictions.

    Examples:
        >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
        >>> results = predictor.predict("path/to/image.jpg")
        >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
    """
    save_feats = getattr(self, "_feats", None) is not None
    preds = nms.non_max_suppression(
        preds,
        self.args.conf,
        self.args.iou,
        self.args.classes,
        self.args.agnostic_nms,
        max_det=self.args.max_det,
        nc=0 if self.args.task == "detect" else len(self.model.names),
        end2end=getattr(self.model, "end2end", False),
        rotated=self.args.task == "obb",
        return_idxs=save_feats,
    )

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

    if save_feats:
        obj_feats = self.get_obj_feats(self._feats, preds[1])
        preds = preds[0]

    results = self.construct_results(preds, img, orig_imgs, **kwargs)

    if save_feats:
        for r, f in zip(results, obj_feats):
            r.feats = f  # add object features to results

    return results
```
</details>

<br><br>
