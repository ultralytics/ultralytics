---
description: Explore the PoseValidator class for YOLO models. Learn how to extend DetectionValidator for pose validation with example code and detailed methods.
keywords: Ultralytics, YOLO, PoseValidator, pose validation, machine learning, object detection, keypoints, python code, AI, deep learning
---

# Reference for `ultralytics/models/yolo/pose/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`PoseValidator`](#ultralytics.models.yolo.pose.val.PoseValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`PoseValidator.preprocess`](#ultralytics.models.yolo.pose.val.PoseValidator.preprocess)
        - [`PoseValidator.get_desc`](#ultralytics.models.yolo.pose.val.PoseValidator.get_desc)
        - [`PoseValidator.init_metrics`](#ultralytics.models.yolo.pose.val.PoseValidator.init_metrics)
        - [`PoseValidator.postprocess`](#ultralytics.models.yolo.pose.val.PoseValidator.postprocess)
        - [`PoseValidator._prepare_batch`](#ultralytics.models.yolo.pose.val.PoseValidator._prepare_batch)
        - [`PoseValidator._process_batch`](#ultralytics.models.yolo.pose.val.PoseValidator._process_batch)
        - [`PoseValidator.save_one_txt`](#ultralytics.models.yolo.pose.val.PoseValidator.save_one_txt)
        - [`PoseValidator.pred_to_json`](#ultralytics.models.yolo.pose.val.PoseValidator.pred_to_json)
        - [`PoseValidator.scale_preds`](#ultralytics.models.yolo.pose.val.PoseValidator.scale_preds)
        - [`PoseValidator.eval_json`](#ultralytics.models.yolo.pose.val.PoseValidator.eval_json)


## Class `ultralytics.models.yolo.pose.val.PoseValidator` {#ultralytics.models.yolo.pose.val.PoseValidator}

```python
PoseValidator(self, dataloader = None, save_dir = None, args = None, _callbacks = None) -> None
```

**Bases:** `DetectionValidator`

A class extending the DetectionValidator class for validation based on a pose model.

This validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized metrics for pose evaluation.

This validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized metrics for pose evaluation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` | `torch.utils.data.DataLoader, optional` | DataLoader to be used for validation. | `None` |
| `save_dir` | `Path | str, optional` | Directory to save results. | `None` |
| `args` | `dict, optional` | Arguments for the validator including task set to "pose". | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be executed during validation. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `sigma` | `np.ndarray` | Sigma values for OKS calculation, either OKS_SIGMA or ones divided by number of keypoints. |
| `kpt_shape` | `list[int]` | Shape of the keypoints, typically [17, 3] for COCO format. |
| `args` | `dict` | Arguments for the validator including task set to "pose". |
| `metrics` | `PoseMetrics` | Metrics object for pose evaluation. |

**Methods**

| Name | Description |
| --- | --- |
| [`_prepare_batch`](#ultralytics.models.yolo.pose.val.PoseValidator._prepare_batch) | Prepare a batch for processing by converting keypoints to float and scaling to original dimensions. |
| [`_process_batch`](#ultralytics.models.yolo.pose.val.PoseValidator._process_batch) | Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground |
| [`eval_json`](#ultralytics.models.yolo.pose.val.PoseValidator.eval_json) | Evaluate object detection model using COCO JSON format. |
| [`get_desc`](#ultralytics.models.yolo.pose.val.PoseValidator.get_desc) | Return description of evaluation metrics in string format. |
| [`init_metrics`](#ultralytics.models.yolo.pose.val.PoseValidator.init_metrics) | Initialize evaluation metrics for YOLO pose validation. |
| [`postprocess`](#ultralytics.models.yolo.pose.val.PoseValidator.postprocess) | Postprocess YOLO predictions to extract and reshape keypoints for pose estimation. |
| [`pred_to_json`](#ultralytics.models.yolo.pose.val.PoseValidator.pred_to_json) | Convert YOLO predictions to COCO JSON format. |
| [`preprocess`](#ultralytics.models.yolo.pose.val.PoseValidator.preprocess) | Preprocess batch by converting keypoints data to float and moving it to the device. |
| [`save_one_txt`](#ultralytics.models.yolo.pose.val.PoseValidator.save_one_txt) | Save YOLO pose detections to a text file in normalized coordinates. |
| [`scale_preds`](#ultralytics.models.yolo.pose.val.PoseValidator.scale_preds) | Scales predictions to the original image size. |

**Examples**

```python
>>> from ultralytics.models.yolo.pose import PoseValidator
>>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml")
>>> validator = PoseValidator(args=args)
>>> validator()
```

!!! note "Notes"

    This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
    for OKS calculation and sets up PoseMetrics for evaluation. A warning is displayed when using Apple MPS
    due to a known bug with pose models.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L16-L248"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PoseValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a pose model.

    This validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized
    metrics for pose evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation, either OKS_SIGMA or ones divided by number of keypoints.
        kpt_shape (list[int]): Shape of the keypoints, typically [17, 3] for COCO format.
        args (dict): Arguments for the validator including task set to "pose".
        metrics (PoseMetrics): Metrics object for pose evaluation.

    Methods:
        preprocess: Preprocess batch by converting keypoints data to float and moving it to the device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize pose estimation metrics for YOLO model.
        _prepare_batch: Prepare a batch for processing by converting keypoints to float and scaling to original
            dimensions.
        _prepare_pred: Prepare and scale keypoints in predictions for pose processing.
        _process_batch: Return correct prediction matrix by computing Intersection over Union (IoU) between detections
            and ground truth.
        plot_val_samples: Plot and save validation set samples with ground truth bounding boxes and keypoints.
        plot_predictions: Plot and save model predictions with bounding boxes and keypoints.
        save_one_txt: Save YOLO pose detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate object detection model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()

    Notes:
        This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
        for OKS calculation and sets up PoseMetrics for evaluation. A warning is displayed when using Apple MPS
        due to a known bug with pose models.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize a PoseValidator object for pose estimation validation.

        This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
        specialized metrics for pose evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "pose".
            _callbacks (list, optional): List of callback functions to be executed during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator._prepare_batch` {#ultralytics.models.yolo.pose.val.PoseValidator.\_prepare\_batch}

```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]
```

Prepare a batch for processing by converting keypoints to float and scaling to original dimensions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `si` | `int` | Sample index within the batch. | *required* |
| `batch` | `dict[str, Any]` | Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Prepared batch with keypoints scaled to original image dimensions. |

!!! note "Notes"

    This method extends the parent class's _prepare_batch method by adding keypoint processing.
    Keypoints are scaled from normalized coordinates to original image dimensions.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L135-L156"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
    """Prepare a batch for processing by converting keypoints to float and scaling to original dimensions.

    Args:
        si (int): Sample index within the batch.
        batch (dict[str, Any]): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

    Returns:
        (dict[str, Any]): Prepared batch with keypoints scaled to original image dimensions.

    Notes:
        This method extends the parent class's _prepare_batch method by adding keypoint processing.
        Keypoints are scaled from normalized coordinates to original image dimensions.
    """
    pbatch = super()._prepare_batch(si, batch)
    kpts = batch["keypoints"][batch["batch_idx"] == si]
    h, w = pbatch["imgsz"]
    kpts = kpts.clone()
    kpts[..., 0] *= w
    kpts[..., 1] *= h
    pbatch["keypoints"] = kpts
    return pbatch
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator._process_batch` {#ultralytics.models.yolo.pose.val.PoseValidator.\_process\_batch}

```python
def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]
```

Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground

truth.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` | Dictionary containing prediction data with keys 'cls' for class predictions<br>    and 'keypoints' for keypoint predictions. | *required* |
| `batch` | `dict[str, Any]` | Dictionary containing ground truth data with keys 'cls' for class labels, 'bboxes'<br>    for bounding boxes, and 'keypoints' for keypoint annotations. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, np.ndarray]` | Dictionary containing the correct prediction matrix including 'tp_p' for pose true |

!!! note "Notes"

    `0.53` scale factor used in area computation is referenced from
    https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L158-L186"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground
    truth.

    Args:
        preds (dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
            and 'keypoints' for keypoint predictions.
        batch (dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels, 'bboxes'
            for bounding boxes, and 'keypoints' for keypoint annotations.

    Returns:
        (dict[str, np.ndarray]): Dictionary containing the correct prediction matrix including 'tp_p' for pose true
            positives across 10 IoU levels.

    Notes:
        `0.53` scale factor used in area computation is referenced from
        https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
    """
    tp = super()._process_batch(preds, batch)
    gt_cls = batch["cls"]
    if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
        tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
    else:
        # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
        area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
        iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
        tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
    tp.update({"tp_p": tp_p})  # update tp with kpts IoU
    return tp
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator.eval_json` {#ultralytics.models.yolo.pose.val.PoseValidator.eval\_json}

```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]
```

Evaluate object detection model using COCO JSON format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stats` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L244-L248"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
    """Evaluate object detection model using COCO JSON format."""
    anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
    pred_json = self.save_dir / "predictions.json"  # predictions
    return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator.get_desc` {#ultralytics.models.yolo.pose.val.PoseValidator.get\_desc}

```python
def get_desc(self) -> str
```

Return description of evaluation metrics in string format.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L79-L93"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_desc(self) -> str:
    """Return description of evaluation metrics in string format."""
    return ("%22s" + "%11s" * 10) % (
        "Class",
        "Images",
        "Instances",
        "Box(P",
        "R",
        "mAP50",
        "mAP50-95)",
        "Pose(P",
        "R",
        "mAP50",
        "mAP50-95)",
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator.init_metrics` {#ultralytics.models.yolo.pose.val.PoseValidator.init\_metrics}

```python
def init_metrics(self, model: torch.nn.Module) -> None
```

Initialize evaluation metrics for YOLO pose validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | Model to validate. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L95-L105"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_metrics(self, model: torch.nn.Module) -> None:
    """Initialize evaluation metrics for YOLO pose validation.

    Args:
        model (torch.nn.Module): Model to validate.
    """
    super().init_metrics(model)
    self.kpt_shape = self.data["kpt_shape"]
    is_pose = self.kpt_shape == [17, 3]
    nkpt = self.kpt_shape[0]
    self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator.postprocess` {#ultralytics.models.yolo.pose.val.PoseValidator.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]
```

Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.

This method extends the parent class postprocessing by extracting keypoints from the 'extra' field of predictions and reshaping them according to the keypoint shape configuration. The keypoints are reshaped from a flattened format to the proper dimensional structure (typically [N, 17, 3] for COCO pose format).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw prediction tensor from the YOLO pose model containing bounding boxes, confidence<br>    scores, class predictions, and keypoint data. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, torch.Tensor]]` | List of processed prediction dictionaries, each containing: |

!!! note "Notes"

    If no keypoints are present in a prediction (empty keypoints), that prediction is skipped and continues
    to the next one. The keypoints are extracted from the 'extra' field which contains additional
    task-specific data beyond basic detection.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L107-L133"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.

    This method extends the parent class postprocessing by extracting keypoints from the 'extra' field of
    predictions and reshaping them according to the keypoint shape configuration. The keypoints are reshaped from a
    flattened format to the proper dimensional structure (typically [N, 17, 3] for COCO pose format).

    Args:
        preds (torch.Tensor): Raw prediction tensor from the YOLO pose model containing bounding boxes, confidence
            scores, class predictions, and keypoint data.

    Returns:
        (list[dict[str, torch.Tensor]]): List of processed prediction dictionaries, each containing:
            - 'bboxes': Bounding box coordinates
            - 'conf': Confidence scores
            - 'cls': Class predictions
            - 'keypoints': Reshaped keypoint coordinates with shape (-1, *self.kpt_shape)

    Notes:
        If no keypoints are present in a prediction (empty keypoints), that prediction is skipped and continues
        to the next one. The keypoints are extracted from the 'extra' field which contains additional
        task-specific data beyond basic detection.
    """
    preds = super().postprocess(preds)
    for pred in preds:
        pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # remove extra if exists
    return preds
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator.pred_to_json` {#ultralytics.models.yolo.pose.val.PoseValidator.pred\_to\_json}

```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None
```

Convert YOLO predictions to COCO JSON format.

This method takes prediction tensors and batch data, converts the bounding boxes from YOLO format to COCO format, and appends the results with keypoints to the internal JSON dictionary (self.jdict).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'kpts'<br>    tensors. | *required* |
| `pbatch` | `dict[str, Any]` | Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'. | *required* |

!!! note "Notes"

    The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
    converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
    before saving to the JSON dictionary.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L211-L230"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
    """Convert YOLO predictions to COCO JSON format.

    This method takes prediction tensors and batch data, converts the bounding boxes from YOLO format to COCO
    format, and appends the results with keypoints to the internal JSON dictionary (self.jdict).

    Args:
        predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'kpts'
            tensors.
        pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

    Notes:
        The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
        converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
        before saving to the JSON dictionary.
    """
    super().pred_to_json(predn, pbatch)
    kpts = predn["kpts"]
    for i, k in enumerate(kpts.flatten(1, 2).tolist()):
        self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator.preprocess` {#ultralytics.models.yolo.pose.val.PoseValidator.preprocess}

```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]
```

Preprocess batch by converting keypoints data to float and moving it to the device.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L73-L77"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
    """Preprocess batch by converting keypoints data to float and moving it to the device."""
    batch = super().preprocess(batch)
    batch["keypoints"] = batch["keypoints"].float()
    return batch
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator.save_one_txt` {#ultralytics.models.yolo.pose.val.PoseValidator.save\_one\_txt}

```python
def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None
```

Save YOLO pose detections to a text file in normalized coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Prediction dict with keys 'bboxes', 'conf', 'cls', and 'keypoints'. | *required* |
| `save_conf` | `bool` | Whether to save confidence scores. | *required* |
| `shape` | `tuple[int, int]` | Shape of the original image (height, width). | *required* |
| `file` | `Path` | Output file path to save detections. | *required* |

!!! note "Notes"

    The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
    normalized (x, y, visibility) values for each point.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L188-L209"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
    """Save YOLO pose detections to a text file in normalized coordinates.

    Args:
        predn (dict[str, torch.Tensor]): Prediction dict with keys 'bboxes', 'conf', 'cls', and 'keypoints'.
        save_conf (bool): Whether to save confidence scores.
        shape (tuple[int, int]): Shape of the original image (height, width).
        file (Path): Output file path to save detections.

    Notes:
        The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
        normalized (x, y, visibility) values for each point.
    """
    from ultralytics.engine.results import Results

    Results(
        np.zeros((shape[0], shape[1]), dtype=np.uint8),
        path=None,
        names=self.names,
        boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        keypoints=predn["keypoints"],
    ).save_txt(file, save_conf=save_conf)
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.val.PoseValidator.scale_preds` {#ultralytics.models.yolo.pose.val.PoseValidator.scale\_preds}

```python
def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]
```

Scales predictions to the original image size.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` |  | *required* |
| `pbatch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/val.py#L232-L242"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Scales predictions to the original image size."""
    return {
        **super().scale_preds(predn, pbatch),
        "kpts": ops.scale_coords(
            pbatch["imgsz"],
            predn["keypoints"].clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        ),
    }
```
</details>

<br><br>
