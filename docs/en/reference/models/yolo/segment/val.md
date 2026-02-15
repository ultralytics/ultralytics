---
description: Explore the YOLO Segmentation Validator module for validating segment models. Understand its usage, metrics, and implementation within the Ultralytics framework.
keywords: YOLO, segmentation, validator, Ultralytics, model validation, machine learning, deep learning, AI, computer vision
---

# Reference for `ultralytics/models/yolo/segment/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`SegmentationValidator`](#ultralytics.models.yolo.segment.val.SegmentationValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`SegmentationValidator.preprocess`](#ultralytics.models.yolo.segment.val.SegmentationValidator.preprocess)
        - [`SegmentationValidator.init_metrics`](#ultralytics.models.yolo.segment.val.SegmentationValidator.init_metrics)
        - [`SegmentationValidator.get_desc`](#ultralytics.models.yolo.segment.val.SegmentationValidator.get_desc)
        - [`SegmentationValidator.postprocess`](#ultralytics.models.yolo.segment.val.SegmentationValidator.postprocess)
        - [`SegmentationValidator._prepare_batch`](#ultralytics.models.yolo.segment.val.SegmentationValidator._prepare_batch)
        - [`SegmentationValidator._process_batch`](#ultralytics.models.yolo.segment.val.SegmentationValidator._process_batch)
        - [`SegmentationValidator.plot_predictions`](#ultralytics.models.yolo.segment.val.SegmentationValidator.plot_predictions)
        - [`SegmentationValidator.save_one_txt`](#ultralytics.models.yolo.segment.val.SegmentationValidator.save_one_txt)
        - [`SegmentationValidator.pred_to_json`](#ultralytics.models.yolo.segment.val.SegmentationValidator.pred_to_json)
        - [`SegmentationValidator.scale_preds`](#ultralytics.models.yolo.segment.val.SegmentationValidator.scale_preds)
        - [`SegmentationValidator.eval_json`](#ultralytics.models.yolo.segment.val.SegmentationValidator.eval_json)


## Class `ultralytics.models.yolo.segment.val.SegmentationValidator` {#ultralytics.models.yolo.segment.val.SegmentationValidator}

```python
SegmentationValidator(self, dataloader = None, save_dir = None, args = None, _callbacks = None) -> None
```

**Bases:** `DetectionValidator`

A class extending the DetectionValidator class for validation based on a segmentation model.

This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions to compute metrics such as mAP for both detection and segmentation tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` | `torch.utils.data.DataLoader, optional` | DataLoader to use for validation. | `None` |
| `save_dir` | `Path, optional` | Directory to save results. | `None` |
| `args` | `dict, optional` | Arguments for the validator. | `None` |
| `_callbacks` | `list, optional` | List of callback functions. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `plot_masks` | `list` | List to store masks for plotting. |
| `process` | `callable` | Function to process masks based on save_json and save_txt flags. |
| `args` | `SimpleNamespace` | Arguments for the validator. |
| `metrics` | `SegmentMetrics` | Metrics calculator for segmentation tasks. |
| `stats` | `dict` | Dictionary to store statistics during validation. |

**Methods**

| Name | Description |
| --- | --- |
| [`_prepare_batch`](#ultralytics.models.yolo.segment.val.SegmentationValidator._prepare_batch) | Prepare a batch for validation by processing images and targets. |
| [`_process_batch`](#ultralytics.models.yolo.segment.val.SegmentationValidator._process_batch) | Compute correct prediction matrix for a batch based on bounding boxes and optional masks. |
| [`eval_json`](#ultralytics.models.yolo.segment.val.SegmentationValidator.eval_json) | Return COCO-style instance segmentation evaluation metrics. |
| [`get_desc`](#ultralytics.models.yolo.segment.val.SegmentationValidator.get_desc) | Return a formatted description of evaluation metrics. |
| [`init_metrics`](#ultralytics.models.yolo.segment.val.SegmentationValidator.init_metrics) | Initialize metrics and select mask processing function based on save_json flag. |
| [`plot_predictions`](#ultralytics.models.yolo.segment.val.SegmentationValidator.plot_predictions) | Plot batch predictions with masks and bounding boxes. |
| [`postprocess`](#ultralytics.models.yolo.segment.val.SegmentationValidator.postprocess) | Post-process YOLO predictions and return output detections with proto. |
| [`pred_to_json`](#ultralytics.models.yolo.segment.val.SegmentationValidator.pred_to_json) | Save one JSON result for COCO evaluation. |
| [`preprocess`](#ultralytics.models.yolo.segment.val.SegmentationValidator.preprocess) | Preprocess batch of images for YOLO segmentation validation. |
| [`save_one_txt`](#ultralytics.models.yolo.segment.val.SegmentationValidator.save_one_txt) | Save YOLO detections to a txt file in normalized coordinates in a specific format. |
| [`scale_preds`](#ultralytics.models.yolo.segment.val.SegmentationValidator.scale_preds) | Scales predictions to the original image size. |

**Examples**

```python
>>> from ultralytics.models.yolo.segment import SegmentationValidator
>>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml")
>>> validator = SegmentationValidator(args=args)
>>> validator()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L18-L307"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SegmentationValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a segmentation model.

    This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions to
    compute metrics such as mAP for both detection and segmentation tasks.

    Attributes:
        plot_masks (list): List to store masks for plotting.
        process (callable): Function to process masks based on save_json and save_txt flags.
        args (SimpleNamespace): Arguments for the validator.
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks.
        stats (dict): Dictionary to store statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator._prepare_batch` {#ultralytics.models.yolo.segment.val.SegmentationValidator.\_prepare\_batch}

```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]
```

Prepare a batch for validation by processing images and targets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `si` | `int` | Sample index within the batch. | *required* |
| `batch` | `dict[str, Any]` | Batch data containing images and annotations. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Prepared batch with processed annotations. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L118-L142"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
    """Prepare a batch for validation by processing images and targets.

    Args:
        si (int): Sample index within the batch.
        batch (dict[str, Any]): Batch data containing images and annotations.

    Returns:
        (dict[str, Any]): Prepared batch with processed annotations.
    """
    prepared_batch = super()._prepare_batch(si, batch)
    nl = prepared_batch["cls"].shape[0]
    if self.args.overlap_mask:
        masks = batch["masks"][si]
        index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
        masks = (masks == index).float()
    else:
        masks = batch["masks"][batch["batch_idx"] == si]
    if nl:
        mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
        if masks.shape[1:] != mask_size:
            masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
            masks = masks.gt_(0.5)
    prepared_batch["masks"] = masks
    return prepared_batch
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator._process_batch` {#ultralytics.models.yolo.segment.val.SegmentationValidator.\_process\_batch}

```python
def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]
```

Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` | Dictionary containing predictions with keys like 'cls' and 'masks'. | *required* |
| `batch` | `dict[str, Any]` | Dictionary containing batch data with keys like 'cls' and 'masks'. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, np.ndarray]` | A dictionary containing correct prediction matrices including 'tp_m' for mask IoU. |

**Examples**

```python
>>> preds = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
>>> batch = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
>>> correct_preds = validator._process_batch(preds, batch)
```

!!! note "Notes"

    - This method computes IoU between predicted and ground truth masks.
    - Overlapping masks are handled based on the overlap_mask argument setting.

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L144-L171"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

    Args:
        preds (dict[str, torch.Tensor]): Dictionary containing predictions with keys like 'cls' and 'masks'.
        batch (dict[str, Any]): Dictionary containing batch data with keys like 'cls' and 'masks'.

    Returns:
        (dict[str, np.ndarray]): A dictionary containing correct prediction matrices including 'tp_m' for mask IoU.

    Examples:
        >>> preds = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
        >>> batch = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
        >>> correct_preds = validator._process_batch(preds, batch)

    Notes:
        - This method computes IoU between predicted and ground truth masks.
        - Overlapping masks are handled based on the overlap_mask argument setting.
    """
    tp = super()._process_batch(preds, batch)
    gt_cls = batch["cls"]
    if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
        tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
    else:
        iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())  # float, uint8
        tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
    tp.update({"tp_m": tp_m})  # update tp with mask IoU
    return tp
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.eval_json` {#ultralytics.models.yolo.segment.val.SegmentationValidator.eval\_json}

```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]
```

Return COCO-style instance segmentation evaluation metrics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stats` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L299-L307"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
    """Return COCO-style instance segmentation evaluation metrics."""
    pred_json = self.save_dir / "predictions.json"  # predictions
    anno_json = (
        self.data["path"]
        / "annotations"
        / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
    )  # annotations
    return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.get_desc` {#ultralytics.models.yolo.segment.val.SegmentationValidator.get\_desc}

```python
def get_desc(self) -> str
```

Return a formatted description of evaluation metrics.

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L77-L91"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_desc(self) -> str:
    """Return a formatted description of evaluation metrics."""
    return ("%22s" + "%11s" * 10) % (
        "Class",
        "Images",
        "Instances",
        "Box(P",
        "R",
        "mAP50",
        "mAP50-95)",
        "Mask(P",
        "R",
        "mAP50",
        "mAP50-95)",
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.init_metrics` {#ultralytics.models.yolo.segment.val.SegmentationValidator.init\_metrics}

```python
def init_metrics(self, model: torch.nn.Module) -> None
```

Initialize metrics and select mask processing function based on save_json flag.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | Model to validate. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L65-L75"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_metrics(self, model: torch.nn.Module) -> None:
    """Initialize metrics and select mask processing function based on save_json flag.

    Args:
        model (torch.nn.Module): Model to validate.
    """
    super().init_metrics(model)
    if self.args.save_json:
        check_requirements("faster-coco-eval>=1.6.7")
    # More accurate vs faster
    self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.plot_predictions` {#ultralytics.models.yolo.segment.val.SegmentationValidator.plot\_predictions}

```python
def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None
```

Plot batch predictions with masks and bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Batch containing images and annotations. | *required* |
| `preds` | `list[dict[str, torch.Tensor]]` | List of predictions from the model. | *required* |
| `ni` | `int` | Batch index. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L173-L186"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
    """Plot batch predictions with masks and bounding boxes.

    Args:
        batch (dict[str, Any]): Batch containing images and annotations.
        preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
        ni (int): Batch index.
    """
    for p in preds:
        masks = p["masks"]
        if masks.shape[0] > self.args.max_det:
            LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
        p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
    super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)  # plot bboxes
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.postprocess` {#ultralytics.models.yolo.segment.val.SegmentationValidator.postprocess}

```python
def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]
```

Post-process YOLO predictions and return output detections with proto.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `list[torch.Tensor]` | Raw predictions from the model. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, torch.Tensor]]` | Processed detection predictions with masks. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L93-L116"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Post-process YOLO predictions and return output detections with proto.

    Args:
        preds (list[torch.Tensor]): Raw predictions from the model.

    Returns:
        (list[dict[str, torch.Tensor]]): Processed detection predictions with masks.
    """
    proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
    preds = super().postprocess(preds[0])
    imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
    for i, pred in enumerate(preds):
        coefficient = pred.pop("extra")
        pred["masks"] = (
            self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
            if coefficient.shape[0]
            else torch.zeros(
                (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                dtype=torch.uint8,
                device=pred["bboxes"].device,
            )
        )
    return preds
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.pred_to_json` {#ultralytics.models.yolo.segment.val.SegmentationValidator.pred\_to\_json}

```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None
```

Save one JSON result for COCO evaluation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Predictions containing bboxes, masks, confidence scores, and classes. | *required* |
| `pbatch` | `dict[str, Any]` | Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L207-L288"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
    """Save one JSON result for COCO evaluation.

    Args:
        predn (dict[str, torch.Tensor]): Predictions containing bboxes, masks, confidence scores, and classes.
        pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
    """

    def to_string(counts: list[int]) -> str:
        """Converts the RLE object into a compact string representation. Each count is delta-encoded and
        variable-length encoded as a string.

        Args:
            counts (list[int]): List of RLE counts.
        """
        result = []

        for i in range(len(counts)):
            x = int(counts[i])

            # Apply delta encoding for all counts after the second entry
            if i > 2:
                x -= int(counts[i - 2])

            # Variable-length encode the value
            while True:
                c = x & 0x1F  # Take 5 bits
                x >>= 5

                # If the sign bit (0x10) is set, continue if x != -1;
                # otherwise, continue if x != 0
                more = (x != -1) if (c & 0x10) else (x != 0)
                if more:
                    c |= 0x20  # Set continuation bit
                c += 48  # Shift to ASCII
                result.append(chr(c))
                if not more:
                    break

        return "".join(result)

    def multi_encode(pixels: torch.Tensor) -> list[int]:
        """Convert multiple binary masks using Run-Length Encoding (RLE).

        Args:
            pixels (torch.Tensor): A 2D tensor where each row represents a flattened binary mask with shape [N,
                H*W].

        Returns:
            (list[list[int]]): A list of RLE counts for each mask.
        """
        transitions = pixels[:, 1:] != pixels[:, :-1]
        row_idx, col_idx = torch.where(transitions)
        col_idx = col_idx + 1

        # Compute run lengths
        counts = []
        for i in range(pixels.shape[0]):
            positions = col_idx[row_idx == i]
            if len(positions):
                count = torch.diff(positions).tolist()
                count.insert(0, positions[0].item())
                count.append(len(pixels[i]) - positions[-1].item())
            else:
                count = [len(pixels[i])]

            # Ensure starting with background (0) count
            if pixels[i][0].item() == 1:
                count = [0, *count]
            counts.append(count)

        return counts

    pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)  # N, H*W
    h, w = predn["masks"].shape[1:3]
    counts = multi_encode(pred_masks)
    rles = []
    for c in counts:
        rles.append({"size": [h, w], "counts": to_string(c)})
    super().pred_to_json(predn, pbatch)
    for i, r in enumerate(rles):
        self.jdict[-len(rles) + i]["segmentation"] = r  # segmentation
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.preprocess` {#ultralytics.models.yolo.segment.val.SegmentationValidator.preprocess}

```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]
```

Preprocess batch of images for YOLO segmentation validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Batch containing images and annotations. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Preprocessed batch. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L52-L63"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
    """Preprocess batch of images for YOLO segmentation validation.

    Args:
        batch (dict[str, Any]): Batch containing images and annotations.

    Returns:
        (dict[str, Any]): Preprocessed batch.
    """
    batch = super().preprocess(batch)
    batch["masks"] = batch["masks"].float()
    return batch
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.save_one_txt` {#ultralytics.models.yolo.segment.val.SegmentationValidator.save\_one\_txt}

```python
def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None
```

Save YOLO detections to a txt file in normalized coordinates in a specific format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'masks' keys. | *required* |
| `save_conf` | `bool` | Whether to save confidence scores. | *required* |
| `shape` | `tuple[int, int]` | Shape of the original image. | *required* |
| `file` | `Path` | File path to save the detections. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L188-L205"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
    """Save YOLO detections to a txt file in normalized coordinates in a specific format.

    Args:
        predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'masks' keys.
        save_conf (bool): Whether to save confidence scores.
        shape (tuple[int, int]): Shape of the original image.
        file (Path): File path to save the detections.
    """
    from ultralytics.engine.results import Results

    Results(
        np.zeros((shape[0], shape[1]), dtype=np.uint8),
        path=None,
        names=self.names,
        boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
    ).save_txt(file, save_conf=save_conf)
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.val.SegmentationValidator.scale_preds` {#ultralytics.models.yolo.segment.val.SegmentationValidator.scale\_preds}

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
<summary>Source code in <code>ultralytics/models/yolo/segment/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/val.py#L290-L297"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Scales predictions to the original image size."""
    return {
        **super().scale_preds(predn, pbatch),
        "masks": ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[
            0
        ].byte(),
    }
```
</details>

<br><br>
