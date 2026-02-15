---
description: Explore the DetectionValidator class for YOLO models in Ultralytics. Learn validation techniques, metrics, and dataset handling for object detection.
keywords: YOLO validation, detection validation, YOLO metrics, Ultralytics, object detection, machine learning, AI
---

# Reference for `ultralytics/models/yolo/detect/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DetectionValidator`](#ultralytics.models.yolo.detect.val.DetectionValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DetectionValidator.preprocess`](#ultralytics.models.yolo.detect.val.DetectionValidator.preprocess)
        - [`DetectionValidator.init_metrics`](#ultralytics.models.yolo.detect.val.DetectionValidator.init_metrics)
        - [`DetectionValidator.get_desc`](#ultralytics.models.yolo.detect.val.DetectionValidator.get_desc)
        - [`DetectionValidator.postprocess`](#ultralytics.models.yolo.detect.val.DetectionValidator.postprocess)
        - [`DetectionValidator._prepare_batch`](#ultralytics.models.yolo.detect.val.DetectionValidator._prepare_batch)
        - [`DetectionValidator._prepare_pred`](#ultralytics.models.yolo.detect.val.DetectionValidator._prepare_pred)
        - [`DetectionValidator.update_metrics`](#ultralytics.models.yolo.detect.val.DetectionValidator.update_metrics)
        - [`DetectionValidator.finalize_metrics`](#ultralytics.models.yolo.detect.val.DetectionValidator.finalize_metrics)
        - [`DetectionValidator.gather_stats`](#ultralytics.models.yolo.detect.val.DetectionValidator.gather_stats)
        - [`DetectionValidator.get_stats`](#ultralytics.models.yolo.detect.val.DetectionValidator.get_stats)
        - [`DetectionValidator.print_results`](#ultralytics.models.yolo.detect.val.DetectionValidator.print_results)
        - [`DetectionValidator._process_batch`](#ultralytics.models.yolo.detect.val.DetectionValidator._process_batch)
        - [`DetectionValidator.build_dataset`](#ultralytics.models.yolo.detect.val.DetectionValidator.build_dataset)
        - [`DetectionValidator.get_dataloader`](#ultralytics.models.yolo.detect.val.DetectionValidator.get_dataloader)
        - [`DetectionValidator.plot_val_samples`](#ultralytics.models.yolo.detect.val.DetectionValidator.plot_val_samples)
        - [`DetectionValidator.plot_predictions`](#ultralytics.models.yolo.detect.val.DetectionValidator.plot_predictions)
        - [`DetectionValidator.save_one_txt`](#ultralytics.models.yolo.detect.val.DetectionValidator.save_one_txt)
        - [`DetectionValidator.pred_to_json`](#ultralytics.models.yolo.detect.val.DetectionValidator.pred_to_json)
        - [`DetectionValidator.scale_preds`](#ultralytics.models.yolo.detect.val.DetectionValidator.scale_preds)
        - [`DetectionValidator.eval_json`](#ultralytics.models.yolo.detect.val.DetectionValidator.eval_json)
        - [`DetectionValidator.coco_evaluate`](#ultralytics.models.yolo.detect.val.DetectionValidator.coco_evaluate)


## Class `ultralytics.models.yolo.detect.val.DetectionValidator` {#ultralytics.models.yolo.detect.val.DetectionValidator}

```python
DetectionValidator(self, dataloader = None, save_dir = None, args = None, _callbacks = None) -> None
```

**Bases:** `BaseValidator`

A class extending the BaseValidator class for validation based on a detection model.

This class implements validation functionality specific to object detection tasks, including metrics calculation, prediction processing, and visualization of results.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` | `torch.utils.data.DataLoader, optional` | DataLoader to use for validation. | `None` |
| `save_dir` | `Path, optional` | Directory to save results. | `None` |
| `args` | `dict[str, Any], optional` | Arguments for the validator. | `None` |
| `_callbacks` | `list[Any], optional` | List of callback functions. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `is_coco` | `bool` | Whether the dataset is COCO. |
| `is_lvis` | `bool` | Whether the dataset is LVIS. |
| `class_map` | `list[int]` | Mapping from model class indices to dataset class indices. |
| `metrics` | `DetMetrics` | Object detection metrics calculator. |
| `iouv` | `torch.Tensor` | IoU thresholds for mAP calculation. |
| `niou` | `int` | Number of IoU thresholds. |
| `lb` | `list[Any]` | List for storing ground truth labels for hybrid saving. |
| `jdict` | `list[dict[str, Any]]` | List for storing JSON detection results. |
| `stats` | `dict[str, list[torch.Tensor]]` | Dictionary for storing statistics during validation. |

**Methods**

| Name | Description |
| --- | --- |
| [`_prepare_batch`](#ultralytics.models.yolo.detect.val.DetectionValidator._prepare_batch) | Prepare a batch of images and annotations for validation. |
| [`_prepare_pred`](#ultralytics.models.yolo.detect.val.DetectionValidator._prepare_pred) | Prepare predictions for evaluation against ground truth. |
| [`_process_batch`](#ultralytics.models.yolo.detect.val.DetectionValidator._process_batch) | Return correct prediction matrix. |
| [`build_dataset`](#ultralytics.models.yolo.detect.val.DetectionValidator.build_dataset) | Build YOLO Dataset. |
| [`coco_evaluate`](#ultralytics.models.yolo.detect.val.DetectionValidator.coco_evaluate) | Evaluate COCO/LVIS metrics using faster-coco-eval library. |
| [`eval_json`](#ultralytics.models.yolo.detect.val.DetectionValidator.eval_json) | Evaluate YOLO output in JSON format and return performance statistics. |
| [`finalize_metrics`](#ultralytics.models.yolo.detect.val.DetectionValidator.finalize_metrics) | Set final values for metrics speed and confusion matrix. |
| [`gather_stats`](#ultralytics.models.yolo.detect.val.DetectionValidator.gather_stats) | Gather stats from all GPUs. |
| [`get_dataloader`](#ultralytics.models.yolo.detect.val.DetectionValidator.get_dataloader) | Construct and return dataloader. |
| [`get_desc`](#ultralytics.models.yolo.detect.val.DetectionValidator.get_desc) | Return a formatted string summarizing class metrics of YOLO model. |
| [`get_stats`](#ultralytics.models.yolo.detect.val.DetectionValidator.get_stats) | Calculate and return metrics statistics. |
| [`init_metrics`](#ultralytics.models.yolo.detect.val.DetectionValidator.init_metrics) | Initialize evaluation metrics for YOLO detection validation. |
| [`plot_predictions`](#ultralytics.models.yolo.detect.val.DetectionValidator.plot_predictions) | Plot predicted bounding boxes on input images and save the result. |
| [`plot_val_samples`](#ultralytics.models.yolo.detect.val.DetectionValidator.plot_val_samples) | Plot validation image samples. |
| [`postprocess`](#ultralytics.models.yolo.detect.val.DetectionValidator.postprocess) | Apply Non-maximum suppression to prediction outputs. |
| [`pred_to_json`](#ultralytics.models.yolo.detect.val.DetectionValidator.pred_to_json) | Serialize YOLO predictions to COCO json format. |
| [`preprocess`](#ultralytics.models.yolo.detect.val.DetectionValidator.preprocess) | Preprocess batch of images for YOLO validation. |
| [`print_results`](#ultralytics.models.yolo.detect.val.DetectionValidator.print_results) | Print training/validation set metrics per class. |
| [`save_one_txt`](#ultralytics.models.yolo.detect.val.DetectionValidator.save_one_txt) | Save YOLO detections to a txt file in normalized coordinates in a specific format. |
| [`scale_preds`](#ultralytics.models.yolo.detect.val.DetectionValidator.scale_preds) | Scales predictions to the original image size. |
| [`update_metrics`](#ultralytics.models.yolo.detect.val.DetectionValidator.update_metrics) | Update metrics with new predictions and ground truth. |

**Examples**

```python
>>> from ultralytics.models.yolo.detect import DetectionValidator
>>> args = dict(model="yolo26n.pt", data="coco8.yaml")
>>> validator = DetectionValidator(args=args)
>>> validator()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L21-L513"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DetectionValidator(BaseValidator):
    """A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list[int]): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (list[Any]): List for storing ground truth labels for hybrid saving.
        jdict (list[dict[str, Any]]): List for storing JSON detection results.
        stats (dict[str, list[torch.Tensor]]): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator.
            _callbacks (list[Any], optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator._prepare_batch` {#ultralytics.models.yolo.detect.val.DetectionValidator.\_prepare\_batch}

```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]
```

Prepare a batch of images and annotations for validation.

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
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L128-L153"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
    """Prepare a batch of images and annotations for validation.

    Args:
        si (int): Sample index within the batch.
        batch (dict[str, Any]): Batch data containing images and annotations.

    Returns:
        (dict[str, Any]): Prepared batch with processed annotations.
    """
    idx = batch["batch_idx"] == si
    cls = batch["cls"][idx].squeeze(-1)
    bbox = batch["bboxes"][idx]
    ori_shape = batch["ori_shape"][si]
    imgsz = batch["img"].shape[2:]
    ratio_pad = batch["ratio_pad"][si]
    if cls.shape[0]:
        bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
    return {
        "cls": cls,
        "bboxes": bbox,
        "ori_shape": ori_shape,
        "imgsz": imgsz,
        "ratio_pad": ratio_pad,
        "im_file": batch["im_file"][si],
    }
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator._prepare_pred` {#ultralytics.models.yolo.detect.val.DetectionValidator.\_prepare\_pred}

```python
def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]
```

Prepare predictions for evaluation against ground truth.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred` | `dict[str, torch.Tensor]` | Post-processed predictions from the model. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, torch.Tensor]` | Prepared predictions in native space. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L155-L166"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Prepare predictions for evaluation against ground truth.

    Args:
        pred (dict[str, torch.Tensor]): Post-processed predictions from the model.

    Returns:
        (dict[str, torch.Tensor]): Prepared predictions in native space.
    """
    if self.args.single_cls:
        pred["cls"] *= 0
    return pred
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator._process_batch` {#ultralytics.models.yolo.detect.val.DetectionValidator.\_process\_batch}

```python
def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]
```

Return correct prediction matrix.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` | Dictionary containing prediction data with 'bboxes' and 'cls' keys. | *required* |
| `batch` | `dict[str, Any]` | Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, np.ndarray]` | Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L274-L288"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return correct prediction matrix.

    Args:
        preds (dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
        batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

    Returns:
        (dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for
            10 IoU levels.
    """
    if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
        return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
    iou = box_iou(batch["bboxes"], preds["bboxes"])
    return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.build_dataset` {#ultralytics.models.yolo.detect.val.DetectionValidator.build\_dataset}

```python
def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset
```

Build YOLO Dataset.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `str` | Path to the folder containing images. | *required* |
| `mode` | `str` | `train` mode or `val` mode, users are able to customize different augmentations for each mode. | `"val"` |
| `batch` | `int, optional` | Size of batches, this is for `rect`. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `Dataset` | YOLO dataset. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L290-L301"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
    """Build YOLO Dataset.

    Args:
        img_path (str): Path to the folder containing images.
        mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for `rect`.

    Returns:
        (Dataset): YOLO dataset.
    """
    return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.coco_evaluate` {#ultralytics.models.yolo.detect.val.DetectionValidator.coco\_evaluate}

```python
def coco_evaluate(
    self,
    stats: dict[str, Any],
    pred_json: str,
    anno_json: str,
    iou_types: str | list[str] = "bbox",
    suffix: str | list[str] = "Box",
) -> dict[str, Any]
```

Evaluate COCO/LVIS metrics using faster-coco-eval library.

Performs evaluation using the faster-coco-eval library to compute mAP metrics for object detection. Updates the provided stats dictionary with computed metrics including mAP50, mAP50-95, and LVIS-specific metrics if applicable.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stats` | `dict[str, Any]` | Dictionary to store computed metrics and statistics. | *required* |
| `pred_json` | `str | Path` | Path to JSON file containing predictions in COCO format. | *required* |
| `anno_json` | `str | Path` | Path to JSON file containing ground truth annotations in COCO format. | *required* |
| `iou_types` | `str | list[str]` | IoU type(s) for evaluation. Can be single string or list of strings. Common<br>    values include "bbox", "segm", "keypoints". Defaults to "bbox". | `"bbox"` |
| `suffix` | `str | list[str]` | Suffix to append to metric names in stats dictionary. Should correspond to<br>    iou_types if multiple types provided. Defaults to "Box". | `"Box"` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Updated stats dictionary containing the computed COCO/LVIS evaluation metrics. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L447-L513"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def coco_evaluate(
    self,
    stats: dict[str, Any],
    pred_json: str,
    anno_json: str,
    iou_types: str | list[str] = "bbox",
    suffix: str | list[str] = "Box",
) -> dict[str, Any]:
    """Evaluate COCO/LVIS metrics using faster-coco-eval library.

    Performs evaluation using the faster-coco-eval library to compute mAP metrics for object detection. Updates the
    provided stats dictionary with computed metrics including mAP50, mAP50-95, and LVIS-specific metrics if
    applicable.

    Args:
        stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
        pred_json (str | Path): Path to JSON file containing predictions in COCO format.
        anno_json (str | Path): Path to JSON file containing ground truth annotations in COCO format.
        iou_types (str | list[str]): IoU type(s) for evaluation. Can be single string or list of strings. Common
            values include "bbox", "segm", "keypoints". Defaults to "bbox".
        suffix (str | list[str]): Suffix to append to metric names in stats dictionary. Should correspond to
            iou_types if multiple types provided. Defaults to "Box".

    Returns:
        (dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
    """
    if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
        LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
        try:
            for x in pred_json, anno_json:
                assert x.is_file(), f"{x} file not found"
            iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
            suffix = [suffix] if isinstance(suffix, str) else suffix
            check_requirements("faster-coco-eval>=1.6.7")
            from faster_coco_eval import COCO, COCOeval_faster

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            for i, iou_type in enumerate(iou_types):
                val = COCOeval_faster(
                    anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                )
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()

                # update mAP50-95 and mAP50
                stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]
                # record mAP for small, medium, large objects as well
                stats["metrics/mAP_small(B)"] = val.stats_as_dict["AP_small"]
                stats["metrics/mAP_medium(B)"] = val.stats_as_dict["AP_medium"]
                stats["metrics/mAP_large(B)"] = val.stats_as_dict["AP_large"]
                # update fitness
                stats["fitness"] = 0.9 * val.stats_as_dict["AP_all"] + 0.1 * val.stats_as_dict["AP_50"]

                if self.is_lvis:
                    stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                    stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                    stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

            if self.is_lvis:
                stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
        except Exception as e:
            LOGGER.warning(f"faster-coco-eval unable to run: {e}")
    return stats
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.eval_json` {#ultralytics.models.yolo.detect.val.DetectionValidator.eval\_json}

```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]
```

Evaluate YOLO output in JSON format and return performance statistics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stats` | `dict[str, Any]` | Current statistics dictionary. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Updated statistics dictionary with COCO/LVIS evaluation results. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L430-L445"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
    """Evaluate YOLO output in JSON format and return performance statistics.

    Args:
        stats (dict[str, Any]): Current statistics dictionary.

    Returns:
        (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
    """
    pred_json = self.save_dir / "predictions.json"  # predictions
    anno_json = (
        self.data["path"]
        / "annotations"
        / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
    )  # annotations
    return self.coco_evaluate(stats, pred_json, anno_json)
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.finalize_metrics` {#ultralytics.models.yolo.detect.val.DetectionValidator.finalize\_metrics}

```python
def finalize_metrics(self) -> None
```

Set final values for metrics speed and confusion matrix.

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L213-L220"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def finalize_metrics(self) -> None:
    """Set final values for metrics speed and confusion matrix."""
    if self.args.plots:
        for normalize in True, False:
            self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
    self.metrics.speed = self.speed
    self.metrics.confusion_matrix = self.confusion_matrix
    self.metrics.save_dir = self.save_dir
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.gather_stats` {#ultralytics.models.yolo.detect.val.DetectionValidator.gather\_stats}

```python
def gather_stats(self) -> None
```

Gather stats from all GPUs.

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L222-L242"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def gather_stats(self) -> None:
    """Gather stats from all GPUs."""
    if RANK == 0:
        gathered_stats = [None] * dist.get_world_size()
        dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
        merged_stats = {key: [] for key in self.metrics.stats.keys()}
        for stats_dict in gathered_stats:
            for key in merged_stats:
                merged_stats[key].extend(stats_dict[key])
        gathered_jdict = [None] * dist.get_world_size()
        dist.gather_object(self.jdict, gathered_jdict, dst=0)
        self.jdict = []
        for jdict in gathered_jdict:
            self.jdict.extend(jdict)
        self.metrics.stats = merged_stats
        self.seen = len(self.dataloader.dataset)  # total image count from dataset
    elif RANK > 0:
        dist.gather_object(self.metrics.stats, None, dst=0)
        dist.gather_object(self.jdict, None, dst=0)
        self.jdict = []
        self.metrics.clear_stats()
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.get_dataloader` {#ultralytics.models.yolo.detect.val.DetectionValidator.get\_dataloader}

```python
def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader
```

Construct and return dataloader.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataset_path` | `str` | Path to the dataset. | *required* |
| `batch_size` | `int` | Size of each batch. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.utils.data.DataLoader` | DataLoader for validation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L303-L322"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
    """Construct and return dataloader.

    Args:
        dataset_path (str): Path to the dataset.
        batch_size (int): Size of each batch.

    Returns:
        (torch.utils.data.DataLoader): DataLoader for validation.
    """
    dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
    return build_dataloader(
        dataset,
        batch_size,
        self.args.workers,
        shuffle=False,
        rank=-1,
        drop_last=self.args.compile,
        pin_memory=self.training,
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.get_desc` {#ultralytics.models.yolo.detect.val.DetectionValidator.get\_desc}

```python
def get_desc(self) -> str
```

Return a formatted string summarizing class metrics of YOLO model.

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L101-L103"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_desc(self) -> str:
    """Return a formatted string summarizing class metrics of YOLO model."""
    return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.get_stats` {#ultralytics.models.yolo.detect.val.DetectionValidator.get\_stats}

```python
def get_stats(self) -> dict[str, Any]
```

Calculate and return metrics statistics.

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Dictionary containing metrics results. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L244-L252"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_stats(self) -> dict[str, Any]:
    """Calculate and return metrics statistics.

    Returns:
        (dict[str, Any]): Dictionary containing metrics results.
    """
    self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
    self.metrics.clear_stats()
    return self.metrics.results_dict
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.init_metrics` {#ultralytics.models.yolo.detect.val.DetectionValidator.init\_metrics}

```python
def init_metrics(self, model: torch.nn.Module) -> None
```

Initialize evaluation metrics for YOLO detection validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | Model to validate. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L78-L99"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_metrics(self, model: torch.nn.Module) -> None:
    """Initialize evaluation metrics for YOLO detection validation.

    Args:
        model (torch.nn.Module): Model to validate.
    """
    val = self.data.get(self.args.split, "")  # validation path
    self.is_coco = (
        isinstance(val, str)
        and "coco" in val
        and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
    )  # is COCO
    self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
    self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
    self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
    self.names = model.names
    self.nc = len(model.names)
    self.end2end = getattr(model, "end2end", False)
    self.seen = 0
    self.jdict = []
    self.metrics.names = model.names
    self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.plot_predictions` {#ultralytics.models.yolo.detect.val.DetectionValidator.plot\_predictions}

```python
def plot_predictions(
    self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
) -> None
```

Plot predicted bounding boxes on input images and save the result.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Batch containing images and annotations. | *required* |
| `preds` | `list[dict[str, torch.Tensor]]` | List of predictions from the model. | *required* |
| `ni` | `int` | Batch index. | *required* |
| `max_det` | `int | None` | Maximum number of detections to plot. | `None` |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L339-L365"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_predictions(
    self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
) -> None:
    """Plot predicted bounding boxes on input images and save the result.

    Args:
        batch (dict[str, Any]): Batch containing images and annotations.
        preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
        ni (int): Batch index.
        max_det (int | None): Maximum number of detections to plot.
    """
    if not preds:
        return
    for i, pred in enumerate(preds):
        pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
    keys = preds[0].keys()
    max_det = max_det or self.args.max_det
    batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
    batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])  # convert to xywh format
    plot_images(
        images=batch["img"],
        labels=batched_preds,
        paths=batch["im_file"],
        fname=self.save_dir / f"val_batch{ni}_pred.jpg",
        names=self.names,
        on_plot=self.on_plot,
    )  # pred
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.plot_val_samples` {#ultralytics.models.yolo.detect.val.DetectionValidator.plot\_val\_samples}

```python
def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None
```

Plot validation image samples.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Batch containing images and annotations. | *required* |
| `ni` | `int` | Batch index. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L324-L337"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
    """Plot validation image samples.

    Args:
        batch (dict[str, Any]): Batch containing images and annotations.
        ni (int): Batch index.
    """
    plot_images(
        labels=batch,
        paths=batch["im_file"],
        fname=self.save_dir / f"val_batch{ni}_labels.jpg",
        names=self.names,
        on_plot=self.on_plot,
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.postprocess` {#ultralytics.models.yolo.detect.val.DetectionValidator.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]
```

Apply Non-maximum suppression to prediction outputs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw predictions from the model. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, torch.Tensor]]` | Processed predictions after NMS, where each dict contains 'bboxes', 'conf', |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L105-L126"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Apply Non-maximum suppression to prediction outputs.

    Args:
        preds (torch.Tensor): Raw predictions from the model.

    Returns:
        (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains 'bboxes', 'conf',
            'cls', and 'extra' tensors.
    """
    outputs = nms.non_max_suppression(
        preds,
        self.args.conf,
        self.args.iou,
        nc=0 if self.args.task == "detect" else self.nc,
        multi_label=True,
        agnostic=self.args.single_cls or self.args.agnostic_nms,
        max_det=self.args.max_det,
        end2end=self.end2end,
        rotated=self.args.task == "obb",
    )
    return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.pred_to_json` {#ultralytics.models.yolo.detect.val.DetectionValidator.pred\_to\_json}

```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None
```

Serialize YOLO predictions to COCO json format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with<br>    bounding box coordinates, confidence scores, and class predictions. | *required* |
| `pbatch` | `dict[str, Any]` | Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'. | *required* |

**Examples**

```python
>>> result = {
...     "image_id": 42,
...     "file_name": "42.jpg",
...     "category_id": 18,
...     "bbox": [258.15, 41.29, 348.26, 243.78],
...     "score": 0.236,
... }
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L385-L416"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
    """Serialize YOLO predictions to COCO json format.

    Args:
        predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
            bounding box coordinates, confidence scores, and class predictions.
        pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

    Examples:
         >>> result = {
         ...     "image_id": 42,
         ...     "file_name": "42.jpg",
         ...     "category_id": 18,
         ...     "bbox": [258.15, 41.29, 348.26, 243.78],
         ...     "score": 0.236,
         ... }
    """
    path = Path(pbatch["im_file"])
    stem = path.stem
    image_id = int(stem) if stem.isnumeric() else stem
    box = ops.xyxy2xywh(predn["bboxes"])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
        self.jdict.append(
            {
                "image_id": image_id,
                "file_name": path.name,
                "category_id": self.class_map[int(c)],
                "bbox": [round(x, 3) for x in b],
                "score": round(s, 5),
            }
        )
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.preprocess` {#ultralytics.models.yolo.detect.val.DetectionValidator.preprocess}

```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]
```

Preprocess batch of images for YOLO validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Batch containing images and annotations. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Preprocessed batch. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L63-L76"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
    """Preprocess batch of images for YOLO validation.

    Args:
        batch (dict[str, Any]): Batch containing images and annotations.

    Returns:
        (dict[str, Any]): Preprocessed batch.
    """
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
    batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
    return batch
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.print_results` {#ultralytics.models.yolo.detect.val.DetectionValidator.print\_results}

```python
def print_results(self) -> None
```

Print training/validation set metrics per class.

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L254-L272"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def print_results(self) -> None:
    """Print training/validation set metrics per class."""
    pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
    LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
    if self.metrics.nt_per_class.sum() == 0:
        LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute metrics without labels")

    # Print results per class
    if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
        for i, c in enumerate(self.metrics.ap_class_index):
            LOGGER.info(
                pf
                % (
                    self.names[c],
                    self.metrics.nt_per_image[c],
                    self.metrics.nt_per_class[c],
                    *self.metrics.class_result(i),
                )
            )
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.save_one_txt` {#ultralytics.models.yolo.detect.val.DetectionValidator.save\_one\_txt}

```python
def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None
```

Save YOLO detections to a txt file in normalized coordinates in a specific format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'. | *required* |
| `save_conf` | `bool` | Whether to save confidence scores. | *required* |
| `shape` | `tuple[int, int]` | Shape of the original image (height, width). | *required* |
| `file` | `Path` | File path to save the detections. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L367-L383"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
    """Save YOLO detections to a txt file in normalized coordinates in a specific format.

    Args:
        predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
        save_conf (bool): Whether to save confidence scores.
        shape (tuple[int, int]): Shape of the original image (height, width).
        file (Path): File path to save the detections.
    """
    from ultralytics.engine.results import Results

    Results(
        np.zeros((shape[0], shape[1]), dtype=np.uint8),
        path=None,
        names=self.names,
        boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
    ).save_txt(file, save_conf=save_conf)
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.scale_preds` {#ultralytics.models.yolo.detect.val.DetectionValidator.scale\_preds}

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
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L418-L428"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Scales predictions to the original image size."""
    return {
        **predn,
        "bboxes": ops.scale_boxes(
            pbatch["imgsz"],
            predn["bboxes"].clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        ),
    }
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.val.DetectionValidator.update_metrics` {#ultralytics.models.yolo.detect.val.DetectionValidator.update\_metrics}

```python
def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None
```

Update metrics with new predictions and ground truth.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `list[dict[str, torch.Tensor]]` | List of predictions from the model. | *required* |
| `batch` | `dict[str, Any]` | Batch data containing ground truth. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L168-L211"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
    """Update metrics with new predictions and ground truth.

    Args:
        preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
        batch (dict[str, Any]): Batch data containing ground truth.
    """
    for si, pred in enumerate(preds):
        self.seen += 1
        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred)

        cls = pbatch["cls"].cpu().numpy()
        no_pred = predn["cls"].shape[0] == 0
        self.metrics.update_stats(
            {
                **self._process_batch(predn, pbatch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
            }
        )
        # Evaluate
        if self.args.plots:
            self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
            if self.args.visualize:
                self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

        if no_pred:
            continue

        # Save
        if self.args.save_json or self.args.save_txt:
            predn_scaled = self.scale_preds(predn, pbatch)
        if self.args.save_json:
            self.pred_to_json(predn_scaled, pbatch)
        if self.args.save_txt:
            self.save_one_txt(
                predn_scaled,
                self.args.save_conf,
                pbatch["ori_shape"],
                self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
            )
```
</details>

<br><br>
