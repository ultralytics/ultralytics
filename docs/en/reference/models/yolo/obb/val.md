---
description: Explore the OBBValidator for YOLO, an advanced class for oriented bounding boxes (OBB). Learn initialization, processes, and evaluation methods.
keywords: Ultralytics, YOLO, OBBValidator, Oriented Bounding Boxes, DetectionValidator, validation, Python, deep learning
---

# Reference for `ultralytics/models/yolo/obb/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`OBBValidator`](#ultralytics.models.yolo.obb.val.OBBValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`OBBValidator.init_metrics`](#ultralytics.models.yolo.obb.val.OBBValidator.init_metrics)
        - [`OBBValidator._process_batch`](#ultralytics.models.yolo.obb.val.OBBValidator._process_batch)
        - [`OBBValidator.postprocess`](#ultralytics.models.yolo.obb.val.OBBValidator.postprocess)
        - [`OBBValidator._prepare_batch`](#ultralytics.models.yolo.obb.val.OBBValidator._prepare_batch)
        - [`OBBValidator.plot_predictions`](#ultralytics.models.yolo.obb.val.OBBValidator.plot_predictions)
        - [`OBBValidator.pred_to_json`](#ultralytics.models.yolo.obb.val.OBBValidator.pred_to_json)
        - [`OBBValidator.save_one_txt`](#ultralytics.models.yolo.obb.val.OBBValidator.save_one_txt)
        - [`OBBValidator.scale_preds`](#ultralytics.models.yolo.obb.val.OBBValidator.scale_preds)
        - [`OBBValidator.eval_json`](#ultralytics.models.yolo.obb.val.OBBValidator.eval_json)


## Class `ultralytics.models.yolo.obb.val.OBBValidator` {#ultralytics.models.yolo.obb.val.OBBValidator}

```python
OBBValidator(self, dataloader = None, save_dir = None, args = None, _callbacks = None) -> None
```

**Bases:** `DetectionValidator`

A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

This validator specializes in evaluating models that predict rotated bounding boxes, commonly used for aerial and satellite imagery where objects can appear at various orientations.

This constructor initializes an OBBValidator instance for validating Oriented Bounding Box (OBB) models. It extends the DetectionValidator class and configures it specifically for the OBB task.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` | `torch.utils.data.DataLoader, optional` | DataLoader to be used for validation. | `None` |
| `save_dir` | `str | Path, optional` | Directory to save results. | `None` |
| `args` | `dict, optional` | Arguments containing validation parameters. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be called during validation. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `args` | `dict` | Configuration arguments for the validator. |
| `metrics` | `OBBMetrics` | Metrics object for evaluating OBB model performance. |
| `is_dota` | `bool` | Flag indicating whether the validation dataset is in DOTA format. |

**Methods**

| Name | Description |
| --- | --- |
| [`_prepare_batch`](#ultralytics.models.yolo.obb.val.OBBValidator._prepare_batch) | Prepare batch data for OBB validation with proper scaling and formatting. |
| [`_process_batch`](#ultralytics.models.yolo.obb.val.OBBValidator._process_batch) | Compute the correct prediction matrix for a batch of detections and ground truth bounding boxes. |
| [`eval_json`](#ultralytics.models.yolo.obb.val.OBBValidator.eval_json) | Evaluate YOLO output in JSON format and save predictions in DOTA format. |
| [`init_metrics`](#ultralytics.models.yolo.obb.val.OBBValidator.init_metrics) | Initialize evaluation metrics for YOLO obb validation. |
| [`plot_predictions`](#ultralytics.models.yolo.obb.val.OBBValidator.plot_predictions) | Plot predicted bounding boxes on input images and save the result. |
| [`postprocess`](#ultralytics.models.yolo.obb.val.OBBValidator.postprocess) | Postprocess OBB predictions. |
| [`pred_to_json`](#ultralytics.models.yolo.obb.val.OBBValidator.pred_to_json) | Convert YOLO predictions to COCO JSON format with rotated bounding box information. |
| [`save_one_txt`](#ultralytics.models.yolo.obb.val.OBBValidator.save_one_txt) | Save YOLO OBB detections to a text file in normalized coordinates. |
| [`scale_preds`](#ultralytics.models.yolo.obb.val.OBBValidator.scale_preds) | Scales predictions to the original image size. |

**Examples**

```python
>>> from ultralytics.models.yolo.obb import OBBValidator
>>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml")
>>> validator = OBBValidator(args=args)
>>> validator(model=args["model"])
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L18-L305"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class OBBValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    This validator specializes in evaluating models that predict rotated bounding boxes, commonly used for aerial and
    satellite imagery where objects can appear at various orientations.

    Attributes:
        args (dict): Configuration arguments for the validator.
        metrics (OBBMetrics): Metrics object for evaluating OBB model performance.
        is_dota (bool): Flag indicating whether the validation dataset is in DOTA format.

    Methods:
        init_metrics: Initialize evaluation metrics for YOLO.
        _process_batch: Process batch of detections and ground truth boxes to compute IoU matrix.
        _prepare_batch: Prepare batch data for OBB validation.
        _prepare_pred: Prepare predictions for evaluation against ground truth.
        plot_predictions: Plot predicted bounding boxes on input images.
        pred_to_json: Serialize YOLO predictions to COCO json format.
        save_one_txt: Save YOLO detections to a txt file in normalized coordinates.
        eval_json: Evaluate YOLO output in JSON format and return performance statistics.

    Examples:
        >>> from ultralytics.models.yolo.obb import OBBValidator
        >>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml")
        >>> validator = OBBValidator(args=args)
        >>> validator(model=args["model"])
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics.

        This constructor initializes an OBBValidator instance for validating Oriented Bounding Box (OBB) models. It
        extends the DetectionValidator class and configures it specifically for the OBB task.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Arguments containing validation parameters.
            _callbacks (list, optional): List of callback functions to be called during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics()
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.val.OBBValidator._prepare_batch` {#ultralytics.models.yolo.obb.val.OBBValidator.\_prepare\_batch}

```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]
```

Prepare batch data for OBB validation with proper scaling and formatting.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `si` | `int` | Sample index within the batch. | *required* |
| `batch` | `dict[str, Any]` | Dictionary containing batch data with keys:<br>    - batch_idx: Tensor of batch indices<br>    - cls: Tensor of class labels<br>    - bboxes: Tensor of bounding boxes<br>    - ori_shape: Original image shapes<br>    - img: Batch of images<br>    - ratio_pad: Ratio and padding information | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Prepared batch data with scaled bounding boxes and metadata. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L111-L142"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
    """Prepare batch data for OBB validation with proper scaling and formatting.

    Args:
        si (int): Sample index within the batch.
        batch (dict[str, Any]): Dictionary containing batch data with keys:
            - batch_idx: Tensor of batch indices
            - cls: Tensor of class labels
            - bboxes: Tensor of bounding boxes
            - ori_shape: Original image shapes
            - img: Batch of images
            - ratio_pad: Ratio and padding information

    Returns:
        (dict[str, Any]): Prepared batch data with scaled bounding boxes and metadata.
    """
    idx = batch["batch_idx"] == si
    cls = batch["cls"][idx].squeeze(-1)
    bbox = batch["bboxes"][idx]
    ori_shape = batch["ori_shape"][si]
    imgsz = batch["img"].shape[2:]
    ratio_pad = batch["ratio_pad"][si]
    if cls.shape[0]:
        bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
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

### Method `ultralytics.models.yolo.obb.val.OBBValidator._process_batch` {#ultralytics.models.yolo.obb.val.OBBValidator.\_process\_batch}

```python
def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]
```

Compute the correct prediction matrix for a batch of detections and ground truth bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` | Prediction dictionary containing 'cls' and 'bboxes' keys with detected<br>    class labels and bounding boxes. | *required* |
| `batch` | `dict[str, torch.Tensor]` | Batch dictionary containing 'cls' and 'bboxes' keys with ground truth class<br>    labels and bounding boxes. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, np.ndarray]` | Dictionary containing 'tp' key with the correct prediction matrix as a numpy array |

**Examples**

```python
>>> preds = {"cls": torch.randint(0, 5, (100,)), "bboxes": torch.rand(100, 5)}
>>> batch = {"cls": torch.randint(0, 5, (50,)), "bboxes": torch.rand(50, 5)}
>>> correct_matrix = validator._process_batch(preds, batch)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L73-L95"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Compute the correct prediction matrix for a batch of detections and ground truth bounding boxes.

    Args:
        preds (dict[str, torch.Tensor]): Prediction dictionary containing 'cls' and 'bboxes' keys with detected
            class labels and bounding boxes.
        batch (dict[str, torch.Tensor]): Batch dictionary containing 'cls' and 'bboxes' keys with ground truth class
            labels and bounding boxes.

    Returns:
        (dict[str, np.ndarray]): Dictionary containing 'tp' key with the correct prediction matrix as a numpy array
            with shape (N, 10), which includes 10 IoU levels for each detection, indicating the accuracy of
            predictions compared to the ground truth.

    Examples:
        >>> preds = {"cls": torch.randint(0, 5, (100,)), "bboxes": torch.rand(100, 5)}
        >>> batch = {"cls": torch.randint(0, 5, (50,)), "bboxes": torch.rand(50, 5)}
        >>> correct_matrix = validator._process_batch(preds, batch)
    """
    if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
        return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
    iou = batch_probiou(batch["bboxes"], preds["bboxes"])
    return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.val.OBBValidator.eval_json` {#ultralytics.models.yolo.obb.val.OBBValidator.eval\_json}

```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]
```

Evaluate YOLO output in JSON format and save predictions in DOTA format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stats` | `dict[str, Any]` | Performance statistics dictionary. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, Any]` | Updated performance statistics. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L242-L305"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
    """Evaluate YOLO output in JSON format and save predictions in DOTA format.

    Args:
        stats (dict[str, Any]): Performance statistics dictionary.

    Returns:
        (dict[str, Any]): Updated performance statistics.
    """
    if self.args.save_json and self.is_dota and len(self.jdict):
        import json
        import re
        from collections import defaultdict

        pred_json = self.save_dir / "predictions.json"  # predictions
        pred_txt = self.save_dir / "predictions_txt"  # predictions
        pred_txt.mkdir(parents=True, exist_ok=True)
        data = json.load(open(pred_json))
        # Save split results
        LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
        for d in data:
            image_id = d["image_id"]
            score = d["score"]
            classname = self.names[d["category_id"] - 1].replace(" ", "-")
            p = d["poly"]

            with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
        # Save merged results, this could result slightly lower map than using official merging script,
        # because of the probiou calculation.
        pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
        pred_merged_txt.mkdir(parents=True, exist_ok=True)
        merged_results = defaultdict(list)
        LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
        for d in data:
            image_id = d["image_id"].split("__", 1)[0]
            pattern = re.compile(r"\d+___\d+")
            x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
            bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
            bbox[0] += x
            bbox[1] += y
            bbox.extend([score, cls])
            merged_results[image_id].append(bbox)
        for image_id, bbox in merged_results.items():
            bbox = torch.tensor(bbox)
            max_wh = torch.max(bbox[:, :2]).item() * 2
            c = bbox[:, 6:7] * max_wh  # classes
            scores = bbox[:, 5]  # scores
            b = bbox[:, :5].clone()
            b[:, :2] += c
            # 0.3 could get results close to the ones from official merging script, even slightly better.
            i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_probiou)
            bbox = bbox[i]

            b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
            for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                classname = self.names[int(x[-1])].replace(" ", "-")
                p = [round(i, 3) for i in x[:-2]]  # poly
                score = round(x[-2], 3)

                with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

    return stats
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.val.OBBValidator.init_metrics` {#ultralytics.models.yolo.obb.val.OBBValidator.init\_metrics}

```python
def init_metrics(self, model: torch.nn.Module) -> None
```

Initialize evaluation metrics for YOLO obb validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | Model to validate. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L62-L71"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_metrics(self, model: torch.nn.Module) -> None:
    """Initialize evaluation metrics for YOLO obb validation.

    Args:
        model (torch.nn.Module): Model to validate.
    """
    super().init_metrics(model)
    val = self.data.get(self.args.split, "")  # validation path
    self.is_dota = isinstance(val, str) and "DOTA" in val  # check if dataset is DOTA format
    self.confusion_matrix.task = "obb"  # set confusion matrix task to 'obb'
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.val.OBBValidator.plot_predictions` {#ultralytics.models.yolo.obb.val.OBBValidator.plot\_predictions}

```python
def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None
```

Plot predicted bounding boxes on input images and save the result.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Batch data containing images, file paths, and other metadata. | *required* |
| `preds` | `list[dict[str, torch.Tensor]]` | List of prediction dictionaries for each image in the batch. | *required* |
| `ni` | `int` | Batch index used for naming the output file. | *required* |

**Examples**

```python
>>> validator = OBBValidator()
>>> batch = {"img": images, "im_file": paths}
>>> preds = [{"bboxes": torch.rand(10, 5), "cls": torch.zeros(10), "conf": torch.rand(10)}]
>>> validator.plot_predictions(batch, preds, 0)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L144-L171"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
    """Plot predicted bounding boxes on input images and save the result.

    Args:
        batch (dict[str, Any]): Batch data containing images, file paths, and other metadata.
        preds (list[dict[str, torch.Tensor]]): List of prediction dictionaries for each image in the batch.
        ni (int): Batch index used for naming the output file.

    Examples:
        >>> validator = OBBValidator()
        >>> batch = {"img": images, "im_file": paths}
        >>> preds = [{"bboxes": torch.rand(10, 5), "cls": torch.zeros(10), "conf": torch.rand(10)}]
        >>> validator.plot_predictions(batch, preds, 0)
    """
    if not preds:
        return
    for i, pred in enumerate(preds):
        pred["batch_idx"] = torch.ones_like(pred["conf"]) * i
    keys = preds[0].keys()
    batched_preds = {k: torch.cat([x[k] for x in preds], dim=0) for k in keys}
    plot_images(
        images=batch["img"],
        labels=batched_preds,
        paths=batch["im_file"],
        fname=self.save_dir / f"val_batch{ni}_pred.jpg",
        names=self.names,
        on_plot=self.on_plot,
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.val.OBBValidator.postprocess` {#ultralytics.models.yolo.obb.val.OBBValidator.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]
```

Postprocess OBB predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw predictions from the model. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, torch.Tensor]]` | Processed predictions with angle information concatenated to bboxes. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L97-L109"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Postprocess OBB predictions.

    Args:
        preds (torch.Tensor): Raw predictions from the model.

    Returns:
        (list[dict[str, torch.Tensor]]): Processed predictions with angle information concatenated to bboxes.
    """
    preds = super().postprocess(preds)
    for pred in preds:
        pred["bboxes"] = torch.cat([pred["bboxes"], pred.pop("extra")], dim=-1)  # concatenate angle
    return preds
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.val.OBBValidator.pred_to_json` {#ultralytics.models.yolo.obb.val.OBBValidator.pred\_to\_json}

```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None
```

Convert YOLO predictions to COCO JSON format with rotated bounding box information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Prediction dictionary containing 'bboxes', 'conf', and 'cls' keys with<br>    bounding box coordinates, confidence scores, and class predictions. | *required* |
| `pbatch` | `dict[str, Any]` | Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'. | *required* |

!!! note "Notes"

    This method processes rotated bounding box predictions and converts them to both rbox format
    (x, y, w, h, angle) and polygon format (x1, y1, x2, y2, x3, y3, x4, y4) before adding them
    to the JSON dictionary.

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L173-L201"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
    """Convert YOLO predictions to COCO JSON format with rotated bounding box information.

    Args:
        predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', and 'cls' keys with
            bounding box coordinates, confidence scores, and class predictions.
        pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

    Notes:
        This method processes rotated bounding box predictions and converts them to both rbox format
        (x, y, w, h, angle) and polygon format (x1, y1, x2, y2, x3, y3, x4, y4) before adding them
        to the JSON dictionary.
    """
    path = Path(pbatch["im_file"])
    stem = path.stem
    image_id = int(stem) if stem.isnumeric() else stem
    rbox = predn["bboxes"]
    poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
    for r, b, s, c in zip(rbox.tolist(), poly.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
        self.jdict.append(
            {
                "image_id": image_id,
                "file_name": path.name,
                "category_id": self.class_map[int(c)],
                "score": round(s, 5),
                "rbox": [round(x, 3) for x in r],
                "poly": [round(x, 3) for x in b],
            }
        )
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.val.OBBValidator.save_one_txt` {#ultralytics.models.yolo.obb.val.OBBValidator.save\_one\_txt}

```python
def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None
```

Save YOLO OBB detections to a text file in normalized coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Prediction dictionary containing 'bboxes', 'conf', and 'cls' keys with<br>    bounding box coordinates (including angle), confidence scores, and class predictions. | *required* |
| `save_conf` | `bool` | Whether to save confidence scores in the text file. | *required* |
| `shape` | `tuple[int, int]` | Original image shape in format (height, width). | *required* |
| `file` | `Path` | Output file path to save detections. | *required* |

**Examples**

```python
>>> validator = OBBValidator()
>>> predn = {
...     "bboxes": torch.tensor([[100, 100, 50, 30, 45]]),
...     "conf": torch.tensor([0.9]),
...     "cls": torch.tensor([0]),
... }
>>> validator.save_one_txt(predn, True, (640, 480), Path("detection.txt"))
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L203-L231"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
    """Save YOLO OBB detections to a text file in normalized coordinates.

    Args:
        predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', and 'cls' keys with
            bounding box coordinates (including angle), confidence scores, and class predictions.
        save_conf (bool): Whether to save confidence scores in the text file.
        shape (tuple[int, int]): Original image shape in format (height, width).
        file (Path): Output file path to save detections.

    Examples:
        >>> validator = OBBValidator()
        >>> predn = {
        ...     "bboxes": torch.tensor([[100, 100, 50, 30, 45]]),
        ...     "conf": torch.tensor([0.9]),
        ...     "cls": torch.tensor([0]),
        ... }
        >>> validator.save_one_txt(predn, True, (640, 480), Path("detection.txt"))
    """
    import numpy as np

    from ultralytics.engine.results import Results

    Results(
        np.zeros((shape[0], shape[1]), dtype=np.uint8),
        path=None,
        names=self.names,
        obb=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
    ).save_txt(file, save_conf=save_conf)
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.val.OBBValidator.scale_preds` {#ultralytics.models.yolo.obb.val.OBBValidator.scale\_preds}

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
<summary>Source code in <code>ultralytics/models/yolo/obb/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/val.py#L233-L240"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Scales predictions to the original image size."""
    return {
        **predn,
        "bboxes": ops.scale_boxes(
            pbatch["imgsz"], predn["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        ),
    }
```
</details>

<br><br>
