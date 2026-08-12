---
title: models.rtdetr.val API Reference
description: Explore the RTDETRValidator and RTDETRDataset classes for real-time detection and tracking. Understand initialization, transformations, and post-processing.
keywords: RTDETR, Ultralytics, object detection, tracking, YOLO, RTDETRDataset, RTDETRValidator, real-time detection
---

# Reference for `ultralytics/models/rtdetr/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`RTDETRDataset`](#ultralytics.models.rtdetr.val.RTDETRDataset)
        - [`RTDETRValidator`](#ultralytics.models.rtdetr.val.RTDETRValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`RTDETRDataset.load_image`](#ultralytics.models.rtdetr.val.RTDETRDataset.load_image)
        - [`RTDETRValidator.build_dataset`](#ultralytics.models.rtdetr.val.RTDETRValidator.build_dataset)
        - [`RTDETRValidator.scale_preds`](#ultralytics.models.rtdetr.val.RTDETRValidator.scale_preds)
        - [`RTDETRValidator.postprocess`](#ultralytics.models.rtdetr.val.RTDETRValidator.postprocess)
        - [`RTDETRValidator.pred_to_json`](#ultralytics.models.rtdetr.val.RTDETRValidator.pred_to_json)


## Class `ultralytics.models.rtdetr.val.RTDETRDataset` {#ultralytics.models.rtdetr.val.RTDETRDataset}

```python
RTDETRDataset(*args, data=None, **kwargs)
```

**Bases:** `YOLODataset`

Real-Time DEtection TRansformer (RT-DETR) dataset class extending the base YOLODataset class.

This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for real-time detection and tracking tasks.

This constructor sets up a dataset specifically optimized for the RT-DETR (Real-Time DEtection TRansformer) model, building upon the base YOLODataset functionality.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` | Variable length argument list passed to the parent YOLODataset class. | *required* |
| `data` | `dict \| None` | Dictionary containing dataset information. If None, default values will be used. | `None` |
| `**kwargs` | `Any` | Additional keyword arguments passed to the parent YOLODataset class. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `augment` | `bool` | Whether to apply data augmentation. |
| `rect` | `bool` | Whether to use rectangular training. |
| `use_segments` | `bool` | Whether to use segmentation masks. |
| `use_keypoints` | `bool` | Whether to use keypoint annotations. |
| `imgsz` | `int` | Target image size for training. |

**Methods**

| Name | Description |
| --- | --- |
| [`load_image`](#ultralytics.models.rtdetr.val.RTDETRDataset.load_image) | Load one image from dataset index 'i'. |

**Examples**

Initialize an RT-DETR dataset

```python
>>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
>>> image, hw0, hw = dataset.load_image(0)
```

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L17-L70">View on GitHub</a>
```python
class RTDETRDataset(YOLODataset):
    """Real-Time DEtection TRansformer (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.

    Attributes:
        augment (bool): Whether to apply data augmentation.
        rect (bool): Whether to use rectangular training.
        use_segments (bool): Whether to use segmentation masks.
        use_keypoints (bool): Whether to use keypoint annotations.
        imgsz (int): Target image size for training.

    Methods:
        load_image: Load one image from dataset index.
        build_transforms: Build transformation pipeline for the dataset.

    Examples:
        Initialize an RT-DETR dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw0, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class.

        This constructor sets up a dataset specifically optimized for the RT-DETR (Real-Time DEtection TRansformer)
        model, building upon the base YOLODataset functionality.

        Args:
            *args (Any): Variable length argument list passed to the parent YOLODataset class.
            data (dict | None): Dictionary containing dataset information. If None, default values will be used.
            **kwargs (Any): Additional keyword arguments passed to the parent YOLODataset class.
        """
        super().__init__(*args, data=data, **kwargs)
```
</details>

<br>

### Method `ultralytics.models.rtdetr.val.RTDETRDataset.load_image` {#ultralytics.models.rtdetr.val.RTDETRDataset.load\_image}

```python
def load_image(self, i, rect_mode=False)
```

Load one image from dataset index 'i'.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `i` | `int` | Index of the image to load. | *required* |
| `rect_mode` | `bool, optional` | Whether to use rectangular mode for batch inference. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `im (np.ndarray)` | Loaded image as a NumPy array. |
| `hw_original (tuple[int, int])` | Original image dimensions in (height, width) format. |
| `hw_resized (tuple[int, int])` | Resized image dimensions in (height, width) format. |

**Examples**

Load an image from the dataset

```python
>>> dataset = RTDETRDataset(img_path="path/to/images")
>>> image, hw0, hw = dataset.load_image(0)
```

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L53-L70">View on GitHub</a>
```python
def load_image(self, i, rect_mode=False):
    """Load one image from dataset index 'i'.

    Args:
        i (int): Index of the image to load.
        rect_mode (bool, optional): Whether to use rectangular mode for batch inference.

    Returns:
        im (np.ndarray): Loaded image as a NumPy array.
        hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
        hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

    Examples:
        Load an image from the dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images")
        >>> image, hw0, hw = dataset.load_image(0)
    """
    return super().load_image(i=i, rect_mode=rect_mode)
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.rtdetr.val.RTDETRValidator` {#ultralytics.models.rtdetr.val.RTDETRValidator}

```python
RTDETRValidator()
```

**Bases:** `DetectionValidator`

RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for

the RT-DETR (Real-Time DETR) object detection model.

The class allows building of an RTDETR-specific dataset for validation, applies confidence thresholding for post-processing, and updates evaluation metrics accordingly.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `args` | `Namespace` | Configuration arguments for validation. |
| `data` | `dict` | Dataset configuration dictionary. |

**Methods**

| Name | Description |
| --- | --- |
| [`build_dataset`](#ultralytics.models.rtdetr.val.RTDETRValidator.build_dataset) | Build an RTDETR Dataset. |
| [`postprocess`](#ultralytics.models.rtdetr.val.RTDETRValidator.postprocess) | Apply post-processing to prediction outputs. |
| [`pred_to_json`](#ultralytics.models.rtdetr.val.RTDETRValidator.pred_to_json) | Serialize YOLO predictions to COCO json format. |
| [`scale_preds`](#ultralytics.models.rtdetr.val.RTDETRValidator.scale_preds) | Return predictions unchanged as RT-DETR handles scaling in postprocessing. |

**Examples**

Initialize and run RT-DETR validation

```python
>>> from ultralytics.models.rtdetr import RTDETRValidator
>>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
>>> validator = RTDETRValidator(args=args)
>>> validator()
```

!!! note "Notes"

    For further details on the attributes and methods, refer to the parent DetectionValidator class.

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L73-L185">View on GitHub</a>
```python
class RTDETRValidator(DetectionValidator):
    """RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies confidence thresholding for
    post-processing, and updates evaluation metrics accordingly.

    Attributes:
        args (Namespace): Configuration arguments for validation.
        data (dict): Dataset configuration dictionary.

    Methods:
        build_dataset: Build an RTDETR Dataset for validation.
        postprocess: Apply confidence thresholding to prediction outputs.

    Examples:
        Initialize and run RT-DETR validation
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Notes:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """
```
</details>

<br>

### Method `ultralytics.models.rtdetr.val.RTDETRValidator.build_dataset` {#ultralytics.models.rtdetr.val.RTDETRValidator.build\_dataset}

```python
def build_dataset(self, img_path, mode="val", batch=None)
```

Build an RTDETR Dataset.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `str` | Path to the folder containing images. | *required* |
| `mode` | `str, optional` | `train` mode or `val` mode, users are able to customize different augmentations for each mode. | `"val"` |
| `batch` | `int, optional` | Size of batches, this is for `rect`. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `RTDETRDataset` | Dataset configured for RT-DETR validation. |

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L99-L123">View on GitHub</a>
```python
def build_dataset(self, img_path, mode="val", batch=None):
    """Build an RTDETR Dataset.

    Args:
        img_path (str): Path to the folder containing images.
        mode (str, optional): `train` mode or `val` mode, users are able to customize different augmentations for
            each mode.
        batch (int, optional): Size of batches, this is for `rect`.

    Returns:
        (RTDETRDataset): Dataset configured for RT-DETR validation.
    """
    return RTDETRDataset(
        img_path=img_path,
        imgsz=self.args.imgsz,
        batch_size=batch,
        augment=False,  # no augmentation
        hyp=self.args,
        rect=False,  # no rect
        cache=self.args.cache or None,
        single_cls=self.args.single_cls or False,
        prefix=colorstr(f"{mode}: "),
        classes=self.args.classes,
        data=self.data,
    )
```
</details>

<br>

### Method `ultralytics.models.rtdetr.val.RTDETRValidator.postprocess` {#ultralytics.models.rtdetr.val.RTDETRValidator.postprocess}

```python
def postprocess(self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]) -> list[dict[str, torch.Tensor]]
```

Apply post-processing to prediction outputs.

Top-k selection is already performed inside the decoder head. This method converts normalized xywh coordinates to pixel xyxy format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor \| list \| tuple` | Predictions from the model with shape (batch_size, num_queries, 6), where the last dimension is [cx, cy, w, h, score, class]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, torch.Tensor]]` | List of dictionaries for each image, each containing:<br>- 'bboxes': Tensor of shape (N, 4) with bounding box coordinates in xyxy pixel format<br>- 'conf': Tensor of shape (N,) with confidence scores<br>- 'cls': Tensor of shape (N,) with class indices |

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L129-L158">View on GitHub</a>
```python
def postprocess(
    self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
) -> list[dict[str, torch.Tensor]]:
    """Apply post-processing to prediction outputs.

    Top-k selection is already performed inside the decoder head. This method converts normalized xywh
    coordinates to pixel xyxy format.

    Args:
        preds (torch.Tensor | list | tuple): Predictions from the model with shape (batch_size, num_queries, 6),
            where the last dimension is [cx, cy, w, h, score, class].

    Returns:
        (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing:
            - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates in xyxy pixel format
            - 'conf': Tensor of shape (N,) with confidence scores
            - 'cls': Tensor of shape (N,) with class indices
    """
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
    bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    masks = [(score > self.args.conf).nonzero().squeeze(1)[: self.args.max_det] for score in scores]

    return [
        {"bboxes": bbox[m], "conf": score[m], "cls": label[m]}
        for bbox, score, label, m in zip(bboxes, scores, labels, masks)
    ]
```
</details>

<br>

### Method `ultralytics.models.rtdetr.val.RTDETRValidator.pred_to_json` {#ultralytics.models.rtdetr.val.RTDETRValidator.pred\_to\_json}

```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None
```

Serialize YOLO predictions to COCO json format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` | Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with bounding box coordinates, confidence scores, and class predictions. | *required* |
| `pbatch` | `dict[str, Any]` | Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L160-L185">View on GitHub</a>
```python
def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
    """Serialize YOLO predictions to COCO json format.

    Args:
        predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
            bounding box coordinates, confidence scores, and class predictions.
        pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
    """
    path = Path(pbatch["im_file"])
    stem = path.stem
    image_id = int(stem) if stem.isnumeric() else stem
    box = predn["bboxes"].clone()
    box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
    box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
    box = ops.xyxy2xywh(box)  # xywh
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

### Method `ultralytics.models.rtdetr.val.RTDETRValidator.scale_preds` {#ultralytics.models.rtdetr.val.RTDETRValidator.scale\_preds}

```python
def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]
```

Return predictions unchanged as RT-DETR handles scaling in postprocessing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predn` | `dict[str, torch.Tensor]` |  | *required* |
| `pbatch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/val.py#L125-L127">View on GitHub</a>
```python
def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return predictions unchanged as RT-DETR handles scaling in postprocessing."""
    return predn
```
</details>

<br><br>
