---
description: Documentation for YOLOE validator classes in Ultralytics, supporting both text and visual prompt embeddings for object detection and segmentation models.
keywords: YOLOE, validation, object detection, segmentation, visual prompts, text prompts, embeddings, Ultralytics, YOLOEDetectValidator, YOLOESegValidator, deep learning
---

# Reference for `ultralytics/models/yolo/yoloe/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`YOLOEDetectValidator`](#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator)
        - [`YOLOESegValidator`](#ultralytics.models.yolo.yoloe.val.YOLOESegValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`YOLOEDetectValidator.get_visual_pe`](#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.get_visual_pe)
        - [`YOLOEDetectValidator.get_vpe_dataloader`](#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.get_vpe_dataloader)
        - [`YOLOEDetectValidator.__call__`](#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.__call__)


## Class `ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator` {#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator}

```python
YOLOEDetectValidator()
```

**Bases:** `DetectionValidator`

A validator class for YOLOE detection models that handles both text and visual prompt embeddings.

This class extends DetectionValidator to provide specialized validation functionality for YOLOE models. It supports validation using either text prompts or visual prompt embeddings extracted from training samples, enabling flexible evaluation strategies for prompt-based object detection.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `device` | `torch.device` | The device on which validation is performed. |
| `args` | `namespace` | Configuration arguments for validation. |
| `dataloader` | `DataLoader` | DataLoader for validation data. |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.__call__) | Run validation on the model using either text or visual prompt embeddings. |
| [`get_visual_pe`](#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.get_visual_pe) | Extract visual prompt embeddings from training samples. |
| [`get_vpe_dataloader`](#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.get_vpe_dataloader) | Create a dataloader for LVIS training visual prompt samples. |

**Examples**

```python
Validate with text prompts
>>> validator = YOLOEDetectValidator()
>>> stats = validator(model=model, load_vp=False)

Validate with visual prompts
>>> stats = validator(model=model, refer_data="path/to/data.yaml", load_vp=True)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/val.py#L23-L200"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOEDetectValidator(DetectionValidator):
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.__call__` {#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.\_\_call\_\_}

```python
def __call__(
    self,
    trainer: Any | None = None,
    model: YOLOEModel | str | None = None,
    refer_data: str | None = None,
    load_vp: bool = False,
) -> dict[str, Any]
```

Run validation on the model using either text or visual prompt embeddings.

This method validates the model using either text prompts or visual prompts, depending on the load_vp flag. It supports validation during training (using a trainer object) or standalone validation with a provided model. For visual prompts, reference data can be specified to extract embeddings from a different dataset.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `object, optional` | Trainer object containing the model and device. | `None` |
| `model` | `YOLOEModel | str, optional` | Model to validate. Required if trainer is not provided. | `None` |
| `refer_data` | `str, optional` | Path to reference data for visual prompts. | `None` |
| `load_vp` | `bool` | Whether to load visual prompts. If False, text prompts are used. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Validation statistics containing metrics computed during validation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/val.py#L133-L200"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def __call__(
    self,
    trainer: Any | None = None,
    model: YOLOEModel | str | None = None,
    refer_data: str | None = None,
    load_vp: bool = False,
) -> dict[str, Any]:
    """Run validation on the model using either text or visual prompt embeddings.

    This method validates the model using either text prompts or visual prompts, depending on the load_vp flag. It
    supports validation during training (using a trainer object) or standalone validation with a provided model. For
    visual prompts, reference data can be specified to extract embeddings from a different dataset.

    Args:
        trainer (object, optional): Trainer object containing the model and device.
        model (YOLOEModel | str, optional): Model to validate. Required if trainer is not provided.
        refer_data (str, optional): Path to reference data for visual prompts.
        load_vp (bool): Whether to load visual prompts. If False, text prompts are used.

    Returns:
        (dict): Validation statistics containing metrics computed during validation.
    """
    if trainer is not None:
        self.device = trainer.device
        model = trainer.ema.ema
        names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]

        if load_vp:
            LOGGER.info("Validate using the visual prompt.")
            self.args.half = False
            # Directly use the same dataloader for visual embeddings extracted during training
            vpe = self.get_visual_pe(self.dataloader, model)
            model.set_classes(names, vpe)
        else:
            LOGGER.info("Validate using the text prompt.")
            tpe = model.get_text_pe(names)
            model.set_classes(names, tpe)
        stats = super().__call__(trainer, model)
    else:
        if refer_data is not None:
            assert load_vp, "Refer data is only used for visual prompt validation."
        self.device = select_device(self.args.device, verbose=False)

        if isinstance(model, (str, Path)):
            from ultralytics.nn.tasks import load_checkpoint

            model, _ = load_checkpoint(model, device=self.device)  # model, ckpt
        model.eval().to(self.device)
        data = check_det_dataset(refer_data or self.args.data)
        names = [name.split("/", 1)[0] for name in list(data["names"].values())]

        if load_vp:
            LOGGER.info("Validate using the visual prompt.")
            self.args.half = False
            # TODO: need to check if the names from refer data is consistent with the evaluated dataset
            # could use same dataset or refer to extract visual prompt embeddings
            dataloader = self.get_vpe_dataloader(data)
            vpe = self.get_visual_pe(dataloader, model)
            model.set_classes(names, vpe)
            stats = super().__call__(model=deepcopy(model))
        elif isinstance(model.model[-1], YOLOEDetect) and hasattr(model.model[-1], "lrpc"):  # prompt-free
            return super().__call__(trainer, model)
        else:
            LOGGER.info("Validate using the text prompt.")
            tpe = model.get_text_pe(names)
            model.set_classes(names, tpe)
            stats = super().__call__(model=deepcopy(model))
    return stats
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.get_visual_pe` {#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.get\_visual\_pe}

```python
def get_visual_pe(self, dataloader: torch.utils.data.DataLoader, model: YOLOEModel) -> torch.Tensor
```

Extract visual prompt embeddings from training samples.

This method processes a dataloader to compute visual prompt embeddings for each class using a YOLOE model. It normalizes the embeddings and handles cases where no samples exist for a class by setting their embeddings to zero.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` | `torch.utils.data.DataLoader` | The dataloader providing training samples. | *required* |
| `model` | `YOLOEModel` | The YOLOE model from which to extract visual prompt embeddings. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Visual prompt embeddings with shape (1, num_classes, embed_dim). |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/val.py#L51-L97"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def get_visual_pe(self, dataloader: torch.utils.data.DataLoader, model: YOLOEModel) -> torch.Tensor:
    """Extract visual prompt embeddings from training samples.

    This method processes a dataloader to compute visual prompt embeddings for each class using a YOLOE model. It
    normalizes the embeddings and handles cases where no samples exist for a class by setting their embeddings to
    zero.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader providing training samples.
        model (YOLOEModel): The YOLOE model from which to extract visual prompt embeddings.

    Returns:
        (torch.Tensor): Visual prompt embeddings with shape (1, num_classes, embed_dim).
    """
    assert isinstance(model, YOLOEModel)
    names = [name.split("/", 1)[0] for name in list(dataloader.dataset.data["names"].values())]
    visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
    cls_visual_num = torch.zeros(len(names))

    desc = "Get visual prompt embeddings from samples"

    # Count samples per class
    for batch in dataloader:
        cls = batch["cls"].squeeze(-1).to(torch.int).unique()
        count = torch.bincount(cls, minlength=len(names))
        cls_visual_num += count

    cls_visual_num = cls_visual_num.to(self.device)

    # Extract visual prompt embeddings
    pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
    for batch in pbar:
        batch = self.preprocess(batch)
        preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])  # (B, max_n, embed_dim)

        batch_idx = batch["batch_idx"]
        for i in range(preds.shape[0]):
            cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
            pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
            pad_cls[: cls.shape[0]] = cls
            for c in cls:
                visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

    # Normalize embeddings for classes with samples, set others to zero
    visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
    visual_pe[cls_visual_num == 0] = 0
    return visual_pe.unsqueeze(0)
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.get_vpe_dataloader` {#ultralytics.models.yolo.yoloe.val.YOLOEDetectValidator.get\_vpe\_dataloader}

```python
def get_vpe_dataloader(self, data: dict[str, Any]) -> torch.utils.data.DataLoader
```

Create a dataloader for LVIS training visual prompt samples.

This method prepares a dataloader for visual prompt embeddings (VPE) using the specified dataset. It applies necessary transformations including LoadVisualPrompt and configurations to the dataset for validation purposes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data` | `dict` | Dataset configuration dictionary containing paths and settings. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.utils.data.DataLoader` | The dataloader for visual prompt samples. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/val.py#L99-L130"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_vpe_dataloader(self, data: dict[str, Any]) -> torch.utils.data.DataLoader:
    """Create a dataloader for LVIS training visual prompt samples.

    This method prepares a dataloader for visual prompt embeddings (VPE) using the specified dataset. It applies
    necessary transformations including LoadVisualPrompt and configurations to the dataset for validation purposes.

    Args:
        data (dict): Dataset configuration dictionary containing paths and settings.

    Returns:
        (torch.utils.data.DataLoader): The dataloader for visual prompt samples.
    """
    dataset = build_yolo_dataset(
        self.args,
        data.get(self.args.split, data.get("val")),
        self.args.batch,
        data,
        mode="val",
        rect=False,
    )
    if isinstance(dataset, YOLOConcatDataset):
        for d in dataset.datasets:
            d.transforms.append(LoadVisualPrompt())
    else:
        dataset.transforms.append(LoadVisualPrompt())
    return build_dataloader(
        dataset,
        self.args.batch,
        self.args.workers,
        shuffle=False,
        rank=-1,
    )
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.yoloe.val.YOLOESegValidator` {#ultralytics.models.yolo.yoloe.val.YOLOESegValidator}

```python
YOLOESegValidator()
```

**Bases:** `YOLOEDetectValidator`, `SegmentationValidator`

YOLOE segmentation validator that supports both text and visual prompt embeddings.

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/val.py#L203-L206"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOESegValidator(YOLOEDetectValidator, SegmentationValidator):
```
</details>

<br><br>
