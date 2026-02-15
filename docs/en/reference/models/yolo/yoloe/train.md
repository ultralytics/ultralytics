---
description: Learn about YOLOE training classes in Ultralytics, including standard, linear probing, and visual prompt training for detection and segmentation models.
keywords: YOLOE, training, trainers, EVP, visual prompts, computer vision, object detection, segmentation, Ultralytics, deep learning
---

# Reference for `ultralytics/models/yolo/yoloe/train.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`YOLOETrainer`](#ultralytics.models.yolo.yoloe.train.YOLOETrainer)
        - [`YOLOEPETrainer`](#ultralytics.models.yolo.yoloe.train.YOLOEPETrainer)
        - [`YOLOETrainerFromScratch`](#ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch)
        - [`YOLOEPEFreeTrainer`](#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer)
        - [`YOLOEVPTrainer`](#ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`YOLOETrainer.get_model`](#ultralytics.models.yolo.yoloe.train.YOLOETrainer.get_model)
        - [`YOLOETrainer.get_validator`](#ultralytics.models.yolo.yoloe.train.YOLOETrainer.get_validator)
        - [`YOLOETrainer.build_dataset`](#ultralytics.models.yolo.yoloe.train.YOLOETrainer.build_dataset)
        - [`YOLOEPETrainer.get_model`](#ultralytics.models.yolo.yoloe.train.YOLOEPETrainer.get_model)
        - [`YOLOETrainerFromScratch.build_dataset`](#ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch.build_dataset)
        - [`YOLOETrainerFromScratch.generate_text_embeddings`](#ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch.generate_text_embeddings)
        - [`YOLOEPEFreeTrainer.get_validator`](#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.get_validator)
        - [`YOLOEPEFreeTrainer.preprocess_batch`](#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.preprocess_batch)
        - [`YOLOEPEFreeTrainer.set_text_embeddings`](#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.set_text_embeddings)
        - [`YOLOEVPTrainer.build_dataset`](#ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer.build_dataset)
        - [`YOLOEVPTrainer._close_dataloader_mosaic`](#ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer._close_dataloader_mosaic)


## Class `ultralytics.models.yolo.yoloe.train.YOLOETrainer` {#ultralytics.models.yolo.yoloe.train.YOLOETrainer}

```python
YOLOETrainer(self, cfg = DEFAULT_CFG, overrides: dict | None = None, _callbacks = None)
```

**Bases:** `DetectionTrainer`

A trainer class for YOLOE object detection models.

This class extends DetectionTrainer to provide specialized training functionality for YOLOE models, including custom model initialization, validation, and dataset building with multi-modal support.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict` | Configuration dictionary with default training settings from DEFAULT_CFG. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Dictionary of parameter overrides for the default configuration. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be applied during training. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `loss_names` | `tuple` | Names of loss components used during training. |

**Methods**

| Name | Description |
| --- | --- |
| [`build_dataset`](#ultralytics.models.yolo.yoloe.train.YOLOETrainer.build_dataset) | Build YOLO Dataset. |
| [`get_model`](#ultralytics.models.yolo.yoloe.train.YOLOETrainer.get_model) | Return a YOLOEModel initialized with the specified configuration and weights. |
| [`get_validator`](#ultralytics.models.yolo.yoloe.train.YOLOETrainer.get_validator) | Return a YOLOEDetectValidator for YOLOE model validation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L21-L101"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOETrainer(DetectionTrainer):
    """A trainer class for YOLOE object detection models.

    This class extends DetectionTrainer to provide specialized training functionality for YOLOE models, including custom
    model initialization, validation, and dataset building with multi-modal support.

    Attributes:
        loss_names (tuple): Names of loss components used during training.

    Methods:
        get_model: Initialize and return a YOLOEModel with specified configuration.
        get_validator: Return a YOLOEDetectValidator for model validation.
        build_dataset: Build YOLO dataset with multi-modal support for training.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        """Initialize the YOLOE Trainer with specified configurations.

        Args:
            cfg (dict): Configuration dictionary with default training settings from DEFAULT_CFG.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be applied during training.
        """
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        overrides["overlap_mask"] = False
        super().__init__(cfg, overrides, _callbacks)
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOETrainer.build_dataset` {#ultralytics.models.yolo.yoloe.train.YOLOETrainer.build\_dataset}

```python
def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None)
```

Build YOLO Dataset.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `str` | Path to the folder containing images. | *required* |
| `mode` | `str` | 'train' mode or 'val' mode, users are able to customize different augmentations for each mode. | `"train"` |
| `batch` | `int, optional` | Size of batches, this is for rectangular training. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `Dataset` | YOLO dataset configured for training or validation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L87-L101"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
    """Build YOLO Dataset.

    Args:
        img_path (str): Path to the folder containing images.
        mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for rectangular training.

    Returns:
        (Dataset): YOLO dataset configured for training or validation.
    """
    gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
    return build_yolo_dataset(
        self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOETrainer.get_model` {#ultralytics.models.yolo.yoloe.train.YOLOETrainer.get\_model}

```python
def get_model(self, cfg = None, weights = None, verbose: bool = True)
```

Return a YOLOEModel initialized with the specified configuration and weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict | str, optional` | Model configuration. Can be a dictionary containing a 'yaml_file' key, a direct<br>    path to a YAML file, or None to use default configuration. | `None` |
| `weights` | `str | Path, optional` | Path to pretrained weights file to load into the model. | `None` |
| `verbose` | `bool` | Whether to display model information during initialization. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `YOLOEModel` | The initialized YOLOE model. |

!!! note "Notes"

    - The number of classes (nc) is hard-coded to a maximum of 80 following the official configuration.
    - The nc parameter here represents the maximum number of different text samples in one image,
      rather than the actual number of classes.

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L50-L78"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(self, cfg=None, weights=None, verbose: bool = True):
    """Return a YOLOEModel initialized with the specified configuration and weights.

    Args:
        cfg (dict | str, optional): Model configuration. Can be a dictionary containing a 'yaml_file' key, a direct
            path to a YAML file, or None to use default configuration.
        weights (str | Path, optional): Path to pretrained weights file to load into the model.
        verbose (bool): Whether to display model information during initialization.

    Returns:
        (YOLOEModel): The initialized YOLOE model.

    Notes:
        - The number of classes (nc) is hard-coded to a maximum of 80 following the official configuration.
        - The nc parameter here represents the maximum number of different text samples in one image,
          rather than the actual number of classes.
    """
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = YOLOEModel(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=self.data["channels"],
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )
    if weights:
        model.load(weights)

    return model
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOETrainer.get_validator` {#ultralytics.models.yolo.yoloe.train.YOLOETrainer.get\_validator}

```python
def get_validator(self)
```

Return a YOLOEDetectValidator for YOLOE model validation.

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L80-L85"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Return a YOLOEDetectValidator for YOLOE model validation."""
    self.loss_names = "box", "cls", "dfl"
    return YOLOEDetectValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.yoloe.train.YOLOEPETrainer` {#ultralytics.models.yolo.yoloe.train.YOLOEPETrainer}

```python
YOLOEPETrainer()
```

**Bases:** `DetectionTrainer`

Fine-tune YOLOE model using linear probing approach.

This trainer freezes most model layers and only trains specific projection layers for efficient fine-tuning on new datasets while preserving pretrained features.

**Methods**

| Name | Description |
| --- | --- |
| [`get_model`](#ultralytics.models.yolo.yoloe.train.YOLOEPETrainer.get_model) | Return YOLOEModel initialized with specified config and weights. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L104-L158"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOEPETrainer(DetectionTrainer):
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOEPETrainer.get_model` {#ultralytics.models.yolo.yoloe.train.YOLOEPETrainer.get\_model}

```python
def get_model(self, cfg = None, weights = None, verbose: bool = True)
```

Return YOLOEModel initialized with specified config and weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict | str, optional` | Model configuration. | `None` |
| `weights` | `str, optional` | Path to pretrained weights. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `YOLOEModel` | Initialized model with frozen layers except for specific projection layers. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L114-L158"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(self, cfg=None, weights=None, verbose: bool = True):
    """Return YOLOEModel initialized with specified config and weights.

    Args:
        cfg (dict | str, optional): Model configuration.
        weights (str, optional): Path to pretrained weights.
        verbose (bool): Whether to display model information.

    Returns:
        (YOLOEModel): Initialized model with frozen layers except for specific projection layers.
    """
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = YOLOEModel(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=self.data["channels"],
        nc=self.data["nc"],
        verbose=verbose and RANK == -1,
    )

    del model.model[-1].savpe

    assert weights is not None, "Pretrained weights must be provided for linear probing."
    if weights:
        model.load(weights)

    model.eval()
    names = list(self.data["names"].values())
    # NOTE: `get_text_pe` related to text model and YOLOEDetect.reprta,
    # it'd get correct results as long as loading proper pretrained weights.
    tpe = model.get_text_pe(names)
    model.set_classes(names, tpe)
    model.model[-1].fuse(model.pe)  # fuse text embeddings to classify head
    model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
    model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
    model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

    if getattr(model.model[-1], "one2one_cv3", None) is not None:
        model.model[-1].one2one_cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].one2one_cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].one2one_cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

    model.train()

    return model
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch` {#ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch}

```python
YOLOETrainerFromScratch()
```

**Bases:** `YOLOETrainer`, `WorldTrainerFromScratch`

Train YOLOE models from scratch with text embedding support.

This trainer combines YOLOE training capabilities with world training features, enabling training from scratch with text embeddings and grounding datasets.

**Methods**

| Name | Description |
| --- | --- |
| [`build_dataset`](#ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch.build_dataset) | Build YOLO Dataset for training or validation. |
| [`generate_text_embeddings`](#ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch.generate_text_embeddings) | Generate text embeddings for a list of text samples. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L161-L210"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOETrainerFromScratch(YOLOETrainer, WorldTrainerFromScratch):
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch.build_dataset` {#ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch.build\_dataset}

```python
def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None)
```

Build YOLO Dataset for training or validation.

This method constructs appropriate datasets based on the mode and input paths, handling both standard YOLO datasets and grounding datasets with different formats.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `list[str] | str` | Path to the folder containing images or list of paths. | *required* |
| `mode` | `str` | 'train' mode or 'val' mode, allowing customized augmentations for each mode. | `"train"` |
| `batch` | `int, optional` | Size of batches, used for rectangular training/validation. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `YOLOConcatDataset | Dataset` | The constructed dataset for training or validation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L172-L186"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
    """Build YOLO Dataset for training or validation.

    This method constructs appropriate datasets based on the mode and input paths, handling both standard YOLO
    datasets and grounding datasets with different formats.

    Args:
        img_path (list[str] | str): Path to the folder containing images or list of paths.
        mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
        batch (int, optional): Size of batches, used for rectangular training/validation.

    Returns:
        (YOLOConcatDataset | Dataset): The constructed dataset for training or validation.
    """
    return WorldTrainerFromScratch.build_dataset(self, img_path, mode, batch)
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch.generate_text_embeddings` {#ultralytics.models.yolo.yoloe.train.YOLOETrainerFromScratch.generate\_text\_embeddings}

```python
def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path)
```

Generate text embeddings for a list of text samples.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` | `list[str]` | List of text samples to encode. | *required* |
| `batch` | `int` | Batch size for processing. | *required* |
| `cache_dir` | `Path` | Directory to save/load cached embeddings. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Dictionary mapping text samples to their embeddings. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L188-L210"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path):
    """Generate text embeddings for a list of text samples.

    Args:
        texts (list[str]): List of text samples to encode.
        batch (int): Batch size for processing.
        cache_dir (Path): Directory to save/load cached embeddings.

    Returns:
        (dict): Dictionary mapping text samples to their embeddings.
    """
    model = unwrap_model(self.model).text_model
    cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
    if cache_path.exists():
        LOGGER.info(f"Reading existed cache from '{cache_path}'")
        txt_map = torch.load(cache_path, map_location=self.device)
        if sorted(txt_map.keys()) == sorted(texts):
            return txt_map
    LOGGER.info(f"Caching text embeddings to '{cache_path}'")
    txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, without_reprta=True, cache_clip_model=False)
    txt_map = dict(zip(texts, txt_feats.squeeze(0)))
    torch.save(txt_map, cache_path)
    return txt_map
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer` {#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer}

```python
YOLOEPEFreeTrainer()
```

**Bases:** `YOLOEPETrainer`, `YOLOETrainerFromScratch`

Train prompt-free YOLOE model.

This trainer combines linear probing capabilities with from-scratch training for prompt-free YOLOE models that don't require text prompts during inference.

**Methods**

| Name | Description |
| --- | --- |
| [`get_validator`](#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.get_validator) | Return a DetectionValidator for YOLO model validation. |
| [`preprocess_batch`](#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.preprocess_batch) | Preprocess a batch of images for YOLOE training, adjusting formatting and dimensions as needed. |
| [`set_text_embeddings`](#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.set_text_embeddings) | No-op override for prompt-free training that does not require text embeddings. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L213-L243"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOEPEFreeTrainer(YOLOEPETrainer, YOLOETrainerFromScratch):
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.get_validator` {#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.get\_validator}

```python
def get_validator(self)
```

Return a DetectionValidator for YOLO model validation.

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L225-L230"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Return a DetectionValidator for YOLO model validation."""
    self.loss_names = "box", "cls", "dfl"
    return DetectionValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.preprocess_batch` {#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.preprocess\_batch}

```python
def preprocess_batch(self, batch)
```

Preprocess a batch of images for YOLOE training, adjusting formatting and dimensions as needed.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L232-L234"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def preprocess_batch(self, batch):
    """Preprocess a batch of images for YOLOE training, adjusting formatting and dimensions as needed."""
    return DetectionTrainer.preprocess_batch(self, batch)
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.set_text_embeddings` {#ultralytics.models.yolo.yoloe.train.YOLOEPEFreeTrainer.set\_text\_embeddings}

```python
def set_text_embeddings(self, datasets, batch: int)
```

No-op override for prompt-free training that does not require text embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `datasets` | `list[Dataset]` | List of datasets containing category names to process. | *required* |
| `batch` | `int` | Batch size for processing text embeddings. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L236-L243"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_text_embeddings(self, datasets, batch: int):
    """No-op override for prompt-free training that does not require text embeddings.

    Args:
        datasets (list[Dataset]): List of datasets containing category names to process.
        batch (int): Batch size for processing text embeddings.
    """
    pass
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer` {#ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer}

```python
YOLOEVPTrainer()
```

**Bases:** `YOLOETrainerFromScratch`

Train YOLOE model with visual prompts.

This trainer extends YOLOETrainerFromScratch to support visual prompt-based training, where visual cues are provided alongside images to guide the detection process.

**Methods**

| Name | Description |
| --- | --- |
| [`_close_dataloader_mosaic`](#ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer._close_dataloader_mosaic) | Close mosaic augmentation and add visual prompt loading to the training dataset. |
| [`build_dataset`](#ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer.build_dataset) | Build YOLO Dataset for training or validation with visual prompts. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L246-L283"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOEVPTrainer(YOLOETrainerFromScratch):
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer._close_dataloader_mosaic` {#ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer.\_close\_dataloader\_mosaic}

```python
def _close_dataloader_mosaic(self)
```

Close mosaic augmentation and add visual prompt loading to the training dataset.

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L276-L283"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _close_dataloader_mosaic(self):
    """Close mosaic augmentation and add visual prompt loading to the training dataset."""
    super()._close_dataloader_mosaic()
    if isinstance(self.train_loader.dataset, YOLOConcatDataset):
        for d in self.train_loader.dataset.datasets:
            d.transforms.append(LoadVisualPrompt())
    else:
        self.train_loader.dataset.transforms.append(LoadVisualPrompt())
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer.build_dataset` {#ultralytics.models.yolo.yoloe.train.YOLOEVPTrainer.build\_dataset}

```python
def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None)
```

Build YOLO Dataset for training or validation with visual prompts.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `list[str] | str` | Path to the folder containing images or list of paths. | *required* |
| `mode` | `str` | 'train' mode or 'val' mode, allowing customized augmentations for each mode. | `"train"` |
| `batch` | `int, optional` | Size of batches, used for rectangular training/validation. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `YOLOConcatDataset | Dataset` | YOLO dataset configured for training or validation, with visual prompts for |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train.py#L256-L274"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
    """Build YOLO Dataset for training or validation with visual prompts.

    Args:
        img_path (list[str] | str): Path to the folder containing images or list of paths.
        mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
        batch (int, optional): Size of batches, used for rectangular training/validation.

    Returns:
        (YOLOConcatDataset | Dataset): YOLO dataset configured for training or validation, with visual prompts for
            training mode.
    """
    dataset = super().build_dataset(img_path, mode, batch)
    if isinstance(dataset, YOLOConcatDataset):
        for d in dataset.datasets:
            d.transforms.append(LoadVisualPrompt())
    else:
        dataset.transforms.append(LoadVisualPrompt())
    return dataset
```
</details>

<br><br>
