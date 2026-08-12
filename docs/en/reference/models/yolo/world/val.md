---
title: models.yolo.world.val API Reference
description: Reference for `ultralytics.models.yolo.world.val` in the Ultralytics package.
keywords: Ultralytics, ultralytics.models.yolo.world.val, API reference, YOLO, Python
---

# Reference for `ultralytics/models/yolo/world/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`WorldValidator`](#ultralytics.models.yolo.world.val.WorldValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`WorldValidator.__call__`](#ultralytics.models.yolo.world.val.WorldValidator.__call__)


## Class `ultralytics.models.yolo.world.val.WorldValidator` {#ultralytics.models.yolo.world.val.WorldValidator}

```python
WorldValidator()
```

**Bases:** `DetectionValidator`

A validator for YOLO-World models that sets dataset class names before validation.

Open-vocabulary YOLO-World models default to 80 COCO classes, so validating on a dataset with different classes (e.g. LVIS) fails or yields zero metrics. This validator generates text embeddings for the dataset's class names so standalone `model.val()` works. During training, classes are set via the `on_pretrain_routine_end` callback instead.

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.models.yolo.world.val.WorldValidator.__call__) | Set dataset classes for standalone validation, then run validation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/world/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/val.py#L12-L40">View on GitHub</a>
```python
class WorldValidator(DetectionValidator):
    """A validator for YOLO-World models that sets dataset class names before validation.

    Open-vocabulary YOLO-World models default to 80 COCO classes, so validating on a dataset with different classes
    (e.g. LVIS) fails or yields zero metrics. This validator generates text embeddings for the dataset's class names so
    standalone `model.val()` works. During training, classes are set via the `on_pretrain_routine_end` callback instead.
    """
```
</details>

<br>

### Method `ultralytics.models.yolo.world.val.WorldValidator.__call__` {#ultralytics.models.yolo.world.val.WorldValidator.\_\_call\_\_}

```python
def __call__(self, trainer=None, model=None)
```

Set dataset classes for standalone validation, then run validation.

<details>
<summary>Source code in <code>ultralytics/models/yolo/world/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/val.py#L20-L40">View on GitHub</a>
```python
def __call__(self, trainer=None, model=None):
    """Set dataset classes for standalone validation, then run validation."""
    if trainer is None:  # standalone val; training sets classes via on_pretrain_routine_end callback
        self.device = select_device(self.args.device, verbose=False)
        if not isinstance(model, torch.nn.Module):
            from ultralytics.nn.tasks import load_checkpoint

            model = load_checkpoint(model or self.args.model, device=self.device)[0]
        model.eval().to(self.device)
        self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)  # match BaseValidator dataset handling
        names = [name.split("/", 1)[0] for name in check_det_dataset(self.args.data)["names"].values()]
        current = model.names.values() if isinstance(model.names, dict) else model.names  # names may be a list
        if list(current) != names:  # regenerate prompts only if class order differs from dataset
            state = (model.names, model.txt_feats, model.model[-1].nc)  # restore after to avoid leak to caller
            model.set_classes(names, cache_clip_model=False)
            model.names = dict(enumerate(names))  # set_classes updates embeddings/nc but not names
            try:
                return super().__call__(trainer, model)
            finally:
                model.names, model.txt_feats, model.model[-1].nc = state
    return super().__call__(trainer, model)
```
</details>

<br><br>
