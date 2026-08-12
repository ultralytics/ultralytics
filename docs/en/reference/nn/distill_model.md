---
title: nn.distill_model API Reference
description: Reference for `ultralytics.nn.distill_model` in the Ultralytics package.
keywords: Ultralytics, ultralytics.nn.distill_model, API reference, YOLO, Python
---

# Reference for `ultralytics/nn/distill_model.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`FeatureHook`](#ultralytics.nn.distill_model.FeatureHook)
        - [`DistillationModel`](#ultralytics.nn.distill_model.DistillationModel)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`DistillationModel.criterion`](#ultralytics.nn.distill_model.DistillationModel.criterion)
        - [`DistillationModel.end2end`](#ultralytics.nn.distill_model.DistillationModel.end2end)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`FeatureHook.__call__`](#ultralytics.nn.distill_model.FeatureHook.__call__)
        - [`DistillationModel.__getstate__`](#ultralytics.nn.distill_model.DistillationModel.__getstate__)
        - [`DistillationModel.__setstate__`](#ultralytics.nn.distill_model.DistillationModel.__setstate__)
        - [`DistillationModel._remove_feature_hooks`](#ultralytics.nn.distill_model.DistillationModel._remove_feature_hooks)
        - [`DistillationModel._clear_feature_hooks`](#ultralytics.nn.distill_model.DistillationModel._clear_feature_hooks)
        - [`DistillationModel._register_feature_hooks`](#ultralytics.nn.distill_model.DistillationModel._register_feature_hooks)
        - [`DistillationModel.get_distill_layers`](#ultralytics.nn.distill_model.DistillationModel.get_distill_layers)
        - [`DistillationModel._freeze_teacher`](#ultralytics.nn.distill_model.DistillationModel._freeze_teacher)
        - [`DistillationModel.train`](#ultralytics.nn.distill_model.DistillationModel.train)
        - [`DistillationModel.forward`](#ultralytics.nn.distill_model.DistillationModel.forward)
        - [`DistillationModel.fuse`](#ultralytics.nn.distill_model.DistillationModel.fuse)
        - [`DistillationModel.loss`](#ultralytics.nn.distill_model.DistillationModel.loss)
        - [`DistillationModel.loss_sl2`](#ultralytics.nn.distill_model.DistillationModel.loss_sl2)
        - [`DistillationModel.criterion`](#ultralytics.nn.distill_model.DistillationModel.criterion)
        - [`DistillationModel.init_criterion`](#ultralytics.nn.distill_model.DistillationModel.init_criterion)
        - [`DistillationModel.end2end`](#ultralytics.nn.distill_model.DistillationModel.end2end)
        - [`DistillationModel.set_head_attr`](#ultralytics.nn.distill_model.DistillationModel.set_head_attr)
        - [`DistillationModel.decouple_outputs`](#ultralytics.nn.distill_model.DistillationModel.decouple_outputs)


## Class `ultralytics.nn.distill_model.FeatureHook` {#ultralytics.nn.distill\_model.FeatureHook}

```python
FeatureHook(feat_dict: dict, idx: int)
```

Picklable forward hook that stores layer output into a shared dict.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `feat_dict` | `dict` |  | *required* |
| `idx` | `int` |  | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.nn.distill_model.FeatureHook.__call__) | Store the layer's forward output into the shared feature dict under its index. |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L17-L30">View on GitHub</a>
```python
class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        """Initialize the hook with the shared feature dict and the layer index to store outputs under."""
        self.feat_dict = feat_dict
        self.idx = idx
```
</details>

<br>

### Method `ultralytics.nn.distill_model.FeatureHook.__call__` {#ultralytics.nn.distill\_model.FeatureHook.\_\_call\_\_}

```python
def __call__(self, module: nn.Module, inputs: tuple, output) -> None
```

Store the layer's forward output into the shared feature dict under its index.

The output is a tensor for neck layers but a tuple/dict for the Detect head, so it is left untyped.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `module` | `nn.Module` |  | *required* |
| `inputs` | `tuple` |  | *required* |
| `output` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L25-L30">View on GitHub</a>
```python
def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
    """Store the layer's forward output into the shared feature dict under its index.

    The output is a tensor for neck layers but a tuple/dict for the Detect head, so it is left untyped.
    """
    self.feat_dict[self.idx] = output
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.distill_model.DistillationModel` {#ultralytics.nn.distill\_model.DistillationModel}

```python
DistillationModel(teacher_model: str | Path | nn.Module, student_model: nn.Module)
```

**Bases:** `nn.Module`

YOLO knowledge distillation model.

This class wraps a teacher-student pair for knowledge distillation training. Features are extracted from both models via forward hooks for distillation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `teacher_model` | `str \| Path \| nn.Module` | Teacher model checkpoint path or module. | *required* |
| `student_model` | `nn.Module` | Student model module to be trained. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `teacher_model` | `nn.Module` | Frozen teacher model providing features. |
| `student_model` | `nn.Module` | Trainable student model being distilled. |
| `feats_idx` | `list` | Layer indices for feature extraction. |
| `projector` | `nn.ModuleList` | MLP projector aligning student features to teacher dimensions. |
| `dis` | `float` | Distillation loss weight factor. |

**Methods**

| Name | Description |
| --- | --- |
| [`criterion`](#ultralytics.nn.distill_model.DistillationModel.criterion) | Get the criterion from the student model. |
| [`end2end`](#ultralytics.nn.distill_model.DistillationModel.end2end) | Expose student end-to-end mode for validator/predictor control. |
| [`__getstate__`](#ultralytics.nn.distill_model.DistillationModel.__getstate__) | Return a copy of state for pickling without captured features or hook handles. |
| [`__setstate__`](#ultralytics.nn.distill_model.DistillationModel.__setstate__) | Clear stale features and hooks, and re-register forward hooks after unpickling. |
| [`_clear_feature_hooks`](#ultralytics.nn.distill_model.DistillationModel._clear_feature_hooks) | Remove any FeatureHook instances from a module's forward hooks. |
| [`_freeze_teacher`](#ultralytics.nn.distill_model.DistillationModel._freeze_teacher) | Keep teacher fixed for distillation. |
| [`_register_feature_hooks`](#ultralytics.nn.distill_model.DistillationModel._register_feature_hooks) | Register feature-capture hooks, removing stale FeatureHook instances first. |
| [`_remove_feature_hooks`](#ultralytics.nn.distill_model.DistillationModel._remove_feature_hooks) | Remove any previously registered feature-capture hooks. |
| [`criterion`](#ultralytics.nn.distill_model.DistillationModel.criterion) | Set value for student criterion. |
| [`decouple_outputs`](#ultralytics.nn.distill_model.DistillationModel.decouple_outputs) | Decouple outputs for teacher/student models. |
| [`end2end`](#ultralytics.nn.distill_model.DistillationModel.end2end) | Forward end-to-end mode update to the student model. |
| [`forward`](#ultralytics.nn.distill_model.DistillationModel.forward) | Forward pass through the student model. |
| [`fuse`](#ultralytics.nn.distill_model.DistillationModel.fuse) | Fuse and return the student model, dropping the training-only distillation wrapper. |
| [`get_distill_layers`](#ultralytics.nn.distill_model.DistillationModel.get_distill_layers) | Auto-detect distillation feature layers from the model's Detect head. |
| [`init_criterion`](#ultralytics.nn.distill_model.DistillationModel.init_criterion) | Initialize the loss criterion via the student model. |
| [`loss`](#ultralytics.nn.distill_model.DistillationModel.loss) | Compute loss. |
| [`loss_sl2`](#ultralytics.nn.distill_model.DistillationModel.loss_sl2) | Compute score-weighted L2 distillation loss for a feature pair. |
| [`set_head_attr`](#ultralytics.nn.distill_model.DistillationModel.set_head_attr) | Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model. |
| [`train`](#ultralytics.nn.distill_model.DistillationModel.train) | Set model train mode while keeping teacher frozen in eval mode. |

**Examples**

Train a student model with knowledge distillation from a larger teacher (the trainer builds the
DistillationModel internally when the ``distill_model`` argument is set)

```python
>>> from ultralytics import YOLO
>>> model = YOLO("yolo26n.pt")
>>> model.train(data="coco8.yaml", distill_model="yolo26s.pt")
```

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L33-L319">View on GitHub</a>
```python
class DistillationModel(nn.Module):
    """YOLO knowledge distillation model.

    This class wraps a teacher-student pair for knowledge distillation training. Features are extracted from both models
    via forward hooks for distillation.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        feats_idx (list): Layer indices for feature extraction.
        projector (nn.ModuleList): MLP projector aligning student features to teacher dimensions.
        dis (float): Distillation loss weight factor.

    Methods:
        get_distill_layers: Auto-detect distillation feature layers from the Detect head.
        forward: Run the student model, or compute the combined loss when given a training batch.
        loss: Compute combined detection and distillation loss.
        loss_sl2: Compute score-weighted L2 distillation loss for a feature pair.
        decouple_outputs: Normalize teacher/student head outputs across train/val formats.
        fuse: Fuse and return the student model for inference and export.
        train: Set training mode while keeping teacher frozen.

    Examples:
        Train a student model with knowledge distillation from a larger teacher (the trainer builds the
        DistillationModel internally when the ``distill_model`` argument is set)
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="coco8.yaml", distill_model="yolo26s.pt")
    """

    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module):
        """Initialize the distillation model with teacher, student, and feature extraction hooks.

        Args:
            teacher_model (str | Path | nn.Module): Teacher model checkpoint path or module.
            student_model (nn.Module): Student model module to be trained.
        """
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
            if teacher_model.yaml.get("channels", 3) != ch:
                weights = teacher_model
                teacher_model = type(weights)(weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False)
                teacher_model.load(weights)
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.feats_idx = self.get_distill_layers(student_model)

        # Hook-based feature capture: identical for teacher and student
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        # Get feature dimensions via dummy forward pass (hooks capture outputs)
        imgsz = student_model.args.imgsz
        student_model.eval()
        with torch.no_grad():
            im = torch.zeros(2, ch, imgsz, imgsz, device=device)
            teacher_model(im)
            student_model(im)
        student_model.train()
        teacher_output = [self._teacher_feats[idx] for idx in self.feats_idx]
        student_output = [self._student_feats[idx] for idx in self.feats_idx]

        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        projectors = []
        for student_out, teacher_out in zip(student_output[:-1], teacher_output[:-1]):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)
```
</details>

<br>

### Property `ultralytics.nn.distill_model.DistillationModel.criterion` {#ultralytics.nn.distill\_model.DistillationModel.criterion}

```python
def criterion(self)
```

Get the criterion from the student model.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L274-L276">View on GitHub</a>
```python
@property
def criterion(self):
    """Get the criterion from the student model."""
    return self.student_model.criterion
```
</details>

<br>

### Property `ultralytics.nn.distill_model.DistillationModel.end2end` {#ultralytics.nn.distill\_model.DistillationModel.end2end}

```python
def end2end(self)
```

Expose student end-to-end mode for validator/predictor control.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L288-L290">View on GitHub</a>
```python
@property
def end2end(self):
    """Expose student end-to-end mode for validator/predictor control."""
    return getattr(self.student_model, "end2end", False)
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.__getstate__` {#ultralytics.nn.distill\_model.DistillationModel.\_\_getstate\_\_}

```python
def __getstate__(self)
```

Return a copy of state for pickling without captured features or hook handles.

Clears the feature dicts in place (rather than replacing the attributes) because the registered FeatureHooks share these exact dict objects; otherwise a deepcopy/pickle of a mid-training model would still reach the hook-held tensors (which carry grad_fn and cannot be deep-copied).

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L117-L129">View on GitHub</a>
```python
def __getstate__(self):
    """Return a copy of state for pickling without captured features or hook handles.

    Clears the feature dicts in place (rather than replacing the attributes) because the registered
    FeatureHooks share these exact dict objects; otherwise a deepcopy/pickle of a mid-training model would
    still reach the hook-held tensors (which carry grad_fn and cannot be deep-copied).
    """
    self._teacher_feats.clear()
    self._student_feats.clear()
    state = self.__dict__.copy()
    state["_teacher_hooks"] = []
    state["_student_hooks"] = []
    return state
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.__setstate__` {#ultralytics.nn.distill\_model.DistillationModel.\_\_setstate\_\_}

```python
def __setstate__(self, state)
```

Clear stale features and hooks, and re-register forward hooks after unpickling.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L131-L136">View on GitHub</a>
```python
def __setstate__(self, state):
    """Clear stale features and hooks, and re-register forward hooks after unpickling."""
    self.__dict__.update(state)
    self._teacher_feats = {}
    self._student_feats = {}
    self._register_feature_hooks()
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel._clear_feature_hooks` {#ultralytics.nn.distill\_model.DistillationModel.\_clear\_feature\_hooks}

```python
def _clear_feature_hooks(module: nn.Module) -> None
```

Remove any FeatureHook instances from a module's forward hooks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `module` | `nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L149-L153">View on GitHub</a>
```python
@staticmethod
def _clear_feature_hooks(module: nn.Module) -> None:
    """Remove any FeatureHook instances from a module's forward hooks."""
    for handle_id, hook in list(module._forward_hooks.items()):
        if isinstance(hook, FeatureHook):
            del module._forward_hooks[handle_id]
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel._freeze_teacher` {#ultralytics.nn.distill\_model.DistillationModel.\_freeze\_teacher}

```python
def _freeze_teacher(self)
```

Keep teacher fixed for distillation.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L181-L188">View on GitHub</a>
```python
def _freeze_teacher(self):
    """Keep teacher fixed for distillation."""
    if self.teacher_model is None:
        return
    self.teacher_model.eval()
    for v in self.teacher_model.parameters():
        if v.requires_grad:
            v.requires_grad = False
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel._register_feature_hooks` {#ultralytics.nn.distill\_model.DistillationModel.\_register\_feature\_hooks}

```python
def _register_feature_hooks(self) -> None
```

Register feature-capture hooks, removing stale FeatureHook instances first.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L155-L167">View on GitHub</a>
```python
def _register_feature_hooks(self) -> None:
    """Register feature-capture hooks, removing stale FeatureHook instances first."""
    self._remove_feature_hooks()
    for idx in self.feats_idx:
        self._clear_feature_hooks(self.student_model.model[idx])
        self._student_hooks.append(
            self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
        )
        if self.teacher_model is not None:
            self._clear_feature_hooks(self.teacher_model.model[idx])
            self._teacher_hooks.append(
                self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
            )
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel._remove_feature_hooks` {#ultralytics.nn.distill\_model.DistillationModel.\_remove\_feature\_hooks}

```python
def _remove_feature_hooks(self) -> None
```

Remove any previously registered feature-capture hooks.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L138-L146">View on GitHub</a>
```python
def _remove_feature_hooks(self) -> None:
    """Remove any previously registered feature-capture hooks."""
    for handle in self._student_hooks:
        handle.remove()
    self._student_hooks.clear()
    if self.teacher_model is not None:
        for handle in self._teacher_hooks:
            handle.remove()
        self._teacher_hooks.clear()
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.criterion` {#ultralytics.nn.distill\_model.DistillationModel.criterion}

```python
def criterion(self, value) -> None
```

Set value for student criterion.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L279-L281">View on GitHub</a>
```python
@criterion.setter
def criterion(self, value) -> None:
    """Set value for student criterion."""
    self.student_model.criterion = value
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.decouple_outputs` {#ultralytics.nn.distill\_model.DistillationModel.decouple\_outputs}

```python
def decouple_outputs(self, preds, branch: str = "one2one")
```

Decouple outputs for teacher/student models.

This method handles different output formats from YOLO models, including tuple outputs (train/val mode), dict outputs with branches (one2one/one2many), and direct tensor outputs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor \| tuple \| dict` | Model predictions in various formats. | *required* |
| `branch` | `str` | Which branch to extract from dict outputs ("one2one" or "one2many"). | `"one2one"` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor \| dict` | The decoupled predictions. |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L301-L319">View on GitHub</a>
```python
def decouple_outputs(self, preds, branch: str = "one2one"):
    """Decouple outputs for teacher/student models.

    This method handles different output formats from YOLO models, including
    tuple outputs (train/val mode), dict outputs with branches (one2one/one2many),
    and direct tensor outputs.

    Args:
        preds (torch.Tensor | tuple | dict): Model predictions in various formats.
        branch (str): Which branch to extract from dict outputs ("one2one" or "one2many").

    Returns:
        (torch.Tensor | dict): The decoupled predictions.
    """
    if isinstance(preds, tuple):  # decouple for val mode
        preds = preds[1]
    if isinstance(preds, dict) and branch in preds:
        preds = preds[branch]
    return preds
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.end2end` {#ultralytics.nn.distill\_model.DistillationModel.end2end}

```python
def end2end(self, value)
```

Forward end-to-end mode update to the student model.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L293-L295">View on GitHub</a>
```python
@end2end.setter
def end2end(self, value):
    """Forward end-to-end mode update to the student model."""
    self.student_model.end2end = value
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.forward` {#ultralytics.nn.distill\_model.DistillationModel.forward}

```python
def forward(self, x, *args, **kwargs)
```

Forward pass through the student model.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L196-L200">View on GitHub</a>
```python
def forward(self, x, *args, **kwargs):
    """Forward pass through the student model."""
    if isinstance(x, dict):  # for cases of training and validating while training.
        return self.loss(x, *args, **kwargs)
    return self.student_model.predict(x, *args, **kwargs)
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.fuse` {#ultralytics.nn.distill\_model.DistillationModel.fuse}

```python
def fuse(self, verbose: bool = True, imgsz: int | list[int, int] = 640)
```

Fuse and return the student model, dropping the training-only distillation wrapper.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `verbose` | `bool` |  | `True` |
| `imgsz` | `int \| list[int, int]` |  | `640` |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L202-L205">View on GitHub</a>
```python
def fuse(self, verbose: bool = True, imgsz: int | list[int, int] = 640):
    """Fuse and return the student model, dropping the training-only distillation wrapper."""
    self._remove_feature_hooks()
    return self.student_model.fuse(verbose=verbose, imgsz=imgsz)
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.get_distill_layers` {#ultralytics.nn.distill\_model.DistillationModel.get\_distill\_layers}

```python
def get_distill_layers(model: nn.Module) -> list[int]
```

Auto-detect distillation feature layers from the model's Detect head.

Returns the Detect head's input layer indices plus the head layer index itself. E.g. YOLO26 -> [16, 19, 22, 23], YOLOv8 -> [15, 18, 21, 22].

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L170-L179">View on GitHub</a>
```python
@staticmethod
def get_distill_layers(model: nn.Module) -> list[int]:
    """Auto-detect distillation feature layers from the model's Detect head.

    Returns the Detect head's input layer indices plus the head layer index itself.
    E.g. YOLO26 -> [16, 19, 22, 23], YOLOv8 -> [15, 18, 21, 22].
    """
    for m in model.model:
        if isinstance(m, Detect):
            return [*list(m.f), m.i]
    raise ValueError("No Detect head found in model")
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.init_criterion` {#ultralytics.nn.distill\_model.DistillationModel.init\_criterion}

```python
def init_criterion(self)
```

Initialize the loss criterion via the student model.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L283-L285">View on GitHub</a>
```python
def init_criterion(self):
    """Initialize the loss criterion via the student model."""
    return self.student_model.init_criterion()
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.loss` {#ultralytics.nn.distill\_model.DistillationModel.loss}

```python
def loss(self, batch, preds=None)
```

Compute loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict` | Batch to compute loss on. | *required* |
| `preds` | `torch.Tensor \| list[torch.Tensor], optional` | Predictions. | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L207-L249">View on GitHub</a>
```python
def loss(self, batch, preds=None):
    """Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
    """
    loss_distill = torch.zeros(1, device=batch["img"].device)
    if not self.training:  # for loss calculation during validation while training
        if preds is None:
            preds = self.student_model(batch["img"])
        regular_loss, loss_items = self.student_model.loss(batch, preds)
        loss_items["dis_loss"] = loss_distill.detach()
        return torch.cat([regular_loss, loss_distill]), loss_items

    # Clear feature dicts before forward passes
    self._teacher_feats.clear()
    self._student_feats.clear()

    with torch.no_grad():
        self.teacher_model(batch["img"])  # hooks capture teacher features
    preds = self.student_model(batch["img"])  # hooks capture student features

    regular_loss, loss_items = self.student_model.loss(batch, preds)
    teacher_head_feat = self._teacher_feats[self.feats_idx[-1]]
    teacher_scores = (
        self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
        + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
    ) / 2
    # neck feature sizes vary per batch (e.g. multi_scale), so split scores by the live teacher feats
    neck_feats = [self._teacher_feats[idx] for idx in self.feats_idx[:-1]]
    parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in neck_feats], dim=-1)
    teacher_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
    for i, feat_idx in enumerate(self.feats_idx[:-1]):
        teacher_feat = self.decouple_outputs(self._teacher_feats[feat_idx])
        student_feat = self.projector[i](self.decouple_outputs(self._student_feats[feat_idx]))
        loss_distill += (
            self.loss_sl2(student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores) * self.dis
        )

    loss_items["dis_loss"] = loss_distill.detach()
    loss_distill = loss_distill * batch["img"].shape[0]
    return torch.cat([regular_loss, loss_distill]), loss_items
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.loss_sl2` {#ultralytics.nn.distill\_model.DistillationModel.loss\_sl2}

```python
def loss_sl2(
    self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
) -> torch.Tensor
```

Compute score-weighted L2 distillation loss for a feature pair.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `student_feat` | `torch.Tensor` | Student feature tensor of shape (N, C, H, W). | *required* |
| `teacher_feat` | `torch.Tensor` | Teacher feature tensor of shape (N, C, H, W). | *required* |
| `feat_idx` | `int` | Index of the feature level for selecting teacher scores. | *required* |
| `teacher_scores` | `tuple` | Tuple of score tensors for each feature level. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | The computed score-weighted L2 loss. |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L251-L271">View on GitHub</a>
```python
def loss_sl2(
    self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
) -> torch.Tensor:
    """Compute score-weighted L2 distillation loss for a feature pair.

    Args:
        student_feat (torch.Tensor): Student feature tensor of shape (N, C, H, W).
        teacher_feat (torch.Tensor): Teacher feature tensor of shape (N, C, H, W).
        feat_idx (int): Index of the feature level for selecting teacher scores.
        teacher_scores (tuple): Tuple of score tensors for each feature level.

    Returns:
        (torch.Tensor): The computed score-weighted L2 loss.
    """
    teacher_score = teacher_scores[feat_idx]
    n, c = student_feat.shape[:2]
    student_feat = student_feat.view(n, c, -1)
    teacher_feat = teacher_feat.view(n, c, -1)
    mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
    weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
    return weighted_mse
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.set_head_attr` {#ultralytics.nn.distill\_model.DistillationModel.set\_head\_attr}

```python
def set_head_attr(self, **kwargs)
```

Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model.

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L297-L299">View on GitHub</a>
```python
def set_head_attr(self, **kwargs):
    """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
    self.student_model.set_head_attr(**kwargs)
```
</details>

<br>

### Method `ultralytics.nn.distill_model.DistillationModel.train` {#ultralytics.nn.distill\_model.DistillationModel.train}

```python
def train(self, mode: bool = True)
```

Set model train mode while keeping teacher frozen in eval mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `mode` | `bool` |  | `True` |

<details>
<summary>Source code in <code>ultralytics/nn/distill_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py#L190-L194">View on GitHub</a>
```python
def train(self, mode: bool = True):
    """Set model train mode while keeping teacher frozen in eval mode."""
    super().train(mode)
    self._freeze_teacher()
    return self
```
</details>

<br><br>
