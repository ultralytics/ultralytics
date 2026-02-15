---
description: Dive into the intricacies of YOLO tasks.py. Learn about DetectionModel, PoseModel and more for powerful AI development.
keywords: Ultralytics, YOLO, nn tasks, DetectionModel, PoseModel, RTDETRDetectionModel, model weights, parse model, AI development
---

# Reference for `ultralytics/nn/tasks.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`BaseModel`](#ultralytics.nn.tasks.BaseModel)
        - [`DetectionModel`](#ultralytics.nn.tasks.DetectionModel)
        - [`OBBModel`](#ultralytics.nn.tasks.OBBModel)
        - [`SegmentationModel`](#ultralytics.nn.tasks.SegmentationModel)
        - [`PoseModel`](#ultralytics.nn.tasks.PoseModel)
        - [`ClassificationModel`](#ultralytics.nn.tasks.ClassificationModel)
        - [`RTDETRDetectionModel`](#ultralytics.nn.tasks.RTDETRDetectionModel)
        - [`WorldModel`](#ultralytics.nn.tasks.WorldModel)
        - [`YOLOEModel`](#ultralytics.nn.tasks.YOLOEModel)
        - [`YOLOESegModel`](#ultralytics.nn.tasks.YOLOESegModel)
        - [`Ensemble`](#ultralytics.nn.tasks.Ensemble)
        - [`SafeClass`](#ultralytics.nn.tasks.SafeClass)
        - [`SafeUnpickler`](#ultralytics.nn.tasks.SafeUnpickler)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`DetectionModel.end2end`](#ultralytics.nn.tasks.DetectionModel.end2end)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`BaseModel.forward`](#ultralytics.nn.tasks.BaseModel.forward)
        - [`BaseModel.predict`](#ultralytics.nn.tasks.BaseModel.predict)
        - [`BaseModel._predict_once`](#ultralytics.nn.tasks.BaseModel._predict_once)
        - [`BaseModel._predict_augment`](#ultralytics.nn.tasks.BaseModel._predict_augment)
        - [`BaseModel._profile_one_layer`](#ultralytics.nn.tasks.BaseModel._profile_one_layer)
        - [`BaseModel.fuse`](#ultralytics.nn.tasks.BaseModel.fuse)
        - [`BaseModel.is_fused`](#ultralytics.nn.tasks.BaseModel.is_fused)
        - [`BaseModel.info`](#ultralytics.nn.tasks.BaseModel.info)
        - [`BaseModel._apply`](#ultralytics.nn.tasks.BaseModel._apply)
        - [`BaseModel.load`](#ultralytics.nn.tasks.BaseModel.load)
        - [`BaseModel.loss`](#ultralytics.nn.tasks.BaseModel.loss)
        - [`BaseModel.init_criterion`](#ultralytics.nn.tasks.BaseModel.init_criterion)
        - [`DetectionModel.end2end`](#ultralytics.nn.tasks.DetectionModel.end2end)
        - [`DetectionModel.set_head_attr`](#ultralytics.nn.tasks.DetectionModel.set_head_attr)
        - [`DetectionModel._predict_augment`](#ultralytics.nn.tasks.DetectionModel._predict_augment)
        - [`DetectionModel._descale_pred`](#ultralytics.nn.tasks.DetectionModel._descale_pred)
        - [`DetectionModel._clip_augmented`](#ultralytics.nn.tasks.DetectionModel._clip_augmented)
        - [`DetectionModel.init_criterion`](#ultralytics.nn.tasks.DetectionModel.init_criterion)
        - [`OBBModel.init_criterion`](#ultralytics.nn.tasks.OBBModel.init_criterion)
        - [`SegmentationModel.init_criterion`](#ultralytics.nn.tasks.SegmentationModel.init_criterion)
        - [`PoseModel.init_criterion`](#ultralytics.nn.tasks.PoseModel.init_criterion)
        - [`ClassificationModel._from_yaml`](#ultralytics.nn.tasks.ClassificationModel._from_yaml)
        - [`ClassificationModel.reshape_outputs`](#ultralytics.nn.tasks.ClassificationModel.reshape_outputs)
        - [`ClassificationModel.init_criterion`](#ultralytics.nn.tasks.ClassificationModel.init_criterion)
        - [`RTDETRDetectionModel._apply`](#ultralytics.nn.tasks.RTDETRDetectionModel._apply)
        - [`RTDETRDetectionModel.init_criterion`](#ultralytics.nn.tasks.RTDETRDetectionModel.init_criterion)
        - [`RTDETRDetectionModel.loss`](#ultralytics.nn.tasks.RTDETRDetectionModel.loss)
        - [`RTDETRDetectionModel.predict`](#ultralytics.nn.tasks.RTDETRDetectionModel.predict)
        - [`WorldModel.set_classes`](#ultralytics.nn.tasks.WorldModel.set_classes)
        - [`WorldModel.get_text_pe`](#ultralytics.nn.tasks.WorldModel.get_text_pe)
        - [`WorldModel.predict`](#ultralytics.nn.tasks.WorldModel.predict)
        - [`WorldModel.loss`](#ultralytics.nn.tasks.WorldModel.loss)
        - [`YOLOEModel.get_text_pe`](#ultralytics.nn.tasks.YOLOEModel.get_text_pe)
        - [`YOLOEModel.get_visual_pe`](#ultralytics.nn.tasks.YOLOEModel.get_visual_pe)
        - [`YOLOEModel.set_vocab`](#ultralytics.nn.tasks.YOLOEModel.set_vocab)
        - [`YOLOEModel.get_vocab`](#ultralytics.nn.tasks.YOLOEModel.get_vocab)
        - [`YOLOEModel.set_classes`](#ultralytics.nn.tasks.YOLOEModel.set_classes)
        - [`YOLOEModel.get_cls_pe`](#ultralytics.nn.tasks.YOLOEModel.get_cls_pe)
        - [`YOLOEModel.predict`](#ultralytics.nn.tasks.YOLOEModel.predict)
        - [`YOLOEModel.loss`](#ultralytics.nn.tasks.YOLOEModel.loss)
        - [`YOLOESegModel.loss`](#ultralytics.nn.tasks.YOLOESegModel.loss)
        - [`Ensemble.forward`](#ultralytics.nn.tasks.Ensemble.forward)
        - [`SafeClass.__call__`](#ultralytics.nn.tasks.SafeClass.__call__)
        - [`SafeUnpickler.find_class`](#ultralytics.nn.tasks.SafeUnpickler.find_class)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`temporary_modules`](#ultralytics.nn.tasks.temporary_modules)
        - [`torch_safe_load`](#ultralytics.nn.tasks.torch_safe_load)
        - [`load_checkpoint`](#ultralytics.nn.tasks.load_checkpoint)
        - [`parse_model`](#ultralytics.nn.tasks.parse_model)
        - [`yaml_model_load`](#ultralytics.nn.tasks.yaml_model_load)
        - [`guess_model_scale`](#ultralytics.nn.tasks.guess_model_scale)
        - [`guess_model_task`](#ultralytics.nn.tasks.guess_model_task)


## Class `ultralytics.nn.tasks.BaseModel` {#ultralytics.nn.tasks.BaseModel}

```python
BaseModel()
```

**Bases:** `torch.nn.Module`

Base class for all YOLO models in the Ultralytics family.

This class provides common functionality for YOLO models including forward pass handling, model fusion, information display, and weight loading capabilities.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `torch.nn.Sequential` | The neural network model. |
| `save` | `list` | List of layer indices to save outputs from. |
| `stride` | `torch.Tensor` | Model stride values. |

**Methods**

| Name | Description |
| --- | --- |
| [`_apply`](#ultralytics.nn.tasks.BaseModel._apply) | Apply a function to all tensors in the model, including Detect head attributes like stride and anchors. |
| [`_predict_augment`](#ultralytics.nn.tasks.BaseModel._predict_augment) | Perform augmentations on input image x and return augmented inference. |
| [`_predict_once`](#ultralytics.nn.tasks.BaseModel._predict_once) | Perform a forward pass through the network. |
| [`_profile_one_layer`](#ultralytics.nn.tasks.BaseModel._profile_one_layer) | Profile the computation time and FLOPs of a single layer of the model on a given input. |
| [`forward`](#ultralytics.nn.tasks.BaseModel.forward) | Perform forward pass of the model for either training or inference. |
| [`fuse`](#ultralytics.nn.tasks.BaseModel.fuse) | Fuse Conv/ConvTranspose and BatchNorm layers, and reparameterize RepConv/RepVGGDW for improved efficiency. |
| [`info`](#ultralytics.nn.tasks.BaseModel.info) | Print model information. |
| [`init_criterion`](#ultralytics.nn.tasks.BaseModel.init_criterion) | Initialize the loss criterion for the BaseModel. |
| [`is_fused`](#ultralytics.nn.tasks.BaseModel.is_fused) | Check if the model has less than a certain threshold of normalization layers. |
| [`load`](#ultralytics.nn.tasks.BaseModel.load) | Load weights into the model. |
| [`loss`](#ultralytics.nn.tasks.BaseModel.loss) | Compute loss. |
| [`predict`](#ultralytics.nn.tasks.BaseModel.predict) | Perform a forward pass through the network. |

**Examples**

```python
Create a BaseModel instance
>>> model = BaseModel()
>>> model.info()  # Display model information
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L102-L339"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BaseModel(torch.nn.Module):
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel._apply` {#ultralytics.nn.tasks.BaseModel.\_apply}

```python
def _apply(self, fn)
```

Apply a function to all tensors in the model, including Detect head attributes like stride and anchors.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `fn` | `function` | The function to apply to the model. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `BaseModel` | An updated BaseModel object. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L279-L296"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _apply(self, fn):
    """Apply a function to all tensors in the model, including Detect head attributes like stride and anchors.

    Args:
        fn (function): The function to apply to the model.

    Returns:
        (BaseModel): An updated BaseModel object.
    """
    self = super()._apply(fn)
    m = self.model[-1]  # Detect()
    if isinstance(
        m, Detect
    ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
        m.stride = fn(m.stride)
        m.anchors = fn(m.anchors)
        m.strides = fn(m.strides)
    return self
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel._predict_augment` {#ultralytics.nn.tasks.BaseModel.\_predict\_augment}

```python
def _predict_augment(self, x)
```

Perform augmentations on input image x and return augmented inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L191-L197"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _predict_augment(self, x):
    """Perform augmentations on input image x and return augmented inference."""
    LOGGER.warning(
        f"{self.__class__.__name__} does not support 'augment=True' prediction. "
        f"Reverting to single-scale prediction."
    )
    return self._predict_once(x)
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel._predict_once` {#ultralytics.nn.tasks.BaseModel.\_predict\_once}

```python
def _predict_once(self, x, profile = False, visualize = False, embed = None)
```

Perform a forward pass through the network.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | The input tensor to the model. | *required* |
| `profile` | `bool` | Print the computation time of each layer if True. | `False` |
| `visualize` | `bool` | Save the feature maps of the model if True. | `False` |
| `embed` | `list, optional` | A list of layer indices to return embeddings from. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | The last output of the model. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L161-L189"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _predict_once(self, x, profile=False, visualize=False, embed=None):
    """Perform a forward pass through the network.

    Args:
        x (torch.Tensor): The input tensor to the model.
        profile (bool): Print the computation time of each layer if True.
        visualize (bool): Save the feature maps of the model if True.
        embed (list, optional): A list of layer indices to return embeddings from.

    Returns:
        (torch.Tensor): The last output of the model.
    """
    y, dt, embeddings = [], [], []  # outputs
    embed = frozenset(embed) if embed is not None else {-1}
    max_idx = max(embed)
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        if m.i in embed:
            embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            if m.i == max_idx:
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel._profile_one_layer` {#ultralytics.nn.tasks.BaseModel.\_profile\_one\_layer}

```python
def _profile_one_layer(self, m, x, dt)
```

Profile the computation time and FLOPs of a single layer of the model on a given input.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `m` | `torch.nn.Module` | The layer to be profiled. | *required* |
| `x` | `torch.Tensor` | The input data to the layer. | *required* |
| `dt` | `list` | A list to store the computation time of the layer. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L199-L222"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _profile_one_layer(self, m, x, dt):
    """Profile the computation time and FLOPs of a single layer of the model on a given input.

    Args:
        m (torch.nn.Module): The layer to be profiled.
        x (torch.Tensor): The input data to the layer.
        dt (list): A list to store the computation time of the layer.
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda support without 'ultralytics-thop' installed

    c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
    flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
    t = time_sync()
    for _ in range(10):
        m(x.copy() if c else x)
    dt.append((time_sync() - t) * 100)
    if m == self.model[0]:
        LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
    LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
    if c:
        LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel.forward` {#ultralytics.nn.tasks.BaseModel.forward}

```python
def forward(self, x, *args, **kwargs)
```

Perform forward pass of the model for either training or inference.

If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor | dict` | Input tensor for inference, or dict with image tensor and labels for training. | *required* |
| `*args` | `Any` | Variable length argument list. | *required* |
| `**kwargs` | `Any` | Arbitrary keyword arguments. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Loss if x is a dict (training), or network predictions (inference). |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L127-L142"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x, *args, **kwargs):
    """Perform forward pass of the model for either training or inference.

    If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

    Args:
        x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
    """
    if isinstance(x, dict):  # for cases of training and validating while training.
        return self.loss(x, *args, **kwargs)
    return self.predict(x, *args, **kwargs)
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel.fuse` {#ultralytics.nn.tasks.BaseModel.fuse}

```python
def fuse(self, verbose = True)
```

Fuse Conv/ConvTranspose and BatchNorm layers, and reparameterize RepConv/RepVGGDW for improved efficiency.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `verbose` | `bool` | Whether to print model information after fusion. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.nn.Module` | The fused model is returned. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L224-L255"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self, verbose=True):
    """Fuse Conv/ConvTranspose and BatchNorm layers, and reparameterize RepConv/RepVGGDW for improved efficiency.

    Args:
        verbose (bool): Whether to print model information after fusion.

    Returns:
        (torch.nn.Module): The fused model is returned.
    """
    if not self.is_fused():
        for m in self.model.modules():
            if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                if isinstance(m, Conv2):
                    m.fuse_convs()
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, RepConv):
                m.fuse_convs()
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, RepVGGDW):
                m.fuse()
                m.forward = m.forward_fuse
            if isinstance(m, Detect) and getattr(m, "end2end", False):
                m.fuse()  # remove one2many head
        self.info(verbose=verbose)

    return self
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel.info` {#ultralytics.nn.tasks.BaseModel.info}

```python
def info(self, detailed = False, verbose = True, imgsz = 640)
```

Print model information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `detailed` | `bool` | If True, prints out detailed information about the model. | `False` |
| `verbose` | `bool` | If True, prints out the model information. | `True` |
| `imgsz` | `int` | The size of the image used for computing model information. | `640` |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L269-L277"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def info(self, detailed=False, verbose=True, imgsz=640):
    """Print model information.

    Args:
        detailed (bool): If True, prints out detailed information about the model.
        verbose (bool): If True, prints out the model information.
        imgsz (int): The size of the image used for computing model information.
    """
    return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel.init_criterion` {#ultralytics.nn.tasks.BaseModel.init\_criterion}

```python
def init_criterion(self)
```

Initialize the loss criterion for the BaseModel.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L337-L339"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_criterion(self):
    """Initialize the loss criterion for the BaseModel."""
    raise NotImplementedError("compute_loss() needs to be implemented by task heads")
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel.is_fused` {#ultralytics.nn.tasks.BaseModel.is\_fused}

```python
def is_fused(self, thresh = 10)
```

Check if the model has less than a certain threshold of normalization layers.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `thresh` | `int, optional` | The threshold number of normalization layers. | `10` |

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if the number of normalization layers in the model is less than the threshold, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L257-L267"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_fused(self, thresh=10):
    """Check if the model has less than a certain threshold of normalization layers.

    Args:
        thresh (int, optional): The threshold number of normalization layers.

    Returns:
        (bool): True if the number of normalization layers in the model is less than the threshold, False otherwise.
    """
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel.load` {#ultralytics.nn.tasks.BaseModel.load}

```python
def load(self, weights, verbose = True)
```

Load weights into the model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weights` | `dict | torch.nn.Module` | The pre-trained weights to be loaded. | *required* |
| `verbose` | `bool, optional` | Whether to log the transfer progress. | `True` |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L298-L321"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def load(self, weights, verbose=True):
    """Load weights into the model.

    Args:
        weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
        verbose (bool, optional): Whether to log the transfer progress.
    """
    model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
    csd = model.float().state_dict()  # checkpoint state_dict as FP32
    updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
    self.load_state_dict(updated_csd, strict=False)  # load
    len_updated_csd = len(updated_csd)
    first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
    # mostly used to boost multi-channel training
    state_dict = self.state_dict()
    if first_conv not in updated_csd and first_conv in state_dict:
        c1, c2, h, w = state_dict[first_conv].shape
        cc1, cc2, ch, cw = csd[first_conv].shape
        if ch == h and cw == w:
            c1, c2 = min(c1, cc1), min(c2, cc2)
            state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
            len_updated_csd += 1
    if verbose:
        LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel.loss` {#ultralytics.nn.tasks.BaseModel.loss}

```python
def loss(self, batch, preds = None)
```

Compute loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict` | Batch to compute loss on. | *required* |
| `preds` | `torch.Tensor | list[torch.Tensor], optional` | Predictions. | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L323-L335"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def loss(self, batch, preds=None):
    """Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
    """
    if getattr(self, "criterion", None) is None:
        self.criterion = self.init_criterion()

    if preds is None:
        preds = self.forward(batch["img"])
    return self.criterion(preds, batch)
```
</details>

<br>

### Method `ultralytics.nn.tasks.BaseModel.predict` {#ultralytics.nn.tasks.BaseModel.predict}

```python
def predict(self, x, profile = False, visualize = False, augment = False, embed = None)
```

Perform a forward pass through the network.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | The input tensor to the model. | *required* |
| `profile` | `bool` | Print the computation time of each layer if True. | `False` |
| `visualize` | `bool` | Save the feature maps of the model if True. | `False` |
| `augment` | `bool` | Augment image during prediction. | `False` |
| `embed` | `list, optional` | A list of layer indices to return embeddings from. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | The last output of the model. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L144-L159"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
    """Perform a forward pass through the network.

    Args:
        x (torch.Tensor): The input tensor to the model.
        profile (bool): Print the computation time of each layer if True.
        visualize (bool): Save the feature maps of the model if True.
        augment (bool): Augment image during prediction.
        embed (list, optional): A list of layer indices to return embeddings from.

    Returns:
        (torch.Tensor): The last output of the model.
    """
    if augment:
        return self._predict_augment(x)
    return self._predict_once(x, profile, visualize, embed)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.DetectionModel` {#ultralytics.nn.tasks.DetectionModel}

```python
DetectionModel(self, cfg = "yolo26n.yaml", ch = 3, nc = None, verbose = True)
```

**Bases:** `BaseModel`

YOLO detection model.

This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented inference, and loss computation for object detection tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | `"yolo26n.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `yaml` | `dict` | Model configuration dictionary. |
| `model` | `torch.nn.Sequential` | The neural network model. |
| `save` | `list` | List of layer indices to save outputs from. |
| `names` | `dict` | Class names dictionary. |
| `inplace` | `bool` | Whether to use inplace operations. |
| `end2end` | `bool` | Whether the model uses end-to-end detection. |
| `stride` | `torch.Tensor` | Model stride values. |

**Methods**

| Name | Description |
| --- | --- |
| [`end2end`](#ultralytics.nn.tasks.DetectionModel.end2end) | Return whether the model uses end-to-end NMS-free detection. |
| [`_clip_augmented`](#ultralytics.nn.tasks.DetectionModel._clip_augmented) | Clip YOLO augmented inference tails. |
| [`_descale_pred`](#ultralytics.nn.tasks.DetectionModel._descale_pred) | De-scale predictions following augmented inference (inverse operation). |
| [`_predict_augment`](#ultralytics.nn.tasks.DetectionModel._predict_augment) | Perform augmentations on input image x and return augmented inference and train outputs. |
| [`end2end`](#ultralytics.nn.tasks.DetectionModel.end2end) | Override the end-to-end detection mode. |
| [`init_criterion`](#ultralytics.nn.tasks.DetectionModel.init_criterion) | Initialize the loss criterion for the DetectionModel. |
| [`set_head_attr`](#ultralytics.nn.tasks.DetectionModel.set_head_attr) | Set attributes of the model head (last layer). |

**Examples**

```python
Initialize a detection model
>>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
>>> results = model.predict(image_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L342-L514"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DetectionModel(BaseModel):
    """YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented
    inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                output = self.forward(x)
                if self.end2end:
                    output = output["one2many"]
                return output["feats"]

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride, e.g., RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")
```
</details>

<br>

### Property `ultralytics.nn.tasks.DetectionModel.end2end` {#ultralytics.nn.tasks.DetectionModel.end2end}

```python
def end2end(self)
```

Return whether the model uses end-to-end NMS-free detection.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L426-L428"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def end2end(self):
    """Return whether the model uses end-to-end NMS-free detection."""
    return getattr(self.model[-1], "end2end", False)
```
</details>

<br>

### Method `ultralytics.nn.tasks.DetectionModel._clip_augmented` {#ultralytics.nn.tasks.DetectionModel.\_clip\_augmented}

```python
def _clip_augmented(self, y)
```

Clip YOLO augmented inference tails.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `y` | `list[torch.Tensor]` | List of detection tensors. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[torch.Tensor]` | Clipped detection tensors. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L494-L510"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _clip_augmented(self, y):
    """Clip YOLO augmented inference tails.

    Args:
        y (list[torch.Tensor]): List of detection tensors.

    Returns:
        (list[torch.Tensor]): Clipped detection tensors.
    """
    nl = self.model[-1].nl  # number of detection layers (P3-P5)
    g = sum(4**x for x in range(nl))  # grid points
    e = 1  # exclude layer count
    i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
    y[0] = y[0][..., :-i]  # large
    i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
    y[-1] = y[-1][..., i:]  # small
    return y
```
</details>

<br>

### Method `ultralytics.nn.tasks.DetectionModel._descale_pred` {#ultralytics.nn.tasks.DetectionModel.\_descale\_pred}

```python
def _descale_pred(p, flips, scale, img_size, dim = 1)
```

De-scale predictions following augmented inference (inverse operation).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `p` | `torch.Tensor` | Predictions tensor. | *required* |
| `flips` | `int | None` | Flip type (None=none, 2=ud, 3=lr). | *required* |
| `scale` | `float` | Scale factor. | *required* |
| `img_size` | `tuple` | Original image size (height, width). | *required* |
| `dim` | `int` | Dimension to split at. | `1` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | De-scaled predictions. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L473-L492"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _descale_pred(p, flips, scale, img_size, dim=1):
    """De-scale predictions following augmented inference (inverse operation).

    Args:
        p (torch.Tensor): Predictions tensor.
        flips (int | None): Flip type (None=none, 2=ud, 3=lr).
        scale (float): Scale factor.
        img_size (tuple): Original image size (height, width).
        dim (int): Dimension to split at.

    Returns:
        (torch.Tensor): De-scaled predictions.
    """
    p[:, :4] /= scale  # de-scale
    x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
    if flips == 2:
        y = img_size[0] - y  # de-flip ud
    elif flips == 3:
        x = img_size[1] - x  # de-flip lr
    return torch.cat((x, y, wh, cls), dim)
```
</details>

<br>

### Method `ultralytics.nn.tasks.DetectionModel._predict_augment` {#ultralytics.nn.tasks.DetectionModel.\_predict\_augment}

```python
def _predict_augment(self, x)
```

Perform augmentations on input image x and return augmented inference and train outputs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input image tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `tuple[torch.Tensor, None]` | Augmented inference output and None for train output. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L448-L470"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _predict_augment(self, x):
    """Perform augmentations on input image x and return augmented inference and train outputs.

    Args:
        x (torch.Tensor): Input image tensor.

    Returns:
        (tuple[torch.Tensor, None]): Augmented inference output and None for train output.
    """
    if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
        LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
        return self._predict_once(x)
    img_size = x.shape[-2:]  # height, width
    s = [1, 0.83, 0.67]  # scales
    f = [None, 3, None]  # flips (2-ud, 3-lr)
    y = []  # outputs
    for si, fi in zip(s, f):
        xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
        yi = super().predict(xi)[0]  # forward
        yi = self._descale_pred(yi, fi, si, img_size)
        y.append(yi)
    y = self._clip_augmented(y)  # clip augmented tails
    return torch.cat(y, -1), None  # augmented inference, train
```
</details>

<br>

### Method `ultralytics.nn.tasks.DetectionModel.end2end` {#ultralytics.nn.tasks.DetectionModel.end2end}

```python
def end2end(self, value)
```

Override the end-to-end detection mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `value` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L431-L433"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@end2end.setter
def end2end(self, value):
    """Override the end-to-end detection mode."""
    self.set_head_attr(end2end=value)
```
</details>

<br>

### Method `ultralytics.nn.tasks.DetectionModel.init_criterion` {#ultralytics.nn.tasks.DetectionModel.init\_criterion}

```python
def init_criterion(self)
```

Initialize the loss criterion for the DetectionModel.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L512-L514"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_criterion(self):
    """Initialize the loss criterion for the DetectionModel."""
    return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
```
</details>

<br>

### Method `ultralytics.nn.tasks.DetectionModel.set_head_attr` {#ultralytics.nn.tasks.DetectionModel.set\_head\_attr}

```python
def set_head_attr(self, **kwargs)
```

Set attributes of the model head (last layer).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `**kwargs` |  | Arbitrary keyword arguments representing attributes to set. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L435-L446"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_head_attr(self, **kwargs):
    """Set attributes of the model head (last layer).

    Args:
        **kwargs: Arbitrary keyword arguments representing attributes to set.
    """
    head = self.model[-1]
    for k, v in kwargs.items():
        if not hasattr(head, k):
            LOGGER.warning(f"Head has no attribute '{k}'.")
            continue
        setattr(head, k, v)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.OBBModel` {#ultralytics.nn.tasks.OBBModel}

```python
OBBModel(self, cfg = "yolo26n-obb.yaml", ch = 3, nc = None, verbose = True)
```

**Bases:** `DetectionModel`

YOLO Oriented Bounding Box (OBB) model.

This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss computation for rotated object detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | `"yolo26n-obb.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Methods**

| Name | Description |
| --- | --- |
| [`init_criterion`](#ultralytics.nn.tasks.OBBModel.init_criterion) | Initialize the loss criterion for the model. |

**Examples**

```python
Initialize an OBB model
>>> model = OBBModel("yolo26n-obb.yaml", ch=3, nc=80)
>>> results = model.predict(image_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L517-L546"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss
    computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo26n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
```
</details>

<br>

### Method `ultralytics.nn.tasks.OBBModel.init_criterion` {#ultralytics.nn.tasks.OBBModel.init\_criterion}

```python
def init_criterion(self)
```

Initialize the loss criterion for the model.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L544-L546"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_criterion(self):
    """Initialize the loss criterion for the model."""
    return E2ELoss(self, v8OBBLoss) if getattr(self, "end2end", False) else v8OBBLoss(self)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.SegmentationModel` {#ultralytics.nn.tasks.SegmentationModel}

```python
SegmentationModel(self, cfg = "yolo26n-seg.yaml", ch = 3, nc = None, verbose = True)
```

**Bases:** `DetectionModel`

YOLO segmentation model.

This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for pixel-level object detection and segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | `"yolo26n-seg.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Methods**

| Name | Description |
| --- | --- |
| [`init_criterion`](#ultralytics.nn.tasks.SegmentationModel.init_criterion) | Initialize the loss criterion for the SegmentationModel. |

**Examples**

```python
Initialize a segmentation model
>>> model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80)
>>> results = model.predict(image_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L549-L578"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SegmentationModel(DetectionModel):
    """YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for
    pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
```
</details>

<br>

### Method `ultralytics.nn.tasks.SegmentationModel.init_criterion` {#ultralytics.nn.tasks.SegmentationModel.init\_criterion}

```python
def init_criterion(self)
```

Initialize the loss criterion for the SegmentationModel.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L576-L578"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_criterion(self):
    """Initialize the loss criterion for the SegmentationModel."""
    return E2ELoss(self, v8SegmentationLoss) if getattr(self, "end2end", False) else v8SegmentationLoss(self)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.PoseModel` {#ultralytics.nn.tasks.PoseModel}

```python
PoseModel(self, cfg = "yolo26n-pose.yaml", ch = 3, nc = None, data_kpt_shape = (None, None), verbose = True)
```

**Bases:** `DetectionModel`

YOLO pose model.

This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for keypoint detection and pose estimation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | `"yolo26n-pose.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `data_kpt_shape` | `tuple` | Shape of keypoints data. | `(None, None)` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `kpt_shape` | `tuple` | Shape of keypoints data (num_keypoints, num_dimensions). |

**Methods**

| Name | Description |
| --- | --- |
| [`init_criterion`](#ultralytics.nn.tasks.PoseModel.init_criterion) | Initialize the loss criterion for the PoseModel. |

**Examples**

```python
Initialize a pose model
>>> model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
>>> results = model.predict(image_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L581-L619"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PoseModel(DetectionModel):
    """YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for
    keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
```
</details>

<br>

### Method `ultralytics.nn.tasks.PoseModel.init_criterion` {#ultralytics.nn.tasks.PoseModel.init\_criterion}

```python
def init_criterion(self)
```

Initialize the loss criterion for the PoseModel.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L617-L619"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_criterion(self):
    """Initialize the loss criterion for the PoseModel."""
    return E2ELoss(self, PoseLoss26) if getattr(self, "end2end", False) else v8PoseLoss(self)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.ClassificationModel` {#ultralytics.nn.tasks.ClassificationModel}

```python
ClassificationModel(self, cfg = "yolo26n-cls.yaml", ch = 3, nc = None, verbose = True)
```

**Bases:** `BaseModel`

YOLO classification model.

This class implements the YOLO classification architecture for image classification tasks, providing model initialization, configuration, and output reshaping capabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | `"yolo26n-cls.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `yaml` | `dict` | Model configuration dictionary. |
| `model` | `torch.nn.Sequential` | The neural network model. |
| `stride` | `torch.Tensor` | Model stride values. |
| `names` | `dict` | Class names dictionary. |

**Methods**

| Name | Description |
| --- | --- |
| [`_from_yaml`](#ultralytics.nn.tasks.ClassificationModel._from_yaml) | Set Ultralytics YOLO model configurations and define the model architecture. |
| [`init_criterion`](#ultralytics.nn.tasks.ClassificationModel.init_criterion) | Initialize the loss criterion for the ClassificationModel. |
| [`reshape_outputs`](#ultralytics.nn.tasks.ClassificationModel.reshape_outputs) | Update a TorchVision classification model to class count 'nc' if required. |

**Examples**

```python
Initialize a classification model
>>> model = ClassificationModel("yolo26n-cls.yaml", ch=3, nc=1000)
>>> results = model.predict(image_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L622-L711"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ClassificationModel(BaseModel):
    """YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks, providing model
    initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo26n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-cls.yaml", ch=3, nc=None, verbose=True):
        """Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)
```
</details>

<br>

### Method `ultralytics.nn.tasks.ClassificationModel._from_yaml` {#ultralytics.nn.tasks.ClassificationModel.\_from\_yaml}

```python
def _from_yaml(self, cfg, ch, nc, verbose)
```

Set Ultralytics YOLO model configurations and define the model architecture.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | *required* |
| `ch` | `int` | Number of input channels. | *required* |
| `nc` | `int, optional` | Number of classes. | *required* |
| `verbose` | `bool` | Whether to display model information. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L658-L679"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _from_yaml(self, cfg, ch, nc, verbose):
    """Set Ultralytics YOLO model configurations and define the model architecture.

    Args:
        cfg (str | dict): Model configuration file path or dictionary.
        ch (int): Number of input channels.
        nc (int, optional): Number of classes.
        verbose (bool): Whether to display model information.
    """
    self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

    # Define model
    ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
    if nc and nc != self.yaml["nc"]:
        LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        self.yaml["nc"] = nc  # override YAML value
    elif not nc and not self.yaml.get("nc", None):
        raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
    self.stride = torch.Tensor([1])  # no stride constraints
    self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
    self.info()
```
</details>

<br>

### Method `ultralytics.nn.tasks.ClassificationModel.init_criterion` {#ultralytics.nn.tasks.ClassificationModel.init\_criterion}

```python
def init_criterion(self)
```

Initialize the loss criterion for the ClassificationModel.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L709-L711"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_criterion(self):
    """Initialize the loss criterion for the ClassificationModel."""
    return v8ClassificationLoss()
```
</details>

<br>

### Method `ultralytics.nn.tasks.ClassificationModel.reshape_outputs` {#ultralytics.nn.tasks.ClassificationModel.reshape\_outputs}

```python
def reshape_outputs(model, nc)
```

Update a TorchVision classification model to class count 'nc' if required.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | Model to update. | *required* |
| `nc` | `int` | New number of classes. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L682-L707"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def reshape_outputs(model, nc):
    """Update a TorchVision classification model to class count 'nc' if required.

    Args:
        model (torch.nn.Module): Model to update.
        nc (int): New number of classes.
    """
    name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
    if isinstance(m, Classify):  # YOLO Classify() head
        if m.linear.out_features != nc:
            m.linear = torch.nn.Linear(m.linear.in_features, nc)
    elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
        if m.out_features != nc:
            setattr(model, name, torch.nn.Linear(m.in_features, nc))
    elif isinstance(m, torch.nn.Sequential):
        types = [type(x) for x in m]
        if torch.nn.Linear in types:
            i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
            if m[i].out_features != nc:
                m[i] = torch.nn.Linear(m[i].in_features, nc)
        elif torch.nn.Conv2d in types:
            i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
            if m[i].out_channels != nc:
                m[i] = torch.nn.Conv2d(
                    m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                )
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.RTDETRDetectionModel` {#ultralytics.nn.tasks.RTDETRDetectionModel}

```python
RTDETRDetectionModel(self, cfg = "rtdetr-l.yaml", ch = 3, nc = None, verbose = True)
```

**Bases:** `DetectionModel`

RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both the training and inference processes. RTDETR is an object detection and tracking model that extends from the DetectionModel base class.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Configuration file name or path. | `"rtdetr-l.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `verbose` | `bool` | Print additional information during initialization. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `nc` | `int` | Number of classes for detection. |
| `criterion` | `RTDETRDetectionLoss` | Loss function for training. |

**Methods**

| Name | Description |
| --- | --- |
| [`_apply`](#ultralytics.nn.tasks.RTDETRDetectionModel._apply) | Apply a function to all tensors in the model, including decoder anchors and valid mask. |
| [`init_criterion`](#ultralytics.nn.tasks.RTDETRDetectionModel.init_criterion) | Initialize the loss criterion for the RTDETRDetectionModel. |
| [`loss`](#ultralytics.nn.tasks.RTDETRDetectionModel.loss) | Compute the loss for the given batch of data. |
| [`predict`](#ultralytics.nn.tasks.RTDETRDetectionModel.predict) | Perform a forward pass through the model. |

**Examples**

```python
Initialize an RTDETR model
>>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
>>> results = model.predict(image_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L714-L847"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RTDETRDetectionModel(DetectionModel):
    """RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
```
</details>

<br>

### Method `ultralytics.nn.tasks.RTDETRDetectionModel._apply` {#ultralytics.nn.tasks.RTDETRDetectionModel.\_apply}

```python
def _apply(self, fn)
```

Apply a function to all tensors in the model, including decoder anchors and valid mask.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `fn` | `function` | The function to apply to the model. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `RTDETRDetectionModel` | An updated RTDETRDetectionModel object. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L748-L761"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _apply(self, fn):
    """Apply a function to all tensors in the model, including decoder anchors and valid mask.

    Args:
        fn (function): The function to apply to the model.

    Returns:
        (RTDETRDetectionModel): An updated RTDETRDetectionModel object.
    """
    self = super()._apply(fn)
    m = self.model[-1]
    m.anchors = fn(m.anchors)
    m.valid_mask = fn(m.valid_mask)
    return self
```
</details>

<br>

### Method `ultralytics.nn.tasks.RTDETRDetectionModel.init_criterion` {#ultralytics.nn.tasks.RTDETRDetectionModel.init\_criterion}

```python
def init_criterion(self)
```

Initialize the loss criterion for the RTDETRDetectionModel.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L763-L767"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_criterion(self):
    """Initialize the loss criterion for the RTDETRDetectionModel."""
    from ultralytics.models.utils.loss import RTDETRDetectionLoss

    return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)
```
</details>

<br>

### Method `ultralytics.nn.tasks.RTDETRDetectionModel.loss` {#ultralytics.nn.tasks.RTDETRDetectionModel.loss}

```python
def loss(self, batch, preds = None)
```

Compute the loss for the given batch of data.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict` | Dictionary containing image and label data. | *required* |
| `preds` | `tuple, optional` | Precomputed model predictions. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Total loss value. |
| `torch.Tensor` | Main three losses in a tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L769-L813"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def loss(self, batch, preds=None):
    """Compute the loss for the given batch of data.

    Args:
        batch (dict): Dictionary containing image and label data.
        preds (tuple, optional): Precomputed model predictions.

    Returns:
        (torch.Tensor): Total loss value.
        (torch.Tensor): Main three losses in a tensor.
    """
    if not hasattr(self, "criterion"):
        self.criterion = self.init_criterion()

    img = batch["img"]
    # NOTE: preprocess gt_bbox and gt_labels to list.
    bs = img.shape[0]
    batch_idx = batch["batch_idx"]
    gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
    targets = {
        "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
        "bboxes": batch["bboxes"].to(device=img.device),
        "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
        "gt_groups": gt_groups,
    }

    if preds is None:
        preds = self.predict(img, batch=targets)
    dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
    if dn_meta is None:
        dn_bboxes, dn_scores = None, None
    else:
        dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
        dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

    dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
    dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

    loss = self.criterion(
        (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
    )
    # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
    return sum(loss.values()), torch.as_tensor(
        [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
    )
```
</details>

<br>

### Method `ultralytics.nn.tasks.RTDETRDetectionModel.predict` {#ultralytics.nn.tasks.RTDETRDetectionModel.predict}

```python
def predict(self, x, profile = False, visualize = False, batch = None, augment = False, embed = None)
```

Perform a forward pass through the model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | The input tensor. | *required* |
| `profile` | `bool` | If True, profile the computation time for each layer. | `False` |
| `visualize` | `bool` | If True, save feature maps for visualization. | `False` |
| `batch` | `dict, optional` | Ground truth data for evaluation. | `None` |
| `augment` | `bool` | If True, perform data augmentation during inference. | `False` |
| `embed` | `list, optional` | A list of layer indices to return embeddings from. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Model's output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L815-L847"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
    """Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool): If True, profile the computation time for each layer.
        visualize (bool): If True, save feature maps for visualization.
        batch (dict, optional): Ground truth data for evaluation.
        augment (bool): If True, perform data augmentation during inference.
        embed (list, optional): A list of layer indices to return embeddings from.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    y, dt, embeddings = [], [], []  # outputs
    embed = frozenset(embed) if embed is not None else {-1}
    max_idx = max(embed)
    for m in self.model[:-1]:  # except the head part
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        if m.i in embed:
            embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            if m.i == max_idx:
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
    head = self.model[-1]
    x = head([y[j] for j in head.f], batch)  # head inference
    return x
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.WorldModel` {#ultralytics.nn.tasks.WorldModel}

```python
WorldModel(self, cfg = "yolov8s-world.yaml", ch = 3, nc = None, verbose = True)
```

**Bases:** `DetectionModel`

YOLOv8 World Model.

This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class specification and CLIP model integration for zero-shot detection capabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | `"yolov8s-world.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `txt_feats` | `torch.Tensor` | Text feature embeddings for classes. |
| `clip_model` | `torch.nn.Module` | CLIP model for text encoding. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_text_pe`](#ultralytics.nn.tasks.WorldModel.get_text_pe) | Get text positional embeddings using the CLIP model. |
| [`loss`](#ultralytics.nn.tasks.WorldModel.loss) | Compute loss. |
| [`predict`](#ultralytics.nn.tasks.WorldModel.predict) | Perform a forward pass through the model. |
| [`set_classes`](#ultralytics.nn.tasks.WorldModel.set_classes) | Set classes in advance so that model could do offline-inference without clip model. |

**Examples**

```python
Initialize a world model
>>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
>>> model.set_classes(["person", "car", "bicycle"])
>>> results = model.predict(image_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L850-L977"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class WorldModel(DetectionModel):
    """YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class
    specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
```
</details>

<br>

### Method `ultralytics.nn.tasks.WorldModel.get_text_pe` {#ultralytics.nn.tasks.WorldModel.get\_text\_pe}

```python
def get_text_pe(self, text, batch = 80, cache_clip_model = True)
```

Get text positional embeddings using the CLIP model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` | `list[str]` | List of class names. | *required* |
| `batch` | `int` | Batch size for processing text tokens. | `80` |
| `cache_clip_model` | `bool` | Whether to cache the CLIP model. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Text positional embeddings. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L898-L919"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_text_pe(self, text, batch=80, cache_clip_model=True):
    """Get text positional embeddings using the CLIP model.

    Args:
        text (list[str]): List of class names.
        batch (int): Batch size for processing text tokens.
        cache_clip_model (bool): Whether to cache the CLIP model.

    Returns:
        (torch.Tensor): Text positional embeddings.
    """
    from ultralytics.nn.text_model import build_text_model

    device = next(self.model.parameters()).device
    if not getattr(self, "clip_model", None) and cache_clip_model:
        # For backwards compatibility of models lacking clip_model attribute
        self.clip_model = build_text_model("clip:ViT-B/32", device=device)
    model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
    text_token = model.tokenize(text)
    txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
    txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
    return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
```
</details>

<br>

### Method `ultralytics.nn.tasks.WorldModel.loss` {#ultralytics.nn.tasks.WorldModel.loss}

```python
def loss(self, batch, preds = None)
```

Compute loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict` | Batch to compute loss on. | *required* |
| `preds` | `torch.Tensor | list[torch.Tensor], optional` | Predictions. | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L965-L977"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def loss(self, batch, preds=None):
    """Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
    """
    if not hasattr(self, "criterion"):
        self.criterion = self.init_criterion()

    if preds is None:
        preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
    return self.criterion(preds, batch)
```
</details>

<br>

### Method `ultralytics.nn.tasks.WorldModel.predict` {#ultralytics.nn.tasks.WorldModel.predict}

```python
def predict(self, x, profile = False, visualize = False, txt_feats = None, augment = False, embed = None)
```

Perform a forward pass through the model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | The input tensor. | *required* |
| `profile` | `bool` | If True, profile the computation time for each layer. | `False` |
| `visualize` | `bool` | If True, save feature maps for visualization. | `False` |
| `txt_feats` | `torch.Tensor, optional` | The text features, use it if it's given. | `None` |
| `augment` | `bool` | If True, perform data augmentation during inference. | `False` |
| `embed` | `list, optional` | A list of layer indices to return embeddings from. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Model's output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L921-L963"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
    """Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool): If True, profile the computation time for each layer.
        visualize (bool): If True, save feature maps for visualization.
        txt_feats (torch.Tensor, optional): The text features, use it if it's given.
        augment (bool): If True, perform data augmentation during inference.
        embed (list, optional): A list of layer indices to return embeddings from.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
    if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
        txt_feats = txt_feats.expand(x.shape[0], -1, -1)
    ori_txt_feats = txt_feats.clone()
    y, dt, embeddings = [], [], []  # outputs
    embed = frozenset(embed) if embed is not None else {-1}
    max_idx = max(embed)
    for m in self.model:  # except the head part
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        if isinstance(m, C2fAttn):
            x = m(x, txt_feats)
        elif isinstance(m, WorldDetect):
            x = m(x, ori_txt_feats)
        elif isinstance(m, ImagePoolingAttn):
            txt_feats = m(x, txt_feats)
        else:
            x = m(x)  # run

        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        if m.i in embed:
            embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            if m.i == max_idx:
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x
```
</details>

<br>

### Method `ultralytics.nn.tasks.WorldModel.set_classes` {#ultralytics.nn.tasks.WorldModel.set\_classes}

```python
def set_classes(self, text, batch = 80, cache_clip_model = True)
```

Set classes in advance so that model could do offline-inference without clip model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` | `list[str]` | List of class names. | *required* |
| `batch` | `int` | Batch size for processing text tokens. | `80` |
| `cache_clip_model` | `bool` | Whether to cache the CLIP model. | `True` |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L887-L896"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_classes(self, text, batch=80, cache_clip_model=True):
    """Set classes in advance so that model could do offline-inference without clip model.

    Args:
        text (list[str]): List of class names.
        batch (int): Batch size for processing text tokens.
        cache_clip_model (bool): Whether to cache the CLIP model.
    """
    self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
    self.model[-1].nc = len(text)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.YOLOEModel` {#ultralytics.nn.tasks.YOLOEModel}

```python
YOLOEModel(self, cfg = "yoloe-v8s.yaml", ch = 3, nc = None, verbose = True)
```

**Bases:** `DetectionModel`

YOLOE detection model.

This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting both prompt-based and prompt-free inference modes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | `"yoloe-v8s.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `pe` | `torch.Tensor` | Prompt embeddings for classes. |
| `clip_model` | `torch.nn.Module` | CLIP model for text encoding. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_cls_pe`](#ultralytics.nn.tasks.YOLOEModel.get_cls_pe) | Get class positional embeddings. |
| [`get_text_pe`](#ultralytics.nn.tasks.YOLOEModel.get_text_pe) | Get text positional embeddings using the CLIP model. |
| [`get_visual_pe`](#ultralytics.nn.tasks.YOLOEModel.get_visual_pe) | Get visual positional embeddings. |
| [`get_vocab`](#ultralytics.nn.tasks.YOLOEModel.get_vocab) | Get fused vocabulary layer from the model. |
| [`loss`](#ultralytics.nn.tasks.YOLOEModel.loss) | Compute loss. |
| [`predict`](#ultralytics.nn.tasks.YOLOEModel.predict) | Perform a forward pass through the model. |
| [`set_classes`](#ultralytics.nn.tasks.YOLOEModel.set_classes) | Set classes in advance so that model could do offline-inference without clip model. |
| [`set_vocab`](#ultralytics.nn.tasks.YOLOEModel.set_vocab) | Set vocabulary for the prompt-free model. |

**Examples**

```python
Initialize a YOLOE model
>>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
>>> results = model.predict(image_tensor, tpe=text_embeddings)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L980-L1230"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOEModel(DetectionModel):
    """YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting
    both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.text_model = self.yaml.get("text_model", "mobileclip:blt")
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOEModel.get_cls_pe` {#ultralytics.nn.tasks.YOLOEModel.get\_cls\_pe}

```python
def get_cls_pe(self, tpe, vpe)
```

Get class positional embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tpe` | `torch.Tensor | None` | Text positional embeddings. | *required* |
| `vpe` | `torch.Tensor | None` | Visual positional embeddings. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Class positional embeddings. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1139-L1158"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_cls_pe(self, tpe, vpe):
    """Get class positional embeddings.

    Args:
        tpe (torch.Tensor | None): Text positional embeddings.
        vpe (torch.Tensor | None): Visual positional embeddings.

    Returns:
        (torch.Tensor): Class positional embeddings.
    """
    all_pe = []
    if tpe is not None:
        assert tpe.ndim == 3
        all_pe.append(tpe)
    if vpe is not None:
        assert vpe.ndim == 3
        all_pe.append(vpe)
    if not all_pe:
        all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
    return torch.cat(all_pe, dim=1)
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOEModel.get_text_pe` {#ultralytics.nn.tasks.YOLOEModel.get\_text\_pe}

```python
def get_text_pe(self, text, batch = 80, cache_clip_model = False, without_reprta = False)
```

Get text positional embeddings using the CLIP model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` | `list[str]` | List of class names. | *required* |
| `batch` | `int` | Batch size for processing text tokens. | `80` |
| `cache_clip_model` | `bool` | Whether to cache the CLIP model. | `False` |
| `without_reprta` | `bool` | Whether to return text embeddings without reprta module processing. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Text positional embeddings. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1020-L1053"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
    """Get text positional embeddings using the CLIP model.

    Args:
        text (list[str]): List of class names.
        batch (int): Batch size for processing text tokens.
        cache_clip_model (bool): Whether to cache the CLIP model.
        without_reprta (bool): Whether to return text embeddings without reprta module processing.

    Returns:
        (torch.Tensor): Text positional embeddings.
    """
    from ultralytics.nn.text_model import build_text_model

    device = next(self.model.parameters()).device
    if not getattr(self, "clip_model", None) and cache_clip_model:
        # For backwards compatibility of models lacking clip_model attribute
        self.clip_model = build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)

    model = (
        self.clip_model
        if cache_clip_model
        else build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)
    )
    text_token = model.tokenize(text)
    txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
    txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
    txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
    if without_reprta:
        return txt_feats

    head = self.model[-1]
    assert isinstance(head, YOLOEDetect)
    return head.get_tpe(txt_feats)  # run auxiliary text head
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOEModel.get_visual_pe` {#ultralytics.nn.tasks.YOLOEModel.get\_visual\_pe}

```python
def get_visual_pe(self, img, visual)
```

Get visual positional embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img` | `torch.Tensor` | Input image tensor. | *required* |
| `visual` | `torch.Tensor` | Visual features. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Visual positional embeddings. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1056-L1066"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def get_visual_pe(self, img, visual):
    """Get visual positional embeddings.

    Args:
        img (torch.Tensor): Input image tensor.
        visual (torch.Tensor): Visual features.

    Returns:
        (torch.Tensor): Visual positional embeddings.
    """
    return self(img, vpe=visual, return_vpe=True)
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOEModel.get_vocab` {#ultralytics.nn.tasks.YOLOEModel.get\_vocab}

```python
def get_vocab(self, names)
```

Get fused vocabulary layer from the model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` | `list[str]` | List of class names. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `nn.ModuleList` | List of vocabulary modules. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1098-L1122"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_vocab(self, names):
    """Get fused vocabulary layer from the model.

    Args:
        names (list[str]): List of class names.

    Returns:
        (nn.ModuleList): List of vocabulary modules.
    """
    assert not self.training
    head = self.model[-1]
    assert isinstance(head, YOLOEDetect)
    assert not head.is_fused

    tpe = self.get_text_pe(names)
    self.set_classes(names, tpe)
    device = next(self.model.parameters()).device
    head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

    cv3 = getattr(head, "one2one_cv3", head.cv3)
    vocab = nn.ModuleList()
    for cls_head in cv3:
        assert isinstance(cls_head, nn.Sequential)
        vocab.append(cls_head[-1])
    return vocab
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOEModel.loss` {#ultralytics.nn.tasks.YOLOEModel.loss}

```python
def loss(self, batch, preds = None)
```

Compute loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict` | Batch to compute loss on. | *required* |
| `preds` | `torch.Tensor | list[torch.Tensor], optional` | Predictions. | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1208-L1230"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def loss(self, batch, preds=None):
    """Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
    """
    if not hasattr(self, "criterion"):
        from ultralytics.utils.loss import TVPDetectLoss

        visual_prompt = batch.get("visuals", None) is not None  # TODO
        self.criterion = (
            (E2ELoss(self, TVPDetectLoss) if getattr(self, "end2end", False) else TVPDetectLoss(self))
            if visual_prompt
            else self.init_criterion()
        )
    if preds is None:
        preds = self.forward(
            batch["img"],
            tpe=None if "visuals" in batch else batch.get("txt_feats", None),
            vpe=batch.get("visuals", None),
        )
    return self.criterion(preds, batch)
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOEModel.predict` {#ultralytics.nn.tasks.YOLOEModel.predict}

```python
def predict(
    self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
)
```

Perform a forward pass through the model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | The input tensor. | *required* |
| `profile` | `bool` | If True, profile the computation time for each layer. | `False` |
| `visualize` | `bool` | If True, save feature maps for visualization. | `False` |
| `tpe` | `torch.Tensor, optional` | Text positional embeddings. | `None` |
| `augment` | `bool` | If True, perform data augmentation during inference. | `False` |
| `embed` | `list, optional` | A list of layer indices to return embeddings from. | `None` |
| `vpe` | `torch.Tensor, optional` | Visual positional embeddings. | `None` |
| `return_vpe` | `bool` | If True, return visual positional embeddings. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Model's output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1160-L1206"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(
    self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
):
    """Perform a forward pass through the model.

    Args:
        x (torch.Tensor): The input tensor.
        profile (bool): If True, profile the computation time for each layer.
        visualize (bool): If True, save feature maps for visualization.
        tpe (torch.Tensor, optional): Text positional embeddings.
        augment (bool): If True, perform data augmentation during inference.
        embed (list, optional): A list of layer indices to return embeddings from.
        vpe (torch.Tensor, optional): Visual positional embeddings.
        return_vpe (bool): If True, return visual positional embeddings.

    Returns:
        (torch.Tensor): Model's output tensor.
    """
    y, dt, embeddings = [], [], []  # outputs
    b = x.shape[0]
    embed = frozenset(embed) if embed is not None else {-1}
    max_idx = max(embed)
    for m in self.model:  # except the head part
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        if isinstance(m, YOLOEDetect):
            vpe = m.get_vpe(x, vpe) if vpe is not None else None
            if return_vpe:
                assert vpe is not None
                assert not self.training
                return vpe
            cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
            if cls_pe.shape[0] != b or m.export:
                cls_pe = cls_pe.expand(b, -1, -1)
            x.append(cls_pe)  # adding cls embedding
        x = m(x)  # run

        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        if m.i in embed:
            embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            if m.i == max_idx:
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOEModel.set_classes` {#ultralytics.nn.tasks.YOLOEModel.set\_classes}

```python
def set_classes(self, names, embeddings)
```

Set classes in advance so that model could do offline-inference without clip model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` | `list[str]` | List of class names. | *required* |
| `embeddings` | `torch.Tensor` | Embeddings tensor. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1124-L1137"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_classes(self, names, embeddings):
    """Set classes in advance so that model could do offline-inference without clip model.

    Args:
        names (list[str]): List of class names.
        embeddings (torch.Tensor): Embeddings tensor.
    """
    assert not hasattr(self.model[-1], "lrpc"), (
        "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
    )
    assert embeddings.ndim == 3
    self.pe = embeddings
    self.model[-1].nc = len(names)
    self.names = check_class_names(names)
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOEModel.set_vocab` {#ultralytics.nn.tasks.YOLOEModel.set\_vocab}

```python
def set_vocab(self, vocab, names)
```

Set vocabulary for the prompt-free model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `vocab` | `nn.ModuleList` | List of vocabulary items. | *required* |
| `names` | `list[str]` | List of class names. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1068-L1096"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_vocab(self, vocab, names):
    """Set vocabulary for the prompt-free model.

    Args:
        vocab (nn.ModuleList): List of vocabulary items.
        names (list[str]): List of class names.
    """
    assert not self.training
    head = self.model[-1]
    assert isinstance(head, YOLOEDetect)

    # Cache anchors for head
    device = next(self.parameters()).device
    self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

    cv3 = getattr(head, "one2one_cv3", head.cv3)
    cv2 = getattr(head, "one2one_cv2", head.cv2)

    # re-parameterization for prompt-free model
    self.model[-1].lrpc = nn.ModuleList(
        LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2) for i, (cls, pf, loc) in enumerate(zip(vocab, cv3, cv2))
    )
    for loc_head, cls_head in zip(head.cv2, head.cv3):
        assert isinstance(loc_head, nn.Sequential)
        assert isinstance(cls_head, nn.Sequential)
        del loc_head[-1]
        del cls_head[-1]
    self.model[-1].nc = len(names)
    self.names = check_class_names(names)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.YOLOESegModel` {#ultralytics.nn.tasks.YOLOESegModel}

```python
YOLOESegModel(self, cfg = "yoloe-v8s-seg.yaml", ch = 3, nc = None, verbose = True)
```

**Bases:** `YOLOEModel`, `SegmentationModel`

YOLOE segmentation model.

This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing specialized loss computation for pixel-level object detection and segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict` | Model configuration file path or dictionary. | `"yoloe-v8s-seg.yaml"` |
| `ch` | `int` | Number of input channels. | `3` |
| `nc` | `int, optional` | Number of classes. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Methods**

| Name | Description |
| --- | --- |
| [`loss`](#ultralytics.nn.tasks.YOLOESegModel.loss) | Compute loss. |

**Examples**

```python
Initialize a YOLOE segmentation model
>>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
>>> results = model.predict(image_tensor, tpe=text_embeddings)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1233-L1279"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing
    specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
```
</details>

<br>

### Method `ultralytics.nn.tasks.YOLOESegModel.loss` {#ultralytics.nn.tasks.YOLOESegModel.loss}

```python
def loss(self, batch, preds = None)
```

Compute loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict` | Batch to compute loss on. | *required* |
| `preds` | `torch.Tensor | list[torch.Tensor], optional` | Predictions. | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1260-L1279"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def loss(self, batch, preds=None):
    """Compute loss.

    Args:
        batch (dict): Batch to compute loss on.
        preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
    """
    if not hasattr(self, "criterion"):
        from ultralytics.utils.loss import TVPSegmentLoss

        visual_prompt = batch.get("visuals", None) is not None  # TODO
        self.criterion = (
            (E2ELoss(self, TVPSegmentLoss) if getattr(self, "end2end", False) else TVPSegmentLoss(self))
            if visual_prompt
            else self.init_criterion()
        )

    if preds is None:
        preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
    return self.criterion(preds, batch)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.Ensemble` {#ultralytics.nn.tasks.Ensemble}

```python
Ensemble(self)
```

**Bases:** `torch.nn.ModuleList`

Ensemble of models.

This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging or other ensemble techniques.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.tasks.Ensemble.forward) | Run ensemble forward pass and concatenate predictions from all models. |

**Examples**

```python
Create an ensemble of models
>>> ensemble = Ensemble()
>>> ensemble.append(model1)
>>> ensemble.append(model2)
>>> results = ensemble(image_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1282-L1321"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Ensemble(torch.nn.ModuleList):
    """Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging
    or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()
```
</details>

<br>

### Method `ultralytics.nn.tasks.Ensemble.forward` {#ultralytics.nn.tasks.Ensemble.forward}

```python
def forward(self, x, augment = False, profile = False, visualize = False)
```

Run ensemble forward pass and concatenate predictions from all models.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |
| `augment` | `bool` | Whether to augment the input. | `False` |
| `profile` | `bool` | Whether to profile the model. | `False` |
| `visualize` | `bool` | Whether to visualize the features. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Concatenated predictions from all models. |
| `None` | Always None for ensemble inference. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1304-L1321"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x, augment=False, profile=False, visualize=False):
    """Run ensemble forward pass and concatenate predictions from all models.

    Args:
        x (torch.Tensor): Input tensor.
        augment (bool): Whether to augment the input.
        profile (bool): Whether to profile the model.
        visualize (bool): Whether to visualize the features.

    Returns:
        (torch.Tensor): Concatenated predictions from all models.
        (None): Always None for ensemble inference.
    """
    y = [module(x, augment, profile, visualize)[0] for module in self]
    # y = torch.stack(y).max(0)[0]  # max ensemble
    # y = torch.stack(y).mean(0)  # mean ensemble
    y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C*num_models)
    return y, None  # inference, train output
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.SafeClass` {#ultralytics.nn.tasks.SafeClass}

```python
SafeClass(self, *args, **kwargs)
```

A placeholder class to replace unknown classes during unpickling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` |  |  | *required* |
| `**kwargs` |  |  | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.nn.tasks.SafeClass.__call__) | Run SafeClass instance, ignoring all arguments. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1375-L1384"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass
```
</details>

<br>

### Method `ultralytics.nn.tasks.SafeClass.__call__` {#ultralytics.nn.tasks.SafeClass.\_\_call\_\_}

```python
def __call__(self, *args, **kwargs)
```

Run SafeClass instance, ignoring all arguments.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` |  |  | *required* |
| `**kwargs` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1382-L1384"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, *args, **kwargs):
    """Run SafeClass instance, ignoring all arguments."""
    pass
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.tasks.SafeUnpickler` {#ultralytics.nn.tasks.SafeUnpickler}

```python
SafeUnpickler()
```

**Bases:** `pickle.Unpickler`

Custom Unpickler that replaces unknown classes with SafeClass.

**Methods**

| Name | Description |
| --- | --- |
| [`find_class`](#ultralytics.nn.tasks.SafeUnpickler.find_class) | Attempt to find a class, returning SafeClass if not among safe modules. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1387-L1412"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SafeUnpickler(pickle.Unpickler):
```
</details>

<br>

### Method `ultralytics.nn.tasks.SafeUnpickler.find_class` {#ultralytics.nn.tasks.SafeUnpickler.find\_class}

```python
def find_class(self, module, name)
```

Attempt to find a class, returning SafeClass if not among safe modules.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `module` | `str` | Module name. | *required* |
| `name` | `str` | Class name. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `type` | Found class or SafeClass. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1390-L1412"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def find_class(self, module, name):
    """Attempt to find a class, returning SafeClass if not among safe modules.

    Args:
        module (str): Module name.
        name (str): Class name.

    Returns:
        (type): Found class or SafeClass.
    """
    safe_modules = (
        "torch",
        "collections",
        "collections.abc",
        "builtins",
        "math",
        "numpy",
        # Add other modules considered safe
    )
    if module in safe_modules:
        return super().find_class(module, name)
    else:
        return SafeClass
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.tasks.temporary_modules` {#ultralytics.nn.tasks.temporary\_modules}

```python
def temporary_modules(modules = None, attributes = None)
```

Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've moved a module from one location to another, but you still want to support the old import paths for backwards compatibility.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `modules` | `dict, optional` | A dictionary mapping old module paths to new module paths. | `None` |
| `attributes` | `dict, optional` | A dictionary mapping old module attributes to new module attributes. | `None` |

**Examples**

```python
>>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
>>> import old.module  # this will now import new.module
>>> from old.module import attribute  # this will now import new.module.attribute
```

!!! note "Notes"

    The changes are only in effect inside the context manager and are undone once the context manager exits.
    Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
    applications or libraries. Use this function with caution.

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1328-L1372"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've
    moved a module from one location to another, but you still want to support the old import paths for backwards
    compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Notes:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.tasks.torch_safe_load` {#ultralytics.nn.tasks.torch\_safe\_load}

```python
def torch_safe_load(weight, safe_only = False)
```

Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches

the error, logs a warning message, and attempts to install the missing module via the check_requirements() function. After installation, the function again attempts to load the model using torch.load().

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str | Path` | The file path of the PyTorch model. | *required* |
| `safe_only` | `bool` | If True, replace unknown classes with SafeClass during loading. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | The loaded model checkpoint. |
| `str` | The loaded filename. |

**Examples**

```python
>>> from ultralytics.nn.tasks import torch_safe_load
>>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
```

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1415-L1493"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def torch_safe_load(weight, safe_only=False):
    """Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches
    the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str | Path): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        (dict): The loaded model checkpoint.
        (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.tasks.load_checkpoint` {#ultralytics.nn.tasks.load\_checkpoint}

```python
def load_checkpoint(weight, device = None, inplace = True, fuse = False)
```

Load single model weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str | Path` | Model weight path. | *required* |
| `device` | `torch.device, optional` | Device to load model to. | `None` |
| `inplace` | `bool` | Whether to do inplace operations. | `True` |
| `fuse` | `bool` | Whether to fuse model. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.nn.Module` | Loaded model. |
| `dict` | Model checkpoint dictionary. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1496-L1530"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """Load single model weights.

    Args:
        weight (str | Path): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        (torch.nn.Module): Loaded model.
        (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model

    # Model compatibility updates
    model.args = args  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.tasks.parse_model` {#ultralytics.nn.tasks.parse\_model}

```python
def parse_model(d, ch, verbose = True)
```

Parse a YOLO model.yaml dictionary into a PyTorch model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `d` | `dict` | Model dictionary. | *required* |
| `ch` | `int` | Input channels. | *required* |
| `verbose` | `bool` | Whether to print model details. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.nn.Sequential` | PyTorch model. |
| `list` | Sorted list of layer indices whose outputs need to be saved. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1533-L1725"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        (torch.nn.Sequential): PyTorch model.
        (list): Sorted list of layer indices whose outputs need to be saved.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 != nc (e.g., Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
                m.legacy = legacy
        elif m is v10Detect:
            args.append([ch[x] for x in f])
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])  # channels as second arg
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.tasks.yaml_model_load` {#ultralytics.nn.tasks.yaml\_model\_load}

```python
def yaml_model_load(path)
```

Load a YOLO model from a YAML file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `str | Path` | Path to the YAML file. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Model dictionary. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1728-L1748"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def yaml_model_load(path):
    """Load a YOLO model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.tasks.guess_model_scale` {#ultralytics.nn.tasks.guess\_model\_scale}

```python
def guess_model_scale(model_path)
```

Extract the size character n, s, m, l, or x of the model's scale from the model path.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model_path` | `str | Path` | The path to the YOLO model's YAML file. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | The size character of the model's scale (n, s, m, l, or x), or empty string if not found. |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1751-L1763"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def guess_model_scale(model_path):
    """Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x), or empty string if not found.
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.tasks.guess_model_task` {#ultralytics.nn.tasks.guess\_model\_task}

```python
def guess_model_task(model)
```

Guess the task of a PyTorch model from its architecture or configuration.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module | dict | str | Path` | PyTorch model, model configuration dict, or model file path. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Task of the model ('detect', 'segment', 'classify', 'pose', 'obb'). |

<details>
<summary>Source code in <code>ultralytics/nn/tasks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L1766-L1833"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def guess_model_task(model):
    """Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict | str | Path): PyTorch model, model configuration dict, or model file path.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]  # nosec B307: safe eval of known attribute paths
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))  # nosec B307: safe eval of known attribute paths
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
```
</details>

<br><br>
