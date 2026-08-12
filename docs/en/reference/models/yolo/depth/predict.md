---
title: models.yolo.depth.predict API Reference
description: Reference for `ultralytics.models.yolo.depth.predict` in the Ultralytics package.
keywords: Ultralytics, ultralytics.models.yolo.depth.predict, API reference, YOLO, Python
---

# Reference for `ultralytics/models/yolo/depth/predict.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/predict.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/predict.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DepthPredictor`](#ultralytics.models.yolo.depth.predict.DepthPredictor)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DepthPredictor.postprocess`](#ultralytics.models.yolo.depth.predict.DepthPredictor.postprocess)


## Class `ultralytics.models.yolo.depth.predict.DepthPredictor` {#ultralytics.models.yolo.depth.predict.DepthPredictor}

```python
DepthPredictor(cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None)
```

**Bases:** `BasePredictor`

Predictor for YOLO depth estimation models.

Produces per-pixel depth maps from RGB images.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` |  |  | `DEFAULT_CFG` |
| `overrides` | `dict[str, Any] \| None` |  | `None` |
| `_callbacks` | `dict \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`postprocess`](#ultralytics.models.yolo.depth.predict.DepthPredictor.postprocess) | Post-process depth predictions to Results objects. |

**Examples**

```python
>>> from ultralytics.models.yolo.depth import DepthPredictor
>>> predictor = DepthPredictor(overrides=dict(model="yolo26n-depth.pt"))
>>> results = predictor("image.jpg")
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/predict.py#L16-L53">View on GitHub</a>
```python
class DepthPredictor(BasePredictor):
    """Predictor for YOLO depth estimation models.

    Produces per-pixel depth maps from RGB images.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthPredictor
        >>> predictor = DepthPredictor(overrides=dict(model="yolo26n-depth.pt"))
        >>> results = predictor("image.jpg")
    """

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None
    ) -> None:
        """Initialize DepthPredictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "depth"
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.predict.DepthPredictor.postprocess` {#ultralytics.models.yolo.depth.predict.DepthPredictor.postprocess}

```python
def postprocess(
    self, preds: torch.Tensor | tuple | list, img: torch.Tensor, orig_imgs: list[np.ndarray] | torch.Tensor
) -> list[Results]
```

Post-process depth predictions to Results objects.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor \| tuple \| list` |  | *required* |
| `img` | `torch.Tensor` |  | *required* |
| `orig_imgs` | `list[np.ndarray] \| torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/predict.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/predict.py#L34-L53">View on GitHub</a>
```python
def postprocess(
    self, preds: torch.Tensor | tuple | list, img: torch.Tensor, orig_imgs: list[np.ndarray] | torch.Tensor
) -> list[Results]:
    """Post-process depth predictions to Results objects."""
    depth_maps = preds[0] if isinstance(preds, (tuple, list)) else preds  # (B, 1, H, W)
    if depth_maps.ndim == 3:
        depth_maps = depth_maps.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
    # Restore model-input resolution so all backends crop letterbox padding before scaling to the original image.
    depth_maps = ops.scale_masks(depth_maps, img.shape[2:], padding=False)

    if not isinstance(orig_imgs, list):  # torch.Tensor source (B, 3, H, W)
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, orig_img in enumerate(orig_imgs):
        img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
        depth = ops.scale_masks(depth_maps[i : i + 1].float(), orig_img.shape[:2])
        results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, depth=depth.squeeze()))

    return results
```
</details>

<br><br>
