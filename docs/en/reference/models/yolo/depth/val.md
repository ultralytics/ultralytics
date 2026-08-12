---
title: models.yolo.depth.val API Reference
description: Reference for `ultralytics.models.yolo.depth.val` in the Ultralytics package.
keywords: Ultralytics, ultralytics.models.yolo.depth.val, API reference, YOLO, Python
---

# Reference for `ultralytics/models/yolo/depth/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DepthValidator`](#ultralytics.models.yolo.depth.val.DepthValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DepthValidator.init_metrics`](#ultralytics.models.yolo.depth.val.DepthValidator.init_metrics)
        - [`DepthValidator.preprocess`](#ultralytics.models.yolo.depth.val.DepthValidator.preprocess)
        - [`DepthValidator.postprocess`](#ultralytics.models.yolo.depth.val.DepthValidator.postprocess)
        - [`DepthValidator.update_metrics`](#ultralytics.models.yolo.depth.val.DepthValidator.update_metrics)
        - [`DepthValidator.get_stats`](#ultralytics.models.yolo.depth.val.DepthValidator.get_stats)
        - [`DepthValidator.gather_stats`](#ultralytics.models.yolo.depth.val.DepthValidator.gather_stats)
        - [`DepthValidator.print_results`](#ultralytics.models.yolo.depth.val.DepthValidator.print_results)
        - [`DepthValidator.finalize_metrics`](#ultralytics.models.yolo.depth.val.DepthValidator.finalize_metrics)
        - [`DepthValidator.get_desc`](#ultralytics.models.yolo.depth.val.DepthValidator.get_desc)
        - [`DepthValidator.plot_predictions`](#ultralytics.models.yolo.depth.val.DepthValidator.plot_predictions)


## Class `ultralytics.models.yolo.depth.val.DepthValidator` {#ultralytics.models.yolo.depth.val.DepthValidator}

```python
DepthValidator(dataloader=None, save_dir: str | Path | None = None, args=None, _callbacks: dict | None = None)
```

**Bases:** `DetectionValidator`

Validator for YOLO depth estimation models.

Computes standard depth metrics: delta1, abs_rel, rmse, silog. Uses validation loss as the primary training signal.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` |  |  | `None` |
| `save_dir` | `str \| Path \| None` |  | `None` |
| `args` |  |  | `None` |
| `_callbacks` | `dict \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`finalize_metrics`](#ultralytics.models.yolo.depth.val.DepthValidator.finalize_metrics) | Set final values for metrics speed. |
| [`gather_stats`](#ultralytics.models.yolo.depth.val.DepthValidator.gather_stats) | Sum depth metric accumulators across DDP ranks onto rank 0. |
| [`get_desc`](#ultralytics.models.yolo.depth.val.DepthValidator.get_desc) | Return description for progress bar. |
| [`get_stats`](#ultralytics.models.yolo.depth.val.DepthValidator.get_stats) | Finalize and return the metrics dict. |
| [`init_metrics`](#ultralytics.models.yolo.depth.val.DepthValidator.init_metrics) | Initialize the DepthMetrics accumulator with the dataset's depth range. |
| [`plot_predictions`](#ultralytics.models.yolo.depth.val.DepthValidator.plot_predictions) | Save predicted depth overlays to val_batch{ni}_pred.jpg. |
| [`postprocess`](#ultralytics.models.yolo.depth.val.DepthValidator.postprocess) | No NMS needed for depth — return predictions as-is. |
| [`preprocess`](#ultralytics.models.yolo.depth.val.DepthValidator.preprocess) | Preprocess batch — move to device, normalize images, and keep depth as float32. |
| [`print_results`](#ultralytics.models.yolo.depth.val.DepthValidator.print_results) | Log the headline depth metrics in the detection-style aligned table format. |
| [`update_metrics`](#ultralytics.models.yolo.depth.val.DepthValidator.update_metrics) | Accumulate depth metrics for a batch. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L19-L135">View on GitHub</a>
```python
class DepthValidator(DetectionValidator):
    """Validator for YOLO depth estimation models.

    Computes standard depth metrics: delta1, abs_rel, rmse, silog. Uses validation loss as the primary training signal.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir: str | Path | None = None,
        args=None,
        _callbacks: dict | None = None,
    ) -> None:
        """Initialize DepthValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.finalize_metrics` {#ultralytics.models.yolo.depth.val.DepthValidator.finalize\_metrics}

```python
def finalize_metrics(self) -> None
```

Set final values for metrics speed.

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L113-L116">View on GitHub</a>
```python
def finalize_metrics(self) -> None:
    """Set final values for metrics speed."""
    self.metrics.speed = self.speed
    self.metrics.save_dir = self.save_dir
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.gather_stats` {#ultralytics.models.yolo.depth.val.DepthValidator.gather\_stats}

```python
def gather_stats(self) -> None
```

Sum depth metric accumulators across DDP ranks onto rank 0.

Validation is sharded (ContiguousDistributedSampler gives each rank a distinct chunk of the val set), so each rank holds only its shard's summed statistics. All-reduce the sums so rank 0's get_stats() computes metrics over the full val set instead of a single shard. Overrides DetectionValidator.gather_stats(), which reduces detection-specific stats/box attributes that DepthMetrics does not have.

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L71-L90">View on GitHub</a>
```python
def gather_stats(self) -> None:
    """Sum depth metric accumulators across DDP ranks onto rank 0.

    Validation is sharded (ContiguousDistributedSampler gives each rank a distinct chunk of the
    val set), so each rank holds only its shard's summed statistics. All-reduce the sums so
    rank 0's get_stats() computes metrics over the full val set instead of a single shard.
    Overrides DetectionValidator.gather_stats(), which reduces detection-specific stats/box
    attributes that DepthMetrics does not have.
    """
    if RANK == -1 or not dist.is_initialized():
        return
    totals = self.metrics._totals
    totals = (
        totals.to(self.device) if totals is not None else torch.zeros(6, dtype=torch.float64, device=self.device)
    )
    count = torch.tensor([self.metrics._count], dtype=torch.float64, device=self.device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    self.metrics._totals = totals
    self.metrics._count = float(count.item())
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.get_desc` {#ultralytics.models.yolo.depth.val.DepthValidator.get\_desc}

```python
def get_desc(self) -> str
```

Return description for progress bar.

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L118-L120">View on GitHub</a>
```python
def get_desc(self) -> str:
    """Return description for progress bar."""
    return ("%22s" + "%11s" * 5) % ("Class", "Images", "delta1", "abs_rel", "rmse", "silog")
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.get_stats` {#ultralytics.models.yolo.depth.val.DepthValidator.get\_stats}

```python
def get_stats(self) -> dict[str, float]
```

Finalize and return the metrics dict.

Cross-rank metric reduction is handled by gather_stats() (called before this on all ranks); this runs on rank 0 with the already-summed accumulators.

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L62-L69">View on GitHub</a>
```python
def get_stats(self) -> dict[str, float]:
    """Finalize and return the metrics dict.

    Cross-rank metric reduction is handled by gather_stats() (called before this on all ranks);
    this runs on rank 0 with the already-summed accumulators.
    """
    self.metrics.process()
    return self.metrics.results_dict
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.init_metrics` {#ultralytics.models.yolo.depth.val.DepthValidator.init\_metrics}

```python
def init_metrics(self, model: torch.nn.Module) -> None
```

Initialize the DepthMetrics accumulator with the dataset's depth range.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L36-L39">View on GitHub</a>
```python
def init_metrics(self, model: torch.nn.Module) -> None:
    """Initialize the DepthMetrics accumulator with the dataset's depth range."""
    self.metrics = DepthMetrics(max_depth=self.data.get("max_depth") or 100.0)
    self.metrics.clear_stats()
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.plot_predictions` {#ultralytics.models.yolo.depth.val.DepthValidator.plot\_predictions}

```python
def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None
```

Save predicted depth overlays to val_batch{ni}_pred.jpg.

Depth has no boxes/classes, so the detection-style plotter is replaced with a depth heatmap overlay through the shared ``plot_images`` path, matching the semantic-segmentation visualization style.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` |  | *required* |
| `preds` | `torch.Tensor` |  | *required* |
| `ni` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L122-L135">View on GitHub</a>
```python
def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
    """Save predicted depth overlays to val_batch{ni}_pred.jpg.

    Depth has no boxes/classes, so the detection-style plotter is replaced with a depth heatmap overlay
    through the shared ``plot_images`` path, matching the semantic-segmentation visualization style.
    """
    plot_images(
        labels={"depth": preds},
        images=batch["img"],
        paths=batch["im_file"],
        fname=self.save_dir / f"val_batch{ni}_pred.jpg",
        names=self.names,
        on_plot=self.on_plot,
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.postprocess` {#ultralytics.models.yolo.depth.val.DepthValidator.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> torch.Tensor
```

No NMS needed for depth — return predictions as-is.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L47-L49">View on GitHub</a>
```python
def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
    """No NMS needed for depth — return predictions as-is."""
    return preds
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.preprocess` {#ultralytics.models.yolo.depth.val.DepthValidator.preprocess}

```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]
```

Preprocess batch — move to device, normalize images, and keep depth as float32.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L41-L45">View on GitHub</a>
```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
    """Preprocess batch — move to device, normalize images, and keep depth as float32."""
    batch = super().preprocess(batch)
    batch["depth"] = batch["depth"].float()
    return batch
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.print_results` {#ultralytics.models.yolo.depth.val.DepthValidator.print\_results}

```python
def print_results(self) -> None
```

Log the headline depth metrics in the detection-style aligned table format.

Columns line up with get_desc(): Class, Images, delta1, abs_rel, rmse, silog. Uses "depth_val" as the row label (depth has no classes, where detection prints "all").

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L92-L111">View on GitHub</a>
```python
def print_results(self) -> None:
    """Log the headline depth metrics in the detection-style aligned table format.

    Columns line up with get_desc(): Class, Images, delta1, abs_rel, rmse, silog.
    Uses "depth_val" as the row label (depth has no classes, where detection prints "all").
    """
    r = self.metrics.results_dict
    n_images = len(self.dataloader.dataset) if self.dataloader is not None else (self.seen or 0)
    pf = "%22s" + "%11i" + "%11.4g" * 4  # label, Images, delta1, abs_rel, rmse, silog
    LOGGER.info(
        pf
        % (
            "depth_val",
            n_images,
            r.get("metrics/delta1", 0.0),
            r.get("metrics/abs_rel", 0.0),
            r.get("metrics/rmse", 0.0),
            r.get("metrics/silog", 0.0),
        )
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.val.DepthValidator.update_metrics` {#ultralytics.models.yolo.depth.val.DepthValidator.update\_metrics}

```python
def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None
```

Accumulate depth metrics for a batch.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` |  | *required* |
| `batch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/val.py#L51-L60">View on GitHub</a>
```python
def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
    """Accumulate depth metrics for a batch."""
    gt_depth = batch["depth"]
    if gt_depth.ndim == 3:
        gt_depth = gt_depth.unsqueeze(1)
    if preds.ndim == 3:
        preds = preds.unsqueeze(1)
    if preds.shape[-2:] != gt_depth.shape[-2:]:
        preds = F.interpolate(preds.float(), size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)
    self.metrics.update_stats(preds, gt_depth)
```
</details>

<br><br>
