---
title: models.yolo.depth.train API Reference
description: Reference for `ultralytics.models.yolo.depth.train` in the Ultralytics package.
keywords: Ultralytics, ultralytics.models.yolo.depth.train, API reference, YOLO, Python
---

# Reference for `ultralytics/models/yolo/depth/train.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/train.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DepthTrainer`](#ultralytics.models.yolo.depth.train.DepthTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DepthTrainer.get_model`](#ultralytics.models.yolo.depth.train.DepthTrainer.get_model)
        - [`DepthTrainer.preprocess_batch`](#ultralytics.models.yolo.depth.train.DepthTrainer.preprocess_batch)
        - [`DepthTrainer.get_validator`](#ultralytics.models.yolo.depth.train.DepthTrainer.get_validator)
        - [`DepthTrainer.plot_training_labels`](#ultralytics.models.yolo.depth.train.DepthTrainer.plot_training_labels)
        - [`DepthTrainer.final_eval`](#ultralytics.models.yolo.depth.train.DepthTrainer.final_eval)


## Class `ultralytics.models.yolo.depth.train.DepthTrainer` {#ultralytics.models.yolo.depth.train.DepthTrainer}

```python
DepthTrainer(cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None)
```

**Bases:** `DetectionTrainer`

Trainer for YOLO depth estimation models.

Multi-source training (list of img_paths) is handled transparently by the base DetectionTrainer/BaseDataset.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` |  |  | `DEFAULT_CFG` |
| `overrides` | `dict[str, Any] \| None` |  | `None` |
| `_callbacks` | `dict \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`final_eval`](#ultralytics.models.yolo.depth.train.DepthTrainer.final_eval) | Run the standard final evaluation, then calibrate the saved checkpoints. |
| [`get_model`](#ultralytics.models.yolo.depth.train.DepthTrainer.get_model) | Return a DepthModel initialized with the given config and weights. |
| [`get_validator`](#ultralytics.models.yolo.depth.train.DepthTrainer.get_validator) | Return a DepthValidator for model validation. |
| [`plot_training_labels`](#ultralytics.models.yolo.depth.train.DepthTrainer.plot_training_labels) | Plot the training-set GT depth distribution to ``labels.jpg``. |
| [`preprocess_batch`](#ultralytics.models.yolo.depth.train.DepthTrainer.preprocess_batch) | Preprocess batch: normalize images and keep depth as float32. |

**Examples**

```python
>>> from ultralytics.models.yolo.depth import DepthTrainer
>>> args = dict(model="yolo26s-depth.yaml", data="nyu-depth.yaml", epochs=100)
>>> trainer = DepthTrainer(overrides=args)
>>> trainer.train()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/train.py#L17-L173">View on GitHub</a>
```python
class DepthTrainer(DetectionTrainer):
    """Trainer for YOLO depth estimation models.

    Multi-source training (list of img_paths) is handled transparently by the base DetectionTrainer/BaseDataset.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthTrainer
        >>> args = dict(model="yolo26s-depth.yaml", data="nyu-depth.yaml", epochs=100)
        >>> trainer = DepthTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None
    ) -> None:
        """Initialize DepthTrainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "depth"
        super().__init__(cfg, overrides, _callbacks)
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.train.DepthTrainer.final_eval` {#ultralytics.models.yolo.depth.train.DepthTrainer.final\_eval}

```python
def final_eval(self) -> None
```

Run the standard final evaluation, then calibrate the saved checkpoints.

After training, fits the scale-only log-affine (``cal_a``/``cal_b``) on the validation set and writes it into best.pt/last.pt, so the model outputs metric-scaled depth out of the box. When ``plots`` is set, also writes ``val_batch{ni}_calibrated.jpg`` (RGB | GT | raw | calibrated) comparison panels.

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/train.py#L130-L173">View on GitHub</a>
```python
def final_eval(self) -> None:
    """Run the standard final evaluation, then calibrate the saved checkpoints.

    After training, fits the scale-only log-affine (``cal_a``/``cal_b``) on the validation
    set and writes it into best.pt/last.pt, so the model outputs metric-scaled depth out of
    the box. When ``plots`` is set, also writes ``val_batch{ni}_calibrated.jpg``
    (RGB | GT | raw | calibrated) comparison panels.
    """
    super().final_eval()
    if RANK not in {-1, 0}:
        return
    try:
        from .calibrate import calibrate_checkpoint

        LOGGER.info("Calibrating depth output scale on the validation set...")
        plot_ckpt = self.best if self.best.exists() else self.last
        for ckpt in (self.best, self.last):
            if ckpt.exists():
                plot_dir = self.save_dir if self.args.plots and ckpt == plot_ckpt else None
                validation_path = self.data.get("val") or self.data.get("test")
                validation_split = None
                if isinstance(validation_path, (str, Path)):
                    try:
                        validation_split = (
                            Path(validation_path)
                            .resolve()
                            .relative_to(Path(self.data["path"]).resolve())
                            .as_posix()
                        )
                    except ValueError:
                        pass  # External validation paths have no portable dataset-root-relative identifier.
                provenance = calibrate_checkpoint(
                    ckpt,
                    self.test_loader,
                    self.device,
                    plot_dir=plot_dir,
                    dataset_hash=self.data.get("hash"),
                    validation_split=validation_split,
                    max_depth=self.data.get("max_depth") or 100.0,
                )
                if ckpt == plot_ckpt and provenance is not None:
                    self.depth_calibration = provenance
    except Exception as e:
        LOGGER.warning(f"Calibration skipped ({type(e).__name__}: {e}); checkpoints left uncalibrated.")
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.train.DepthTrainer.get_model` {#ultralytics.models.yolo.depth.train.DepthTrainer.get\_model}

```python
def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True) -> DepthModel
```

Return a DepthModel initialized with the given config and weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str \| None` |  | `None` |
| `weights` | `str \| None` |  | `None` |
| `verbose` | `bool` |  | `True` |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/train.py#L38-L45">View on GitHub</a>
```python
def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True) -> DepthModel:
    """Return a DepthModel initialized with the given config and weights."""
    model = DepthModel(
        cfg, ch=self.data.get("channels", 3), nc=self.data["nc"], verbose=verbose and RANK in {-1, 0}
    )
    if weights:
        model.load(weights)
    return model
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.train.DepthTrainer.get_validator` {#ultralytics.models.yolo.depth.train.DepthTrainer.get\_validator}

```python
def get_validator(self) -> yolo.depth.DepthValidator
```

Return a DepthValidator for model validation.

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/train.py#L53-L57">View on GitHub</a>
```python
def get_validator(self) -> yolo.depth.DepthValidator:
    """Return a DepthValidator for model validation."""
    return yolo.depth.DepthValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.train.DepthTrainer.plot_training_labels` {#ultralytics.models.yolo.depth.train.DepthTrainer.plot\_training\_labels}

```python
def plot_training_labels(self) -> None
```

Plot the training-set GT depth distribution to ``labels.jpg``.

The depth analog of the detection/semantic label plots. The inherited DetectionTrainer version concatenates per-image ``bboxes``/``cls`` (all empty for depth) and hands them to ``plot_labels``, whose reductions raise "zero-size array to reduction operation maximum which has no identity". Instead, sample GT depth maps from the training set and plot a histogram of valid (``> 0``) depth values, annotated with basic statistics.

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/train.py#L60-L128">View on GitHub</a>
```python
@plt_settings()
def plot_training_labels(self) -> None:
    """Plot the training-set GT depth distribution to ``labels.jpg``.

    The depth analog of the detection/semantic label plots. The inherited DetectionTrainer
    version concatenates per-image ``bboxes``/``cls`` (all empty for depth) and hands them to
    ``plot_labels``, whose reductions raise "zero-size array to reduction operation maximum
    which has no identity". Instead, sample GT depth maps from the training set and plot a
    histogram of valid (``> 0``) depth values, annotated with basic statistics.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    LOGGER.info(f"Plotting labels to {self.save_dir / 'labels.jpg'}...")
    dataset = self.train_loader.dataset
    n = len(dataset.im_files)
    if n == 0:
        LOGGER.warning("No depth maps found, skipping label plot.")
        return

    sample_size = min(1000, n)
    indices = np.linspace(0, n - 1, sample_size).astype(int)
    per_map_cap = max(1, 1_000_000 // sample_size)  # bound total memory to ~1M values
    values = []
    for idx in indices:
        d = dataset._load_depth(idx)  # shared loader sanitizes non-finite GT to 0
        if d is None:
            continue
        v = d[d > 0].ravel()
        if v.size == 0:
            continue
        if v.size > per_map_cap:  # uniform stride keeps the spatial distribution unbiased
            v = v[np.linspace(0, v.size - 1, per_map_cap).astype(int)]
        values.append(v)

    if not values:
        LOGGER.warning("No valid depth values found, skipping label plot.")
        return

    values = np.concatenate(values)
    vmin, vmax = float(values.min()), float(np.percentile(values, 99.5))
    mean, median, std = float(values.mean()), float(np.median(values)), float(values.std())

    _, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    ax.hist(values, bins=100, range=(vmin, max(vmax, vmin + 1e-6)), color="#3b7dd8")
    ax.axvline(mean, color="#d8643b", linestyle="--", linewidth=1.5, label=f"mean {mean:.2f} m")
    ax.axvline(median, color="#3bd86b", linestyle="--", linewidth=1.5, label=f"median {median:.2f} m")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Pixels")
    ax.set_title("Training Labels Depth Distribution")
    ax.legend(loc="upper right", frameon=False)
    stats = f"images: {sample_size}\nmin: {vmin:.2f} m\nmax: {values.max():.2f} m\nstd: {std:.2f} m"
    ax.text(
        0.98,
        0.7,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    fname = self.save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if self.on_plot:
        self.on_plot(fname)
```
</details>

<br>

### Method `ultralytics.models.yolo.depth.train.DepthTrainer.preprocess_batch` {#ultralytics.models.yolo.depth.train.DepthTrainer.preprocess\_batch}

```python
def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]
```

Preprocess batch: normalize images and keep depth as float32.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/train.py#L47-L51">View on GitHub</a>
```python
def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
    """Preprocess batch: normalize images and keep depth as float32."""
    batch = super().preprocess_batch(batch)
    batch["depth"] = batch["depth"].float()
    return batch
```
</details>

<br><br>
