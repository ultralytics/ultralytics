---
title: models.yolo.depth.calibrate API Reference
description: Reference for `ultralytics.models.yolo.depth.calibrate` in the Ultralytics package.
keywords: Ultralytics, ultralytics.models.yolo.depth.calibrate, API reference, YOLO, Python
---

# Reference for `ultralytics/models/yolo/depth/calibrate.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`_depth_head`](#ultralytics.models.yolo.depth.calibrate._depth_head)
        - [`_rewind`](#ultralytics.models.yolo.depth.calibrate._rewind)
        - [`_delta1_none`](#ultralytics.models.yolo.depth.calibrate._delta1_none)
        - [`select_calibration`](#ultralytics.models.yolo.depth.calibrate.select_calibration)
        - [`select_calibration_cv`](#ultralytics.models.yolo.depth.calibrate.select_calibration_cv)
        - [`_collect_logpairs`](#ultralytics.models.yolo.depth.calibrate._collect_logpairs)
        - [`fit_calibration_selective`](#ultralytics.models.yolo.depth.calibrate.fit_calibration_selective)
        - [`_plot_calibrated_batches`](#ultralytics.models.yolo.depth.calibrate._plot_calibrated_batches)
        - [`calibrate_checkpoint`](#ultralytics.models.yolo.depth.calibrate.calibrate_checkpoint)


## Function `ultralytics.models.yolo.depth.calibrate._depth_head` {#ultralytics.models.yolo.depth.calibrate.\_depth\_head}

```python
def _depth_head(model: torch.nn.Module) -> torch.nn.Module | None
```

Return the Depth head module that carries ``cal_a``/``cal_b`` buffers, or None.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L28-L35">View on GitHub</a>
```python
def _depth_head(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the Depth head module that carries ``cal_a``/``cal_b`` buffers, or None."""
    m = model.module if hasattr(model, "module") else model  # unwrap DDP
    seq = getattr(m, "model", None)
    if seq is not None and not isinstance(seq, torch.nn.Sequential):
        seq = getattr(seq, "model", None)
    head = seq[-1] if isinstance(seq, torch.nn.Sequential) else m
    return head if hasattr(head, "cal_a") else None
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.yolo.depth.calibrate._rewind` {#ultralytics.models.yolo.depth.calibrate.\_rewind}

```python
def _rewind(dataloader) -> None
```

Rewind a stateful loader to the epoch start before a partial pass.

The trainer's InfiniteDataLoader keeps one persistent iterator across ``for`` loops, so a previous pass that broke out early (e.g. a fit stopping at ``max_images``) leaves it mid-epoch and the next loop silently resumes there. Every pass here that wants "the first N" must rewind first — the plot pass in particular must see the same leading batches BaseValidator plotted as ``val_batch{ni}.jpg``. Plain DataLoaders and list fixtures restart on every ``__iter__``.

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L38-L49">View on GitHub</a>
```python
def _rewind(dataloader) -> None:
    """Rewind a stateful loader to the epoch start before a partial pass.

    The trainer's InfiniteDataLoader keeps one persistent iterator across ``for`` loops, so a previous pass that broke
    out early (e.g. a fit stopping at ``max_images``) leaves it mid-epoch and the next loop silently resumes there.
    Every pass here that wants "the first N" must rewind first — the plot pass in particular must see the same leading
    batches BaseValidator plotted as ``val_batch{ni}.jpg``. Plain DataLoaders and list fixtures restart on every
    ``__iter__``.
    """
    reset = getattr(dataloader, "reset", None)
    if callable(reset):
        reset()
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.yolo.depth.calibrate._delta1_none` {#ultralytics.models.yolo.depth.calibrate.\_delta1\_none}

```python
def _delta1_none(log_pred: np.ndarray, log_gt: np.ndarray, a: float, b: float) -> float
```

δ1 under no per-image alignment after applying ``d' = exp(a·log_pred + b)``.

δ1 is the fraction of pixels with ``max(d'/gt, gt/d') < 1.25``; in log space that is ``|a·log_pred + b − log_gt| < log(1.25)``. This is the deployment metric (raw absolute scale, the ``align="none"`` protocol) the policy optimizes — the val scoreboard's default ``align="median"`` is scale-invariant and cannot see calibration.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `log_pred` | `np.ndarray` |  | *required* |
| `log_gt` | `np.ndarray` |  | *required* |
| `a` | `float` |  | *required* |
| `b` | `float` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L52-L60">View on GitHub</a>
```python
def _delta1_none(log_pred: np.ndarray, log_gt: np.ndarray, a: float, b: float) -> float:
    """δ1 under no per-image alignment after applying ``d' = exp(a·log_pred + b)``.

    δ1 is the fraction of pixels with ``max(d'/gt, gt/d') < 1.25``; in log space that is ``|a·log_pred + b − log_gt| <
    log(1.25)``. This is the deployment metric (raw absolute scale, the ``align="none"`` protocol) the policy optimizes
    — the val scoreboard's default ``align="median"`` is scale-invariant and cannot see calibration.
    """
    ld = (a * np.asarray(log_pred, dtype=np.float64) + b) - np.asarray(log_gt, dtype=np.float64)
    return float(np.mean(np.abs(ld) < np.log(1.25)))
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.yolo.depth.calibrate.select_calibration` {#ultralytics.models.yolo.depth.calibrate.select\_calibration}

```python
def select_calibration(
    lp_fit: np.ndarray,
    lg_fit: np.ndarray,
    lp_score: np.ndarray,
    lg_score: np.ndarray,
) -> dict[str, Any]
```

Pick the calibration that best improves *held-out* raw-scale δ1 — "calibrate only if it helps".

Two candidates are fit on the ``*_fit`` log-pixel arrays and scored on the independent ``*_score`` arrays under
:func:`_delta1_none`:
- ``identity`` (a=1, b=0): no calibration,
- ``scale-only`` (a=1, b=mean(log_gt − log_pred)): a global scale.
The winner is the highest held-out δ1, with ties favoring identity — so a calibration that does not generalize is
rejected and auto-cal does no harm. (An affine log-slope candidate was evaluated and removed: the extra parameter
overfits within-dataset cross-validation and harms cross-distribution generalization.)

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `lp_fit` | `np.ndarray` |  | *required* |
| `lg_fit` | `np.ndarray` |  | *required* |
| `lp_score` | `np.ndarray` |  | *required* |
| `lg_score` | `np.ndarray` |  | *required* |

**Returns**

| Type | Description |
| --- | --- |
|  | dict with ``a``, ``b`` (floats of the winner), ``name``, and ``scores`` (per-candidate δ1). |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L63-L90">View on GitHub</a>
```python
def select_calibration(
    lp_fit: np.ndarray,
    lg_fit: np.ndarray,
    lp_score: np.ndarray,
    lg_score: np.ndarray,
) -> dict[str, Any]:
    """Pick the calibration that best improves *held-out* raw-scale δ1 — "calibrate only if it helps".

    Two candidates are fit on the ``*_fit`` log-pixel arrays and scored on the independent ``*_score`` arrays under
    :func:`_delta1_none`:
    - ``identity`` (a=1, b=0): no calibration,
    - ``scale-only`` (a=1, b=mean(log_gt − log_pred)): a global scale.
    The winner is the highest held-out δ1, with ties favoring identity — so a calibration that does not generalize is
    rejected and auto-cal does no harm. (An affine log-slope candidate was evaluated and removed: the extra parameter
    overfits within-dataset cross-validation and harms cross-distribution generalization.)

    Returns:
        dict with ``a``, ``b`` (floats of the winner), ``name``, and ``scores`` (per-candidate δ1).
    """
    lp_fit = np.asarray(lp_fit, dtype=np.float64)
    lg_fit = np.asarray(lg_fit, dtype=np.float64)
    candidates = [
        ("identity", 1.0, 0.0),
        ("scale-only", 1.0, float(np.mean(lg_fit - lp_fit))),
    ]
    scored = [(name, a, b, _delta1_none(lp_score, lg_score, a, b)) for name, a, b in candidates]
    best = max(scored, key=lambda s: s[3])  # identity first, so exact ties favor it
    return {"a": best[1], "b": best[2], "name": best[0], "scores": {s[0]: s[3] for s in scored}}
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.yolo.depth.calibrate.select_calibration_cv` {#ultralytics.models.yolo.depth.calibrate.select\_calibration\_cv}

```python
def select_calibration_cv(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    margin: float = 0.0,
    folds: int = 2,
) -> dict[str, Any]
```

Cross-validated "calibrate only if it helps": choose a candidate by K-fold held-out δ1.

``pairs`` is a list of per-image ``(log_pred, log_gt)`` arrays. Each candidate type is scored on every fold while held out (fit on the rest) via :func:`select_calibration`, and the per-type held-out δ1 is averaged across folds — so every image contributes to scoring exactly once and a candidate that only wins on one noisy split cannot be selected. The winning *type* must beat identity's mean held-out δ1 by ``margin`` (ties favor the simpler type); the final ``(a, b)`` is then refit on all pairs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pairs` | `list[tuple[np.ndarray, np.ndarray]]` |  | *required* |
| `margin` | `float` |  | `0.0` |
| `folds` | `int` |  | `2` |

**Returns**

| Type | Description |
| --- | --- |
|  | dict with ``a``, ``b`` (floats), ``name``, and ``cv_scores`` (mean held-out δ1 per type). |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L93-L137">View on GitHub</a>
```python
def select_calibration_cv(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    margin: float = 0.0,
    folds: int = 2,
) -> dict[str, Any]:
    """Cross-validated "calibrate only if it helps": choose a candidate by K-fold held-out δ1.

    ``pairs`` is a list of per-image ``(log_pred, log_gt)`` arrays. Each candidate type is scored on every fold while
    held out (fit on the rest) via :func:`select_calibration`, and the per-type held-out δ1 is averaged across folds —
    so every image contributes to scoring exactly once and a candidate that only wins on one noisy split cannot be
    selected. The winning *type* must beat identity's mean held-out δ1 by ``margin`` (ties favor the simpler type); the
    final ``(a, b)`` is then refit on all pairs.

    Returns:
        dict with ``a``, ``b`` (floats), ``name``, and ``cv_scores`` (mean held-out δ1 per type).
    """
    names = ["identity", "scale-only"]
    k = max(2, min(folds, len(pairs)))
    per_fold = {n: [] for n in names}
    for f in range(k):
        fit = [pairs[i] for i in range(len(pairs)) if i % k != f]
        score = [pairs[i] for i in range(len(pairs)) if i % k == f]
        if not fit or not score:
            continue
        s = select_calibration(
            np.concatenate([p[0] for p in fit]),
            np.concatenate([p[1] for p in fit]),
            np.concatenate([p[0] for p in score]),
            np.concatenate([p[1] for p in score]),
        )["scores"]
        for n in names:
            per_fold[n].append(s[n])
    cv = {n: float(np.mean(per_fold[n])) for n in names}
    best = "identity"
    for n in names[1:]:
        if cv[n] > cv[best] + margin:
            best = n
    # refit the chosen type on all pairs (CV selects the type; all data sets the parameters)
    if best == "identity":
        a, b = 1.0, 0.0
    else:  # scale-only
        lp = np.concatenate([p[0] for p in pairs])
        lg = np.concatenate([p[1] for p in pairs])
        a, b = 1.0, float(np.mean(lg - lp))
    return {"a": a, "b": b, "name": best, "cv_scores": cv}
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.yolo.depth.calibrate._collect_logpairs` {#ultralytics.models.yolo.depth.calibrate.\_collect\_logpairs}

```python
def _collect_logpairs(
    model: torch.nn.Module, dataloader, device: torch.device | str, max_images: int, max_depth: float = 100.0
) -> list[tuple[np.ndarray, np.ndarray]]
```

Run the model over the loader and return a list of per-image ``(log_pred, log_gt)`` arrays.

One entry per image (each subsampled to ≤20k valid pixels) so callers can split images into independent fit/score sets. Only GT inside ``(0.001, max_depth)`` is collected — the same population ``DepthMetrics`` evaluates (Eigen protocol), so invalid far GT cannot steer the fitted scale or the held-out δ1 the selection policy trusts. Calibration buffers are reset to identity for the duration so the fit sees the raw output, then restored.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `dataloader` |  |  | *required* |
| `device` | `torch.device \| str` |  | *required* |
| `max_images` | `int` |  | *required* |
| `max_depth` | `float` |  | `100.0` |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L141-L188">View on GitHub</a>
```python
@smart_inference_mode()
def _collect_logpairs(
    model: torch.nn.Module, dataloader, device: torch.device | str, max_images: int, max_depth: float = 100.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the model over the loader and return a list of per-image ``(log_pred, log_gt)`` arrays.

    One entry per image (each subsampled to ≤20k valid pixels) so callers can split images into independent fit/score
    sets. Only GT inside ``(0.001, max_depth)`` is collected — the same population ``DepthMetrics`` evaluates (Eigen
    protocol), so invalid far GT cannot steer the fitted scale or the held-out δ1 the selection policy trusts.
    Calibration buffers are reset to identity for the duration so the fit sees the raw output, then restored.
    """
    head = _depth_head(model)
    a0, b0 = float(head.cal_a), float(head.cal_b)
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)
    model = model.to(device).eval()
    _rewind(dataloader)
    rng = np.random.default_rng(0)
    pairs = []
    seen = 0
    try:
        for batch in dataloader:
            img = batch["img"].to(device).float() / 255
            gt = batch["depth"].to(device).float()
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)
            pred = model(img).float()
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)
            for pi, gi in zip(pred, gt):
                valid = (gi > 1e-3) & (gi < max_depth) & (pi > 1e-3) & torch.isfinite(pi)
                if not valid.any():
                    continue
                lp = torch.log(pi[valid]).detach().cpu().numpy()
                lg = torch.log(gi[valid]).detach().cpu().numpy()
                if lp.size > 20_000:
                    idx = rng.choice(lp.size, 20_000, replace=False)
                    lp, lg = lp[idx], lg[idx]
                pairs.append((lp, lg))
            seen += img.shape[0]
            if seen >= max_images:
                break
    finally:
        # A failed pass must not wipe the model's existing calibration; callers set the chosen value explicitly.
        head.cal_a.fill_(a0)
        head.cal_b.fill_(b0)
    return pairs
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.yolo.depth.calibrate.fit_calibration_selective` {#ultralytics.models.yolo.depth.calibrate.fit\_calibration\_selective}

```python
def fit_calibration_selective(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str,
    max_images: int = 200,
    margin: float = 0.002,
    max_depth: float = 100.0,
) -> dict[str, Any] | None
```

Select and apply calibration via "calibrate only if it helps" (see :func:`select_calibration_cv`).

Collects per-image ``(log_pred, log_gt)`` over the loader, splits images into independent fit/score folds (no leakage), chooses identity / scale-only by cross-validated raw-scale δ1, and writes the winner into the head's ``cal_a``/``cal_b``.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `dataloader` |  |  | *required* |
| `device` | `torch.device \| str` |  | *required* |
| `max_images` | `int` |  | `200` |
| `margin` | `float` |  | `0.002` |
| `max_depth` | `float` |  | `100.0` |

**Returns**

| Type | Description |
| --- | --- |
| `dict \| None` | The :func:`select_calibration_cv` result dict, or None if no Depth head / too few images. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L191-L226">View on GitHub</a>
```python
def fit_calibration_selective(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str,
    max_images: int = 200,
    margin: float = 0.002,
    max_depth: float = 100.0,
) -> dict[str, Any] | None:
    """Select and apply calibration via "calibrate only if it helps" (see :func:`select_calibration_cv`).

    Collects per-image ``(log_pred, log_gt)`` over the loader, splits images into independent fit/score folds (no
    leakage), chooses identity / scale-only by cross-validated raw-scale δ1, and writes the winner into the head's
    ``cal_a``/``cal_b``.

    Returns:
        (dict | None): The :func:`select_calibration_cv` result dict, or None if no Depth head / too few images.
    """
    head = _depth_head(model)
    if head is None:
        LOGGER.warning("calibrate: no Depth head with cal buffers found; skipping.")
        return None
    pairs = _collect_logpairs(model, dataloader, device, max_images, max_depth)
    if len(pairs) < 2:
        LOGGER.warning("calibrate: fewer than 2 valid images for fit/score split; calibration skipped.")
        return None
    res = select_calibration_cv(pairs, margin=margin)
    res["images"] = len(pairs)
    # On CUDA the buffers are inference tensors (the device move ran under inference_mode); reassign instead
    # of an in-place fill_ so the write is legal and the buffers stay normal/saveable for model.save().
    head.cal_a = torch.full_like(head.cal_a, res["a"])
    head.cal_b = torch.full_like(head.cal_b, res["b"])
    scores = " ".join(f"{n}={v:.4f}" for n, v in res["cv_scores"].items())
    LOGGER.info(
        f"Depth calibration selected '{res['name']}' (a={res['a']:.4f} b={res['b']:.4f}); CV held-out δ1 {scores}"
    )
    return res
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.yolo.depth.calibrate._plot_calibrated_batches` {#ultralytics.models.yolo.depth.calibrate.\_plot\_calibrated\_batches}

```python
def _plot_calibrated_batches(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str,
    a: float,
    b: float,
    name: str,
    plot_dir: str | Path,
    max_batches: int = 3,
    max_images: int = 4,
) -> None
```

Write ``val_batch{ni}_calibrated.jpg`` panels (RGB | GT | raw | calibrated) to ``plot_dir``.

Runs the model with calibration buffers at identity to get the raw prediction; the calibrated column is its deterministic affine ``exp(a·log(raw) + b)`` — no second forward. The first ``max_batches`` batches are the same ones BaseValidator plots as ``val_batch{ni}.jpg`` (val loaders are not shuffled), so the files are directly comparable. With the "only if it helps" policy the selected ``name`` may be ``identity``; the panels are still written (raw == calibrated), which documents that calibration was a no-op. Buffers are restored afterwards.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `dataloader` |  |  | *required* |
| `device` | `torch.device \| str` |  | *required* |
| `a` | `float` |  | *required* |
| `b` | `float` |  | *required* |
| `name` | `str` |  | *required* |
| `plot_dir` | `str \| Path` |  | *required* |
| `max_batches` | `int` |  | `3` |
| `max_images` | `int` |  | `4` |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L229-L276">View on GitHub</a>
```python
def _plot_calibrated_batches(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str,
    a: float,
    b: float,
    name: str,
    plot_dir: str | Path,
    max_batches: int = 3,
    max_images: int = 4,
) -> None:
    """Write ``val_batch{ni}_calibrated.jpg`` panels (RGB | GT | raw | calibrated) to ``plot_dir``.

    Runs the model with calibration buffers at identity to get the raw prediction; the calibrated column is its
    deterministic affine ``exp(a·log(raw) + b)`` — no second forward. The first ``max_batches`` batches are the same
    ones BaseValidator plots as ``val_batch{ni}.jpg`` (val loaders are not shuffled), so the files are directly
    comparable. With the "only if it helps" policy the selected ``name`` may be ``identity``; the panels are still
    written (raw == calibrated), which documents that calibration was a no-op. Buffers are restored afterwards.
    """
    from ultralytics.utils.plotting import plot_depth_panels

    head = _depth_head(model)
    a0, b0 = float(head.cal_a), float(head.cal_b)
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)
    model = model.to(device).eval()
    titles = ["RGB", "GT", "raw", f"calibrated ({name} x{np.exp(b):.2f})"]
    plot_dir = Path(plot_dir)
    _rewind(dataloader)
    with torch.no_grad():
        # zip stops on range exhaustion without pulling an extra batch from the stateful iterator
        for ni, batch in zip(range(max_batches), dataloader):
            img = batch["img"].to(device).float() / 255
            gt = batch["depth"].to(device).float()
            raw = model(img).float()
            if raw.ndim == 3:
                raw = raw.unsqueeze(1)
            cal = torch.exp(a * torch.log(raw.clamp(min=1e-3)) + b)
            plot_depth_panels(
                img,
                [raw, cal],
                plot_dir / f"val_batch{ni}_calibrated.jpg",
                gt=gt,
                titles=titles,
                max_images=max_images,
            )
    head.cal_a.fill_(a0)
    head.cal_b.fill_(b0)
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.yolo.depth.calibrate.calibrate_checkpoint` {#ultralytics.models.yolo.depth.calibrate.calibrate\_checkpoint}

```python
def calibrate_checkpoint(
    ckpt_path: str | Path,
    dataloader,
    device: torch.device | str,
    plot_dir: str | Path | None = None,
    *,
    dataset_hash: str | None = None,
    validation_split: str | None = None,
    max_depth: float = 100.0,
) -> dict | None
```

Fit calibration for a saved checkpoint in place (used by automatic post-training calibration).

Loads the checkpoint, selects calibration with the "calibrate only if it helps" policy (:func:`fit_calibration_selective`) on ``dataloader`` using a float copy on ``device``, writes the chosen buffers into the stored model, and re-saves — preserving the rest of the checkpoint.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `ckpt_path` | `str \| Path` | Path to the ``.pt`` checkpoint file to calibrate in place. | *required* |
| `dataloader` | `object` | Yields batches with ``img`` (uint8, Bx3xHxW) and ``depth`` (BxHxW meters). | *required* |
| `device` | `str \| torch.device` | Torch device to run inference on. | *required* |
| `plot_dir` | `str \| Path, optional` | If set, also write ``val_batch{ni}_calibrated.jpg`` comparison panels (RGB \| GT<br>\| raw \| calibrated) for the first val batches into this directory. | `None` |
| `dataset_hash` | `str, optional` | Immutable dataset manifest identity used for calibration. | `None` |
| `validation_split` | `str, optional` | Dataset-root-relative split used to collect calibration images. | `None` |
| `max_depth` | `float` | Maximum valid GT depth in meters; pixels beyond it are excluded from the fit and the held-out δ1 scoring, matching the val metrics' Eigen protocol. | `100.0` |

<details>
<summary>Source code in <code>ultralytics/models/yolo/depth/calibrate.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/depth/calibrate.py#L279-L348">View on GitHub</a>
```python
def calibrate_checkpoint(
    ckpt_path: str | Path,
    dataloader,
    device: torch.device | str,
    plot_dir: str | Path | None = None,
    *,
    dataset_hash: str | None = None,
    validation_split: str | None = None,
    max_depth: float = 100.0,
) -> dict | None:
    """Fit calibration for a saved checkpoint in place (used by automatic post-training calibration).

    Loads the checkpoint, selects calibration with the "calibrate only if it helps" policy
    (:func:`fit_calibration_selective`) on ``dataloader`` using a float copy on ``device``, writes the chosen buffers
    into the stored model, and re-saves — preserving the rest of the checkpoint.

    Args:
        ckpt_path (str | Path): Path to the ``.pt`` checkpoint file to calibrate in place.
        dataloader (object): Yields batches with ``img`` (uint8, Bx3xHxW) and ``depth`` (BxHxW meters).
        device (str | torch.device): Torch device to run inference on.
        plot_dir (str | Path, optional): If set, also write ``val_batch{ni}_calibrated.jpg`` comparison panels (RGB | GT
            | raw | calibrated) for the first val batches into this directory.
        dataset_hash (str, optional): Immutable dataset manifest identity used for calibration.
        validation_split (str, optional): Dataset-root-relative split used to collect calibration images.
        max_depth (float): Maximum valid GT depth in meters; pixels beyond it are excluded from the fit and the held-out
            δ1 scoring, matching the val metrics' Eigen protocol.
    """
    from copy import deepcopy

    from ultralytics.utils.patches import torch_load

    ckpt = torch_load(ckpt_path, map_location="cpu")
    saved = ckpt.get("ema") or ckpt.get("model")
    if saved is None or _depth_head(saved) is None:
        return
    work = deepcopy(saved).float()
    res = fit_calibration_selective(work, dataloader, device, max_depth=max_depth)
    if res is None:
        return None
    a, b = res["a"], res["b"]
    for key in ("ema", "model"):
        m = ckpt.get(key)
        if m is not None and _depth_head(m) is not None:
            _depth_head(m).cal_a.fill_(a)
            _depth_head(m).cal_b.fill_(b)
    stored_head = _depth_head(ckpt.get("ema") or ckpt.get("model"))
    a, b = float(stored_head.cal_a), float(stored_head.cal_b)
    provenance = {
        "status": "selected",
        "candidate": res["name"],
        "images": res["images"],
        "a": a,
        "b": b,
        "dataset_hash": dataset_hash,
        "validation_split": validation_split,
        "strategy": "two-fold-held-out-delta1",
        "scores": res["cv_scores"],
    }
    ckpt["depth_calibration"] = provenance
    torch.save(ckpt, ckpt_path)
    LOGGER.info(
        f"Auto-calibration written to {getattr(ckpt_path, 'name', ckpt_path)}: '{res['name']}' a={a:.4f} b={b:.4f}"
    )
    if plot_dir is not None:
        try:
            _plot_calibrated_batches(work, dataloader, device, a, b, res["name"], plot_dir)
            LOGGER.info(f"Calibrated val_batch plots written to {plot_dir}")
        except Exception as e:
            LOGGER.warning(f"Calibrated val plots skipped ({type(e).__name__}: {e})")
    return provenance
```
</details>

<br><br>
