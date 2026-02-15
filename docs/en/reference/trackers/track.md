---
description: Explore the track.py script for Ultralytics object tracking. Learn how on_predict_start, on_predict_postprocess_end, and register_tracker functions work.
keywords: Ultralytics, YOLO, object tracking, track.py, on_predict_start, on_predict_postprocess_end, register_tracker
---

# Reference for `ultralytics/trackers/track.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/track.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/track.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`on_predict_start`](#ultralytics.trackers.track.on_predict_start)
        - [`on_predict_postprocess_end`](#ultralytics.trackers.track.on_predict_postprocess_end)
        - [`register_tracker`](#ultralytics.trackers.track.register_tracker)


## Function `ultralytics.trackers.track.on_predict_start` {#ultralytics.trackers.track.on\_predict\_start}

```python
def on_predict_start(predictor: object, persist: bool = False) -> None
```

Initialize trackers for object tracking during prediction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predictor` | `ultralytics.engine.predictor.BasePredictor` | The predictor object to initialize trackers for. | *required* |
| `persist` | `bool, optional` | Whether to persist the trackers if they already exist. | `False` |

**Examples**

```python
Initialize trackers for a predictor object
>>> predictor = SomePredictorClass()
>>> on_predict_start(predictor, persist=True)
```

<details>
<summary>Source code in <code>ultralytics/trackers/track.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/track.py#L18-L68"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_predict_start(predictor: object, persist: bool = False) -> None:
    """Initialize trackers for object tracking during prediction.

    Args:
        predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Initialize trackers for a predictor object
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if predictor.args.task == "classify":
        raise ValueError("❌ Classification doesn't support 'mode=track'")

    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    predictor._feats = None  # reset in case used earlier
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()
    if cfg.tracker_type == "botsort" and cfg.with_reid and cfg.model == "auto":
        from ultralytics.nn.modules.head import Detect

        if not (
            isinstance(predictor.model.model, torch.nn.Module)
            and isinstance(predictor.model.model.model[-1], Detect)
            and not predictor.model.model.model[-1].end2end
        ):
            cfg.model = "yolo26n-cls.pt"
        else:
            # Register hook to extract input of Detect layer
            def pre_hook(module, input):
                predictor._feats = list(input[0])  # unroll to new list to avoid mutation in forward

            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video
```
</details>


<br><br><hr><br>

## Function `ultralytics.trackers.track.on_predict_postprocess_end` {#ultralytics.trackers.track.on\_predict\_postprocess\_end}

```python
def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None
```

Postprocess detected boxes and update with object tracking.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `predictor` | `object` | The predictor object containing the predictions. | *required* |
| `persist` | `bool, optional` | Whether to persist the trackers if they already exist. | `False` |

**Examples**

```python
Postprocess predictions and update with tracking
>>> predictor = YourPredictorClass()
>>> on_predict_postprocess_end(predictor, persist=True)
```

<details>
<summary>Source code in <code>ultralytics/trackers/track.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/track.py#L71-L100"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(result.path).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (result.obb if is_obb else result.boxes).cpu().numpy()
        tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None))
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = result[idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)
```
</details>


<br><br><hr><br>

## Function `ultralytics.trackers.track.register_tracker` {#ultralytics.trackers.track.register\_tracker}

```python
def register_tracker(model: object, persist: bool) -> None
```

Register tracking callbacks to the model for object tracking during prediction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `object` | The model object to register tracking callbacks for. | *required* |
| `persist` | `bool` | Whether to persist the trackers if they already exist. | *required* |

**Examples**

```python
Register tracking callbacks to a YOLO model
>>> model = YOLOModel()
>>> register_tracker(model, persist=True)
```

<details>
<summary>Source code in <code>ultralytics/trackers/track.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/track.py#L103-L116"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def register_tracker(model: object, persist: bool) -> None:
    """Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
```
</details>

<br><br>
