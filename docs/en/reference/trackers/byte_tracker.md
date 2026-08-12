---
title: trackers.byte_tracker API Reference
description: Explore the BYTETracker module in Ultralytics for state-of-the-art object tracking using Kalman filtering. Learn about its classes, methods, and attributes.
keywords: Ultralytics, BYTETracker, object tracking, Kalman filter, YOLOv8, documentation
---

# Reference for `ultralytics/trackers/byte_tracker.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`STrack`](#ultralytics.trackers.byte_tracker.STrack)
        - [`BYTETracker`](#ultralytics.trackers.byte_tracker.BYTETracker)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`STrack.tlwh`](#ultralytics.trackers.byte_tracker.STrack.tlwh)
        - [`STrack.xyxy`](#ultralytics.trackers.byte_tracker.STrack.xyxy)
        - [`STrack.xywh`](#ultralytics.trackers.byte_tracker.STrack.xywh)
        - [`STrack.xywha`](#ultralytics.trackers.byte_tracker.STrack.xywha)
        - [`STrack.result`](#ultralytics.trackers.byte_tracker.STrack.result)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`STrack.predict`](#ultralytics.trackers.byte_tracker.STrack.predict)
        - [`STrack.multi_predict`](#ultralytics.trackers.byte_tracker.STrack.multi_predict)
        - [`STrack.activate`](#ultralytics.trackers.byte_tracker.STrack.activate)
        - [`STrack.re_activate`](#ultralytics.trackers.byte_tracker.STrack.re_activate)
        - [`STrack.update`](#ultralytics.trackers.byte_tracker.STrack.update)
        - [`STrack.convert_coords`](#ultralytics.trackers.byte_tracker.STrack.convert_coords)
        - [`STrack.tlwh_to_xyah`](#ultralytics.trackers.byte_tracker.STrack.tlwh_to_xyah)
        - [`STrack.__repr__`](#ultralytics.trackers.byte_tracker.STrack.__repr__)
        - [`BYTETracker.update`](#ultralytics.trackers.byte_tracker.BYTETracker.update)
        - [`BYTETracker._split_detections`](#ultralytics.trackers.byte_tracker.BYTETracker._split_detections)
        - [`BYTETracker._input_for`](#ultralytics.trackers.byte_tracker.BYTETracker._input_for)
        - [`BYTETracker._split_tracked`](#ultralytics.trackers.byte_tracker.BYTETracker._split_tracked)
        - [`BYTETracker._pre_first_associate`](#ultralytics.trackers.byte_tracker.BYTETracker._pre_first_associate)
        - [`BYTETracker._first_association`](#ultralytics.trackers.byte_tracker.BYTETracker._first_association)
        - [`BYTETracker._post_first_association`](#ultralytics.trackers.byte_tracker.BYTETracker._post_first_association)
        - [`BYTETracker._apply_matches`](#ultralytics.trackers.byte_tracker.BYTETracker._apply_matches)
        - [`BYTETracker._apply_match`](#ultralytics.trackers.byte_tracker.BYTETracker._apply_match)
        - [`BYTETracker._second_association`](#ultralytics.trackers.byte_tracker.BYTETracker._second_association)
        - [`BYTETracker._unconfirmed_association`](#ultralytics.trackers.byte_tracker.BYTETracker._unconfirmed_association)
        - [`BYTETracker._init_new_tracks`](#ultralytics.trackers.byte_tracker.BYTETracker._init_new_tracks)
        - [`BYTETracker._remove_stale_lost`](#ultralytics.trackers.byte_tracker.BYTETracker._remove_stale_lost)
        - [`BYTETracker._format_output`](#ultralytics.trackers.byte_tracker.BYTETracker._format_output)
        - [`BYTETracker.get_kalmanfilter`](#ultralytics.trackers.byte_tracker.BYTETracker.get_kalmanfilter)
        - [`BYTETracker.init_track`](#ultralytics.trackers.byte_tracker.BYTETracker.init_track)
        - [`BYTETracker.get_dists`](#ultralytics.trackers.byte_tracker.BYTETracker.get_dists)
        - [`BYTETracker.multi_predict`](#ultralytics.trackers.byte_tracker.BYTETracker.multi_predict)
        - [`BYTETracker.reset_id`](#ultralytics.trackers.byte_tracker.BYTETracker.reset_id)
        - [`BYTETracker.reset`](#ultralytics.trackers.byte_tracker.BYTETracker.reset)


## Class `ultralytics.trackers.byte_tracker.STrack` {#ultralytics.trackers.byte\_tracker.STrack}

```python
STrack(xywh: np.ndarray, score: float, cls: Any)
```

**Bases:** `BaseTrack`

Single object tracking representation that uses Kalman filtering for state estimation.

This class is responsible for storing all the information regarding individual tracklets and performs state updates and predictions based on Kalman filter.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `xywh` | `np.ndarray` | Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is the center, (w, h) are width and height, and `idx` is the detection index. | *required* |
| `score` | `float` | Confidence score of the detection. | *required* |
| `cls` | `Any` | Class label for the detected object. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `shared_kalman` | `KalmanFilterXYAH` | Shared Kalman filter used across all STrack instances for prediction. |
| `_tlwh` | `np.ndarray` | Private attribute to store top-left corner coordinates and width and height of bounding box. |
| `kalman_filter` | `KalmanFilterXYAH` | Instance of Kalman filter used for this particular object track. |
| `mean` | `np.ndarray` | Mean state estimate vector. |
| `covariance` | `np.ndarray` | Covariance of state estimate. |
| `is_activated` | `bool` | Boolean flag indicating if the track has been activated. |
| `score` | `float` | Confidence score of the track. |
| `tracklet_len` | `int` | Length of the tracklet. |
| `cls` | `Any` | Class label for the object. |
| `idx` | `int` | Index or identifier for the object. |
| `frame_id` | `int` | Current frame ID. |
| `start_frame` | `int` | Frame where the object was first detected. |
| `angle` | `float \| None` | Optional angle information for oriented bounding boxes. |

**Methods**

| Name | Description |
| --- | --- |
| [`tlwh`](#ultralytics.trackers.byte_tracker.STrack.tlwh) | Get the bounding box in top-left-width-height format from the current state estimate. |
| [`xyxy`](#ultralytics.trackers.byte_tracker.STrack.xyxy) | Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format. |
| [`xywh`](#ultralytics.trackers.byte_tracker.STrack.xywh) | Get the current position of the bounding box in (center x, center y, width, height) format. |
| [`xywha`](#ultralytics.trackers.byte_tracker.STrack.xywha) | Get position in (center x, center y, width, height, angle) format, warning if angle is missing. |
| [`result`](#ultralytics.trackers.byte_tracker.STrack.result) | Get the current tracking results in the appropriate bounding box format. |
| [`__repr__`](#ultralytics.trackers.byte_tracker.STrack.__repr__) | Return a string representation of the STrack object including start frame, end frame, and track ID. |
| [`activate`](#ultralytics.trackers.byte_tracker.STrack.activate) | Activate a new tracklet using the provided Kalman filter and initialize its state and covariance. |
| [`convert_coords`](#ultralytics.trackers.byte_tracker.STrack.convert_coords) | Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent. |
| [`multi_predict`](#ultralytics.trackers.byte_tracker.STrack.multi_predict) | Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances. |
| [`predict`](#ultralytics.trackers.byte_tracker.STrack.predict) | Predict the next state (mean and covariance) of the object using the Kalman filter. |
| [`re_activate`](#ultralytics.trackers.byte_tracker.STrack.re_activate) | Reactivate a previously lost track using new detection data and update its state and attributes. |
| [`tlwh_to_xyah`](#ultralytics.trackers.byte_tracker.STrack.tlwh_to_xyah) | Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format. |
| [`update`](#ultralytics.trackers.byte_tracker.STrack.update) | Update the state of a matched track. |

**Examples**

Initialize and activate a new track

```python
>>> track = STrack(xywh=np.array([100, 200, 50, 80, 0]), score=0.9, cls="person")
>>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
```

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L17-L210">View on GitHub</a>
```python
class STrack(BaseTrack):
    """Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.
        angle (float | None): Optional angle information for oriented bounding boxes.

    Methods:
        predict: Predict the next state of the object using Kalman filter.
        multi_predict: Predict the next states for multiple tracks.
        activate: Activate a new tracklet.
        re_activate: Reactivate a previously lost tracklet.
        update: Update the state of a matched track.
        convert_coords: Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah: Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=np.array([100, 200, 50, 80, 0]), score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: np.ndarray, score: float, cls: Any):
        """Initialize a new STrack instance.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is
                the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.
        """
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None
```
</details>

<br>

### Property `ultralytics.trackers.byte_tracker.STrack.tlwh` {#ultralytics.trackers.byte\_tracker.STrack.tlwh}

```python
def tlwh(self) -> np.ndarray
```

Get the bounding box in top-left-width-height format from the current state estimate.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L163-L170">View on GitHub</a>
```python
@property
def tlwh(self) -> np.ndarray:
    """Get the bounding box in top-left-width-height format from the current state estimate."""
    if self.mean is None:
        return self._tlwh.copy()
    ret = self.mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret
```
</details>

<br>

### Property `ultralytics.trackers.byte_tracker.STrack.xyxy` {#ultralytics.trackers.byte\_tracker.STrack.xyxy}

```python
def xyxy(self) -> np.ndarray
```

Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L173-L177">View on GitHub</a>
```python
@property
def xyxy(self) -> np.ndarray:
    """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
    ret = self.tlwh  # already a fresh array, safe to mutate
    ret[2:] += ret[:2]
    return ret
```
</details>

<br>

### Property `ultralytics.trackers.byte_tracker.STrack.xywh` {#ultralytics.trackers.byte\_tracker.STrack.xywh}

```python
def xywh(self) -> np.ndarray
```

Get the current position of the bounding box in (center x, center y, width, height) format.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L188-L192">View on GitHub</a>
```python
@property
def xywh(self) -> np.ndarray:
    """Get the current position of the bounding box in (center x, center y, width, height) format."""
    ret = np.asarray(self.tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret
```
</details>

<br>

### Property `ultralytics.trackers.byte_tracker.STrack.xywha` {#ultralytics.trackers.byte\_tracker.STrack.xywha}

```python
def xywha(self) -> np.ndarray
```

Get position in (center x, center y, width, height, angle) format, warning if angle is missing.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L195-L200">View on GitHub</a>
```python
@property
def xywha(self) -> np.ndarray:
    """Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
    if self.angle is None:
        LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
        return self.xywh
    return np.concatenate([self.xywh, self.angle[None]])
```
</details>

<br>

### Property `ultralytics.trackers.byte_tracker.STrack.result` {#ultralytics.trackers.byte\_tracker.STrack.result}

```python
def result(self) -> list[float]
```

Get the current tracking results in the appropriate bounding box format.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L203-L206">View on GitHub</a>
```python
@property
def result(self) -> list[float]:
    """Get the current tracking results in the appropriate bounding box format."""
    coords = self.xyxy if self.angle is None else self.xywha
    return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.__repr__` {#ultralytics.trackers.byte\_tracker.STrack.\_\_repr\_\_}

```python
def __repr__(self) -> str
```

Return a string representation of the STrack object including start frame, end frame, and track ID.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L208-L210">View on GitHub</a>
```python
def __repr__(self) -> str:
    """Return a string representation of the STrack object including start frame, end frame, and track ID."""
    return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.activate` {#ultralytics.trackers.byte\_tracker.STrack.activate}

```python
def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int)
```

Activate a new tracklet using the provided Kalman filter and initialize its state and covariance.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `kalman_filter` | `KalmanFilterXYAH` |  | *required* |
| `frame_id` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L100-L111">View on GitHub</a>
```python
def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
    """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
    self.kalman_filter = kalman_filter
    self.track_id = self.next_id()
    self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

    self.tracklet_len = 0
    self.state = TrackState.Tracked
    if frame_id == 1:
        self.is_activated = True
    self.frame_id = frame_id
    self.start_frame = frame_id
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.convert_coords` {#ultralytics.trackers.byte\_tracker.STrack.convert\_coords}

```python
def convert_coords(self, tlwh: np.ndarray) -> np.ndarray
```

Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tlwh` | `np.ndarray` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L158-L160">View on GitHub</a>
```python
def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
    """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
    return self.tlwh_to_xyah(tlwh)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.multi_predict` {#ultralytics.trackers.byte\_tracker.STrack.multi\_predict}

```python
def multi_predict(stracks: list[STrack])
```

Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stracks` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L86-L98">View on GitHub</a>
```python
@staticmethod
def multi_predict(stracks: list[STrack]):
    """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
    if not stracks:
        return
    multi_mean = np.asarray([st.mean.copy() for st in stracks])
    multi_covariance = np.asarray([st.covariance for st in stracks])
    for i, st in enumerate(stracks):
        if st.state != TrackState.Tracked:
            multi_mean[i][7] = 0
    multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        stracks[i].mean = mean
        stracks[i].covariance = cov
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.predict` {#ultralytics.trackers.byte\_tracker.STrack.predict}

```python
def predict(self)
```

Predict the next state (mean and covariance) of the object using the Kalman filter.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L78-L83">View on GitHub</a>
```python
def predict(self):
    """Predict the next state (mean and covariance) of the object using the Kalman filter."""
    mean_state = self.mean.copy()
    if self.state != TrackState.Tracked:
        mean_state[7] = 0
    self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.re_activate` {#ultralytics.trackers.byte\_tracker.STrack.re\_activate}

```python
def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False)
```

Reactivate a previously lost track using new detection data and update its state and attributes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_track` | `STrack` |  | *required* |
| `frame_id` | `int` |  | *required* |
| `new_id` | `bool` |  | `False` |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L113-L127">View on GitHub</a>
```python
def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
    """Reactivate a previously lost track using new detection data and update its state and attributes."""
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_track.tlwh)
    )
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_id = frame_id
    if new_id:
        self.track_id = self.next_id()
    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.tlwh_to_xyah` {#ultralytics.trackers.byte\_tracker.STrack.tlwh\_to\_xyah}

```python
def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray
```

Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tlwh` | `np.ndarray` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L180-L185">View on GitHub</a>
```python
@staticmethod
def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.update` {#ultralytics.trackers.byte\_tracker.STrack.update}

```python
def update(self, new_track: STrack, frame_id: int)
```

Update the state of a matched track.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_track` | `STrack` | The new track containing updated information. | *required* |
| `frame_id` | `int` | The ID of the current frame. | *required* |

**Examples**

Update the state of a track with new detection information

```python
>>> track = STrack(np.array([100, 200, 50, 80, 0]), score=0.9, cls=0)
>>> track.activate(KalmanFilterXYAH(), 1)
>>> new_track = STrack(np.array([105, 205, 55, 85, 0]), score=0.95, cls=0)
>>> track.update(new_track, 2)
```

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L129-L156">View on GitHub</a>
```python
def update(self, new_track: STrack, frame_id: int):
    """Update the state of a matched track.

    Args:
        new_track (STrack): The new track containing updated information.
        frame_id (int): The ID of the current frame.

    Examples:
        Update the state of a track with new detection information
        >>> track = STrack(np.array([100, 200, 50, 80, 0]), score=0.9, cls=0)
        >>> track.activate(KalmanFilterXYAH(), 1)
        >>> new_track = STrack(np.array([105, 205, 55, 85, 0]), score=0.95, cls=0)
        >>> track.update(new_track, 2)
    """
    self.frame_id = frame_id
    self.tracklet_len += 1

    new_tlwh = new_track.tlwh
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_tlwh)
    )
    self.state = TrackState.Tracked
    self.is_activated = True

    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx
```
</details>


<br><br><hr><br>

## Class `ultralytics.trackers.byte_tracker.BYTETracker` {#ultralytics.trackers.byte\_tracker.BYTETracker}

```python
BYTETracker(args)
```

BYTETracker: A tracking algorithm built on top of YOLO for object detection and tracking.

This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for predicting the new object locations, and performs data association.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `Namespace` | Command-line arguments containing tracking parameters. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `tracked_stracks` | `list[STrack]` | List of successfully activated tracks. |
| `lost_stracks` | `list[STrack]` | List of lost tracks. |
| `removed_stracks` | `list[STrack]` | List of removed tracks. |
| `frame_id` | `int` | The current frame ID. |
| `args` | `Namespace` | Command-line arguments. |
| `max_frames_lost` | `int` | The maximum frames for a track to be considered as 'lost'. |
| `kalman_filter` | `KalmanFilterXYAH` | Kalman Filter object. |

**Methods**

| Name | Description |
| --- | --- |
| [`_apply_match`](#ultralytics.trackers.byte_tracker.BYTETracker._apply_match) | Update or re-activate a single track with its matched detection. |
| [`_apply_matches`](#ultralytics.trackers.byte_tracker.BYTETracker._apply_matches) | Apply a list of matched (track, detection) pairs from an association stage. |
| [`_first_association`](#ultralytics.trackers.byte_tracker.BYTETracker._first_association) | First-stage association between track pool and high-score detections. |
| [`_format_output`](#ultralytics.trackers.byte_tracker.BYTETracker._format_output) | Format the current tracked objects into the output array. |
| [`_init_new_tracks`](#ultralytics.trackers.byte_tracker.BYTETracker._init_new_tracks) | Activate new tracks from detections that survived all association stages. |
| [`_input_for`](#ultralytics.trackers.byte_tracker.BYTETracker._input_for) | Return the per-detection auxiliary input for ``init_track``. |
| [`_post_first_association`](#ultralytics.trackers.byte_tracker.BYTETracker._post_first_association) | Hook executed after the first association stage and before the second. |
| [`_pre_first_associate`](#ultralytics.trackers.byte_tracker.BYTETracker._pre_first_associate) | Hook called after Kalman predict, before first-stage assignment. Default: GMC if available. |
| [`_remove_stale_lost`](#ultralytics.trackers.byte_tracker.BYTETracker._remove_stale_lost) | Remove lost tracks that have exceeded the maximum allowed frames. |
| [`_second_association`](#ultralytics.trackers.byte_tracker.BYTETracker._second_association) | Second-stage association between remaining tracked tracks and low-score detections. |
| [`_split_detections`](#ultralytics.trackers.byte_tracker.BYTETracker._split_detections) | Split detections into high-confidence and low-confidence subsets. |
| [`_split_tracked`](#ultralytics.trackers.byte_tracker.BYTETracker._split_tracked) | Separate ``self.tracked_stracks`` into confirmed and unconfirmed lists. |
| [`_unconfirmed_association`](#ultralytics.trackers.byte_tracker.BYTETracker._unconfirmed_association) | Associate unconfirmed tracks with leftover high-score detections. |
| [`get_dists`](#ultralytics.trackers.byte_tracker.BYTETracker.get_dists) | Calculate the distance between tracks and detections using IoU and optionally fuse scores. |
| [`get_kalmanfilter`](#ultralytics.trackers.byte_tracker.BYTETracker.get_kalmanfilter) | Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH. |
| [`init_track`](#ultralytics.trackers.byte_tracker.BYTETracker.init_track) | Initialize object tracking with given detections, scores, and class labels as STrack instances. |
| [`multi_predict`](#ultralytics.trackers.byte_tracker.BYTETracker.multi_predict) | Predict the next states for multiple tracks using Kalman filter. |
| [`reset`](#ultralytics.trackers.byte_tracker.BYTETracker.reset) | Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter. |
| [`reset_id`](#ultralytics.trackers.byte_tracker.BYTETracker.reset_id) | Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions. |
| [`update`](#ultralytics.trackers.byte_tracker.BYTETracker.update) | Update the tracker with new detections and return the current list of tracked objects. |

**Examples**

Initialize BYTETracker and update with detection results

```python
>>> tracker = BYTETracker(args)
>>> results = yolo_model.detect(image)
>>> tracked_objects = tracker.update(results)
```

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L213-L528">View on GitHub</a>
```python
class BYTETracker:
    """BYTETracker: A tracking algorithm built on top of YOLO for object detection and tracking.

    This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects
    in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman
    filtering for predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_frames_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.

    Methods:
        update: Update object tracker with new detections.
        get_kalmanfilter: Return a Kalman filter object for tracking bounding boxes.
        init_track: Initialize object tracking with detections.
        get_dists: Calculate the distance between tracks and detections.
        multi_predict: Predict the location of tracks.
        reset_id: Reset the ID counter of STrack.
        reset: Reset the tracker by clearing all tracks.
        joint_stracks: Combine two lists of stracks.
        sub_stracks: Filter out the stracks present in the second list from the first list.
        remove_duplicate_stracks: Remove duplicate stracks based on IoU.

    Examples:
        Initialize BYTETracker and update with detection results
        >>> tracker = BYTETracker(args)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    track_class = STrack

    def __init__(self, args):
        """Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
        """
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.args = args
        self.max_frames_lost = args.track_buffer
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._apply_match` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_apply\_match}

```python
def _apply_match(self, track: STrack, det: STrack, activated: list[STrack], refind: list[STrack]) -> None
```

Update or re-activate a single track with its matched detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `track` | `STrack` |  | *required* |
| `det` | `STrack` |  | *required* |
| `activated` | `list[STrack]` |  | *required* |
| `refind` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L406-L413">View on GitHub</a>
```python
def _apply_match(self, track: STrack, det: STrack, activated: list[STrack], refind: list[STrack]) -> None:
    """Update or re-activate a single track with its matched detection."""
    if track.state == TrackState.Tracked:
        track.update(det, self.frame_id)
        activated.append(track)
    else:
        track.re_activate(det, self.frame_id, new_id=False)
        refind.append(track)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._apply_matches` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_apply\_matches}

```python
def _apply_matches(
    self,
    matches: list[list[int]] | np.ndarray,
    pool: list[STrack],
    detections: list[STrack],
    activated: list[STrack],
    refind: list[STrack],
) -> None
```

Apply a list of matched (track, detection) pairs from an association stage.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `matches` | `list[list[int]] \| np.ndarray` |  | *required* |
| `pool` | `list[STrack]` |  | *required* |
| `detections` | `list[STrack]` |  | *required* |
| `activated` | `list[STrack]` |  | *required* |
| `refind` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L394-L404">View on GitHub</a>
```python
def _apply_matches(
    self,
    matches: list[list[int]] | np.ndarray,
    pool: list[STrack],
    detections: list[STrack],
    activated: list[STrack],
    refind: list[STrack],
) -> None:
    """Apply a list of matched (track, detection) pairs from an association stage."""
    for itracked, idet in matches:
        self._apply_match(pool[itracked], detections[idet], activated, refind)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._first_association` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_first\_association}

```python
def _first_association(
    self, strack_pool: list[STrack], detections: list[STrack], activated: list[STrack], refind: list[STrack]
) -> tuple[list[int], list[int]]
```

First-stage association between track pool and high-score detections.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `strack_pool` | `list[STrack]` |  | *required* |
| `detections` | `list[STrack]` |  | *required* |
| `activated` | `list[STrack]` |  | *required* |
| `refind` | `list[STrack]` |  | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `tuple[list[int], list[int]]` | Unmatched track indices and unmatched detection indices. |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L365-L376">View on GitHub</a>
```python
def _first_association(
    self, strack_pool: list[STrack], detections: list[STrack], activated: list[STrack], refind: list[STrack]
) -> tuple[list[int], list[int]]:
    """First-stage association between track pool and high-score detections.

    Returns:
        (tuple[list[int], list[int]]): Unmatched track indices and unmatched detection indices.
    """
    dists = self.get_dists(strack_pool, detections)
    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
    self._apply_matches(matches, strack_pool, detections, activated, refind)
    return u_track, u_detection
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._format_output` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_format\_output}

```python
def _format_output(self) -> np.ndarray
```

Format the current tracked objects into the output array.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L490-L492">View on GitHub</a>
```python
def _format_output(self) -> np.ndarray:
    """Format the current tracked objects into the output array."""
    return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._init_new_tracks` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_init\_new\_tracks}

```python
def _init_new_tracks(
    self,
    u_detection: list[int],
    detections: list[STrack],
    activated: list[STrack],
    refind: list[STrack] | None = None,
) -> None
```

Activate new tracks from detections that survived all association stages.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `u_detection` | `list[int]` |  | *required* |
| `detections` | `list[STrack]` |  | *required* |
| `activated` | `list[STrack]` |  | *required* |
| `refind` | `list[STrack] \| None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L468-L481">View on GitHub</a>
```python
def _init_new_tracks(
    self,
    u_detection: list[int],
    detections: list[STrack],
    activated: list[STrack],
    refind: list[STrack] | None = None,
) -> None:
    """Activate new tracks from detections that survived all association stages."""
    for inew in u_detection:
        track = detections[inew]
        if track.score < self.args.new_track_thresh:
            continue
        track.activate(self.kalman_filter, self.frame_id)
        activated.append(track)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._input_for` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_input\_for}

```python
def _input_for(self, img: np.ndarray | None, feats: np.ndarray | None, mask: np.ndarray) -> Any
```

Return the per-detection auxiliary input for ``init_track``.

When ``feats`` is provided it is sliced by the detection mask. Trackers with a native (``model="auto"``) ReID encoder get None when feats are missing (e.g. user-supplied detections), so ``init_track`` falls back to the no-encoding path instead of feeding the BGR frame into the auto encoder. External ReID models always take the frame.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img` | `np.ndarray \| None` | Current BGR frame. | *required* |
| `feats` | `np.ndarray \| None` | Optional per-detection features. | *required* |
| `mask` | `np.ndarray` | Boolean mask used to slice ``feats``. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Any` | The auxiliary payload (features, image or None) to hand to ``init_track``. |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L318-L338">View on GitHub</a>
```python
def _input_for(self, img: np.ndarray | None, feats: np.ndarray | None, mask: np.ndarray) -> Any:
    """Return the per-detection auxiliary input for ``init_track``.

    When ``feats`` is provided it is sliced by the detection mask. Trackers with a native
    (``model="auto"``) ReID encoder get None when feats are missing (e.g. user-supplied
    detections), so ``init_track`` falls back to the no-encoding path instead of feeding the
    BGR frame into the auto encoder. External ReID models always take the frame.

    Args:
        img (np.ndarray | None): Current BGR frame.
        feats (np.ndarray | None): Optional per-detection features.
        mask (np.ndarray): Boolean mask used to slice ``feats``.

    Returns:
        (Any): The auxiliary payload (features, image or None) to hand to ``init_track``.
    """
    if feats is not None and len(feats):
        return feats[mask]
    if getattr(self, "encoder", None) is not None and getattr(self.args, "model", "auto") == "auto":
        return None
    return img
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._post_first_association` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_post\_first\_association}

```python
def _post_first_association(
    self,
    strack_pool: list[STrack],
    detections: list[STrack],
    u_track: list[int],
    u_detection: list[int],
    activated: list[STrack],
    refind: list[STrack],
) -> tuple[list[int], list[int]]
```

Hook executed after the first association stage and before the second.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `strack_pool` | `list[STrack]` |  | *required* |
| `detections` | `list[STrack]` |  | *required* |
| `u_track` | `list[int]` |  | *required* |
| `u_detection` | `list[int]` |  | *required* |
| `activated` | `list[STrack]` |  | *required* |
| `refind` | `list[STrack]` |  | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `tuple[list[int], list[int]]` | Potentially modified unmatched track and detection indices. |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L378-L392">View on GitHub</a>
```python
def _post_first_association(
    self,
    strack_pool: list[STrack],
    detections: list[STrack],
    u_track: list[int],
    u_detection: list[int],
    activated: list[STrack],
    refind: list[STrack],
) -> tuple[list[int], list[int]]:
    """Hook executed after the first association stage and before the second.

    Returns:
        (tuple[list[int], list[int]]): Potentially modified unmatched track and detection indices.
    """
    return u_track, u_detection
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._pre_first_associate` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_pre\_first\_associate}

```python
def _pre_first_associate(
    self, strack_pool: list[STrack], unconfirmed: list[STrack], img: np.ndarray | None, results_high: Any
) -> None
```

Hook called after Kalman predict, before first-stage assignment. Default: GMC if available.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `strack_pool` | `list[STrack]` |  | *required* |
| `unconfirmed` | `list[STrack]` |  | *required* |
| `img` | `np.ndarray \| None` |  | *required* |
| `results_high` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L352-L363">View on GitHub</a>
```python
def _pre_first_associate(
    self, strack_pool: list[STrack], unconfirmed: list[STrack], img: np.ndarray | None, results_high: Any
) -> None:
    """Hook called after Kalman predict, before first-stage assignment. Default: GMC if available."""
    if hasattr(self, "gmc") and self.gmc.method is not None and img is not None:
        try:
            warp = self.gmc.apply(img, results_high.xyxy)
        except Exception as e:
            LOGGER.warning(f"GMC failed, falling back to identity: {e}")
            warp = np.eye(2, 3)
        multi_gmc(strack_pool, warp)
        multi_gmc(unconfirmed, warp)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._remove_stale_lost` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_remove\_stale\_lost}

```python
def _remove_stale_lost(self, removed: list[STrack]) -> None
```

Remove lost tracks that have exceeded the maximum allowed frames.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `removed` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L483-L488">View on GitHub</a>
```python
def _remove_stale_lost(self, removed: list[STrack]) -> None:
    """Remove lost tracks that have exceeded the maximum allowed frames."""
    for track in self.lost_stracks:
        if self.frame_id - track.end_frame > self.max_frames_lost:
            track.mark_removed()
            removed.append(track)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._second_association` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_second\_association}

```python
def _second_association(
    self,
    strack_pool: list[STrack],
    u_track: list[int],
    detections_second: list[STrack],
    activated: list[STrack],
    refind: list[STrack],
    lost: list[STrack],
) -> None
```

Second-stage association between remaining tracked tracks and low-score detections.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `strack_pool` | `list[STrack]` |  | *required* |
| `u_track` | `list[int]` |  | *required* |
| `detections_second` | `list[STrack]` |  | *required* |
| `activated` | `list[STrack]` |  | *required* |
| `refind` | `list[STrack]` |  | *required* |
| `lost` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L415-L438">View on GitHub</a>
```python
def _second_association(
    self,
    strack_pool: list[STrack],
    u_track: list[int],
    detections_second: list[STrack],
    activated: list[STrack],
    refind: list[STrack],
    lost: list[STrack],
) -> None:
    """Second-stage association between remaining tracked tracks and low-score detections."""
    r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
    if r_tracked_stracks and detections_second:
        # IoU-only by design (ByteTrack paper sec. 3.2): fusing low scores pushes costs above the 0.5 threshold
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        self._apply_matches(matches, r_tracked_stracks, detections_second, activated, refind)
    else:
        u_track = list(range(len(r_tracked_stracks)))

    for it in u_track:
        track = r_tracked_stracks[it]
        if track.state != TrackState.Lost:
            track.mark_lost()
            lost.append(track)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._split_detections` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_split\_detections}

```python
def _split_detections(self, results: Any) -> tuple[Any, Any, np.ndarray, np.ndarray]
```

Split detections into high-confidence and low-confidence subsets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` | `Any` | Results-like object with ``conf`` attribute supporting boolean indexing. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `tuple[Any, Any, np.ndarray, np.ndarray]` | High-confidence results, low-confidence results, high mask, and low mask. |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L302-L316">View on GitHub</a>
```python
def _split_detections(self, results: Any) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    """Split detections into high-confidence and low-confidence subsets.

    Args:
        results (Any): Results-like object with ``conf`` attribute supporting boolean indexing.

    Returns:
        (tuple[Any, Any, np.ndarray, np.ndarray]): High-confidence results, low-confidence results, high mask, and
            low mask.
    """
    scores = results.conf
    remain_inds = scores >= self.args.track_high_thresh
    inds_low = scores > self.args.track_low_thresh
    inds_below_high = scores < self.args.track_high_thresh
    return results[remain_inds], results[inds_low & inds_below_high], remain_inds, inds_low & inds_below_high
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._split_tracked` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_split\_tracked}

```python
def _split_tracked(self) -> tuple[list[STrack], list[STrack]]
```

Separate ``self.tracked_stracks`` into confirmed and unconfirmed lists.

**Returns**

| Type | Description |
| --- | --- |
| `tuple[list[STrack], list[STrack]]` | ``(unconfirmed, tracked)`` where ``unconfirmed`` holds tracks whose ``is_activated`` flag is False. |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L340-L350">View on GitHub</a>
```python
def _split_tracked(self) -> tuple[list[STrack], list[STrack]]:
    """Separate ``self.tracked_stracks`` into confirmed and unconfirmed lists.

    Returns:
        (tuple[list[STrack], list[STrack]]): ``(unconfirmed, tracked)`` where ``unconfirmed`` holds tracks whose
            ``is_activated`` flag is False.
    """
    unconfirmed, tracked = [], []
    for track in self.tracked_stracks:
        (unconfirmed if not track.is_activated else tracked).append(track)
    return unconfirmed, tracked
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker._unconfirmed_association` {#ultralytics.trackers.byte\_tracker.BYTETracker.\_unconfirmed\_association}

```python
def _unconfirmed_association(
    self,
    unconfirmed: list[STrack],
    u_detection: list[int],
    detections: list[STrack],
    activated: list[STrack],
    removed: list[STrack],
) -> tuple[list[int], list[STrack]]
```

Associate unconfirmed tracks with leftover high-score detections.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `unconfirmed` | `list[STrack]` |  | *required* |
| `u_detection` | `list[int]` |  | *required* |
| `detections` | `list[STrack]` |  | *required* |
| `activated` | `list[STrack]` |  | *required* |
| `removed` | `list[STrack]` |  | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `tuple[list[int], list[STrack]]` | Unmatched detection indices after association, and the filtered detection list those indices refer to. |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L440-L466">View on GitHub</a>
```python
def _unconfirmed_association(
    self,
    unconfirmed: list[STrack],
    u_detection: list[int],
    detections: list[STrack],
    activated: list[STrack],
    removed: list[STrack],
) -> tuple[list[int], list[STrack]]:
    """Associate unconfirmed tracks with leftover high-score detections.

    Returns:
        (tuple[list[int], list[STrack]]): Unmatched detection indices after association, and the filtered detection
            list those indices refer to.
    """
    detections = [detections[i] for i in u_detection]
    if not unconfirmed:
        return list(range(len(detections))), detections
    dists = self.get_dists(unconfirmed, detections)
    matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    for itracked, idet in matches:
        unconfirmed[itracked].update(detections[idet], self.frame_id)
        activated.append(unconfirmed[itracked])
    for it in u_unconfirmed:
        track = unconfirmed[it]
        track.mark_removed()
        removed.append(track)
    return u_detection, detections
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.get_dists` {#ultralytics.trackers.byte\_tracker.BYTETracker.get\_dists}

```python
def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray
```

Calculate the distance between tracks and detections using IoU and optionally fuse scores.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracks` | `list[STrack]` |  | *required* |
| `detections` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L505-L510">View on GitHub</a>
```python
def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
    """Calculate the distance between tracks and detections using IoU and optionally fuse scores."""
    dists = matching.iou_distance(tracks, detections)
    if self.args.fuse_score:
        dists = matching.fuse_score(dists, detections)
    return dists
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.get_kalmanfilter` {#ultralytics.trackers.byte\_tracker.BYTETracker.get\_kalmanfilter}

```python
def get_kalmanfilter(self) -> KalmanFilterXYAH
```

Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L494-L496">View on GitHub</a>
```python
def get_kalmanfilter(self) -> KalmanFilterXYAH:
    """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
    return KalmanFilterXYAH()
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.init_track` {#ultralytics.trackers.byte\_tracker.BYTETracker.init\_track}

```python
def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]
```

Initialize object tracking with given detections, scores, and class labels as STrack instances.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` |  |  | *required* |
| `img` | `np.ndarray \| None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L498-L503">View on GitHub</a>
```python
def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
    """Initialize object tracking with given detections, scores, and class labels as STrack instances."""
    if len(results) == 0:
        return []
    bboxes = parse_bboxes(results)
    return [self.track_class(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.multi_predict` {#ultralytics.trackers.byte\_tracker.BYTETracker.multi\_predict}

```python
def multi_predict(self, tracks: list[STrack])
```

Predict the next states for multiple tracks using Kalman filter.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracks` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L512-L514">View on GitHub</a>
```python
def multi_predict(self, tracks: list[STrack]):
    """Predict the next states for multiple tracks using Kalman filter."""
    STrack.multi_predict(tracks)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.reset` {#ultralytics.trackers.byte\_tracker.BYTETracker.reset}

```python
def reset(self)
```

Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L521-L528">View on GitHub</a>
```python
def reset(self):
    """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
    self.tracked_stracks = []
    self.lost_stracks = []
    self.removed_stracks = []
    self.frame_id = 0
    self.kalman_filter = self.get_kalmanfilter()
    self.reset_id()
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.reset_id` {#ultralytics.trackers.byte\_tracker.BYTETracker.reset\_id}

```python
def reset_id()
```

Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions.

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L517-L519">View on GitHub</a>
```python
@staticmethod
def reset_id():
    """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
    STrack.reset_id()
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.update` {#ultralytics.trackers.byte\_tracker.BYTETracker.update}

```python
def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, **kwargs) -> np.ndarray
```

Update the tracker with new detections and return the current list of tracked objects.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` |  |  | *required* |
| `img` | `np.ndarray \| None` |  | `None` |
| `feats` | `np.ndarray \| None` |  | `None` |
| `**kwargs` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L266-L300">View on GitHub</a>
```python
def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, **kwargs) -> np.ndarray:
    """Update the tracker with new detections and return the current list of tracked objects."""
    self.frame_id += 1
    activated_stracks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []

    results_high, results_low, mask_high, mask_low = self._split_detections(results)
    detections = self.init_track(results_high, self._input_for(img, feats, mask_high))
    detections_second = self.init_track(results_low, self._input_for(img, feats, mask_low))
    for tracks, mask in ((detections, mask_high), (detections_second, mask_low)):
        for track, i in zip(tracks, np.flatnonzero(mask)):
            track.idx = i  # idx must be in full detection-set space; parse_bboxes only sees the subset

    unconfirmed, tracked_stracks = self._split_tracked()
    strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    self.multi_predict(strack_pool)
    self._pre_first_associate(strack_pool, unconfirmed, img, results_high)

    u_track, u_detection = self._first_association(strack_pool, detections, activated_stracks, refind_stracks)
    u_track, u_detection = self._post_first_association(
        strack_pool, detections, u_track, u_detection, activated_stracks, refind_stracks
    )
    self._second_association(
        strack_pool, u_track, detections_second, activated_stracks, refind_stracks, lost_stracks
    )
    u_detection, detections = self._unconfirmed_association(
        unconfirmed, u_detection, detections, activated_stracks, removed_stracks
    )
    self._init_new_tracks(u_detection, detections, activated_stracks, refind_stracks)
    self._remove_stale_lost(removed_stracks)

    merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)
    return self._format_output()
```
</details>

<br><br>
