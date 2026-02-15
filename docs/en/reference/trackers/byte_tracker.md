---
description: Explore the BYTETracker module in Ultralytics for state-of-the-art object tracking using Kalman filtering. Learn about its classes, methods, and attributes.
keywords: Ultralytics, BYTETracker, object tracking, Kalman filter, YOLOv8, documentation
---

# Reference for `ultralytics/trackers/byte_tracker.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

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
        - [`STrack.multi_gmc`](#ultralytics.trackers.byte_tracker.STrack.multi_gmc)
        - [`STrack.activate`](#ultralytics.trackers.byte_tracker.STrack.activate)
        - [`STrack.re_activate`](#ultralytics.trackers.byte_tracker.STrack.re_activate)
        - [`STrack.update`](#ultralytics.trackers.byte_tracker.STrack.update)
        - [`STrack.convert_coords`](#ultralytics.trackers.byte_tracker.STrack.convert_coords)
        - [`STrack.tlwh_to_xyah`](#ultralytics.trackers.byte_tracker.STrack.tlwh_to_xyah)
        - [`STrack.__repr__`](#ultralytics.trackers.byte_tracker.STrack.__repr__)
        - [`BYTETracker.update`](#ultralytics.trackers.byte_tracker.BYTETracker.update)
        - [`BYTETracker.get_kalmanfilter`](#ultralytics.trackers.byte_tracker.BYTETracker.get_kalmanfilter)
        - [`BYTETracker.init_track`](#ultralytics.trackers.byte_tracker.BYTETracker.init_track)
        - [`BYTETracker.get_dists`](#ultralytics.trackers.byte_tracker.BYTETracker.get_dists)
        - [`BYTETracker.multi_predict`](#ultralytics.trackers.byte_tracker.BYTETracker.multi_predict)
        - [`BYTETracker.reset_id`](#ultralytics.trackers.byte_tracker.BYTETracker.reset_id)
        - [`BYTETracker.reset`](#ultralytics.trackers.byte_tracker.BYTETracker.reset)
        - [`BYTETracker.joint_stracks`](#ultralytics.trackers.byte_tracker.BYTETracker.joint_stracks)
        - [`BYTETracker.sub_stracks`](#ultralytics.trackers.byte_tracker.BYTETracker.sub_stracks)
        - [`BYTETracker.remove_duplicate_stracks`](#ultralytics.trackers.byte_tracker.BYTETracker.remove_duplicate_stracks)


## Class `ultralytics.trackers.byte_tracker.STrack` {#ultralytics.trackers.byte\_tracker.STrack}

```python
STrack(self, xywh: list[float], score: float, cls: Any)
```

**Bases:** `BaseTrack`

Single object tracking representation that uses Kalman filtering for state estimation.

This class is responsible for storing all the information regarding individual tracklets and performs state updates and predictions based on Kalman filter.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `xywh` | `list[float]` | Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y)<br>    is the center, (w, h) are width and height, and `idx` is the detection index. | *required* |
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
| `angle` | `float | None` | Optional angle information for oriented bounding boxes. |

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
| [`multi_gmc`](#ultralytics.trackers.byte_tracker.STrack.multi_gmc) | Update multiple track positions and covariances using a homography matrix. |
| [`multi_predict`](#ultralytics.trackers.byte_tracker.STrack.multi_predict) | Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances. |
| [`predict`](#ultralytics.trackers.byte_tracker.STrack.predict) | Predict the next state (mean and covariance) of the object using the Kalman filter. |
| [`re_activate`](#ultralytics.trackers.byte_tracker.STrack.re_activate) | Reactivate a previously lost track using new detection data and update its state and attributes. |
| [`tlwh_to_xyah`](#ultralytics.trackers.byte_tracker.STrack.tlwh_to_xyah) | Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format. |
| [`update`](#ultralytics.trackers.byte_tracker.STrack.update) | Update the state of a matched track. |

**Examples**

```python
Initialize and activate a new track
>>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
>>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
```

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L16-L228"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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
        multi_gmc: Update multiple track states using a homography matrix.
        activate: Activate a new tracklet.
        re_activate: Reactivate a previously lost tracklet.
        update: Update the state of a matched track.
        convert_coords: Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah: Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: list[float], score: float, cls: Any):
        """Initialize a new STrack instance.

        Args:
            xywh (list[float]): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y)
                is the center, (w, h) are width and height, and `idx` is the detection index.
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L181-L188"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L191-L195"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def xyxy(self) -> np.ndarray:
    """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
    ret = self.tlwh.copy()
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L206-L210"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L213-L218"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L221-L224"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L226-L228"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L119-L130"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L176-L178"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
    """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
    return self.tlwh_to_xyah(tlwh)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.STrack.multi_gmc` {#ultralytics.trackers.byte\_tracker.STrack.multi\_gmc}

```python
def multi_gmc(stracks: list[STrack], H: np.ndarray = np.eye(2, 3))
```

Update multiple track positions and covariances using a homography matrix.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stracks` | `list[STrack]` |  | *required* |
| `H` | `np.ndarray` |  | `np.eye(2, 3)` |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L101-L117"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def multi_gmc(stracks: list[STrack], H: np.ndarray = np.eye(2, 3)):
    """Update multiple track positions and covariances using a homography matrix."""
    if stracks:
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4, dtype=float), R)
        t = H[:2, 2]

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            mean = R8x8.dot(mean)
            mean[:2] += t
            cov = R8x8.dot(cov).dot(R8x8.transpose())

            stracks[i].mean = mean
            stracks[i].covariance = cov
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L86-L98"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def multi_predict(stracks: list[STrack]):
    """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
    if len(stracks) <= 0:
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L78-L83"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L132-L146"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L198-L203"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

```python
Update the state of a track with new detection information
>>> track = STrack([100, 200, 50, 80, 0], score=0.9, cls=0)
>>> new_track = STrack([105, 205, 55, 85, 0], score=0.95, cls=0)
>>> track.update(new_track, 2)
```

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L148-L174"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, new_track: STrack, frame_id: int):
    """Update the state of a matched track.

    Args:
        new_track (STrack): The new track containing updated information.
        frame_id (int): The ID of the current frame.

    Examples:
        Update the state of a track with new detection information
        >>> track = STrack([100, 200, 50, 80, 0], score=0.9, cls=0)
        >>> new_track = STrack([105, 205, 55, 85, 0], score=0.95, cls=0)
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
BYTETracker(self, args, frame_rate: int = 30)
```

BYTETracker: A tracking algorithm built on top of YOLO for object detection and tracking.

This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for predicting the new object locations, and performs data association.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `Namespace` | Command-line arguments containing tracking parameters. | *required* |
| `frame_rate` | `int` | Frame rate of the video sequence. | `30` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `tracked_stracks` | `list[STrack]` | List of successfully activated tracks. |
| `lost_stracks` | `list[STrack]` | List of lost tracks. |
| `removed_stracks` | `list[STrack]` | List of removed tracks. |
| `frame_id` | `int` | The current frame ID. |
| `args` | `Namespace` | Command-line arguments. |
| `max_time_lost` | `int` | The maximum frames for a track to be considered as 'lost'. |
| `kalman_filter` | `KalmanFilterXYAH` | Kalman Filter object. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_dists`](#ultralytics.trackers.byte_tracker.BYTETracker.get_dists) | Calculate the distance between tracks and detections using IoU and optionally fuse scores. |
| [`get_kalmanfilter`](#ultralytics.trackers.byte_tracker.BYTETracker.get_kalmanfilter) | Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH. |
| [`init_track`](#ultralytics.trackers.byte_tracker.BYTETracker.init_track) | Initialize object tracking with given detections, scores, and class labels as STrack instances. |
| [`joint_stracks`](#ultralytics.trackers.byte_tracker.BYTETracker.joint_stracks) | Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs. |
| [`multi_predict`](#ultralytics.trackers.byte_tracker.BYTETracker.multi_predict) | Predict the next states for multiple tracks using Kalman filter. |
| [`remove_duplicate_stracks`](#ultralytics.trackers.byte_tracker.BYTETracker.remove_duplicate_stracks) | Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance. |
| [`reset`](#ultralytics.trackers.byte_tracker.BYTETracker.reset) | Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter. |
| [`reset_id`](#ultralytics.trackers.byte_tracker.BYTETracker.reset_id) | Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions. |
| [`sub_stracks`](#ultralytics.trackers.byte_tracker.BYTETracker.sub_stracks) | Filter out the stracks present in the second list from the first list. |
| [`update`](#ultralytics.trackers.byte_tracker.BYTETracker.update) | Update the tracker with new detections and return the current list of tracked objects. |

**Examples**

```python
Initialize BYTETracker and update with detection results
>>> tracker = BYTETracker(args, frame_rate=30)
>>> results = yolo_model.detect(image)
>>> tracked_objects = tracker.update(results)
```

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L231-L469"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
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
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.
        """
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L408-L413"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L396-L398"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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
| `img` | `np.ndarray | None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L400-L406"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
    """Initialize object tracking with given detections, scores, and class labels as STrack instances."""
    if len(results) == 0:
        return []
    bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
    bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
    return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.joint_stracks` {#ultralytics.trackers.byte\_tracker.BYTETracker.joint\_stracks}

```python
def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]
```

Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tlista` | `list[STrack]` |  | *required* |
| `tlistb` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L434-L446"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
    """Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L415-L417"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def multi_predict(self, tracks: list[STrack]):
    """Predict the next states for multiple tracks using Kalman filter."""
    STrack.multi_predict(tracks)
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.remove_duplicate_stracks` {#ultralytics.trackers.byte\_tracker.BYTETracker.remove\_duplicate\_stracks}

```python
def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]
```

Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stracksa` | `list[STrack]` |  | *required* |
| `stracksb` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L455-L469"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
    """Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L424-L431"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def reset(self):
    """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
    self.tracked_stracks: list[STrack] = []
    self.lost_stracks: list[STrack] = []
    self.removed_stracks: list[STrack] = []
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L420-L422"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def reset_id():
    """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
    STrack.reset_id()
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.sub_stracks` {#ultralytics.trackers.byte\_tracker.BYTETracker.sub\_stracks}

```python
def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]
```

Filter out the stracks present in the second list from the first list.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tlista` | `list[STrack]` |  | *required* |
| `tlistb` | `list[STrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L449-L452"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
    """Filter out the stracks present in the second list from the first list."""
    track_ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in track_ids_b]
```
</details>

<br>

### Method `ultralytics.trackers.byte_tracker.BYTETracker.update` {#ultralytics.trackers.byte\_tracker.BYTETracker.update}

```python
def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray
```

Update the tracker with new detections and return the current list of tracked objects.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` |  |  | *required* |
| `img` | `np.ndarray | None` |  | `None` |
| `feats` | `np.ndarray | None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/byte_tracker.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py#L283-L394"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
    """Update the tracker with new detections and return the current list of tracked objects."""
    self.frame_id += 1
    activated_stracks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []

    scores = results.conf
    remain_inds = scores >= self.args.track_high_thresh
    inds_low = scores > self.args.track_low_thresh
    inds_high = scores < self.args.track_high_thresh

    inds_second = inds_low & inds_high
    results_second = results[inds_second]
    results = results[remain_inds]
    feats_keep = feats_second = img
    if feats is not None and len(feats):
        feats_keep = feats[remain_inds]
        feats_second = feats[inds_second]

    detections = self.init_track(results, feats_keep)
    # Add newly detected tracklets to tracked_stracks
    unconfirmed = []
    tracked_stracks: list[STrack] = []
    for track in self.tracked_stracks:
        if not track.is_activated:
            unconfirmed.append(track)
        else:
            tracked_stracks.append(track)
    # Step 2: First association, with high score detection boxes
    strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
    # Predict the current location with KF
    self.multi_predict(strack_pool)
    if hasattr(self, "gmc") and img is not None:
        # use try-except here to bypass errors from gmc module
        try:
            warp = self.gmc.apply(img, results.xyxy)
        except Exception:
            warp = np.eye(2, 3)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

    dists = self.get_dists(strack_pool, detections)
    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

    for itracked, idet in matches:
        track = strack_pool[itracked]
        det = detections[idet]
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
    detections_second = self.init_track(results_second, feats_second)
    r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
    # TODO: consider fusing scores or appearance features for second association.
    dists = matching.iou_distance(r_tracked_stracks, detections_second)
    matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
    for itracked, idet in matches:
        track = r_tracked_stracks[itracked]
        det = detections_second[idet]
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)

    for it in u_track:
        track = r_tracked_stracks[it]
        if track.state != TrackState.Lost:
            track.mark_lost()
            lost_stracks.append(track)
    # Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections = [detections[i] for i in u_detection]
    dists = self.get_dists(unconfirmed, detections)
    matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    for itracked, idet in matches:
        unconfirmed[itracked].update(detections[idet], self.frame_id)
        activated_stracks.append(unconfirmed[itracked])
    for it in u_unconfirmed:
        track = unconfirmed[it]
        track.mark_removed()
        removed_stracks.append(track)
    # Step 4: Init new stracks
    for inew in u_detection:
        track = detections[inew]
        if track.score < self.args.new_track_thresh:
            continue
        track.activate(self.kalman_filter, self.frame_id)
        activated_stracks.append(track)
    # Step 5: Update state
    for track in self.lost_stracks:
        if self.frame_id - track.end_frame > self.max_time_lost:
            track.mark_removed()
            removed_stracks.append(track)

    self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
    self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
    self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
    self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
    self.removed_stracks.extend(removed_stracks)
    if len(self.removed_stracks) > 1000:
        self.removed_stracks = self.removed_stracks[-1000:]  # clip removed stracks to 1000 maximum

    return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
```
</details>

<br><br>
