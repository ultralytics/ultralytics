---
description: Explore the robust object tracking capabilities of the BOTrack and BOTSORT classes in the Ultralytics Bot SORT tracker API. Enhance your YOLOv8 projects.
keywords: Ultralytics, Bot SORT, BOTrack, BOTSORT, YOLOv8, object tracking, Kalman filter, ReID, GMC algorithm
---

# Reference for `ultralytics/trackers/bot_sort.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`BOTrack`](#ultralytics.trackers.bot_sort.BOTrack)
        - [`BOTSORT`](#ultralytics.trackers.bot_sort.BOTSORT)
        - [`ReID`](#ultralytics.trackers.bot_sort.ReID)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`BOTrack.tlwh`](#ultralytics.trackers.bot_sort.BOTrack.tlwh)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`BOTrack.update_features`](#ultralytics.trackers.bot_sort.BOTrack.update_features)
        - [`BOTrack.predict`](#ultralytics.trackers.bot_sort.BOTrack.predict)
        - [`BOTrack.re_activate`](#ultralytics.trackers.bot_sort.BOTrack.re_activate)
        - [`BOTrack.update`](#ultralytics.trackers.bot_sort.BOTrack.update)
        - [`BOTrack.multi_predict`](#ultralytics.trackers.bot_sort.BOTrack.multi_predict)
        - [`BOTrack.convert_coords`](#ultralytics.trackers.bot_sort.BOTrack.convert_coords)
        - [`BOTrack.tlwh_to_xywh`](#ultralytics.trackers.bot_sort.BOTrack.tlwh_to_xywh)
        - [`BOTSORT.get_kalmanfilter`](#ultralytics.trackers.bot_sort.BOTSORT.get_kalmanfilter)
        - [`BOTSORT.init_track`](#ultralytics.trackers.bot_sort.BOTSORT.init_track)
        - [`BOTSORT.get_dists`](#ultralytics.trackers.bot_sort.BOTSORT.get_dists)
        - [`BOTSORT.multi_predict`](#ultralytics.trackers.bot_sort.BOTSORT.multi_predict)
        - [`BOTSORT.reset`](#ultralytics.trackers.bot_sort.BOTSORT.reset)
        - [`ReID.__call__`](#ultralytics.trackers.bot_sort.ReID.__call__)


## Class `ultralytics.trackers.bot_sort.BOTrack` {#ultralytics.trackers.bot\_sort.BOTrack}

```python
BOTrack(self, xywh: np.ndarray, score: float, cls: int, feat: np.ndarray | None = None, feat_history: int = 50)
```

**Bases:** `STrack`

An extended version of the STrack class for YOLO, adding object tracking features.

This class extends the STrack class to include additional functionalities for object tracking, such as feature smoothing, Kalman filter prediction, and reactivation of tracks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `xywh` | `np.ndarray` | Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is<br>    the center, (w, h) are width and height, and `idx` is the detection index. | *required* |
| `score` | `float` | Confidence score of the detection. | *required* |
| `cls` | `int` | Class ID of the detected object. | *required* |
| `feat` | `np.ndarray, optional` | Feature vector associated with the detection. | `None` |
| `feat_history` | `int` | Maximum length of the feature history deque. | `50` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `shared_kalman` | `KalmanFilterXYWH` | A shared Kalman filter for all instances of BOTrack. |
| `smooth_feat` | `np.ndarray` | Smoothed feature vector. |
| `curr_feat` | `np.ndarray` | Current feature vector. |
| `features` | `deque` | A deque to store feature vectors with a maximum length defined by `feat_history`. |
| `alpha` | `float` | Smoothing factor for the exponential moving average of features. |
| `mean` | `np.ndarray` | The mean state of the Kalman filter. |
| `covariance` | `np.ndarray` | The covariance matrix of the Kalman filter. |

**Methods**

| Name | Description |
| --- | --- |
| [`tlwh`](#ultralytics.trackers.bot_sort.BOTrack.tlwh) | Return the current bounding box position in `(top left x, top left y, width, height)` format. |
| [`convert_coords`](#ultralytics.trackers.bot_sort.BOTrack.convert_coords) | Convert tlwh bounding box coordinates to xywh format. |
| [`multi_predict`](#ultralytics.trackers.bot_sort.BOTrack.multi_predict) | Predict the mean and covariance for multiple object tracks using a shared Kalman filter. |
| [`predict`](#ultralytics.trackers.bot_sort.BOTrack.predict) | Predict the object's future state using the Kalman filter to update its mean and covariance. |
| [`re_activate`](#ultralytics.trackers.bot_sort.BOTrack.re_activate) | Reactivate a track with updated features and optionally assign a new ID. |
| [`tlwh_to_xywh`](#ultralytics.trackers.bot_sort.BOTrack.tlwh_to_xywh) | Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format. |
| [`update`](#ultralytics.trackers.bot_sort.BOTrack.update) | Update the track with new detection information and the current frame ID. |
| [`update_features`](#ultralytics.trackers.bot_sort.BOTrack.update_features) | Update the feature vector and apply exponential moving average smoothing. |

**Examples**

```python
Create a BOTrack instance and update its features
>>> bo_track = BOTrack(xywh=np.array([100, 50, 80, 40, 0]), score=0.9, cls=1, feat=np.random.rand(128))
>>> bo_track.predict()
>>> new_track = BOTrack(xywh=np.array([110, 60, 80, 40, 0]), score=0.85, cls=1, feat=np.random.rand(128))
>>> bo_track.update(new_track, frame_id=2)
```

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L21-L144"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BOTrack(STrack):
    """An extended version of the STrack class for YOLO, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features: Update features vector and smooth it using exponential moving average.
        predict: Predict the mean and covariance using Kalman filter.
        re_activate: Reactivate a track with updated features and optionally new ID.
        update: Update the track with new detection and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict: Predict the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords: Convert tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh: Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(xywh=np.array([100, 50, 80, 40, 0]), score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(xywh=np.array([110, 60, 80, 40, 0]), score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(
        self, xywh: np.ndarray, score: float, cls: int, feat: np.ndarray | None = None, feat_history: int = 50
    ):
        """Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is
                the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            feat (np.ndarray, optional): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.
        """
        super().__init__(xywh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque(maxlen=feat_history)
        self.alpha = 0.9
```
</details>

<br>

### Property `ultralytics.trackers.bot_sort.BOTrack.tlwh` {#ultralytics.trackers.bot\_sort.BOTrack.tlwh}

```python
def tlwh(self) -> np.ndarray
```

Return the current bounding box position in `(top left x, top left y, width, height)` format.

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L111-L117"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def tlwh(self) -> np.ndarray:
    """Return the current bounding box position in `(top left x, top left y, width, height)` format."""
    if self.mean is None:
        return self._tlwh.copy()
    ret = self.mean[:4].copy()
    ret[:2] -= ret[2:] / 2
    return ret
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTrack.convert_coords` {#ultralytics.trackers.bot\_sort.BOTrack.convert\_coords}

```python
def convert_coords(self, tlwh: np.ndarray) -> np.ndarray
```

Convert tlwh bounding box coordinates to xywh format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tlwh` | `np.ndarray` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L135-L137"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
    """Convert tlwh bounding box coordinates to xywh format."""
    return self.tlwh_to_xywh(tlwh)
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTrack.multi_predict` {#ultralytics.trackers.bot\_sort.BOTrack.multi\_predict}

```python
def multi_predict(stracks: list[BOTrack]) -> None
```

Predict the mean and covariance for multiple object tracks using a shared Kalman filter.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stracks` | `list[BOTrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L120-L133"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def multi_predict(stracks: list[BOTrack]) -> None:
    """Predict the mean and covariance for multiple object tracks using a shared Kalman filter."""
    if len(stracks) <= 0:
        return
    multi_mean = np.asarray([st.mean.copy() for st in stracks])
    multi_covariance = np.asarray([st.covariance for st in stracks])
    for i, st in enumerate(stracks):
        if st.state != TrackState.Tracked:
            multi_mean[i][6] = 0
            multi_mean[i][7] = 0
    multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        stracks[i].mean = mean
        stracks[i].covariance = cov
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTrack.predict` {#ultralytics.trackers.bot\_sort.BOTrack.predict}

```python
def predict(self) -> None
```

Predict the object's future state using the Kalman filter to update its mean and covariance.

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L89-L96"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(self) -> None:
    """Predict the object's future state using the Kalman filter to update its mean and covariance."""
    mean_state = self.mean.copy()
    if self.state != TrackState.Tracked:
        mean_state[6] = 0
        mean_state[7] = 0

    self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTrack.re_activate` {#ultralytics.trackers.bot\_sort.BOTrack.re\_activate}

```python
def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None
```

Reactivate a track with updated features and optionally assign a new ID.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_track` | `BOTrack` |  | *required* |
| `frame_id` | `int` |  | *required* |
| `new_id` | `bool` |  | `False` |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L98-L102"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None:
    """Reactivate a track with updated features and optionally assign a new ID."""
    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat)
    super().re_activate(new_track, frame_id, new_id)
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTrack.tlwh_to_xywh` {#ultralytics.trackers.bot\_sort.BOTrack.tlwh\_to\_xywh}

```python
def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray
```

Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tlwh` | `np.ndarray` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L140-L144"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
    """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTrack.update` {#ultralytics.trackers.bot\_sort.BOTrack.update}

```python
def update(self, new_track: BOTrack, frame_id: int) -> None
```

Update the track with new detection information and the current frame ID.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_track` | `BOTrack` |  | *required* |
| `frame_id` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L104-L108"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, new_track: BOTrack, frame_id: int) -> None:
    """Update the track with new detection information and the current frame ID."""
    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat)
    super().update(new_track, frame_id)
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTrack.update_features` {#ultralytics.trackers.bot\_sort.BOTrack.update\_features}

```python
def update_features(self, feat: np.ndarray) -> None
```

Update the feature vector and apply exponential moving average smoothing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `feat` | `np.ndarray` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L78-L87"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update_features(self, feat: np.ndarray) -> None:
    """Update the feature vector and apply exponential moving average smoothing."""
    feat /= np.linalg.norm(feat)
    self.curr_feat = feat
    if self.smooth_feat is None:
        self.smooth_feat = feat
    else:
        self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
    self.features.append(feat)
    self.smooth_feat /= np.linalg.norm(self.smooth_feat)
```
</details>


<br><br><hr><br>

## Class `ultralytics.trackers.bot_sort.BOTSORT` {#ultralytics.trackers.bot\_sort.BOTSORT}

```python
BOTSORT(self, args: Any, frame_rate: int = 30)
```

**Bases:** `BYTETracker`

An extended version of the BYTETracker class for YOLO, designed for object tracking with ReID and GMC algorithm.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `Any` | Parsed command-line arguments containing tracking parameters. | *required* |
| `frame_rate` | `int` | Frame rate of the video being processed. | `30` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `proximity_thresh` | `float` | Threshold for spatial proximity (IoU) between tracks and detections. |
| `appearance_thresh` | `float` | Threshold for appearance similarity (ReID embeddings) between tracks and detections. |
| `encoder` | `Any` | Object to handle ReID embeddings, set to None if ReID is not enabled. |
| `gmc` | `GMC` | An instance of the GMC algorithm for data association. |
| `args` | `Any` | Parsed command-line arguments containing tracking parameters. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_dists`](#ultralytics.trackers.bot_sort.BOTSORT.get_dists) | Calculate distances between tracks and detections using IoU and optionally ReID embeddings. |
| [`get_kalmanfilter`](#ultralytics.trackers.bot_sort.BOTSORT.get_kalmanfilter) | Return an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process. |
| [`init_track`](#ultralytics.trackers.bot_sort.BOTSORT.init_track) | Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features. |
| [`multi_predict`](#ultralytics.trackers.bot_sort.BOTSORT.multi_predict) | Predict the mean and covariance of multiple object tracks using a shared Kalman filter. |
| [`reset`](#ultralytics.trackers.bot_sort.BOTSORT.reset) | Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states. |

**Examples**

```python
Initialize BOTSORT and process detections
>>> bot_sort = BOTSORT(args, frame_rate=30)
>>> bot_sort.init_track(results, img)
>>> bot_sort.multi_predict(tracks)
```

!!! note "Notes"

    The class is designed to work with a YOLO object detection model and supports ReID only if enabled via args.

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L147-L233"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BOTSORT(BYTETracker):
    """An extended version of the BYTETracker class for YOLO, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (Any): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter: Return an instance of KalmanFilterXYWH for object tracking.
        init_track: Initialize track with detection results and optional image for ReID.
        get_dists: Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict: Predict the mean and covariance of multiple object tracks using a shared Kalman filter.
        reset: Reset the BOTSORT tracker to its initial state.

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(results, img)
        >>> bot_sort.multi_predict(tracks)

    Notes:
        The class is designed to work with a YOLO object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args: Any, frame_rate: int = 30):
        """Initialize BOTSORT object with ReID module and GMC algorithm.

        Args:
            args (Any): Parsed command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video being processed.
        """
        super().__init__(args, frame_rate)
        self.gmc = GMC(method=args.gmc_method)

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = (
            (lambda feats, s: [f.cpu().numpy() for f in feats])  # native features do not require any model
            if args.with_reid and self.args.model == "auto"
            else ReID(args.model)
            if args.with_reid
            else None
        )
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTSORT.get_dists` {#ultralytics.trackers.bot\_sort.BOTSORT.get\_dists}

```python
def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray
```

Calculate distances between tracks and detections using IoU and optionally ReID embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracks` | `list[BOTrack]` |  | *required* |
| `detections` | `list[BOTrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L211-L224"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
    """Calculate distances between tracks and detections using IoU and optionally ReID embeddings."""
    dists = matching.iou_distance(tracks, detections)
    dists_mask = dists > (1 - self.proximity_thresh)

    if self.args.fuse_score:
        dists = matching.fuse_score(dists, detections)

    if self.args.with_reid and self.encoder is not None:
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
        emb_dists[dists_mask] = 1.0
        dists = np.minimum(dists, emb_dists)
    return dists
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTSORT.get_kalmanfilter` {#ultralytics.trackers.bot\_sort.BOTSORT.get\_kalmanfilter}

```python
def get_kalmanfilter(self) -> KalmanFilterXYWH
```

Return an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process.

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L195-L197"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_kalmanfilter(self) -> KalmanFilterXYWH:
    """Return an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
    return KalmanFilterXYWH()
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTSORT.init_track` {#ultralytics.trackers.bot\_sort.BOTSORT.init\_track}

```python
def init_track(self, results, img: np.ndarray | None = None) -> list[BOTrack]
```

Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` |  |  | *required* |
| `img` | `np.ndarray | None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L199-L209"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_track(self, results, img: np.ndarray | None = None) -> list[BOTrack]:
    """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
    if len(results) == 0:
        return []
    bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
    bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
    if self.args.with_reid and self.encoder is not None:
        features_keep = self.encoder(img, bboxes)
        return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features_keep)]
    else:
        return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTSORT.multi_predict` {#ultralytics.trackers.bot\_sort.BOTSORT.multi\_predict}

```python
def multi_predict(self, tracks: list[BOTrack]) -> None
```

Predict the mean and covariance of multiple object tracks using a shared Kalman filter.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracks` | `list[BOTrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L226-L228"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def multi_predict(self, tracks: list[BOTrack]) -> None:
    """Predict the mean and covariance of multiple object tracks using a shared Kalman filter."""
    BOTrack.multi_predict(tracks)
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.BOTSORT.reset` {#ultralytics.trackers.bot\_sort.BOTSORT.reset}

```python
def reset(self) -> None
```

Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states.

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L230-L233"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def reset(self) -> None:
    """Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
    super().reset()
    self.gmc.reset_params()
```
</details>


<br><br><hr><br>

## Class `ultralytics.trackers.bot_sort.ReID` {#ultralytics.trackers.bot\_sort.ReID}

```python
ReID(self, model: str)
```

YOLO model as encoder for re-identification.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str` | Path to the YOLO model for re-identification. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.trackers.bot_sort.ReID.__call__) | Extract embeddings for detected objects. |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L236-L257"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ReID:
    """YOLO model as encoder for re-identification."""

    def __init__(self, model: str):
        """Initialize encoder for re-identification.

        Args:
            model (str): Path to the YOLO model for re-identification.
        """
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.model(embed=[len(self.model.model.model) - 2 if ".pt" in model else -1], verbose=False, save=False)  # init
```
</details>

<br>

### Method `ultralytics.trackers.bot_sort.ReID.__call__` {#ultralytics.trackers.bot\_sort.ReID.\_\_call\_\_}

```python
def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]
```

Extract embeddings for detected objects.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img` | `np.ndarray` |  | *required* |
| `dets` | `np.ndarray` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/bot_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py#L250-L257"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
    """Extract embeddings for detected objects."""
    feats = self.model.predictor(
        [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
    )
    if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
        feats = feats[0]  # batched prediction with non-PyTorch backend
    return [f.cpu().numpy() for f in feats]
```
</details>

<br><br>
