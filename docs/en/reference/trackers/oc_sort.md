---
title: trackers.oc_sort API Reference
description: Explore the OC-SORT module in Ultralytics for observation-centric multi-object tracking with Kalman filtering. Learn about its classes, methods, and attributes.
keywords: Ultralytics, OC-SORT, OCSORT, observation-centric, object tracking, Kalman filter, YOLO, documentation
---

# Reference for `ultralytics/trackers/oc_sort.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`OCSortTrack`](#ultralytics.trackers.oc_sort.OCSortTrack)
        - [`OCSORT`](#ultralytics.trackers.oc_sort.OCSORT)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`OCSortTrack.activate`](#ultralytics.trackers.oc_sort.OCSortTrack.activate)
        - [`OCSortTrack.update`](#ultralytics.trackers.oc_sort.OCSortTrack.update)
        - [`OCSortTrack.re_activate`](#ultralytics.trackers.oc_sort.OCSortTrack.re_activate)
        - [`OCSortTrack._xyxy_center`](#ultralytics.trackers.oc_sort.OCSortTrack._xyxy_center)
        - [`OCSortTrack._record_observation`](#ultralytics.trackers.oc_sort.OCSortTrack._record_observation)
        - [`OCSortTrack._compute_velocity`](#ultralytics.trackers.oc_sort.OCSortTrack._compute_velocity)
        - [`OCSortTrack.apply_oru`](#ultralytics.trackers.oc_sort.OCSortTrack.apply_oru)
        - [`OCSORT.init_track`](#ultralytics.trackers.oc_sort.OCSORT.init_track)
        - [`OCSORT._fuse_appearance`](#ultralytics.trackers.oc_sort.OCSORT._fuse_appearance)
        - [`OCSORT.get_dists`](#ultralytics.trackers.oc_sort.OCSORT.get_dists)
        - [`OCSORT._ocr_associate`](#ultralytics.trackers.oc_sort.OCSORT._ocr_associate)
        - [`OCSORT._post_first_association`](#ultralytics.trackers.oc_sort.OCSORT._post_first_association)
        - [`OCSORT._second_association`](#ultralytics.trackers.oc_sort.OCSORT._second_association)
        - [`OCSORT._velocity_direction_cost`](#ultralytics.trackers.oc_sort.OCSORT._velocity_direction_cost)
        - [`OCSORT._ocr_distance`](#ultralytics.trackers.oc_sort.OCSORT._ocr_distance)


## Class `ultralytics.trackers.oc_sort.OCSortTrack` {#ultralytics.trackers.oc\_sort.OCSortTrack}

```python
OCSortTrack(xywh: np.ndarray, score: float, cls: Any, delta_t: int = 3)
```

**Bases:** `STrack`

Track object for OC-SORT with observation-centric state management.

Extends STrack with storage of real detector observations and velocity computation, enabling the three OC-SORT components: ORU, OCM, and OCR.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `xywh` | `np.ndarray` | Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format. | *required* |
| `score` | `float` | Detection confidence in `[0, 1]`. | *required* |
| `cls` | `Any` | Class label for the detection. | *required* |
| `delta_t` | `int` | Temporal window (in frames) used for velocity direction computation. | `3` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `last_observation` | `np.ndarray` | Last real detection in xyxy format. |
| `observations` | `dict` | Maps frame_id to xyxy observation arrays. |
| `velocity` | `np.ndarray \| None` | Observation-centric velocity direction vector (dx, dy). |
| `delta_t` | `int` | Temporal window for velocity computation. |

**Methods**

| Name | Description |
| --- | --- |
| [`_compute_velocity`](#ultralytics.trackers.oc_sort.OCSortTrack._compute_velocity) | Compute the observation-centric velocity direction from stored observations. |
| [`_record_observation`](#ultralytics.trackers.oc_sort.OCSortTrack._record_observation) | Store `obs` for `frame_id`, dropping history beyond `delta_t + 2` to bound memory. |
| [`_xyxy_center`](#ultralytics.trackers.oc_sort.OCSortTrack._xyxy_center) | Return `(cx, cy)` center of an xyxy bounding box. |
| [`activate`](#ultralytics.trackers.oc_sort.OCSortTrack.activate) | Activate a new tracklet and seed its observation history. |
| [`apply_oru`](#ultralytics.trackers.oc_sort.OCSortTrack.apply_oru) | Repair Kalman state across an occlusion gap by replaying predict-updates on virtual observations. |
| [`re_activate`](#ultralytics.trackers.oc_sort.OCSortTrack.re_activate) | Re-activate a previously lost track with a new detection. |
| [`update`](#ultralytics.trackers.oc_sort.OCSortTrack.update) | Update the track with a matched detection and record the observation. |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L16-L162">View on GitHub</a>
```python
class OCSortTrack(STrack):
    """Track object for OC-SORT with observation-centric state management.

    Extends STrack with storage of real detector observations and velocity computation, enabling the three OC-SORT
    components: ORU, OCM, and OCR.

    Attributes:
        last_observation (np.ndarray): Last real detection in xyxy format.
        observations (dict): Maps frame_id to xyxy observation arrays.
        velocity (np.ndarray | None): Observation-centric velocity direction vector (dx, dy).
        delta_t (int): Temporal window for velocity computation.
    """

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, delta_t: int = 3):
        """Initialize an OCSortTrack with observation storage.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format.
            score (float): Detection confidence in `[0, 1]`.
            cls (Any): Class label for the detection.
            delta_t (int): Temporal window (in frames) used for velocity direction computation.
        """
        super().__init__(xywh, score, cls)
        self.last_observation = np.array([-1, -1, -1, -1], dtype=np.float32)
        self.observations: dict[int, np.ndarray] = {}
        self.velocity: np.ndarray | None = None
        self.delta_t = delta_t
        self._saved_mean: np.ndarray | None = None
        self._saved_covariance: np.ndarray | None = None
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSortTrack._compute_velocity` {#ultralytics.trackers.oc\_sort.OCSortTrack.\_compute\_velocity}

```python
def _compute_velocity(self) -> np.ndarray | None
```

Compute the observation-centric velocity direction from stored observations.

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray \| None` | Normalized `(dx, dy)` direction vector, or None if there are fewer than two usable observations. |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L103-L134">View on GitHub</a>
```python
def _compute_velocity(self) -> np.ndarray | None:
    """Compute the observation-centric velocity direction from stored observations.

    Returns:
        (np.ndarray | None): Normalized `(dx, dy)` direction vector, or None if there are fewer than two usable
            observations.
    """
    if len(self.observations) < 2:
        return None

    current_frame = max(self.observations.keys())
    current_center = self._xyxy_center(self.observations[current_frame])

    # Find the most recent observation at least delta_t frames before current
    prev_obs = None
    for frame in sorted(self.observations.keys(), reverse=True):
        if frame < current_frame - self.delta_t + 1:
            prev_obs = self.observations[frame]
            break

    # Fallback: use the earliest observation if nothing is delta_t frames back
    if prev_obs is None:
        earliest_frame = min(self.observations.keys())
        if earliest_frame == current_frame:
            return None
        prev_obs = self.observations[earliest_frame]

    direction = current_center - self._xyxy_center(prev_obs)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (direction / norm).astype(np.float32)
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSortTrack._record_observation` {#ultralytics.trackers.oc\_sort.OCSortTrack.\_record\_observation}

```python
def _record_observation(self, obs: np.ndarray, frame_id: int) -> None
```

Store `obs` for `frame_id`, dropping history beyond `delta_t + 2` to bound memory.

The retained window always covers the frame `_compute_velocity` reaches back to, since at most `delta_t` distinct frames fall inside `(frame_id - delta_t, frame_id]`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `obs` | `np.ndarray` |  | *required* |
| `frame_id` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L90-L101">View on GitHub</a>
```python
def _record_observation(self, obs: np.ndarray, frame_id: int) -> None:
    """Store `obs` for `frame_id`, dropping history beyond `delta_t + 2` to bound memory.

    The retained window always covers the frame `_compute_velocity` reaches back to, since at most `delta_t`
    distinct frames fall inside `(frame_id - delta_t, frame_id]`.
    """
    self.last_observation = obs
    self.observations[frame_id] = obs
    max_keep = self.delta_t + 2
    if len(self.observations) > max_keep:
        for frame in sorted(self.observations)[:-max_keep]:
            del self.observations[frame]
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSortTrack._xyxy_center` {#ultralytics.trackers.oc\_sort.OCSortTrack.\_xyxy\_center}

```python
def _xyxy_center(xyxy: np.ndarray) -> np.ndarray
```

Return `(cx, cy)` center of an xyxy bounding box.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `xyxy` | `np.ndarray` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L86-L88">View on GitHub</a>
```python
@staticmethod
def _xyxy_center(xyxy: np.ndarray) -> np.ndarray:
    """Return `(cx, cy)` center of an xyxy bounding box."""
    return np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSortTrack.activate` {#ultralytics.trackers.oc\_sort.OCSortTrack.activate}

```python
def activate(self, kalman_filter, frame_id: int) -> None
```

Activate a new tracklet and seed its observation history.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `kalman_filter` | `KalmanFilterXYAH` | Shared Kalman filter instance. | *required* |
| `frame_id` | `int` | Frame id at which the track is created. | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L46-L56">View on GitHub</a>
```python
def activate(self, kalman_filter, frame_id: int) -> None:
    """Activate a new tracklet and seed its observation history.

    Args:
        kalman_filter (KalmanFilterXYAH): Shared Kalman filter instance.
        frame_id (int): Frame id at which the track is created.
    """
    super().activate(kalman_filter, frame_id)
    self._record_observation(self.xyxy.astype(np.float32), frame_id)  # detection-space precision
    self._saved_mean = self.mean.copy()
    self._saved_covariance = self.covariance.copy()
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSortTrack.apply_oru` {#ultralytics.trackers.oc\_sort.OCSortTrack.apply\_oru}

```python
def apply_oru(self, new_observation_xyxy: np.ndarray, current_frame_id: int) -> None
```

Repair Kalman state across an occlusion gap by replaying predict-updates on virtual observations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_observation_xyxy` | `np.ndarray` |  | *required* |
| `current_frame_id` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L136-L162">View on GitHub</a>
```python
def apply_oru(self, new_observation_xyxy: np.ndarray, current_frame_id: int) -> None:
    """Repair Kalman state across an occlusion gap by replaying predict-updates on virtual observations."""
    if self._saved_mean is None or not self.observations:
        return

    last_frame = max(self.observations.keys())
    gap = current_frame_id - last_frame
    if gap <= 1:
        return

    # Restore Kalman state to last observation point
    self.mean = self._saved_mean.copy()
    self.covariance = self._saved_covariance.copy()

    last_obs = self.observations[last_frame]

    # Replay with virtual observations
    for t in range(1, gap):
        alpha = t / gap
        virtual_xyxy = (1 - alpha) * last_obs + alpha * new_observation_xyxy
        # Convert xyxy to tlwh then to xyah for Kalman measurement
        virtual_xyah = self.tlwh_to_xyah(xyxy2ltwh(virtual_xyxy))
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, virtual_xyah)

    # Final predict to reach current frame
    self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSortTrack.re_activate` {#ultralytics.trackers.oc\_sort.OCSortTrack.re\_activate}

```python
def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None
```

Re-activate a previously lost track with a new detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_track` | `STrack` | Detection used to revive this track. | *required* |
| `frame_id` | `int` | Current frame id. | *required* |
| `new_id` | `bool` | If True, assign a fresh track id instead of reusing the old one. | `False` |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L71-L83">View on GitHub</a>
```python
def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
    """Re-activate a previously lost track with a new detection.

    Args:
        new_track (STrack): Detection used to revive this track.
        frame_id (int): Current frame id.
        new_id (bool): If True, assign a fresh track id instead of reusing the old one.
    """
    self._record_observation(new_track.xyxy.copy(), frame_id)
    super().re_activate(new_track, frame_id, new_id)
    self._saved_mean = self.mean.copy()
    self._saved_covariance = self.covariance.copy()
    self.velocity = self._compute_velocity()
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSortTrack.update` {#ultralytics.trackers.oc\_sort.OCSortTrack.update}

```python
def update(self, new_track: STrack, frame_id: int) -> None
```

Update the track with a matched detection and record the observation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_track` | `STrack` | Matched detection for this frame. | *required* |
| `frame_id` | `int` | Current frame id. | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L58-L69">View on GitHub</a>
```python
def update(self, new_track: STrack, frame_id: int) -> None:
    """Update the track with a matched detection and record the observation.

    Args:
        new_track (STrack): Matched detection for this frame.
        frame_id (int): Current frame id.
    """
    self._record_observation(new_track.xyxy.copy(), frame_id)
    super().update(new_track, frame_id)
    self._saved_mean = self.mean.copy()
    self._saved_covariance = self.covariance.copy()
    self.velocity = self._compute_velocity()
```
</details>


<br><br><hr><br>

## Class `ultralytics.trackers.oc_sort.OCSORT` {#ultralytics.trackers.oc\_sort.OCSORT}

```python
OCSORT(args: Any)
```

**Bases:** `BYTETracker`

OC-SORT multi-object tracker with observation-centric association.

Implements three key components on top of BYTETracker:
- Observation-Centric Re-Update (ORU): repairs Kalman state after occlusion
- Observation-Centric Momentum (OCM): velocity direction consistency cost
- Observation-Centric Recovery (OCR): re-association using last observation position

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `Namespace \| IterableSimpleNamespace` | Parsed tracker config providing the BYTE keys plus `delta_t`, `inertia`, and `use_byte`. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `delta_t` | `int` | Temporal window for velocity direction computation. |
| `inertia` | `float` | Weight of velocity consistency cost in association. |
| `use_byte` | `bool` | Whether to use ByteTrack-style low-confidence second pass. |

**Methods**

| Name | Description |
| --- | --- |
| [`_fuse_appearance`](#ultralytics.trackers.oc_sort.OCSORT._fuse_appearance) | Hook combining motion cost with appearance cost. Default: pass-through (no ReID). |
| [`_ocr_associate`](#ultralytics.trackers.oc_sort.OCSORT._ocr_associate) | Run one OCR (last-observation IoU) pass, applying matches in place. |
| [`_ocr_distance`](#ultralytics.trackers.oc_sort.OCSORT._ocr_distance) | Compute IoU distance using tracks' last observation positions instead of Kalman predictions. |
| [`_post_first_association`](#ultralytics.trackers.oc_sort.OCSORT._post_first_association) | Observation-Centric Recovery (OCR) pass after first-stage association. |
| [`_second_association`](#ultralytics.trackers.oc_sort.OCSORT._second_association) | Run ByteTrack-style second pass only when ``use_byte`` is enabled. |
| [`_velocity_direction_cost`](#ultralytics.trackers.oc_sort.OCSORT._velocity_direction_cost) | Compute OCM velocity direction consistency cost matrix (vectorized). |
| [`get_dists`](#ultralytics.trackers.oc_sort.OCSORT.get_dists) | Cost matrix = IoU (+score-fuse) + inertia·OCM (+ optional appearance, via hook). |
| [`init_track`](#ultralytics.trackers.oc_sort.OCSORT.init_track) | Build `OCSortTrack` instances from a `Results`-like object. |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L165-L352">View on GitHub</a>
```python
class OCSORT(BYTETracker):
    """OC-SORT multi-object tracker with observation-centric association.

    Implements three key components on top of BYTETracker:
    - Observation-Centric Re-Update (ORU): repairs Kalman state after occlusion
    - Observation-Centric Momentum (OCM): velocity direction consistency cost
    - Observation-Centric Recovery (OCR): re-association using last observation position

    Attributes:
        delta_t (int): Temporal window for velocity direction computation.
        inertia (float): Weight of velocity consistency cost in association.
        use_byte (bool): Whether to use ByteTrack-style low-confidence second pass.
    """

    track_class = OCSortTrack

    def __init__(self, args: Any):
        """Initialize OC-SORT tracker.

        Args:
            args (Namespace | IterableSimpleNamespace): Parsed tracker config providing the BYTE keys plus `delta_t`,
                `inertia`, and `use_byte`.
        """
        super().__init__(args)
        self.delta_t = getattr(args, "delta_t", 3)
        self.inertia = getattr(args, "inertia", 0.2)
        self.use_byte = getattr(args, "use_byte", False)
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSORT._fuse_appearance` {#ultralytics.trackers.oc\_sort.OCSORT.\_fuse\_appearance}

```python
def _fuse_appearance(
    self,
    dists: np.ndarray,
    tracks: list[OCSortTrack],
    detections: list[OCSortTrack],
    iou_dists: np.ndarray | None = None,
) -> np.ndarray
```

Hook combining motion cost with appearance cost. Default: pass-through (no ReID).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dists` | `np.ndarray` |  | *required* |
| `tracks` | `list[OCSortTrack]` |  | *required* |
| `detections` | `list[OCSortTrack]` |  | *required* |
| `iou_dists` | `np.ndarray \| None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L200-L208">View on GitHub</a>
```python
def _fuse_appearance(
    self,
    dists: np.ndarray,
    tracks: list[OCSortTrack],
    detections: list[OCSortTrack],
    iou_dists: np.ndarray | None = None,
) -> np.ndarray:
    """Hook combining motion cost with appearance cost. Default: pass-through (no ReID)."""
    return dists
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSORT._ocr_associate` {#ultralytics.trackers.oc\_sort.OCSORT.\_ocr\_associate}

```python
def _ocr_associate(
    self,
    tracks: list[OCSortTrack],
    dets: list[OCSortTrack],
    activated: list[OCSortTrack],
    refind: list[OCSortTrack],
) -> tuple[list[int], list[int]]
```

Run one OCR (last-observation IoU) pass, applying matches in place.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracks` | `list[OCSortTrack]` |  | *required* |
| `dets` | `list[OCSortTrack]` |  | *required* |
| `activated` | `list[OCSortTrack]` |  | *required* |
| `refind` | `list[OCSortTrack]` |  | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `tuple[list[int], list[int]]` | Local indices of unmatched ``tracks`` and unmatched ``dets``. |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L217-L245">View on GitHub</a>
```python
def _ocr_associate(
    self,
    tracks: list[OCSortTrack],
    dets: list[OCSortTrack],
    activated: list[OCSortTrack],
    refind: list[OCSortTrack],
) -> tuple[list[int], list[int]]:
    """Run one OCR (last-observation IoU) pass, applying matches in place.

    Returns:
        (tuple[list[int], list[int]]): Local indices of unmatched ``tracks`` and unmatched ``dets``.
    """
    if not tracks or not dets:
        return list(range(len(tracks))), list(range(len(dets)))
    ocr_dists = self._ocr_distance(tracks, dets)
    if self.args.fuse_score:
        ocr_dists = matching.fuse_score(ocr_dists, dets)
    ocr_dists = self._fuse_appearance(ocr_dists, tracks, dets)
    matches, u_track, u_det = matching.linear_assignment(ocr_dists, thresh=self.args.match_thresh)
    for itracked, idet in matches:
        track, det = tracks[itracked], dets[idet]
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated.append(track)
        else:
            track.apply_oru(det.xyxy, self.frame_id)
            track.re_activate(det, self.frame_id, new_id=False)
            refind.append(track)
    return list(u_track), list(u_det)
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSORT._ocr_distance` {#ultralytics.trackers.oc\_sort.OCSORT.\_ocr\_distance}

```python
def _ocr_distance(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray
```

Compute IoU distance using tracks' last observation positions instead of Kalman predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracks` | `list[OCSortTrack]` | List of tracks with last_observation attributes. | *required* |
| `detections` | `list[OCSortTrack]` | List of detections. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Cost matrix based on IoU with last observations (or xywha for OBB). |

!!! note "Notes"

    `last_observation` is stored in xyxy form. For oriented (OBB) tracks no oriented
    last-observation is kept, so this method falls back to the Kalman-predicted `xywha`
    and the OCR pass degenerates to standard IoU on the predicted box. Standard
    (axis-aligned) tracking gets the full OCR benefit.

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L330-L352">View on GitHub</a>
```python
def _ocr_distance(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
    """Compute IoU distance using tracks' last observation positions instead of Kalman predictions.

    Args:
        tracks (list[OCSortTrack]): List of tracks with last_observation attributes.
        detections (list[OCSortTrack]): List of detections.

    Returns:
        (np.ndarray): Cost matrix based on IoU with last observations (or xywha for OBB).

    Notes:
        `last_observation` is stored in xyxy form. For oriented (OBB) tracks no oriented
        last-observation is kept, so this method falls back to the Kalman-predicted `xywha`
        and the OCR pass degenerates to standard IoU on the predicted box. Standard
        (axis-aligned) tracking gets the full OCR benefit.
    """
    if tracks and tracks[0].angle is not None:
        atlbrs = [t.xywha for t in tracks]
        btlbrs = [d.xywha for d in detections]
    else:
        atlbrs = [t.last_observation if t.last_observation[0] >= 0 else t.xyxy for t in tracks]
        btlbrs = [d.xyxy for d in detections]
    return matching.iou_distance(atlbrs, btlbrs)
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSORT._post_first_association` {#ultralytics.trackers.oc\_sort.OCSORT.\_post\_first\_association}

```python
def _post_first_association(
    self,
    strack_pool: list[OCSortTrack],
    detections: list[OCSortTrack],
    u_track: list[int],
    u_detection: list[int],
    activated: list[OCSortTrack],
    refind: list[OCSortTrack],
) -> tuple[list[int], list[int]]
```

Observation-Centric Recovery (OCR) pass after first-stage association.

Runs OCR for still-Tracked unmatched tracks first to preserve active-track matching priority, then for Lost tracks on the detections still unmatched, so a recently-lost track cannot outbid an active one.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `strack_pool` | `list[OCSortTrack]` |  | *required* |
| `detections` | `list[OCSortTrack]` |  | *required* |
| `u_track` | `list[int]` |  | *required* |
| `u_detection` | `list[int]` |  | *required* |
| `activated` | `list[OCSortTrack]` |  | *required* |
| `refind` | `list[OCSortTrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L247-L274">View on GitHub</a>
```python
def _post_first_association(
    self,
    strack_pool: list[OCSortTrack],
    detections: list[OCSortTrack],
    u_track: list[int],
    u_detection: list[int],
    activated: list[OCSortTrack],
    refind: list[OCSortTrack],
) -> tuple[list[int], list[int]]:
    """Observation-Centric Recovery (OCR) pass after first-stage association.

    Runs OCR for still-Tracked unmatched tracks first to preserve active-track matching priority, then for
    Lost tracks on the detections still unmatched, so a recently-lost track cannot outbid an active one.
    """
    ocr_dets = [detections[i] for i in u_detection]
    if not ocr_dets:
        return u_track, u_detection

    tracked = [i for i in u_track if strack_pool[i].state == TrackState.Tracked]
    other = [i for i in u_track if strack_pool[i].state != TrackState.Tracked]

    u_t1, u_d1 = self._ocr_associate([strack_pool[i] for i in tracked], ocr_dets, activated, refind)
    remaining = [ocr_dets[j] for j in u_d1]
    u_t2, u_d2 = self._ocr_associate([strack_pool[i] for i in other], remaining, activated, refind)

    u_track = [tracked[i] for i in u_t1] + [other[i] for i in u_t2]
    u_detection = [u_detection[u_d1[j]] for j in u_d2]
    return u_track, u_detection
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSORT._second_association` {#ultralytics.trackers.oc\_sort.OCSORT.\_second\_association}

```python
def _second_association(
    self,
    strack_pool: list[OCSortTrack],
    u_track: list[int],
    detections_second: list[OCSortTrack],
    activated: list[OCSortTrack],
    refind: list[OCSortTrack],
    lost: list[OCSortTrack],
) -> None
```

Run ByteTrack-style second pass only when ``use_byte`` is enabled.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `strack_pool` | `list[OCSortTrack]` |  | *required* |
| `u_track` | `list[int]` |  | *required* |
| `detections_second` | `list[OCSortTrack]` |  | *required* |
| `activated` | `list[OCSortTrack]` |  | *required* |
| `refind` | `list[OCSortTrack]` |  | *required* |
| `lost` | `list[OCSortTrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L276-L293">View on GitHub</a>
```python
def _second_association(
    self,
    strack_pool: list[OCSortTrack],
    u_track: list[int],
    detections_second: list[OCSortTrack],
    activated: list[OCSortTrack],
    refind: list[OCSortTrack],
    lost: list[OCSortTrack],
) -> None:
    """Run ByteTrack-style second pass only when ``use_byte`` is enabled."""
    if not self.use_byte:
        for i in u_track:
            track = strack_pool[i]
            if track.state == TrackState.Tracked:
                track.mark_lost()
                lost.append(track)
        return
    super()._second_association(strack_pool, u_track, detections_second, activated, refind, lost)
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSORT._velocity_direction_cost` {#ultralytics.trackers.oc\_sort.OCSORT.\_velocity\_direction\_cost}

```python
def _velocity_direction_cost(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray
```

Compute OCM velocity direction consistency cost matrix (vectorized).

For each track-detection pair, measures the angular difference between the track's historical motion direction and the direction to the candidate detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracks` | `list[OCSortTrack]` | List of tracks. | *required* |
| `detections` | `list[OCSortTrack]` | List of detections. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Cost matrix of shape (len(tracks), len(detections)). |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L295-L328">View on GitHub</a>
```python
def _velocity_direction_cost(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
    """Compute OCM velocity direction consistency cost matrix (vectorized).

    For each track-detection pair, measures the angular difference between
    the track's historical motion direction and the direction to the candidate detection.

    Args:
        tracks (list[OCSortTrack]): List of tracks.
        detections (list[OCSortTrack]): List of detections.

    Returns:
        (np.ndarray): Cost matrix of shape (len(tracks), len(detections)).
    """
    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost.size == 0:
        return cost

    # Pre-extract detection centers as (M, 2) array
    det_centers = np.array([OCSortTrack._xyxy_center(det.xyxy) for det in detections], dtype=np.float32)

    for i, track in enumerate(tracks):
        if track.velocity is None or track.last_observation[0] < 0:
            continue
        track_center = OCSortTrack._xyxy_center(track.last_observation)
        directions = det_centers - track_center  # (M, 2)
        norms = np.linalg.norm(directions, axis=1)  # (M,)
        valid = norms > 1e-6
        if not valid.any():
            continue
        directions[valid] /= norms[valid, None]
        dots = np.clip(directions[valid] @ track.velocity, -1.0, 1.0)
        cost[i, valid] = np.arccos(dots) / np.pi

    return cost
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSORT.get_dists` {#ultralytics.trackers.oc\_sort.OCSORT.get\_dists}

```python
def get_dists(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray
```

Cost matrix = IoU (+score-fuse) + inertia·OCM (+ optional appearance, via hook).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracks` | `list[OCSortTrack]` |  | *required* |
| `detections` | `list[OCSortTrack]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L210-L215">View on GitHub</a>
```python
def get_dists(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
    """Cost matrix = IoU (+score-fuse) + inertia·OCM (+ optional appearance, via hook)."""
    iou_dists = matching.iou_distance(tracks, detections)
    dists = matching.fuse_score(iou_dists, detections) if self.args.fuse_score else iou_dists.copy()
    dists = dists + self.inertia * self._velocity_direction_cost(tracks, detections)
    return self._fuse_appearance(dists, tracks, detections, iou_dists=iou_dists)
```
</details>

<br>

### Method `ultralytics.trackers.oc_sort.OCSORT.init_track` {#ultralytics.trackers.oc\_sort.OCSORT.init\_track}

```python
def init_track(self, results, img: np.ndarray | None = None) -> list[OCSortTrack]
```

Build `OCSortTrack` instances from a `Results`-like object.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` |  |  | *required* |
| `img` | `np.ndarray \| None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py#L193-L198">View on GitHub</a>
```python
def init_track(self, results, img: np.ndarray | None = None) -> list[OCSortTrack]:
    """Build `OCSortTrack` instances from a `Results`-like object."""
    if len(results) == 0:
        return []
    bboxes = parse_bboxes(results)
    return [OCSortTrack(xywh, s, c, self.delta_t) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]
```
</details>

<br><br>
