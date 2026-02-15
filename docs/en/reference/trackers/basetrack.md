---
description: Discover the BaseTrack classes and methods for object tracking in YOLO by Ultralytics. Learn about TrackState, BaseTrack attributes, and methods.
keywords: Ultralytics, YOLO, object tracking, BaseTrack, TrackState, tracking methods, TrackState enumeration, object detection
---

# Reference for `ultralytics/trackers/basetrack.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`TrackState`](#ultralytics.trackers.basetrack.TrackState)
        - [`BaseTrack`](#ultralytics.trackers.basetrack.BaseTrack)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`BaseTrack.end_frame`](#ultralytics.trackers.basetrack.BaseTrack.end_frame)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`BaseTrack.next_id`](#ultralytics.trackers.basetrack.BaseTrack.next_id)
        - [`BaseTrack.activate`](#ultralytics.trackers.basetrack.BaseTrack.activate)
        - [`BaseTrack.predict`](#ultralytics.trackers.basetrack.BaseTrack.predict)
        - [`BaseTrack.update`](#ultralytics.trackers.basetrack.BaseTrack.update)
        - [`BaseTrack.mark_lost`](#ultralytics.trackers.basetrack.BaseTrack.mark_lost)
        - [`BaseTrack.mark_removed`](#ultralytics.trackers.basetrack.BaseTrack.mark_removed)
        - [`BaseTrack.reset_id`](#ultralytics.trackers.basetrack.BaseTrack.reset_id)


## Class `ultralytics.trackers.basetrack.TrackState` {#ultralytics.trackers.basetrack.TrackState}

```python
TrackState()
```

Enumeration class representing the possible states of an object being tracked.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `New` | `int` | State when the object is newly detected. |
| `Tracked` | `int` | State when the object is successfully tracked in subsequent frames. |
| `Lost` | `int` | State when the object is no longer tracked. |
| `Removed` | `int` | State when the object is removed from tracking. |

**Examples**

```python
>>> state = TrackState.New
>>> if state == TrackState.New:
...     print("Object is newly detected.")
```

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L10-L28"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class TrackState:
```
</details>


<br><br><hr><br>

## Class `ultralytics.trackers.basetrack.BaseTrack` {#ultralytics.trackers.basetrack.BaseTrack}

```python
BaseTrack(self)
```

Base class for object tracking, providing foundational attributes and methods.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `_count` | `int` | Class-level counter for unique track IDs. |
| `track_id` | `int` | Unique identifier for the track. |
| `is_activated` | `bool` | Flag indicating whether the track is currently active. |
| `state` | `TrackState` | Current state of the track. |
| `history` | `OrderedDict` | Ordered history of the track's states. |
| `features` | `list` | List of features extracted from the object for tracking. |
| `curr_feature` | `Any` | The current feature of the object being tracked. |
| `score` | `float` | The confidence score of the tracking. |
| `start_frame` | `int` | The frame number where tracking started. |
| `frame_id` | `int` | The most recent frame ID processed by the track. |
| `time_since_update` | `int` | Frames passed since the last update. |
| `location` | `tuple` | The location of the object in the context of multi-camera tracking. |

**Methods**

| Name | Description |
| --- | --- |
| [`end_frame`](#ultralytics.trackers.basetrack.BaseTrack.end_frame) | Return the ID of the most recent frame where the object was tracked. |
| [`activate`](#ultralytics.trackers.basetrack.BaseTrack.activate) | Activate the track with provided arguments, initializing necessary attributes for tracking. |
| [`mark_lost`](#ultralytics.trackers.basetrack.BaseTrack.mark_lost) | Mark the track as lost by updating its state to TrackState.Lost. |
| [`mark_removed`](#ultralytics.trackers.basetrack.BaseTrack.mark_removed) | Mark the track as removed by setting its state to TrackState.Removed. |
| [`next_id`](#ultralytics.trackers.basetrack.BaseTrack.next_id) | Increment and return the next unique global track ID for object tracking. |
| [`predict`](#ultralytics.trackers.basetrack.BaseTrack.predict) | Predict the next state of the track based on the current state and tracking model. |
| [`reset_id`](#ultralytics.trackers.basetrack.BaseTrack.reset_id) | Reset the global track ID counter to its initial value. |
| [`update`](#ultralytics.trackers.basetrack.BaseTrack.update) | Update the track with new observations and data, modifying its state and attributes accordingly. |

**Examples**

```python
Initialize a new track and mark it as lost:
>>> track = BaseTrack()
>>> track.mark_lost()
>>> print(track.state)  # Output: 2 (TrackState.Lost)
```

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L31-L115"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BaseTrack:
    """Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count (int): Class-level counter for unique track IDs.
        track_id (int): Unique identifier for the track.
        is_activated (bool): Flag indicating whether the track is currently active.
        state (TrackState): Current state of the track.
        history (OrderedDict): Ordered history of the track's states.
        features (list): List of features extracted from the object for tracking.
        curr_feature (Any): The current feature of the object being tracked.
        score (float): The confidence score of the tracking.
        start_frame (int): The frame number where tracking started.
        frame_id (int): The most recent frame ID processed by the track.
        time_since_update (int): Frames passed since the last update.
        location (tuple): The location of the object in the context of multi-camera tracking.

    Methods:
        end_frame: Returns the ID of the last frame where the object was tracked.
        next_id: Increments and returns the next global track ID.
        activate: Abstract method to activate the track.
        predict: Abstract method to predict the next state of the track.
        update: Abstract method to update the track with new data.
        mark_lost: Marks the track as lost.
        mark_removed: Marks the track as removed.
        reset_id: Resets the global track ID counter.

    Examples:
        Initialize a new track and mark it as lost:
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # Output: 2 (TrackState.Lost)
    """

    _count = 0

    def __init__(self):
        """Initialize a new track with a unique ID and foundational tracking attributes."""
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)
```
</details>

<br>

### Property `ultralytics.trackers.basetrack.BaseTrack.end_frame` {#ultralytics.trackers.basetrack.BaseTrack.end\_frame}

```python
def end_frame(self) -> int
```

Return the ID of the most recent frame where the object was tracked.

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L82-L84"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def end_frame(self) -> int:
    """Return the ID of the most recent frame where the object was tracked."""
    return self.frame_id
```
</details>

<br>

### Method `ultralytics.trackers.basetrack.BaseTrack.activate` {#ultralytics.trackers.basetrack.BaseTrack.activate}

```python
def activate(self, *args: Any) -> None
```

Activate the track with provided arguments, initializing necessary attributes for tracking.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L92-L94"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def activate(self, *args: Any) -> None:
    """Activate the track with provided arguments, initializing necessary attributes for tracking."""
    raise NotImplementedError
```
</details>

<br>

### Method `ultralytics.trackers.basetrack.BaseTrack.mark_lost` {#ultralytics.trackers.basetrack.BaseTrack.mark\_lost}

```python
def mark_lost(self) -> None
```

Mark the track as lost by updating its state to TrackState.Lost.

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L104-L106"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def mark_lost(self) -> None:
    """Mark the track as lost by updating its state to TrackState.Lost."""
    self.state = TrackState.Lost
```
</details>

<br>

### Method `ultralytics.trackers.basetrack.BaseTrack.mark_removed` {#ultralytics.trackers.basetrack.BaseTrack.mark\_removed}

```python
def mark_removed(self) -> None
```

Mark the track as removed by setting its state to TrackState.Removed.

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L108-L110"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def mark_removed(self) -> None:
    """Mark the track as removed by setting its state to TrackState.Removed."""
    self.state = TrackState.Removed
```
</details>

<br>

### Method `ultralytics.trackers.basetrack.BaseTrack.next_id` {#ultralytics.trackers.basetrack.BaseTrack.next\_id}

```python
def next_id() -> int
```

Increment and return the next unique global track ID for object tracking.

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L87-L90"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def next_id() -> int:
    """Increment and return the next unique global track ID for object tracking."""
    BaseTrack._count += 1
    return BaseTrack._count
```
</details>

<br>

### Method `ultralytics.trackers.basetrack.BaseTrack.predict` {#ultralytics.trackers.basetrack.BaseTrack.predict}

```python
def predict(self) -> None
```

Predict the next state of the track based on the current state and tracking model.

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L96-L98"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(self) -> None:
    """Predict the next state of the track based on the current state and tracking model."""
    raise NotImplementedError
```
</details>

<br>

### Method `ultralytics.trackers.basetrack.BaseTrack.reset_id` {#ultralytics.trackers.basetrack.BaseTrack.reset\_id}

```python
def reset_id() -> None
```

Reset the global track ID counter to its initial value.

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L113-L115"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def reset_id() -> None:
    """Reset the global track ID counter to its initial value."""
    BaseTrack._count = 0
```
</details>

<br>

### Method `ultralytics.trackers.basetrack.BaseTrack.update` {#ultralytics.trackers.basetrack.BaseTrack.update}

```python
def update(self, *args: Any, **kwargs: Any) -> None
```

Update the track with new observations and data, modifying its state and attributes accordingly.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` |  | *required* |
| `**kwargs` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/basetrack.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py#L100-L102"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, *args: Any, **kwargs: Any) -> None:
    """Update the track with new observations and data, modifying its state and attributes accordingly."""
    raise NotImplementedError
```
</details>

<br><br>
