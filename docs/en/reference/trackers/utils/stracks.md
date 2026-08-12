---
title: trackers.utils.stracks API Reference
description: Reference for shared track-pool utilities including merge, joint, subtract, duplicate removal, and global motion compensation helpers.
keywords: Ultralytics, stracks, track pools, multi-object tracking, GMC, STrack utilities
---

# Reference for `ultralytics/trackers/utils/stracks.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/stracks.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/stracks.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`merge_track_pools`](#ultralytics.trackers.utils.stracks.merge_track_pools)
        - [`parse_bboxes`](#ultralytics.trackers.utils.stracks.parse_bboxes)
        - [`joint_stracks`](#ultralytics.trackers.utils.stracks.joint_stracks)
        - [`sub_stracks`](#ultralytics.trackers.utils.stracks.sub_stracks)
        - [`remove_duplicate_stracks`](#ultralytics.trackers.utils.stracks.remove_duplicate_stracks)
        - [`multi_gmc`](#ultralytics.trackers.utils.stracks.multi_gmc)


## Function `ultralytics.trackers.utils.stracks.merge_track_pools` {#ultralytics.trackers.utils.stracks.merge\_track\_pools}

```python
def merge_track_pools(
    tracker,
    activated: list,
    refind: list,
    lost: list,
    removed: list,
    removed_buffer: int = 1000,
) -> None
```

Apply the standard end-of-frame bookkeeping to a tracker's persistent pools in place.

Merges newly activated and re-found tracks into `tracker.tracked_stracks`, moves the transitioned tracks into `tracker.lost_stracks`, dedups by IoU, appends removals to `tracker.removed_stracks`, and trims the removed buffer to `removed_buffer` entries.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tracker` | `Any` | Object exposing `tracked_stracks`, `lost_stracks`, `removed_stracks` lists. | *required* |
| `activated` | `list` | Tracks updated from the Tracked state this frame. | *required* |
| `refind` | `list` | Tracks re-activated from the Lost state this frame. | *required* |
| `lost` | `list` | Tracks transitioned to Lost this frame. | *required* |
| `removed` | `list` | Tracks transitioned to Removed this frame. | *required* |
| `removed_buffer` | `int` | Maximum number of historical removed tracks to retain. | `1000` |

**Examples**

Run end-of-frame bookkeeping inside a tracker's `update` method

```python
>>> merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)
```

<details>
<summary>Source code in <code>ultralytics/trackers/utils/stracks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/stracks.py#L18-L56">View on GitHub</a>
```python
def merge_track_pools(
    tracker,
    activated: list,
    refind: list,
    lost: list,
    removed: list,
    removed_buffer: int = 1000,
) -> None:
    """Apply the standard end-of-frame bookkeeping to a tracker's persistent pools in place.

    Merges newly activated and re-found tracks into `tracker.tracked_stracks`, moves the transitioned tracks into
    `tracker.lost_stracks`, dedups by IoU, appends removals to `tracker.removed_stracks`, and trims the removed buffer
    to `removed_buffer` entries.

    Args:
        tracker (Any): Object exposing `tracked_stracks`, `lost_stracks`, `removed_stracks` lists.
        activated (list): Tracks updated from the Tracked state this frame.
        refind (list): Tracks re-activated from the Lost state this frame.
        lost (list): Tracks transitioned to Lost this frame.
        removed (list): Tracks transitioned to Removed this frame.
        removed_buffer (int): Maximum number of historical removed tracks to retain.

    Examples:
        Run end-of-frame bookkeeping inside a tracker's `update` method
        >>> merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)
    """
    tracker.tracked_stracks = [t for t in tracker.tracked_stracks if t.state == TrackState.Tracked]
    tracker.tracked_stracks = joint_stracks(tracker.tracked_stracks, activated)
    tracker.tracked_stracks = joint_stracks(tracker.tracked_stracks, refind)
    tracker.lost_stracks = sub_stracks(tracker.lost_stracks, tracker.tracked_stracks)
    tracker.lost_stracks.extend(lost)
    tracker.lost_stracks = sub_stracks(tracker.lost_stracks, tracker.removed_stracks)
    tracker.tracked_stracks, tracker.lost_stracks = remove_duplicate_stracks(
        tracker.tracked_stracks, tracker.lost_stracks
    )
    tracker.removed_stracks_frame = removed
    tracker.removed_stracks.extend(removed)
    if len(tracker.removed_stracks) > removed_buffer:
        tracker.removed_stracks = tracker.removed_stracks[-removed_buffer:]
```
</details>


<br><br><hr><br>

## Function `ultralytics.trackers.utils.stracks.parse_bboxes` {#ultralytics.trackers.utils.stracks.parse\_bboxes}

```python
def parse_bboxes(results) -> np.ndarray
```

Return detection bounding boxes with appended indices from a Results-like object.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` | `Any` | Object exposing ``xywh`` (or ``xywhr``), ``conf``, and ``cls``. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Array of shape ``(N, 5)`` for ``xywh`` or ``(N, 6)`` for ``xywhr``, with the last column containing the original detection index. |

<details>
<summary>Source code in <code>ultralytics/trackers/utils/stracks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/stracks.py#L59-L70">View on GitHub</a>
```python
def parse_bboxes(results) -> np.ndarray:
    """Return detection bounding boxes with appended indices from a Results-like object.

    Args:
        results (Any): Object exposing ``xywh`` (or ``xywhr``), ``conf``, and ``cls``.

    Returns:
        (np.ndarray): Array of shape ``(N, 5)`` for ``xywh`` or ``(N, 6)`` for ``xywhr``, with the last column
            containing the original detection index.
    """
    bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
    return np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
```
</details>


<br><br><hr><br>

## Function `ultralytics.trackers.utils.stracks.joint_stracks` {#ultralytics.trackers.utils.stracks.joint\_stracks}

```python
def joint_stracks(atracks: list, btracks: list) -> list
```

Combine two track lists into one, de-duplicating by `track_id`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `atracks` | `list[STrack]` | First list of tracks; entries win on `track_id` collisions. | *required* |
| `btracks` | `list[STrack]` | Second list of tracks. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[STrack]` | Union of `atracks` and `btracks` with duplicate `track_id`s removed. |

**Examples**

Merge the currently tracked pool with newly activated tracks

```python
>>> merged = joint_stracks(tracked_stracks, activated_stracks)
```

<details>
<summary>Source code in <code>ultralytics/trackers/utils/stracks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/stracks.py#L73-L88">View on GitHub</a>
```python
def joint_stracks(atracks: list, btracks: list) -> list:
    """Combine two track lists into one, de-duplicating by `track_id`.

    Args:
        atracks (list[STrack]): First list of tracks; entries win on `track_id` collisions.
        btracks (list[STrack]): Second list of tracks.

    Returns:
        (list[STrack]): Union of `atracks` and `btracks` with duplicate `track_id`s removed.

    Examples:
        Merge the currently tracked pool with newly activated tracks
        >>> merged = joint_stracks(tracked_stracks, activated_stracks)
    """
    a_ids = {t.track_id for t in atracks}
    return atracks + [t for t in btracks if t.track_id not in a_ids]
```
</details>


<br><br><hr><br>

## Function `ultralytics.trackers.utils.stracks.sub_stracks` {#ultralytics.trackers.utils.stracks.sub\_stracks}

```python
def sub_stracks(atracks: list, btracks: list) -> list
```

Filter out tracks from `atracks` whose `track_id` appears in `btracks`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `atracks` | `list[STrack]` | Source list of tracks to filter. | *required* |
| `btracks` | `list[STrack]` | Tracks whose `track_id`s should be excluded from the output. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[STrack]` | Elements of `atracks` whose `track_id` is not present in `btracks`. |

**Examples**

Remove any re-tracked objects from the lost pool

```python
>>> lost_stracks = sub_stracks(lost_stracks, tracked_stracks)
```

<details>
<summary>Source code in <code>ultralytics/trackers/utils/stracks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/stracks.py#L91-L106">View on GitHub</a>
```python
def sub_stracks(atracks: list, btracks: list) -> list:
    """Filter out tracks from `atracks` whose `track_id` appears in `btracks`.

    Args:
        atracks (list[STrack]): Source list of tracks to filter.
        btracks (list[STrack]): Tracks whose `track_id`s should be excluded from the output.

    Returns:
        (list[STrack]): Elements of `atracks` whose `track_id` is not present in `btracks`.

    Examples:
        Remove any re-tracked objects from the lost pool
        >>> lost_stracks = sub_stracks(lost_stracks, tracked_stracks)
    """
    btrack_ids = {t.track_id for t in btracks}
    return [t for t in atracks if t.track_id not in btrack_ids]
```
</details>


<br><br><hr><br>

## Function `ultralytics.trackers.utils.stracks.remove_duplicate_stracks` {#ultralytics.trackers.utils.stracks.remove\_duplicate\_stracks}

```python
def remove_duplicate_stracks(atracks: list, btracks: list, dup_thresh: float = 0.15) -> tuple[list, list]
```

Remove duplicate tracks across two lists based on Intersection over Union (IoU) distance.

Track pairs with IoU distance < `dup_thresh` (IoU > `1 - dup_thresh`) are treated as duplicates of the same object. The shorter-lived track (smaller `frame_id - start_frame`) is dropped; ties drop from `atracks`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `atracks` | `list[STrack]` | First list of tracks; entries must expose `xyxy`, `frame_id`, and `start_frame`. | *required* |
| `btracks` | `list[STrack]` | Second list of tracks with the same attribute requirements. | *required* |
| `dup_thresh` | `float` | IoU-distance ceiling for treating two tracks as duplicates. Default 0.15 (IoU > 0.85). | `0.15` |

**Returns**

| Type | Description |
| --- | --- |
| `resa (list[STrack])` | `atracks` with duplicate tracks removed. |
| `resb (list[STrack])` | `btracks` with duplicate tracks removed. |

**Examples**

De-duplicate the tracked and lost pools at the end of a frame

```python
>>> tracked, lost = remove_duplicate_stracks(tracked_stracks, lost_stracks)
```

<details>
<summary>Source code in <code>ultralytics/trackers/utils/stracks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/stracks.py#L109-L142">View on GitHub</a>
```python
def remove_duplicate_stracks(atracks: list, btracks: list, dup_thresh: float = 0.15) -> tuple[list, list]:
    """Remove duplicate tracks across two lists based on Intersection over Union (IoU) distance.

    Track pairs with IoU distance < `dup_thresh` (IoU > `1 - dup_thresh`) are treated as duplicates of the same
    object. The shorter-lived track (smaller `frame_id - start_frame`) is dropped; ties drop
    from `atracks`.

    Args:
        atracks (list[STrack]): First list of tracks; entries must expose `xyxy`, `frame_id`, and `start_frame`.
        btracks (list[STrack]): Second list of tracks with the same attribute requirements.
        dup_thresh (float): IoU-distance ceiling for treating two tracks as duplicates. Default 0.15 (IoU > 0.85).

    Returns:
        resa (list[STrack]): `atracks` with duplicate tracks removed.
        resb (list[STrack]): `btracks` with duplicate tracks removed.

    Examples:
        De-duplicate the tracked and lost pools at the end of a frame
        >>> tracked, lost = remove_duplicate_stracks(tracked_stracks, lost_stracks)
    """
    pdist = matching.iou_distance(atracks, btracks)
    pairs = np.where(pdist < dup_thresh)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = atracks[p].frame_id - atracks[p].start_frame
        timeq = btracks[q].frame_id - btracks[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    dupa_set, dupb_set = set(dupa), set(dupb)
    resa = [t for i, t in enumerate(atracks) if i not in dupa_set]
    resb = [t for i, t in enumerate(btracks) if i not in dupb_set]
    return resa, resb
```
</details>


<br><br><hr><br>

## Function `ultralytics.trackers.utils.stracks.multi_gmc` {#ultralytics.trackers.utils.stracks.multi\_gmc}

```python
def multi_gmc(stracks: list, H: np.ndarray) -> None
```

Update multiple track positions and covariances using a 2x3 affine homography.

The Kalman state is assumed to be laid out as `(*box, *box_velocity)` with the box center `(x, y)` in the first two dims. `R8x8` rotates all four 2-d pairs block-diagonally; the translation `t` is applied only to the position. This assumes the state layout is four spatial/velocity pairs (e.g. XYWH); XYAH trackers must override this.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stracks` | `list[STrack]` | Tracks to warp in place; each must expose `mean` (shape (8,)) and `covariance` (shape (8, 8)). | *required* |
| `H` | `np.ndarray` | 2x3 affine homography mapping the previous frame to the current one. | *required* |

**Examples**

Apply camera-motion compensation to the active track pool

```python
>>> warp = gmc.apply(frame, detection_boxes)
>>> multi_gmc(tracked_stracks, warp)
```

<details>
<summary>Source code in <code>ultralytics/trackers/utils/stracks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/stracks.py#L145-L179">View on GitHub</a>
```python
def multi_gmc(stracks: list, H: np.ndarray) -> None:
    """Update multiple track positions and covariances using a 2x3 affine homography.

    The Kalman state is assumed to be laid out as `(*box, *box_velocity)` with the box center `(x, y)` in the first two
    dims. `R8x8` rotates all four 2-d pairs block-diagonally; the translation `t` is applied only to the position.
    This assumes the state layout is four spatial/velocity pairs (e.g. XYWH); XYAH trackers must override this.

    Args:
        stracks (list[STrack]): Tracks to warp in place; each must expose `mean` (shape (8,)) and `covariance` (shape
            (8, 8)).
        H (np.ndarray): 2x3 affine homography mapping the previous frame to the current one.

    Examples:
        Apply camera-motion compensation to the active track pool
        >>> warp = gmc.apply(frame, detection_boxes)
        >>> multi_gmc(tracked_stracks, warp)
    """
    if not stracks:
        return
    multi_mean = np.asarray([st.mean for st in stracks])
    multi_covariance = np.asarray([st.covariance for st in stracks])

    R = H[:2, :2]
    R8x8 = np.kron(np.eye(4, dtype=np.float32), R)
    t = H[:2, 2]

    multi_mean = np.matmul(R8x8, multi_mean[..., None])[..., 0]
    multi_mean[:, :2] += t
    # Keep the right operand C-contiguous. An F-contiguous one sends matmul into BLAS's transposed-gemm kernel,
    # which on macOS Accelerate leaves FP-exception flags set even for finite inputs — numpy then reports spurious
    # divide/overflow/invalid RuntimeWarnings — and measures slower here than the untransposed kernel.
    multi_covariance = np.matmul(np.matmul(R8x8, multi_covariance), np.ascontiguousarray(R8x8.T))
    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        stracks[i].mean = mean
        stracks[i].covariance = cov
```
</details>

<br><br>
