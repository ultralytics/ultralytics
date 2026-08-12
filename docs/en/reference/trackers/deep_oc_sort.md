---
title: trackers.deep_oc_sort API Reference
description: Explore the Deep OC-SORT module in Ultralytics for observation-centric multi-object tracking with ReID appearance features and camera motion compensation. Learn about its classes, methods, and attributes.
keywords: Ultralytics, Deep OC-SORT, DeepOCSORT, observation-centric, object tracking, Kalman filter, ReID, GMC, YOLO, documentation
---

# Reference for `ultralytics/trackers/deep_oc_sort.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DeepOCSortTrack`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack)
        - [`DeepOCSORT`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DeepOCSortTrack.update_features`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.update_features)
        - [`DeepOCSortTrack.update`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.update)
        - [`DeepOCSortTrack.re_activate`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.re_activate)
        - [`DeepOCSortTrack.multi_gmc`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.multi_gmc)
        - [`DeepOCSORT.init_track`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT.init_track)
        - [`DeepOCSORT._pre_first_associate`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT._pre_first_associate)
        - [`DeepOCSORT._fuse_appearance`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT._fuse_appearance)
        - [`DeepOCSORT.reset`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT.reset)


## Class `ultralytics.trackers.deep_oc_sort.DeepOCSortTrack` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSortTrack}

```python
DeepOCSortTrack(
    xywh: np.ndarray,
    score: float,
    cls: Any,
    delta_t: int = 3,
    feat: np.ndarray | None = None,
    alpha_fixed_emb: float = 0.95,
    det_thresh: float = 0.25,
)
```

**Bases:** `OCSortTrack`

Track object for Deep OC-SORT with appearance features and observation-centric state management.

Extends OCSortTrack with ReID embedding storage and exponential moving average smoothing, plus confidence-adaptive embedding update rates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `xywh` | `np.ndarray` | Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format. | *required* |
| `score` | `float` | Detection confidence in `[0, 1]`. | *required* |
| `cls` | `Any` | Class label for the detection. | *required* |
| `delta_t` | `int` | Temporal window for OCM velocity direction computation. | `3` |
| `feat` | `np.ndarray \| None` | Optional appearance feature vector for this detection. | `None` |
| `alpha_fixed_emb` | `float` | Base EMA factor for embedding updates; higher = slower updates. | `0.95` |
| `det_thresh` | `float` | Detection-confidence threshold below which the embedding is replaced rather than blended. | `0.25` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `smooth_feat` | `np.ndarray \| None` | Smoothed feature vector via EMA. |
| `curr_feat` | `np.ndarray \| None` | Current frame's feature vector. |
| `alpha_fixed_emb` | `float` | Base EMA factor for embedding updates. |
| `det_thresh` | `float` | Confidence threshold below which a new embedding is ignored rather than blended. |

**Methods**

| Name | Description |
| --- | --- |
| [`multi_gmc`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.multi_gmc) | Apply global motion compensation correctly for the XYAH Kalman state. |
| [`re_activate`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.re_activate) | Re-activate a lost track and refresh appearance features. |
| [`update`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.update) | Update track state with a matched detection and refresh appearance features. |
| [`update_features`](#ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.update_features) | Blend a new appearance feature into `smooth_feat` with confidence-adaptive EMA. |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L17-L156">View on GitHub</a>
```python
class DeepOCSortTrack(OCSortTrack):
    """Track object for Deep OC-SORT with appearance features and observation-centric state management.

    Extends OCSortTrack with ReID embedding storage and exponential moving average smoothing, plus confidence-adaptive
    embedding update rates.

    Attributes:
        smooth_feat (np.ndarray | None): Smoothed feature vector via EMA.
        curr_feat (np.ndarray | None): Current frame's feature vector.
        alpha_fixed_emb (float): Base EMA factor for embedding updates.
        det_thresh (float): Confidence threshold below which a new embedding is ignored rather than blended.
    """

    def __init__(
        self,
        xywh: np.ndarray,
        score: float,
        cls: Any,
        delta_t: int = 3,
        feat: np.ndarray | None = None,
        alpha_fixed_emb: float = 0.95,
        det_thresh: float = 0.25,
    ):
        """Initialize a `DeepOCSortTrack` with optional appearance features.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format.
            score (float): Detection confidence in `[0, 1]`.
            cls (Any): Class label for the detection.
            delta_t (int): Temporal window for OCM velocity direction computation.
            feat (np.ndarray | None): Optional appearance feature vector for this detection.
            alpha_fixed_emb (float): Base EMA factor for embedding updates; higher = slower updates.
            det_thresh (float): Detection-confidence threshold below which the embedding is replaced rather than
                blended.
        """
        super().__init__(xywh, score, cls, delta_t)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha_fixed_emb = alpha_fixed_emb
        self.det_thresh = det_thresh
        if feat is not None:
            self.update_features(feat, score)
```
</details>

<br>

### Method `ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.multi_gmc` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSortTrack.multi\_gmc}

```python
def multi_gmc(stracks: list[DeepOCSortTrack], H: np.ndarray) -> None
```

Apply global motion compensation correctly for the XYAH Kalman state.

The standard `multi_gmc` helper from `utils.stracks` would rotate `(a, h)` along with `(x, y)`, which is wrong for an aspect-ratio dim. This variant only rotates the position `(x, y)` and the velocity `(vx, vy)` blocks, leaves `(a, h)` and `(va, vh)` untouched, and also rotates the stored `last_observation` so OCR/ORU stay consistent.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stracks` | `list[DeepOCSortTrack]` | Tracks to warp in place. | *required* |
| `H` | `np.ndarray` | 2x3 affine homography mapping the previous frame to the current one. | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L105-L156">View on GitHub</a>
```python
@staticmethod
def multi_gmc(stracks: list[DeepOCSortTrack], H: np.ndarray) -> None:
    """Apply global motion compensation correctly for the XYAH Kalman state.

    The standard `multi_gmc` helper from `utils.stracks` would rotate `(a, h)` along with
    `(x, y)`, which is wrong for an aspect-ratio dim. This variant only rotates the
    position `(x, y)` and the velocity `(vx, vy)` blocks, leaves `(a, h)` and `(va, vh)`
    untouched, and also rotates the stored `last_observation` so OCR/ORU stay consistent.

    Args:
        stracks (list[DeepOCSortTrack]): Tracks to warp in place.
        H (np.ndarray): 2x3 affine homography mapping the previous frame to the current one.
    """
    if not stracks:
        return
    multi_mean = np.asarray([st.mean for st in stracks])
    multi_covariance = np.asarray([st.covariance for st in stracks])

    R = H[:2, :2]
    t = H[:2, 2]

    # Build 8x8 transform: rotate (x,y) and (vx,vy), identity for (a,h) and (va,vh)
    R8x8 = np.eye(8)  # float64 holds any homography exactly; a narrower one would round the float64 methods
    R8x8[:2, :2] = R  # rotate position (x, y)
    R8x8[4:6, 4:6] = R  # rotate velocity (vx, vy)
    # indices 2,3 (a,h) and 6,7 (va,vh) remain identity

    multi_mean = np.matmul(R8x8, multi_mean[..., None])[..., 0]
    multi_mean[:, :2] += t
    # Keep the right operand C-contiguous, as the `utils.stracks` sibling does: an F-contiguous one sends matmul
    # into BLAS's transposed-gemm kernel, which on macOS Accelerate leaves FP-exception flags set even for finite
    # inputs, and numpy then reports spurious divide-by-zero RuntimeWarnings.
    multi_covariance = np.matmul(np.matmul(R8x8, multi_covariance), np.ascontiguousarray(R8x8.T))
    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        stracks[i].mean = mean
        stracks[i].covariance = cov

        # Also transform stored observations for OCR/ORU consistency
        if stracks[i].last_observation[0] >= 0:
            obs = stracks[i].last_observation
            # Transform xyxy observation centers
            cx, cy = DeepOCSortTrack._xyxy_center(obs)
            w, h = obs[2] - obs[0], obs[3] - obs[1]
            new_c = R @ np.array([cx, cy]) + t
            stracks[i].last_observation = np.array(
                [
                    new_c[0] - w / 2,
                    new_c[1] - h / 2,
                    new_c[0] + w / 2,
                    new_c[1] + h / 2,
                ],
                dtype=np.float32,
            )
```
</details>

<br>

### Method `ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.re_activate` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSortTrack.re\_activate}

```python
def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None
```

Re-activate a lost track and refresh appearance features.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_track` | `STrack` | Detection used to revive this track. | *required* |
| `frame_id` | `int` | Current frame id. | *required* |
| `new_id` | `bool` | If True, assign a fresh track id instead of reusing the old one. | `False` |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L92-L102">View on GitHub</a>
```python
def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
    """Re-activate a lost track and refresh appearance features.

    Args:
        new_track (STrack): Detection used to revive this track.
        frame_id (int): Current frame id.
        new_id (bool): If True, assign a fresh track id instead of reusing the old one.
    """
    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat, new_track.score)
    super().re_activate(new_track, frame_id, new_id)
```
</details>

<br>

### Method `ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.update` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSortTrack.update}

```python
def update(self, new_track: STrack, frame_id: int) -> None
```

Update track state with a matched detection and refresh appearance features.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_track` | `STrack` | Matched detection for this frame, with optional `curr_feat`. | *required* |
| `frame_id` | `int` | Current frame id. | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L81-L90">View on GitHub</a>
```python
def update(self, new_track: STrack, frame_id: int) -> None:
    """Update track state with a matched detection and refresh appearance features.

    Args:
        new_track (STrack): Matched detection for this frame, with optional `curr_feat`.
        frame_id (int): Current frame id.
    """
    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat, new_track.score)
    super().update(new_track, frame_id)
```
</details>

<br>

### Method `ultralytics.trackers.deep_oc_sort.DeepOCSortTrack.update_features` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSortTrack.update\_features}

```python
def update_features(self, feat: np.ndarray, score: float | None = None) -> None
```

Blend a new appearance feature into `smooth_feat` with confidence-adaptive EMA.

When `score` exceeds `det_thresh`, the EMA factor `alpha = alpha_fixed_emb + (1 - alpha_fixed_emb) * (1 - trust)` with `trust = (score - det_thresh) / (1 - det_thresh)`, so a high-confidence detection blends in more strongly. At or below `det_thresh`, `alpha = 1.0` so the existing `smooth_feat` is kept and the new (low-trust) feature is ignored — same behavior used by the reference Deep OC-SORT implementation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `feat` | `np.ndarray` | New (un-normalized) appearance feature vector. | *required* |
| `score` | `float \| None` | Detection confidence used to modulate the EMA factor. | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L60-L79">View on GitHub</a>
```python
def update_features(self, feat: np.ndarray, score: float | None = None) -> None:
    """Blend a new appearance feature into `smooth_feat` with confidence-adaptive EMA.

    When `score` exceeds `det_thresh`, the EMA factor `alpha = alpha_fixed_emb + (1 - alpha_fixed_emb) * (1 - trust)`
    with `trust = (score - det_thresh) / (1 - det_thresh)`, so a high-confidence detection blends in more strongly.
    At or below `det_thresh`, `alpha = 1.0` so the existing `smooth_feat` is kept and the new (low-trust) feature is
    ignored — same behavior used by the reference Deep OC-SORT implementation.

    Args:
        feat (np.ndarray): New (un-normalized) appearance feature vector.
        score (float | None): Detection confidence used to modulate the EMA factor.
    """
    if score is not None and score > self.det_thresh:
        trust = (score - self.det_thresh) / max(1 - self.det_thresh, 1e-9)
        alpha = self.alpha_fixed_emb + (1 - self.alpha_fixed_emb) * (1 - trust)
    else:
        alpha = 1.0  # keep the existing smooth_feat for low-trust detections
    curr, smooth = smooth_feature(feat, self.smooth_feat, alpha)
    if curr is not None:
        self.curr_feat, self.smooth_feat = curr, smooth
```
</details>


<br><br><hr><br>

## Class `ultralytics.trackers.deep_oc_sort.DeepOCSORT` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSORT}

```python
DeepOCSORT(args: Any)
```

**Bases:** `OCSORT`

Deep OC-SORT: OC-SORT enhanced with appearance features, GMC, and adaptive weighting.

Fixes over naive integration:
- GMC correctly handles XYAH state (rotates only x,y positions, not aspect ratio/height)
- Cost combination uses min(IoU, appearance) following BOTSORT's proven approach
- OCR recovery pass also uses appearance features
- ByteTrack-style low-confidence second pass disabled by default

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `Namespace \| IterableSimpleNamespace` | Parsed tracker config providing the OC-SORT keys plus `gmc_method`, `proximity_thresh`, `appearance_thresh`, `alpha_fixed_emb`, `with_reid`, and `model`. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`_fuse_appearance`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT._fuse_appearance) | Min-fuse appearance distance into the motion cost (BoT-SORT-style). |
| [`_pre_first_associate`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT._pre_first_associate) | Apply GMC warp to Kalman state before first-stage association. |
| [`init_track`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT.init_track) | Build `DeepOCSortTrack` instances, attaching ReID features when enabled. |
| [`reset`](#ultralytics.trackers.deep_oc_sort.DeepOCSORT.reset) | Reset the Deep OC-SORT tracker, also clearing the GMC warp state. |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L159-L266">View on GitHub</a>
```python
class DeepOCSORT(OCSORT):
    """Deep OC-SORT: OC-SORT enhanced with appearance features, GMC, and adaptive weighting.

    Fixes over naive integration:
    - GMC correctly handles XYAH state (rotates only x,y positions, not aspect ratio/height)
    - Cost combination uses min(IoU, appearance) following BOTSORT's proven approach
    - OCR recovery pass also uses appearance features
    - ByteTrack-style low-confidence second pass disabled by default
    """

    def __init__(self, args: Any):
        """Initialize Deep OC-SORT tracker.

        Args:
            args (Namespace | IterableSimpleNamespace): Parsed tracker config providing the OC-SORT keys plus
                `gmc_method`, `proximity_thresh`, `appearance_thresh`, `alpha_fixed_emb`, `with_reid`, and `model`.
        """
        super().__init__(args)

        # GMC for camera motion compensation
        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))

        # Appearance parameters
        self.proximity_thresh = getattr(args, "proximity_thresh", 0.5)
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.75)
        self.alpha_fixed_emb = getattr(args, "alpha_fixed_emb", 0.95)

        self.encoder = build_encoder(
            getattr(args, "with_reid", False), getattr(args, "model", "auto"), getattr(args, "device", None)
        )
```
</details>

<br>

### Method `ultralytics.trackers.deep_oc_sort.DeepOCSORT._fuse_appearance` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSORT.\_fuse\_appearance}

```python
def _fuse_appearance(
    self,
    dists: np.ndarray,
    tracks: list[DeepOCSortTrack],
    detections: list[DeepOCSortTrack],
    iou_dists: np.ndarray | None = None,
) -> np.ndarray
```

Min-fuse appearance distance into the motion cost (BoT-SORT-style).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dists` | `np.ndarray` |  | *required* |
| `tracks` | `list[DeepOCSortTrack]` |  | *required* |
| `detections` | `list[DeepOCSortTrack]` |  | *required* |
| `iou_dists` | `np.ndarray \| None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L247-L261">View on GitHub</a>
```python
def _fuse_appearance(
    self,
    dists: np.ndarray,
    tracks: list[DeepOCSortTrack],
    detections: list[DeepOCSortTrack],
    iou_dists: np.ndarray | None = None,
) -> np.ndarray:
    """Min-fuse appearance distance into the motion cost (BoT-SORT-style)."""
    if self.encoder is None or not tracks or not detections:
        return dists
    emb_dists = matching.embedding_distance(tracks, detections) / 2.0
    emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
    if iou_dists is not None:
        emb_dists[iou_dists > (1 - self.proximity_thresh)] = 1.0
    return np.minimum(dists, emb_dists)
```
</details>

<br>

### Method `ultralytics.trackers.deep_oc_sort.DeepOCSORT._pre_first_associate` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSORT.\_pre\_first\_associate}

```python
def _pre_first_associate(
    self,
    strack_pool: list[DeepOCSortTrack],
    unconfirmed: list[DeepOCSortTrack],
    img: np.ndarray | None,
    results_high: Any,
) -> None
```

Apply GMC warp to Kalman state before first-stage association.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `strack_pool` | `list[DeepOCSortTrack]` |  | *required* |
| `unconfirmed` | `list[DeepOCSortTrack]` |  | *required* |
| `img` | `np.ndarray \| None` |  | *required* |
| `results_high` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L230-L245">View on GitHub</a>
```python
def _pre_first_associate(
    self,
    strack_pool: list[DeepOCSortTrack],
    unconfirmed: list[DeepOCSortTrack],
    img: np.ndarray | None,
    results_high: Any,
) -> None:
    """Apply GMC warp to Kalman state before first-stage association."""
    if img is None or self.gmc.method is None:
        return
    try:
        warp = self.gmc.apply(img, results_high.xyxy if len(results_high) else np.empty((0, 4)))
    except Exception:
        warp = np.eye(2, 3)
    DeepOCSortTrack.multi_gmc(strack_pool, warp)
    DeepOCSortTrack.multi_gmc(unconfirmed, warp)
```
</details>

<br>

### Method `ultralytics.trackers.deep_oc_sort.DeepOCSORT.init_track` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSORT.init\_track}

```python
def init_track(self, results, img: np.ndarray | None = None) -> list[DeepOCSortTrack]
```

Build `DeepOCSortTrack` instances, attaching ReID features when enabled.

When `with_reid=True` and `model="auto"`, `img` should already be a list of native backbone features (one per detection); for any other `model`, `img` is the source frame and the configured external ReID encoder is invoked on detection crops.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` | `Any` | Object exposing `xywh` (or `xywhr`), `conf`, and `cls`. | *required* |
| `img` | `np.ndarray \| None` | Either the BGR frame or pre-extracted feature tensors, depending on the ReID configuration. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `list[DeepOCSortTrack]` | One track per detection, empty if no detections. |

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L190-L228">View on GitHub</a>
```python
def init_track(self, results, img: np.ndarray | None = None) -> list[DeepOCSortTrack]:
    """Build `DeepOCSortTrack` instances, attaching ReID features when enabled.

    When `with_reid=True` and `model="auto"`, `img` should already be a list of native
    backbone features (one per detection); for any other `model`, `img` is the source frame
    and the configured external ReID encoder is invoked on detection crops.

    Args:
        results (Any): Object exposing `xywh` (or `xywhr`), `conf`, and `cls`.
        img (np.ndarray | None): Either the BGR frame or pre-extracted feature tensors, depending on the ReID
            configuration.

    Returns:
        (list[DeepOCSortTrack]): One track per detection, empty if no detections.
    """
    if len(results) == 0:
        return []
    bboxes = parse_bboxes(results)

    if self.encoder is not None and img is not None:
        features = self.encoder(img, bboxes)
        return [
            DeepOCSortTrack(
                xywh,
                s,
                c,
                self.delta_t,
                feat=f,
                alpha_fixed_emb=self.alpha_fixed_emb,
                det_thresh=self.args.track_high_thresh,
            )
            for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features)
        ]
    return [
        DeepOCSortTrack(
            xywh, s, c, self.delta_t, alpha_fixed_emb=self.alpha_fixed_emb, det_thresh=self.args.track_high_thresh
        )
        for (xywh, s, c) in zip(bboxes, results.conf, results.cls)
    ]
```
</details>

<br>

### Method `ultralytics.trackers.deep_oc_sort.DeepOCSORT.reset` {#ultralytics.trackers.deep\_oc\_sort.DeepOCSORT.reset}

```python
def reset(self) -> None
```

Reset the Deep OC-SORT tracker, also clearing the GMC warp state.

<details>
<summary>Source code in <code>ultralytics/trackers/deep_oc_sort.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py#L263-L266">View on GitHub</a>
```python
def reset(self) -> None:
    """Reset the Deep OC-SORT tracker, also clearing the GMC warp state."""
    super().reset()
    self.gmc.reset_params()
```
</details>

<br><br>
