# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Shared helpers for operating on lists of track objects across trackers.

These functions are intentionally generic: they only touch attributes that every track
implementation exposes (`track_id`, `frame_id`, `start_frame`, `xyxy`, `mean`, `covariance`).
"""

from __future__ import annotations

__all__ = ("joint_stracks", "merge_track_pools", "multi_gmc", "remove_duplicate_stracks", "sub_stracks")

import numpy as np

from ..basetrack import TrackState
from . import matching


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
    tracker.removed_stracks.extend(removed)
    if len(tracker.removed_stracks) > removed_buffer:
        tracker.removed_stracks = tracker.removed_stracks[-removed_buffer:]


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


def multi_gmc(stracks: list, H: np.ndarray | None = None) -> None:
    """Update multiple track positions and covariances using a 2x3 affine homography.

    The Kalman state is assumed to be laid out as `(*box, *box_velocity)` with the box center `(x, y)` in the first two
    dims. `R8x8` rotates all four 2-d pairs block- diagonally; the translation `t` is applied only to the position.

    .. warning::
        This helper assumes the state layout is four spatial/velocity pairs (e.g., XYWH).
        XYAH trackers must override this behavior.

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
    if H is None:
        H = np.eye(2, 3)
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
