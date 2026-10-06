# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.ops import linear_sum_assignment
from .basetrack import BaseTrack

COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO keypoints, left <-> right
COCO_BONES = np.array([(5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)])  # torso and legs


def _bones(kpts: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Return torso and leg bone lengths (..., 8) of COCO keypoints (..., 17, 2), NaN where an end is not `seen`.

    Other keypoint layouts have no bones: (..., 0).
    """
    if kpts.shape[-2] != 17:
        return np.zeros((*kpts.shape[:-2], 0))
    i, j = COCO_BONES.T
    return np.where(seen[..., i] & seen[..., j], np.linalg.norm(kpts[..., i, :] - kpts[..., j, :], axis=-1), np.nan)


class KeypointTrack(BaseTrack):
    """A person followed by their keypoints in KPTTracker, with `frame_id` the frame of the last match.

    Attributes:
        kpts (np.ndarray): Keypoint coordinates (K, 2) at the last match; unseen keypoints are carried along.
        conf (np.ndarray): Keypoint confidences (K,) when each was last seen.
        seen (np.ndarray): Frame each keypoint was last seen (K,), -inf if never.
        bones (np.ndarray): Torso and leg bone lengths, smoothed over time (NaN where never seen).
        v (np.ndarray): Velocity in pixels per frame, the median keypoint motion.
        wobble (float): Match distance from one frame to the next, smoothed over time (NaN until measured).
        size (float): The person's size in pixels (box's longer side), smoothed over time.
    """

    def __init__(self, kpts: np.ndarray, size: float, frame: int, kpt_conf: float):
        """Start a tentative track from a detection's keypoints (K, 3), size, frame, and keypoint seen threshold."""
        super().__init__()
        seen = kpts[:, 2] > kpt_conf
        self.kpts, self.conf = kpts[:, :2].copy(), kpts[:, 2].copy()
        self.seen = np.where(seen, frame, -np.inf)
        self.bones = _bones(kpts[:, :2], seen)
        self.v = np.zeros(2)
        self.wobble = np.nan
        self.size = size
        self.frame_id = frame


class KPTTracker:
    """KPTTracker: multi-person tracking for pose models, matching people by their keypoints instead of their boxes.

    Two people side by side, or one half hidden behind the other, can have heavily overlapping boxes but look nothing
    alike by their keypoints. Detections are matched to tracks by their mean keypoint distance over the person's size,
    also left-right flipped, in ByteTrack's stages and within per-track gates that adapt to how much each person moves.
    See the KPTTrack section of https://docs.ultralytics.com/modes/track for the full method and arguments.

    Attributes:
        tracked_stracks (list[KeypointTrack]): Confirmed tracks, matched or lost, and tentative tracks.
        removed_stracks_frame (list[KeypointTrack]): Confirmed tracks removed in the most recent frame.
        frame_id (int): The current frame number.
        args (Namespace): Tracker configuration parsed from the tracker YAML (kpttrack.yaml).
        next_id (int): The ID the next confirmed track gets.

    Methods:
        update: Update the tracker with new detections and their keypoints.
        setup_predictor: Check that the predictor runs a pose model.
        reset: Reset the tracker by clearing all tracks.

    Examples:
        Initialize KPTTracker and update with pose results
        >>> from ultralytics import YOLO
        >>> from ultralytics.utils import YAML, IterableSimpleNamespace
        >>> from ultralytics.utils.checks import check_yaml
        >>> args = IterableSimpleNamespace(**YAML.load(check_yaml("kpttrack.yaml")))
        >>> tracker = KPTTracker(args)
        >>> result = YOLO("yolo26n-pose.pt")("https://ultralytics.com/images/bus.jpg")[0]
        >>> tracks = tracker.update(result.boxes.cpu().numpy(), kpts=result.keypoints.data.cpu().numpy())
    """

    def __init__(self, args: Any):
        """Initialize a KPTTracker instance with the tracker configuration from kpttrack.yaml."""
        self.args = args
        self.reset()

    @staticmethod
    def setup_predictor(predictor: Any) -> None:
        """Raise a ValueError unless the predictor runs a pose model, as the tracker matches keypoints."""
        if predictor.args.task != "pose":
            raise ValueError(f"❌ tracker_type 'kpttrack' needs a pose model, got task '{predictor.args.task}'")

    def update(self, results, img: np.ndarray | None = None, *, kpts: np.ndarray, **kwargs) -> np.ndarray:
        """Update the tracker with new detections and their keypoints.

        Args:
            results (Any): NumPy-backed detections (e.g. `Boxes` after `.cpu().numpy()`) exposing `xyxy`, `conf`, and
                `cls`.
            img (np.ndarray | None): Current frame, unused.
            kpts (np.ndarray): Keypoints of shape (N, K, 3) as x, y, confidence, or (N, K, 2) without confidence (all
                seen), one row per detection.
            **kwargs (Any): Additional tracker-specific inputs, ignored by KPTTracker.

        Returns:
            (np.ndarray): Array of shape (M, 8) with `[x1, y1, x2, y2, track_id, score, cls, idx]` rows for the
                confirmed tracks matched this frame, where `idx` is the detection index.
        """
        a = self.args
        self.frame_id += 1
        xyxy, score = results.xyxy, results.conf
        size = (xyxy[:, 2:] - xyxy[:, :2]).max(1)  # box's longer side, which also fits people lying down
        if kpts.shape[-1] == 2:  # pose models without keypoint visibility: every keypoint counts as seen
            kpts = np.concatenate([kpts, np.ones_like(kpts[..., :1])], axis=-1)
        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        valid[valid] = ~self._duplicates(score[valid], size[valid], kpts[valid])
        high = np.flatnonzero(valid & (score >= a.track_high_thresh))
        low = np.flatnonzero(valid & (score > a.track_low_thresh) & (score < a.track_high_thresh))

        tracks = self.tracked_stracks
        matches, left, high_left = self._match([t for t in tracks if t.is_activated], high, kpts)
        found, tracked_left, _ = self._match([t for t in left if t.frame_id == self.frame_id - 1], low, kpts)
        left = [t for t in left if t.frame_id < self.frame_id - 1 or t in tracked_left]
        floor = [a.recover_gate if t.frame_id == self.frame_id - 1 else a.lost_gate for t in left]
        found_left, _, high_left = self._match(left, high_left, kpts, floor)
        found_new, _, high_left = self._match([t for t in tracks if not t.is_activated], high_left, kpts)
        matches += found + found_left + found_new
        for t, d, flip, distance in matches:
            self._hit(t, float(size[d]), kpts[d], flip, distance)

        cutoff = self.frame_id - a.track_buffer  # confirmed tracks unmatched since before this are removed
        self.removed_stracks_frame = [t for t in tracks if t.is_activated and t.frame_id < cutoff]  # read by Solutions
        self.tracked_stracks = [t for t in tracks if t.frame_id >= (cutoff if t.is_activated else self.frame_id)]
        for d in high_left[score[high_left] >= a.new_track_thresh]:
            self.tracked_stracks.append(KeypointTrack(kpts[d], float(size[d]), self.frame_id, a.kpt_conf))
        out = [[*xyxy[d], t.track_id, score[d], results.cls[d], d] for t, d, *_ in matches]  # all confirmed by now
        return np.asarray(out, dtype=np.float32).reshape(-1, 8)

    def _duplicates(self, score: np.ndarray, size: np.ndarray, kpts: np.ndarray) -> np.ndarray:
        """Return a mask of detections whose seen keypoints lie within `kpt_nms` of a more confident detection's.

        Keypoint NMS: box NMS keeps such duplicates when their boxes differ, e.g. a box around part of a person.
        """
        seen = kpts[..., 2] > self.args.kpt_conf
        both = seen[:, None] & seen[None]  # (N, N, K)
        d = np.minimum(np.linalg.norm(kpts[:, None, :, :2] - kpts[None, :, :, :2], axis=-1) / size[None, :, None], 1)
        n = both.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            on = (n >= 2) & ((d * both).sum(axis=2) / n < self.args.kpt_nms)  # [i, j]: i's keypoints on j's
        dup = np.zeros(len(score), dtype=bool)
        for j in np.argsort(-score):  # greedy, as box NMS: a duplicate suppresses no one
            if not dup[j]:
                dup |= on[:, j] & (score < score[j])
        return dup

    def _match(
        self, tracks: list[KeypointTrack], dets: np.ndarray, kpts: np.ndarray, floor: float | list[float] = 0.0
    ) -> tuple[list[tuple], list[KeypointTrack], np.ndarray]:
        """Hungarian-match tracks to detections (indices into `kpts`) within each track's gate, at least `floor`.

        A track seen last frame takes a detection within `wobble_gate` times its wobble and at least `match_gate`; the
        gate widens to `lost_gate` over `lost_ramp` frames once lost. Returns (track, detection, flipped, distance)
        matches, the unmatched tracks, and the unmatched detections.
        """
        if not len(tracks) or not len(dets):
            return [], tracks, dets
        a = self.args
        distance, cost, flip = self._costs(tracks, kpts[dets])
        base = np.fmax(a.match_gate, a.wobble_gate * np.array([t.wobble for t in tracks]))
        lost = np.minimum((self.frame_id - np.array([t.frame_id for t in tracks]) - 1) / a.lost_ramp, 1)
        inside = distance < np.maximum(base + (np.maximum(a.lost_gate, base) - base) * lost, floor)[:, None]
        i, j = linear_sum_assignment(np.where(inside, cost, 1e6))
        i, j = i[inside[i, j]], j[inside[i, j]]
        matches = [(tracks[t], dets[d], flip[t, d], distance[t, d]) for t, d in zip(i, j)]
        return matches, [t for k, t in enumerate(tracks) if k not in i], np.delete(dets, j)

    def _costs(self, tracks: list[KeypointTrack], kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return track-detection distances, assignment costs, and flips (T, D), flipping where it costs less.

        The distance is the mean of the capped keypoint distances over the person's size, over the keypoints the
        detection sees, weighted by the track's keypoint confidences fading over `joint_memory` frames unseen, plus
        `flip_cost` when flipped. The cost adds, for lost tracks, the skeleton scale's mismatch.
        """
        a = self.args
        P = np.array([self._predict(t) for t in tracks])  # (T, K, 2)
        W = np.array([t.conf * np.exp(-(self.frame_id - t.seen) / a.joint_memory) for t in tracks])  # (T, K)
        size = np.array([t.size for t in tracks])[:, None, None]
        coco = kpts.shape[1] == 17  # flip and skeleton scale need COCO keypoints
        k = np.stack([kpts, kpts[:, COCO_FLIP]]) if coco else kpts[None]  # (O, D, K, 3): as is, then flipped
        seen = k[..., 2] > a.kpt_conf
        w = W[:, None] * seen[:, None]  # (O, T, D, K)
        d = np.linalg.norm(P[:, None] - k[:, None, ..., :2], axis=-1) / size
        total = w.sum(axis=-1)  # 0: no keypoint to compare, so no match (the flipped order may have one)
        with np.errstate(invalid="ignore", divide="ignore"):
            distance = np.where(total > 0, (w * np.minimum(d, 1)).sum(-1) / total, np.inf)
        distance[1:] += a.flip_cost
        cost = distance.copy()
        lost = np.flatnonzero([self.frame_id - t.frame_id > 1 for t in tracks])
        if coco and len(lost):  # skeleton scale mismatch: median absolute log ratio of the bones both have, 0 under two
            det_bones = _bones(k[..., :2], seen)[:, None]  # (O, 1, D, B)
            track_bones = np.array([tracks[i].bones for i in lost])[:, None]  # (L, 1, B)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.sort(np.abs(np.log(det_bones / track_bones)), axis=-1)  # (O, L, D, B), NaN and inf last
            m = np.isfinite(ratio).sum(axis=-1)
            median = np.take_along_axis(ratio, (m - 1)[..., None] // 2, axis=-1)[..., 0]
            cost[:, lost] += a.scale_weight * np.where(m >= 2, median, 0)
        flip = cost[-1] < cost[0]  # all False without COCO keypoints
        return np.where(flip, distance[-1], distance[0]), np.where(flip, cost[-1], cost[0]), flip

    def _predict(self, t: KeypointTrack) -> np.ndarray:
        """Return a track's keypoints now, moved at its velocity, which decays over `lost_ramp` frames once lost."""
        tau = self.args.lost_ramp
        return t.kpts + t.v * (1 + tau * (1 - np.exp(-(self.frame_id - t.frame_id - 1) / tau)))

    def _hit(self, t: KeypointTrack, size: float, kpts: np.ndarray, flip: bool, distance: float) -> None:
        """Update a track with its matched detection's size, keypoints (K, 3), flip, and distance; confirm it if new."""
        a = self.args
        if self.frame_id - t.frame_id == 1:
            t.wobble = distance if np.isnan(t.wobble) else t.wobble + a.velocity_smoothing * (distance - t.wobble)
        k = kpts[COCO_FLIP] if flip else kpts
        seen = k[:, 2] > a.kpt_conf
        pred = self._predict(t)
        both = seen & (t.seen == t.frame_id)
        if both.any():
            moved = np.median(k[both, :2] - t.kpts[both], axis=0)
            t.v += a.velocity_smoothing * (moved / (self.frame_id - t.frame_id) - t.v)
        b = _bones(k[:, :2], seen)
        t.bones = np.where(np.isnan(t.bones), b, t.bones + a.size_smoothing * np.nan_to_num(b - t.bones))
        t.kpts = np.where(seen[:, None], k[:, :2], pred)
        t.conf = np.where(seen, k[:, 2], t.conf)
        t.seen = np.where(seen, self.frame_id, t.seen)
        t.size += a.size_smoothing * (size - t.size)
        t.frame_id = self.frame_id
        if not t.is_activated:  # per-tracker IDs: BaseTrack's shared counter is reset by every new tracker
            t.track_id, t.is_activated, self.next_id = self.next_id, True, self.next_id + 1

    def reset(self) -> None:
        """Reset the tracker by clearing all tracks and restarting the frame and ID counters."""
        self.tracked_stracks: list[KeypointTrack] = []
        self.frame_id = 0
        self.next_id = 1
