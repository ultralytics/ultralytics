# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO keypoints, left <-> right
COCO_BONES = np.array([(5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)])  # torso and legs


def _length(v: np.ndarray) -> np.ndarray:
    """Return the Euclidean lengths of vectors along the last axis, faster than np.linalg.norm on small arrays."""
    return np.sqrt((v * v).sum(axis=-1))


def _median(x: np.ndarray) -> np.ndarray:
    """Return the median of the rows of a small (N, 2) array, faster than np.median."""
    x = np.sort(x, axis=0)
    return (x[(len(x) - 1) // 2] + x[len(x) // 2]) / 2


def _size(box: np.ndarray) -> float:
    """Return a person's size in pixels: their box's longer side, which also fits people lying down."""
    return float(max(box[2] - box[0], box[3] - box[1]))


def _bones(kpts: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Return torso and leg bone lengths of COCO keypoints, NaN where an end is unseen; empty for other layouts.

    Args:
        kpts (np.ndarray): Keypoint coordinates of shape (..., K, 2).
        seen (np.ndarray): Boolean keypoint visibility of shape (..., K).

    Returns:
        (np.ndarray): Bone lengths of shape (..., 8) for COCO (K == 17), or (..., 0) otherwise.
    """
    if kpts.shape[-2] != 17:
        return np.zeros((*kpts.shape[:-2], 0))
    a, b = kpts[..., COCO_BONES[:, 0], :], kpts[..., COCO_BONES[:, 1], :]
    both = seen[..., COCO_BONES[:, 0]] & seen[..., COCO_BONES[:, 1]]
    return np.where(both, _length(a - b), np.nan)


class KeypointTrack:
    """A person followed by their keypoints in KPTTracker.

    Attributes:
        id (int | None): Track ID, assigned once the track is confirmed.
        kpts (np.ndarray): Keypoint coordinates (K, 2) at the last match; unseen keypoints are carried along.
        conf (np.ndarray): Keypoint confidences (K,) when each was last seen.
        seen (np.ndarray): Frame each keypoint was last seen (K,), -inf if never.
        bones (np.ndarray): Torso and leg bone lengths, smoothed over time (NaN where never seen).
        v (np.ndarray): Velocity in pixels per frame, the median keypoint motion.
        wobble (float): Match distance from one frame to the next, smoothed over time (NaN until measured).
        size (float): The person's size in pixels (box's longer side), smoothed over time.
        last (int): Frame of the last match.
        hits (int): Number of matches.
    """

    def __init__(self, kpts: np.ndarray, size: float, frame: int, kpt_conf: float):
        """Start a tentative track from a detection.

        Args:
            kpts (np.ndarray): Detection keypoints (K, 3) as x, y, confidence.
            size (float): The detection's size in pixels (box's longer side).
            frame (int): Current frame number.
            kpt_conf (float): Confidence above which a keypoint counts as seen.
        """
        seen = kpts[:, 2] > kpt_conf
        self.id = None
        self.kpts, self.conf = kpts[:, :2].copy(), kpts[:, 2].copy()
        self.seen = np.where(seen, frame, -np.inf)
        self.bones = _bones(kpts[:, :2], seen)
        self.v = np.zeros(2)
        self.wobble = np.nan
        self.size = size
        self.last, self.hits = frame, 1


class KPTTracker:
    """KPTTracker: multi-person tracking for pose models, matching people by their keypoints instead of their boxes.

    A detection is matched to a track by the mean distance between their keypoints, each capped and divided by the
    person's size (their box's longer side), over the keypoints both see, with left and right swapped as well (pose
    models flip people who turn). Two people side by side, or one half hidden behind the other, can have heavily
    overlapping boxes but look nothing alike by their keypoints. A track seen last frame takes a detection within
    `wobble_gate` times its wobble, how far its own matches usually land (a dancer's limbs land further from the
    prediction than a walker's), and at least `match_gate`. A track's keypoints move with the person's velocity; a
    lost person's velocity decays while the gate around them widens with time, so someone hidden for a moment comes
    back close to where they vanished. Between the tracks a detection could go to, a lost track also
    weighs the skeleton's scale (COCO keypoints).

    A detection whose keypoints lie on a more confident detection's keypoints is first dropped as a duplicate
    (`kpt_nms`), which box NMS keeps when the boxes differ, e.g. a box around part of a person. Association follows
    ByteTrack's stages: confident detections to every confirmed track, lost ones included; low-confidence detections
    to the tracks seen last frame; the confident detections left to the tracks left, in a wider gate (the full
    `lost_gate` for lost tracks: someone back sooner than their gate had widened); tentative tracks
    to the confident detections left. A new tentative track is confirmed (given an ID) if matched the very next
    frame. Lost tracks are forgotten after `track_buffer` frames, never for being close to somebody: that is where
    hidden people are.

    Attributes:
        tracks (list[KeypointTrack]): Confirmed tracks, matched or lost, and tentative tracks.
        frame_id (int): The current frame number.
        args (Namespace): Tracker configuration parsed from the tracker YAML (kpttrack.yaml).
        max_frames_lost (int): Frames a lost track is kept before removal.
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

    def __init__(self, args):
        """Initialize a KPTTracker instance.

        Args:
            args (Namespace): Tracker configuration from kpttrack.yaml.
        """
        self.args = args
        self.max_frames_lost = args.track_buffer
        self.reset()

    @staticmethod
    def setup_predictor(predictor: Any) -> None:
        """Check that the predictor runs a pose model, as the tracker matches people by their keypoints.

        Args:
            predictor (ultralytics.engine.predictor.BasePredictor): The predictor the tracker is attached to.

        Raises:
            ValueError: If the predictor's task is not 'pose'.
        """
        if predictor.args.task != "pose":
            raise ValueError(
                f"❌ tracker_type 'kpttrack' matches keypoints and needs a pose model, got task '{predictor.args.task}'"
            )

    def update(
        self,
        results,
        img: np.ndarray | None = None,
        feats: np.ndarray | None = None,
        kpts: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Update the tracker with new detections and their keypoints.

        Args:
            results (Any): NumPy-backed detections (e.g. `Boxes` after `.cpu().numpy()`) exposing `xyxy`, `conf`, and
                `cls`.
            img (np.ndarray | None): Current frame, unused.
            feats (np.ndarray | None): Per-detection features, unused.
            kpts (np.ndarray | None): Keypoints of shape (N, K, 3) as x, y, confidence, or (N, K, 2) without
                confidence (all seen), one row per detection.
            **kwargs (Any): Additional tracker-specific inputs, ignored by KPTTracker.

        Returns:
            (np.ndarray): Array of shape (M, 8) with `[x1, y1, x2, y2, track_id, score, cls, idx]` rows for the
                confirmed tracks matched this frame, where `idx` is the detection index.

        Raises:
            ValueError: If `kpts` is not given.
        """
        if kpts is None:
            raise ValueError("KPTTracker needs the detections' keypoints: update(results, kpts=...)")
        a = self.args
        self.frame_id += 1
        boxes = np.concatenate(
            [np.asarray(results.xyxy).reshape(-1, 4), np.c_[results.conf, results.cls].reshape(-1, 2)], axis=1
        )
        kpts = np.asarray(kpts)
        if kpts.shape[-1] == 2:  # pose models without keypoint visibility: every keypoint counts as seen
            kpts = np.concatenate([kpts, np.ones_like(kpts[..., :1])], axis=-1)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        idx = np.flatnonzero(valid)
        valid[idx[self._duplicates(boxes[idx], kpts[idx])]] = False
        score = boxes[:, 4]
        high = np.flatnonzero(valid & (score >= a.track_high_thresh))
        low = np.flatnonzero(valid & (score > a.track_low_thresh) & (score < a.track_high_thresh))
        confirmed = [t for t in self.tracks if t.id is not None]
        tentative = [t for t in self.tracks if t.id is None]

        matches, left, high_left = self._match(confirmed, high, kpts)
        tracked = [t for t in left if t.last == self.frame_id - 1]
        found = self._match(tracked, low, kpts)[0]
        taken = {id(t) for t, *_ in found}
        left = [t for t in left if id(t) not in taken]
        floor = [a.recover_gate if t.last == self.frame_id - 1 else a.lost_gate for t in left]
        found_left, _, high_left = self._match(left, high_left, kpts, floor)
        matches += found + found_left
        found, _, high_left = self._match(tentative, high_left, kpts)
        matches += found
        for t, d, flip, distance in matches:
            self._hit(t, boxes[d], kpts[d], flip, distance)

        self.tracks = [
            t
            for t in self.tracks
            if (t.id is not None or t.last == self.frame_id) and self.frame_id - t.last <= self.max_frames_lost
        ]
        for d in high_left[score[high_left] >= a.new_track_thresh]:
            self.tracks.append(KeypointTrack(kpts[d], _size(boxes[d]), self.frame_id, a.kpt_conf))
        out = [[*boxes[d, :4], t.id, boxes[d, 4], boxes[d, 5], d] for t, d, *_ in matches if t.id is not None]
        return np.asarray(out, dtype=np.float32).reshape(-1, 8)

    def _duplicates(self, boxes: np.ndarray, kpts: np.ndarray) -> np.ndarray:
        """Find the detections that duplicate a more confident one, as keypoint NMS.

        A detection is a duplicate when the keypoints both see (two at least) lie within `kpt_nms` of the more confident
        one's, by their mean distance, each capped and divided by its size. Box NMS keeps such duplicates when their
        boxes differ, e.g. a box around part of a person.

        Args:
            boxes (np.ndarray): Detections (N, 6) as x1, y1, x2, y2, score, class.
            kpts (np.ndarray): Detection keypoints (N, K, 3).

        Returns:
            (np.ndarray): Boolean mask (N,), True for the duplicates.
        """
        seen = kpts[..., 2] > self.args.kpt_conf
        both = seen[:, None] & seen[None]  # (N, N, K)
        size = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        d = np.minimum(_length(kpts[:, None, :, :2] - kpts[None, :, :, :2]) / size[None, :, None], 1)
        n = both.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            on = (n >= 2) & ((d * both).sum(axis=2) / n < self.args.kpt_nms)  # [i, j]: i's keypoints on j's
        score = boxes[:, 4]
        dup = np.zeros(len(boxes), dtype=bool)
        for j in np.argsort(-score):  # greedy, as box NMS: a duplicate suppresses no one
            if not dup[j]:
                dup |= on[:, j] & (score < score[j])
        return dup

    def _match(
        self, tracks: list[KeypointTrack], dets: np.ndarray, kpts: np.ndarray, floor: float | list[float] = 0.0
    ) -> tuple[list[tuple[KeypointTrack, int, bool, float]], list[KeypointTrack], np.ndarray]:
        """Match tracks to detections by Hungarian assignment within each track's gate, widened to at least `floor`.

        Args:
            tracks (list[KeypointTrack]): Tracks to match.
            dets (np.ndarray): Indices of the detections to match.
            kpts (np.ndarray): Keypoints of all detections (N, K, 3).
            floor (float | list[float]): Minimum gate, or one per track.

        Returns:
            matches (list[tuple[KeypointTrack, int, bool, float]]): (track, detection index, matched left-right
                flipped, distance).
            tracks_left (list[KeypointTrack]): Unmatched tracks.
            dets_left (np.ndarray): Indices of the unmatched detections.
        """
        if not len(tracks) or not len(dets):
            return [], tracks, dets
        distance, cost, flip = self._costs(tracks, kpts[dets])
        gates = np.maximum([self._gate(t) for t in tracks], floor)
        inside = distance < gates[:, None]
        i, j = linear_sum_assignment(np.where(inside, cost, 1e6))
        ok = inside[i, j]
        i, j = i[ok], j[ok]
        matches = [(tracks[a], int(dets[b]), bool(flip[a, b]), float(distance[a, b])) for a, b in zip(i, j)]
        return matches, [t for k, t in enumerate(tracks) if k not in i], np.delete(dets, j)

    def _costs(self, tracks: list[KeypointTrack], kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute track-detection distances and costs, as they are and left-right flipped.

        The distance is the mean of the capped keypoint distances over the person's size, weighted by the track's
        keypoint weights, over the keypoints the detection sees; infinite if they share under `min_common` keypoint
        confidence or no weighted keypoint. The cost adds, for lost tracks, the skeleton scale's mismatch.

        Args:
            tracks (list[KeypointTrack]): Tracks (T).
            kpts (np.ndarray): Detection keypoints (D, K, 3).

        Returns:
            distance (np.ndarray): Gated distance (T, D).
            cost (np.ndarray): Assignment cost (T, D).
            flip (np.ndarray): Whether the detection matched better left-right flipped (T, D).
        """
        a = self.args
        P = np.array([self._predict(t) for t in tracks])  # (T, K, 2)
        W = np.array([self._weights(t) for t in tracks])  # (T, K)
        C = np.array([t.conf for t in tracks])  # (T, K), not faded: unseen keypoints still count as shared
        size = np.array([t.size for t in tracks])[:, None, None]
        lost = np.flatnonzero([self.frame_id - t.last > 1 for t in tracks])
        track_bones = np.array([tracks[i].bones for i in lost]).reshape(len(lost), 1, len(tracks[0].bones))
        orders = [(slice(None), 0.0)] + ([(COCO_FLIP, a.flip_cost)] if kpts.shape[1] == 17 else [])
        out = []
        for order, extra in orders:
            k = kpts[:, order]
            seen = k[..., 2] > a.kpt_conf
            w = W[:, None] * seen[None]  # (T, D, K)
            d = _length(P[:, None] - k[None, ..., :2]) / size
            common = (C[:, None] * seen[None]).sum(axis=2)
            total = w.sum(axis=2)  # 0: no keypoint to compare, so no match (the flipped order may have one)
            with np.errstate(invalid="ignore", divide="ignore"):
                cost = (w * np.minimum(d, 1)).sum(axis=2) / total
                cost = np.where((common >= a.min_common) & (total > 0), cost, np.inf) + extra
            scale = np.zeros_like(cost)
            if len(lost):
                scale[lost] = self._scale(_bones(k[..., :2], seen), track_bones)
            out.append((cost, cost + a.scale_weight * scale))
        if len(out) == 1:
            d0, c0 = out[0]
            return d0, c0, np.zeros(c0.shape, dtype=bool)
        (d0, c0), (d1, c1) = out
        flip = c1 < c0
        return np.where(flip, d1, d0), np.where(flip, c1, c0), flip

    @staticmethod
    def _scale(det_bones: np.ndarray, track_bones: np.ndarray) -> np.ndarray:
        """Return the skeleton scale's mismatch: the median absolute log ratio of the bones both have, 0 under two.

        Args:
            det_bones (np.ndarray): Detection bone lengths (D, B).
            track_bones (np.ndarray): Lost tracks' bone lengths (L, 1, B).

        Returns:
            (np.ndarray): Mismatch of shape (L, D).
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.abs(np.log(det_bones[None] / track_bones))
        ratio = np.sort(np.where(np.isfinite(ratio), ratio, np.inf), axis=2)
        m = np.isfinite(ratio).sum(axis=2)
        if not ratio.shape[2]:
            return np.zeros(m.shape)
        middle = np.maximum(m - 1, 0)[..., None] // 2
        return np.where(m >= 2, np.take_along_axis(ratio, middle, axis=2)[..., 0], 0)

    def _gate(self, t: KeypointTrack) -> float:
        """Return a track's gate: `wobble_gate` times its wobble, at least `match_gate`, if seen last frame; widening
        to `lost_gate` over `lost_ramp` frames once lost.
        """
        a = self.args
        base = np.fmax(a.match_gate, a.wobble_gate * t.wobble)
        growth = min((self.frame_id - t.last - 1) / a.lost_ramp, 1)
        return base + (max(a.lost_gate, base) - base) * growth

    def _predict(self, t: KeypointTrack) -> np.ndarray:
        """Return a track's keypoints now, moved at its velocity, which decays over `lost_ramp` frames once lost."""
        tau = self.args.lost_ramp
        dt = self.frame_id - t.last
        step = min(dt, 1)
        return t.kpts + t.v * (step + tau * (1 - np.exp(-(dt - step) / tau)))

    def _weights(self, t: KeypointTrack) -> np.ndarray:
        """Return each keypoint's weight in a match: its confidence, fading over `joint_memory` frames unseen."""
        return t.conf * np.exp(-(self.frame_id - t.seen) / self.args.joint_memory)

    def _hit(self, t: KeypointTrack, box: np.ndarray, kpts: np.ndarray, flip: bool, distance: float) -> None:
        """Update a track with its matched detection, confirming it if tentative.

        Args:
            t (KeypointTrack): The matched track.
            box (np.ndarray): Detection row [x1, y1, x2, y2, score, cls].
            kpts (np.ndarray): Detection keypoints (K, 3).
            flip (bool): Whether the detection matched left-right flipped.
            distance (float): The match's distance, which updates the track's wobble if it was seen last frame.
        """
        a = self.args
        if self.frame_id - t.last == 1 and np.isfinite(distance):
            w = t.wobble
            t.wobble = distance if np.isnan(w) else w + a.velocity_smoothing * (distance - w)
        k = kpts[COCO_FLIP] if flip else kpts
        seen = k[:, 2] > a.kpt_conf
        pred = self._predict(t)
        both = seen & (t.seen == t.last)
        if both.any():
            moved = _median(k[both, :2] - t.kpts[both])
            t.v += a.velocity_smoothing * (moved / (self.frame_id - t.last) - t.v)
        b = _bones(k[:, :2], seen)
        smoothed = t.bones + a.size_smoothing * (b - t.bones)
        t.bones = np.where(np.isnan(t.bones), b, np.where(np.isnan(b), t.bones, smoothed))
        t.kpts = np.where(seen[:, None], k[:, :2], pred)
        t.conf = np.where(seen, k[:, 2], t.conf)
        t.seen = np.where(seen, self.frame_id, t.seen)
        t.size += a.size_smoothing * (_size(box) - t.size)
        t.last, t.hits = self.frame_id, t.hits + 1
        if t.id is None:
            t.id, self.next_id = self.next_id, self.next_id + 1

    def reset(self) -> None:
        """Reset the tracker by clearing all tracks and restarting frame and ID counters."""
        self.tracks: list[KeypointTrack] = []
        self.frame_id = 0
        self.next_id = 1
