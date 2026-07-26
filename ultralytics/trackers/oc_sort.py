# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.ops import xyxy2ltwh
from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.stracks import parse_bboxes


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

    def activate(self, kalman_filter, frame_id: int) -> None:
        """Activate a new tracklet and seed its observation history.

        Args:
            kalman_filter (KalmanFilterXYAH): Shared Kalman filter instance.
            frame_id (int): Frame id at which the track is created.
        """
        super().activate(kalman_filter, frame_id)
        self.last_observation = self.xyxy.copy()
        self.observations[frame_id] = self.xyxy.copy()
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()

    def update(self, new_track: STrack, frame_id: int) -> None:
        """Update the track with a matched detection and record the observation.

        Args:
            new_track (STrack): Matched detection for this frame.
            frame_id (int): Current frame id.
        """
        obs = new_track.xyxy.copy()
        self.last_observation = obs
        self.observations[frame_id] = obs
        self._prune_observations()
        super().update(new_track, frame_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()
        self.velocity = self._compute_velocity()

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        """Re-activate a previously lost track with a new detection.

        Args:
            new_track (STrack): Detection used to revive this track.
            frame_id (int): Current frame id.
            new_id (bool): If True, assign a fresh track id instead of reusing the old one.
        """
        obs = new_track.xyxy.copy()
        self.last_observation = obs
        self.observations[frame_id] = obs
        super().re_activate(new_track, frame_id, new_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()
        self.velocity = self._compute_velocity()

    @staticmethod
    def _xyxy_center(xyxy: np.ndarray) -> np.ndarray:
        """Return `(cx, cy)` center of an xyxy bounding box."""
        return np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])

    def _prune_observations(self) -> None:
        """Drop old observations beyond `delta_t + 2` to bound memory while keeping enough for velocity."""
        max_keep = self.delta_t + 2
        if len(self.observations) <= max_keep:
            return
        sorted_frames = sorted(self.observations.keys())
        for frame in sorted_frames[:-max_keep]:
            del self.observations[frame]

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

    def init_track(self, results, img: np.ndarray | None = None) -> list[OCSortTrack]:
        """Build `OCSortTrack` instances from a `Results`-like object."""
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [OCSortTrack(xywh, s, c, self.delta_t) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def _fuse_appearance(
        self,
        dists: np.ndarray,
        tracks: list[OCSortTrack],
        detections: list[OCSortTrack],
        iou_dists: np.ndarray | None = None,
    ) -> np.ndarray:
        """Hook combining motion cost with appearance cost. Default: pass-through (no ReID)."""
        return dists

    def get_dists(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
        """Cost matrix = IoU (+score-fuse) + inertia·OCM (+ optional appearance, via hook)."""
        iou_dists = matching.iou_distance(tracks, detections)
        dists = matching.fuse_score(iou_dists, detections) if self.args.fuse_score else iou_dists.copy()
        dists = dists + self.inertia * self._velocity_direction_cost(tracks, detections)
        return self._fuse_appearance(dists, tracks, detections, iou_dists=iou_dists)

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
