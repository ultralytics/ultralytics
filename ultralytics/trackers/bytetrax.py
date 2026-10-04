# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH
from .utils.stracks import joint_stracks, multi_gmc, parse_bboxes, remove_duplicate_stracks, sub_stracks


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

    def predict(self):
        """Predict the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

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

    def update(self, new_track: STrack, frame_id: int):
        """Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        Examples:
            Update the state of a track with new detection information
            >>> track = STrack([100, 200, 50, 80, 0.9, 1])
            >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])
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

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETRAX:
    """BYTETRAX: An enhanced implementation of the ByteTrack architecture with optimized thresholding.

    A simple enhancement of the ByteTrack algorithm that optimizes track continuity via a single unified matching
    threshold. To further limit identity switches, this includes functions to reconnect lost tracks, and merge
    overlapping same-class tracks into existing trajectories. These modifications improve both accuracy and processing
    speed.

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
        can_reconnect_track: Check if a lost track can be reconnected to a new detection.
        calculate_iou: Calculate Intersection over Union (IoU) between two tracks.

    Examples:
        Initialize BYTETRAX and update with detection results
        >>> tracker = BYTETRAX(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a BYTETRAX instance for object tracking.

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

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Update the tracker with new detections and return the current list of tracked objects.

        Args:
            results (Any): NumPy-backed detections (e.g. `Boxes` or `OBB` after `.cpu().numpy()`) exposing `conf`,
                `cls`, and `xywh` (or `xywhr`), and supporting boolean indexing.
            img (np.ndarray | None): Current BGR frame, used for global motion compensation when a `gmc` estimator is
                attached.
            feats (np.ndarray | None): Optional per-detection features, accepted for interface compatibility.
            **kwargs (Any): Additional tracker-specific inputs, ignored by BYTETRAX.

        Returns:
            (np.ndarray): Array of shape (N, 8) with `[x1, y1, x2, y2, track_id, score, cls, idx]` rows, or (N, 9) with
                `[x, y, w, h, angle, track_id, score, cls, idx]` rows for OBB, where `idx` is the detection index.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_thresh
        results = results[remain_inds]
        feats_keep = feats[remain_inds] if feats is not None and len(feats) else img

        detections = self.init_track(results, feats_keep)
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # Step 2: First association, with unified confidence threshold
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            # Use try-except here to bypass errors from gmc module
            try:
                warp = self.gmc.apply(img, results.xyxy)
            except Exception as e:
                LOGGER.warning(f"GMC failed, falling back to identity: {e}")
                warp = np.eye(2, 3)
            multi_gmc(strack_pool, warp)
            multi_gmc(unconfirmed, warp)

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
        # Step 3: Mark unmatched tracks as lost
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for track in r_tracked_stracks:
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
        # Step 4: Check for track reconnection before initializing new tracks (if enabled)
        reconnected_tracks = []
        remaining_detections = []

        if getattr(self.args, "enable_reconnect", True):
            used_detections = set()

            # For each lost track, find the closest qualifying detection
            for lost_track in self.lost_stracks:
                if lost_track.state != TrackState.Lost:
                    # Already reactivated this frame via the first/second association steps
                    continue
                best_detection_idx = None
                best_distance = float("inf")

                # Find the closest detection that can reconnect
                for det_idx in u_detection:
                    if det_idx in used_detections:
                        continue

                    detection = detections[det_idx]
                    if self.can_reconnect_track(lost_track, detection):
                        # Calculate actual distance
                        lost_center = lost_track.xywh[:2]
                        det_center = detection.xywh[:2]
                        distance = np.linalg.norm(lost_center - det_center)

                        if distance < best_distance:
                            best_distance = distance
                            best_detection_idx = det_idx

                # Reconnect to the closest detection
                if best_detection_idx is not None:
                    best_detection = detections[best_detection_idx]
                    lost_track.re_activate(best_detection, self.frame_id, new_id=False)
                    reconnected_tracks.append(lost_track)
                    refind_stracks.append(lost_track)
                    used_detections.add(best_detection_idx)

            # Update remaining detections that weren't reconnected
            remaining_detections = [det_idx for det_idx in u_detection if det_idx not in used_detections]
        else:
            # If reconnection is disabled, all detections remain as new detections
            remaining_detections = u_detection

        u_detection = remaining_detections

        # Step 5: Initialize new stracks with track merging (if enabled)
        merged_tracks = []
        tracks_to_remove = []

        # Check if track merging is enabled (same parameter as reconnection)
        merging_enabled = getattr(self.args, "enable_reconnect", True)
        merged_track_ids: set = set()

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue

            merge_candidate = None
            max_iou = 0.0

            # Only check for merges if merging is enabled
            if merging_enabled:
                # Check for potential merges with existing tracks not already merged this frame
                for existing_track in self.tracked_stracks:
                    if (
                        existing_track.is_activated
                        and existing_track.state == TrackState.Tracked
                        and existing_track.track_id not in merged_track_ids
                    ):
                        iou = self.calculate_iou(track, existing_track)
                        if existing_track.cls == track.cls and iou > max_iou and iou > 0.5:
                            max_iou = iou
                            merge_candidate = existing_track

            if merge_candidate is not None:
                # Merge new track into existing track by updating the existing track
                merge_candidate.update(track, self.frame_id)
                merged_tracks.append(merge_candidate)
                tracks_to_remove.append(track)
                merged_track_ids.add(merge_candidate.track_id)
            else:
                # Activate as new track
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        # Remove merged tracks from activated_stracks if they were accidentally added
        activated_stracks = [t for t in activated_stracks if t not in tracks_to_remove]
        # Step 6: Update state
        # Remove reconnected tracks from lost_stracks
        self.lost_stracks = [t for t in self.lost_stracks if t not in reconnected_tracks]

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, merged_tracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]  # Limit removed stracks to 1000 maximum

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
        """Initialize object tracking with given detections, scores, and class labels using the STrack algorithm."""
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Calculate the distance between tracks and detections using IoU and optionally fuse scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def can_reconnect_track(self, lost_track: STrack, detection: STrack) -> bool:
        """Check if a lost track can be reconnected to a new detection.

        Args:
            lost_track (STrack): The lost track to check for reconnection.
            detection (STrack): The new detection to potentially reconnect to.

        Returns:
            bool: Returns True if the track can be reconnected, False otherwise.
        """
        # Check if the track disappeared within the track_buffer frame window
        if self.frame_id - lost_track.end_frame > self.max_time_lost:
            return False

        # Check class consistency - tracks can only reconnect to detections of the same class
        if lost_track.cls != detection.cls:
            return False

        # Calculate distance between lost track's last position and new detection
        lost_center = lost_track.xywh[:2]  # Center x and y coordinates of lost track
        det_center = detection.xywh[:2]  # Center x and y coordinates of new detection

        distance = np.linalg.norm(lost_center - det_center)

        # Check if distance is less than 1 times the width of the lost track's bounding box
        lost_width = lost_track.xywh[2]  # Width of lost track's bounding box
        max_distance = 1.0 * lost_width

        return distance < max_distance

    def calculate_iou(self, track1: STrack, track2: STrack) -> float:
        """Calculate Intersection over Union (IoU) between two tracks.

        Args:
            track1 (STrack): First track.
            track2 (STrack): Second track.

        Returns:
            float: IoU value between 0 and 1.
        """

        # Convert xywh to xyxy format for IoU calculation
        def xywh_to_xyxy(xywh):
            x, y, w, h = xywh[:4]
            return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

        box1 = xywh_to_xyxy(track1.xywh)
        box2 = xywh_to_xyxy(track2.xywh)

        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def multi_predict(self, tracks: list[STrack]):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
