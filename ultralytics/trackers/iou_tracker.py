# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching


class IOUTrack(BaseTrack):
    """
    Single object tracking representation.

    This class is responsible for storing all the information regarding individual tracklets. Uses simple IOU matching. Lost tracks are not re-activated.

    Attributes:
        _count (int): Class-level counter for unique track IDs.
        track_id (int): Unique identifier for the track.
        is_activated (bool): Flag indicating whether the track is currently active.
        state (TrackState): Current state of the track.
        history (OrderedDict): Ordered history of the track's states.
        score (float): The confidence score of the tracking.
        start_frame (int): The frame number where tracking started.
        frame_id (int): The most recent frame ID processed by the track.
        time_since_update (int): Frames passed since the last update.
        location (tuple): The location of the object in the context of multi-camera tracking.

    Methods:
        activate: Abstract method to activate the track.
        predict: Abstract method to predict the next state of the track.
        update: Abstract method to update the track with new data.
    """

    def __init__(self, xywh, score, cls):
        """Initialize new IOUTrack instance."""
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in [5, 6], f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def activate(self, frame_id):
        """Start a new tracklet."""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self._tlwh = new_track._tlwh  # updating box

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Get current position in bounding box format (top left x, top left y, width, height)."""

        return self._tlwh.copy()

    @property
    def xyxy(self):
        """Convert bounding box to format (min x, min y, max x, max y), i.e., (top left, bottom right)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod  # TODO: might not need this
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format (center x, center y, aspect ratio, height), where the aspect ratio is width /
        height.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        """Get current position in bounding box format (center x, center y, width, height)."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """Get current position in bounding box format (center x, center y, width, height, angle)."""
        if self.angle is None:
            LOGGER.warning("WARNING ⚠️ `angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """Get current tracking results."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """Return a string representation of the IOUTracker object with start and end frames and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class IOUTracker:
    """
    IOUTracker: Basic IOU tracking algorithm.

    The class is responsible for initializing, updating, and managing the tracks for detected objects in a video
    sequence. It maintains the state of tracked, lost, and removed tracks over frames.

    Attributes:
        tracked_ioutracks (list[STrack]): List of successfully activated tracks.
        lost_ioutracks (list[STrack]): List of lost tracks.
        removed_ioutracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        match_thresh (float): 1 - Min iou overlap to match tracks

    Methods: TODO
        update(results, img=None): Updates object tracker with new detections.
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        reset_id(): Resets the ID counter of STrack.
        joint_ioutracks(tlista, tlistb): Combines two lists of ioutracks.
        sub_ioutracks(tlista, tlistb): Filters out the ioutracks present in the second list from the first list.
        remove_duplicate_ioutracks(ioutracksa, ioutracksb): Removes duplicate ioutracks based on IoU.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
        self.tracked_ioutracks = []  # type: list[IOUTrack]
        self.lost_ioutracks = []  # type: list[IOUTrack]
        self.removed_ioutracks = []  # type: list[IOUTrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.match_thresh = args.match_thresh
        self.reset_id()

    def update(self, results, img=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        activated_ioutracks = []
        refind_ioutracks = []
        lost_ioutracks = []
        removed_ioutracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        remain_inds = scores > self.args.track_thresh
        dets_keep = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        cls_keep = cls[remain_inds]

        detections = self.init_track(dets_keep, scores_keep, cls_keep, img)  # list of CentroidTrack objects

        # matching
        dists = self.get_dists(self.tracked_ioutracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            mtrack = self.tracked_ioutracks[itracked]
            mdet = detections[idet]
            if mtrack.state == TrackState.Tracked:
                mtrack.update(mdet, self.frame_id)
                refind_ioutracks.append(mtrack)

        for itracked in u_track:
            utrack = self.tracked_ioutracks[itracked]
            if utrack.state != TrackState.Lost:
                utrack.mark_lost()
                lost_ioutracks.append(utrack)

        for idet in u_detection:  # create new tracks for all unmatched detections
            udet = detections[idet]
            udet.activate(self.frame_id)
            activated_ioutracks.append(udet)

        for ltrack in self.lost_ioutracks:
            if self.frame_id - ltrack.end_frame > self.max_time_lost:
                ltrack.mark_removed()
                removed_ioutracks.append(ltrack)

        self.tracked_ioutracks = [t for t in self.tracked_ioutracks if t.state == TrackState.Tracked]
        self.tracked_ioutracks = self.joint_ioutracks(self.tracked_ioutracks, activated_ioutracks)
        self.tracked_ioutracks = self.joint_ioutracks(self.tracked_ioutracks, refind_ioutracks)
        self.lost_ioutracks = self.sub_ioutracks(self.lost_ioutracks, self.removed_ioutracks)
        self.lost_ioutracks.extend(lost_ioutracks)
        self.tracked_ioutracks, self.lost_ioutracks = self.remove_duplicate_ioutracks(
            self.tracked_ioutracks, self.lost_ioutracks
        )
        self.removed_ioutracks.extend(removed_ioutracks)
        if len(self.removed_ioutracks) > 1000:
            self.removed_ioutracks = self.removed_ioutracks[-999:]  # clip remove ioutracks to 1000 maximum

        return np.asarray([x.result for x in self.tracked_ioutracks if x.is_activated], dtype=np.float32)

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracking with detections and scores using IOUTrack algorithm."""
        return [IOUTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    @staticmethod
    def reset_id():
        """Resets the ID counter of STrack."""
        IOUTrack.reset_id()

    def reset(self):
        """Reset tracker."""
        self.tracked_ioutracks = []  # type: list[IOUTrack]
        self.lost_ioutracks = []  # type: list[IOUTrack]
        self.removed_ioutracks = []  # type: list[IOUTrack]
        self.frame_id = 0
        self.reset_id()

    @staticmethod
    def joint_ioutracks(tlista, tlistb):
        """Combine two lists of ioutracks into a single one."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_ioutracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_ioutracks(ioutracksa, ioutracksb):
        """Remove duplicate ioutracks with non-maximum IoU distance."""
        pdist = matching.iou_distance(ioutracksa, ioutracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = ioutracksa[p].frame_id - ioutracksa[p].start_frame
            timeq = ioutracksb[q].frame_id - ioutracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(ioutracksa) if i not in dupa]
        resb = [t for i, t in enumerate(ioutracksb) if i not in dupb]
        return resa, resb

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IoU."""
        dists = matching.iou_distance(tracks, detections)
        # TODO: mot20
        # if not self.args.mot20:
        return dists
