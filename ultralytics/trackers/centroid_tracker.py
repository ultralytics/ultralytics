# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
from scipy.spatial.distance import cdist

from ..utils import LOGGER
from .basetrack import TrackState
from .iou_tracker import IOUTrack
from .utils import matching


class CentroidTrack(IOUTrack):
    """
    Single object tracking representation.

    This class is responsible for storing all the information regarding individual tracklets. Uses simple IOU matching. Lost tracks are not re-activated.

    Attributes:
        img_h (int): Pixel height of image used to initialize tracklet
        img_w (int): Pixel width of image used to initialize tracklet
    """

    def __init__(self, xywh, score, cls, img):
        """Initialize new centroidTrack instance."""
        super().__init__(xywh, score, cls)
        self.img_h, self.img_w, _ = img.shape

    @property
    def centroid(self):
        """Get current position as normalized centroid of bounding box."""
        xywh = self.xywh.copy()
        xywh[..., [0, 2]] /= self.img_w
        xywh[..., [1, 3]] /= self.img_h

        return xywh[:2]


class CentroidTracker:
    """
    CentroidTracker: Basic Centroid tracking algorithm.

    The class is responsible for initializing, updating, and managing the tracks for detected objects in a video
    sequence. It maintains the state of tracked, lost, and removed tracks over frames.

    Attributes:
        tracked_centroidtracks (list[STrack]): List of successfully activated tracks.
        lost_centroidtracks (list[STrack]): List of lost tracks.
        removed_centroidtracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.

    Methods: TODO
        update(results, img=None): Updates object tracker with new detections.
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        reset_id(): Resets the ID counter of STrack.
        joint_centroidtracks(tlista, tlistb): Combines two lists of centroidtracks.
        sub_centroidtracks(tlista, tlistb): Filters out the centroidtracks present in the second list from the first list.
        remove_duplicate_centroidtracks(centroidtracksa, centroidtracksb): Removes duplicate centroidtracks based on euclidean distance
    """

    def __init__(self, args, frame_rate=30):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
        self.tracked_centroidtracks = []  # type: list[CentroidTrack]
        self.lost_centroidtracks = []  # type: list[CentroidTrack]
        self.removed_centroidtracks = []  # type: list[CentroidTrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.reset_id()

    def update(self, results, img):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        activated_centroidtracks = []
        refind_centroidtracks = []
        lost_centroidtracks = []
        removed_centroidtracks = []

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
        dists = self.get_dists(self.tracked_centroidtracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            mtrack = self.tracked_centroidtracks[itracked]
            mdet = detections[idet]
            if mtrack.state == TrackState.Tracked:
                mtrack.update(mdet, self.frame_id)
                refind_centroidtracks.append(mtrack)

        for itracked in u_track:
            utrack = self.tracked_centroidtracks[itracked]
            if utrack.state != TrackState.Lost:
                utrack.mark_lost()
                lost_centroidtracks.append(utrack)

        for idet in u_detection:  # create new tracks for all unmatched detections
            udet = detections[idet]
            udet.activate(self.frame_id)
            activated_centroidtracks.append(udet)

        for ltrack in self.lost_centroidtracks:
            if self.frame_id - ltrack.end_frame > self.max_time_lost:
                ltrack.mark_removed()
                removed_centroidtracks.append(ltrack)

        self.tracked_centroidtracks = [t for t in self.tracked_centroidtracks if t.state == TrackState.Tracked]
        self.tracked_centroidtracks = self.joint_centroidtracks(self.tracked_centroidtracks, activated_centroidtracks)
        self.tracked_centroidtracks = self.joint_centroidtracks(self.tracked_centroidtracks, refind_centroidtracks)
        self.lost_centroidtracks = self.sub_centroidtracks(self.lost_centroidtracks, self.removed_centroidtracks)
        self.lost_centroidtracks.extend(lost_centroidtracks)
        self.tracked_centroidtracks, self.lost_centroidtracks = self.remove_duplicate_centroidtracks(
            self.tracked_centroidtracks, self.lost_centroidtracks
        )
        self.removed_centroidtracks.extend(removed_centroidtracks)
        if len(self.removed_centroidtracks) > 1000:
            self.removed_centroidtracks = self.removed_centroidtracks[
                -999:
            ]  # clip remove centroidtracks to 1000 maximum

        return np.asarray([x.result for x in self.tracked_centroidtracks if x.is_activated], dtype=np.float32)

    def init_track(self, dets, scores, cls, img):
        """Initialize object tracking with detections and scores using centroid tracking algorithm."""
        return (
            [CentroidTrack(xyxy, s, c, img) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []
        )  # detections

    @staticmethod
    def reset_id():
        """Resets the ID counter of STrack."""
        CentroidTrack.reset_id()

    def reset(self):
        """Reset tracker."""
        self.tracked_centroidtracks = []  # type: list[CentroidTrack]
        self.lost_centroidtracks = []  # type: list[CentroidTrack]
        self.removed_centroidtracks = []  # type: list[CentroidTrack]
        self.frame_id = 0
        self.reset_id()

    @staticmethod
    def joint_centroidtracks(tlista, tlistb):
        """Combine two lists of centroidtracks into a single one."""
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
    def sub_centroidtracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    def remove_duplicate_centroidtracks(self, centroidtracksa, centroidtracksb):
        """Remove duplicate centroidtracks with non-maximum euclidean distance."""
        pdist = self.get_dists(centroidtracksa, centroidtracksb)
        pairs = np.where(pdist < 0.0001)  # bit unclear what this should be
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = centroidtracksa[p].frame_id - centroidtracksa[p].start_frame
            timeq = centroidtracksb[q].frame_id - centroidtracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(centroidtracksa) if i not in dupa]
        resb = [t for i, t in enumerate(centroidtracksb) if i not in dupb]
        return resa, resb

    @staticmethod
    def get_dists(tracks, detections):
        """Calculates the distance between tracks and detections using euclidean distance."""
        ctracks = [t.centroid for t in tracks]
        cdets = [d.centroid for d in detections]
        dists = np.zeros((len(ctracks), len(cdets)), dtype=np.float32)
        if len(ctracks) and len(cdets):
            dists = cdist(ctracks, cdets)
        # TODO: mot20
        # if not self.args.mot20:
        return dists
