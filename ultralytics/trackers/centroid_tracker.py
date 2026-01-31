# Ultralytics YOLO, AGPL-3.0 license

import numpy as np
from scipy.spatial.distance import cdist

from .iou_tracker import IOUTracker, UTrack


class CentroidTrack(UTrack):
    """Single object tracking representation.

    This class is responsible for storing all the information regarding individual tracklets. Uses simple IOU matching.
    Lost tracks are not re-activated.

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


class CentroidTracker(IOUTracker):
    """CentroidTracker: Basic Centroid tracking algorithm.

    The class is responsible for initializing, updating, and managing the tracks for detected objects in a video
    sequence. It maintains the state of tracked, lost, and removed tracks over frames.

    Attributes:
        tracked_utracks (list[UTrack]): List of successfully activated tracks.
        lost_utracks (list[UTrack]): List of lost tracks.
        removed_utracks (list[UTrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.

    Methods:
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the euclidean distance between tracks and detections.
        remove_duplicate_utracks(centroidtracksa, centroidtracksb): Removes duplicate centroidtracks based on euclidean
            distance
    """

    def __init__(self, args, frame_rate=30):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
        super().__init__(args, frame_rate)

    def init_track(self, dets, scores, cls, img):
        """Initialize object tracking with detections and scores using centroid tracking algorithm."""
        return (
            [CentroidTrack(xyxy, s, c, img) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []
        )  # detections

    def remove_duplicate_utracks(self, centroidtracksa, centroidtracksb):
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

        return dists
