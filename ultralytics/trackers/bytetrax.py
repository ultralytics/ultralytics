# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.stracks import sub_stracks


class BYTETRAX(BYTETracker):
    """ByteTraX: a ByteTrack variant that reconnects lost tracks and merges duplicates to suppress ID switches.

    ByteTraX (arXiv:2609.37801) targets scenes where identity continuity matters most, such as a fixed set of objects
    that stay in view. `bytetrax.yaml` sets `track_low_thresh` equal to `track_high_thresh`, so all kept detections go
    through a single association stage. With `enable_reconnect`, detections still unmatched after association first
    reconnect lost tracks and then merge into overlapping tracks before any new track starts. Lost tracks expire before
    association, so a track lost for longer than `track_buffer` frames is never revived.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.track("https://ultralytics.com/images/bus.jpg", tracker="bytetrax.yaml")
    """

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Expire lost tracks older than `track_buffer`, then update the tracker with this frame's detections.

        Args:
            results (Any): NumPy-backed detections (e.g. `Boxes` or `OBB` after `.cpu().numpy()`) exposing `conf`,
                `cls`, and `xywh` (or `xywhr`), and supporting boolean indexing.
            img (np.ndarray | None): Current BGR frame.
            feats (np.ndarray | None): Optional per-detection features.
            **kwargs (Any): Additional tracker-specific inputs, ignored by BYTETRAX.

        Returns:
            (np.ndarray): Tracked objects in the same format as `BYTETracker.update`.
        """
        # BYTETracker expires lost tracks after association, which lets an expired track match once more first
        expired = [t for t in self.lost_stracks if self.frame_id + 1 - t.end_frame > self.max_frames_lost]
        for track in expired:
            track.mark_removed()
        self.lost_stracks = sub_stracks(self.lost_stracks, expired)
        tracks = super().update(results, img, feats, **kwargs)
        self.removed_stracks_frame += expired
        return tracks

    def _init_new_tracks(
        self,
        u_detection: list[int],
        detections: list[STrack],
        activated: list[STrack],
        refind: list[STrack] | None = None,
    ) -> None:
        """Reconnect lost tracks and merge duplicate detections, then start new tracks from the remaining detections.

        With `enable_reconnect`, each lost track in turn reconnects to the nearest unmatched same-class detection whose
        center lies within one track width of its own. Each remaining detection that could start a track instead
        updates the same-class confirmed track it overlaps most when that IoU exceeds 0.5, once per track.
        """
        if self.args.enable_reconnect:
            used = set()
            lost = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # skip tracks re-found this frame
            dets = [detections[i] for i in u_detection]
            if lost and dets:
                xywh = np.asarray([t.xywh for t in lost])
                dist = np.linalg.norm(xywh[:, None, :2] - np.asarray([d.xywh[:2] for d in dets]), axis=-1)
                other_cls = np.asarray([t.cls for t in lost])[:, None] != [d.cls for d in dets]
                dist[(dist >= xywh[:, 2:3]) | other_cls] = np.inf
                for track, row in zip(lost, dist):
                    j = row.argmin()
                    if row[j] < np.inf:
                        track.re_activate(dets[j], self.frame_id)
                        refind.append(track)
                        dist[:, j] = np.inf
                        used.add(u_detection[j])
            tracked = [t for t in self.tracked_stracks + refind if t.is_activated and t.state == TrackState.Tracked]
            new = [i for i in u_detection if i not in used and detections[i].score >= self.args.new_track_thresh]
            if tracked and new:
                iou = 1 - matching.iou_distance([detections[i] for i in new], tracked)
                iou[np.asarray([detections[i].cls for i in new])[:, None] != [t.cls for t in tracked]] = 0
                for i, row in zip(new, iou):
                    j = row.argmax()
                    if row[j] > 0.5:
                        tracked[j].update(detections[i], self.frame_id)
                        iou[:, j] = 0
                        used.add(i)
            u_detection = [i for i in u_detection if i not in used]
        super()._init_new_tracks(u_detection, detections, activated, refind)
