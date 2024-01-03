# Ultralytics YOLO 🚀, AGPL-3.0 license
# Please follow the repo https://github.com/hustvl/SparseTrack to compile GMC mosule before using SparseTracker
import numpy as np

from . import pbcvt 
from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils.matching import *
from .utils.kalman_filter import KalmanFilterXYWH


class DSTrack(STrack):
    """
    An extended version of the STrack class for YOLOv8, adding pseudo depth features.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of DSTrack.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        predict(): Predicts the mean and covariance using Kalman filter.
        multi_predict(stracks): Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        convert_coords(tlwh): Converts tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh(tlwh): Convert bounding box to xywh format `(center x, center y, width, height)`.

    Usage:
        ds_track = DSTrack(tlbr, score, cls)
        ds_track.predict()
        ds_track.update(new_track, frame_id)
    """
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlbr, score, cls):
        super().__init__(tlbr, score, cls)

    def predict(self):
        """Predicts the mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Predicts the mean and covariance of multiple object tracks using shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = DSTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def deep_vec(self):
        """Obtain the pseudo depth of each object."""
        ret = self.tlwh.copy()
        cx = ret[0] + 0.5 * ret[2]
        y2 = ret[1] +  ret[3]
        lendth = 2000 - y2
        return np.asarray([cx, y2, lendth], dtype=float)

    def convert_coords(self, tlwh):
        """Converts Top-Left-Width-Height bounding box coordinates to X-Y-Width-Height format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width, height)`."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

class SparseTracker(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        det_thresh (float): The threshold used for detection.
        down_scale (int): .
        layers (int): .

    Methods:
        update(results, curr_img=None): Updates object tracker with new detections.
        get_kalmanfilter(): Returns an instance of KalmanFilterXYWH for object tracking.
        get_deep_range(obj, step): Divide the depth intervals and obtain a target subset mask in each interval.
        get_sub_mask(deep_range, col): Obtain a mask of the target subset for the corresponding depth range.
        DCM(detections, tracks, activated_starcks, refind_stracks, levels, thresh, is_fuse): Depth cascade matching.
        multi_predict(tracks): Predict and track multiple objects with YOLOv8 model.
        reset_id(): Resets the ID counter of DSTrack.

    Usage:
        sparse_tracker = SparseTracker(args, frame_rate)
        sparse_tracker.init_track(dets, scores, cls, img)
        sparse_tracker.multi_predict(tracks)

    Note:
        Please follow the repo https://github.com/hustvl/SparseTrack to compile GMC mosule before using SparseTracker.
    """
    def __init__(self, args, frame_rate=30):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
        super().__init__(args, frame_rate)

        self.det_thresh = args.track_thresh + 0.1          
        self.pre_img = None
        self.down_scale = args.down_scale
        self.layers = args.depth_levels

    def get_kalmanfilter(self):
        """Returns an instance of KalmanFilterXYWH for object tracking."""
        return KalmanFilterXYWH()
    
    def get_deep_range(self, obj, step):
        """Divide the depth intervals and obtain a target subset mask in each interval."""
        col = []
        for t in obj:
            lend = (t.deep_vec)[2]
            col.append(lend)
        max_len, mix_len = max(col), min(col)
        if max_len != mix_len:
            deep_range =np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
            if deep_range[-1] < max_len:
                deep_range = np.concatenate([deep_range, np.array([max_len],)])
                deep_range[0] = np.floor(deep_range[0])
                deep_range[-1] = np.ceil(deep_range[-1])
        else:    
            deep_range = [mix_len,] 
        mask = self.get_sub_mask(deep_range, col)      
        return mask
    
    def get_sub_mask(self, deep_range, col):
        """Obtain a mask of the target subset for the corresponding depth range."""
        mix_len=deep_range[0]
        max_len=deep_range[-1]
        if max_len == mix_len:
            lc = mix_len   
        mask = []
        for d in deep_range:
            if d > deep_range[0] and d < deep_range[-1]:
                mask.append((col >= lc) & (col < d)) 
                lc = d
            elif d == deep_range[-1]:
                mask.append((col >= lc) & (col <= d)) 
                lc = d 
            else:
                lc = d
                continue
        return mask
    
    def DCM(self, detections, tracks, activated_starcks, refind_stracks, levels, thresh, is_fuse):
        """Depth cascade matching."""
        if len(detections) > 0:
            det_mask = self.get_deep_range(detections, levels) 
        else:
            det_mask = []

        if len(tracks)!=0:
            track_mask = self.get_deep_range(tracks, levels)
        else:
            track_mask = []

        u_detection, u_tracks, res_det, res_track = [], [], [], []
        if len(track_mask) != 0:
            if  len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    idx = np.argwhere(det_mask[len(track_mask) + i] == True)
                    for idd in idx:
                        res_det.append(detections[idd[0]])
            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    idx = np.argwhere(track_mask[len(det_mask) + i] == True)
                    for idd in idx:
                        res_track.append(tracks[idd[0]])
        
            for dm, tm in zip(det_mask, track_mask):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)
                
                # search det 
                det_ = []
                for idd in det_idx:
                    det_.append(detections[idd[0]])
                det_ = det_ + u_detection
                # search trk
                track_ = []
                for idt in trk_idx:
                    track_.append(tracks[idt[0]])
                # update trk
                track_ = track_ + u_tracks
                
                dists = iou_distance(track_, det_)
                if (not self.args.mot20) and is_fuse:
                    dists = fuse_score(dists, det_)
                matches, u_track_, u_det_ = linear_assignment(dists, thresh)
                for itracked, idet in matches:
                    track = track_[itracked]
                    det = det_[idet]
                    if track.state == TrackState.Tracked:
                        track.update(det_[idet], self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                u_tracks = [track_[t] for t in u_track_]
                u_detection = [det_[t] for t in u_det_]
                
            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det

        else:
            u_detection = detections
            
        return activated_starcks, refind_stracks, u_tracks, u_detection

    def update(self, results, curr_img = None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        if self.frame_id == 1:
            self.pre_img = None
            
        # init stracks
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        # current detections
        scores = results.conf
        bboxes = results.xyxy
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        # divide high-score dets and low-scores dets
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]
        
        # tracks preprocess
        unconfirmed = []
        tracked_stracks = []  # type: list[DSTrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # init high-score dets
        if len(dets) > 0:
            detections = [DSTrack(tlbr, s, c) for
                          (tlbr, s, c) in zip(dets, scores_keep, cls_keep)]   
        else:
            detections = []

        # get strack_pool   
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        
        # predict the current location with KF
        DSTrack.multi_predict(strack_pool)
        
        # use GMC: for mot20 dancetrack--unenabled GMC: 368 - 373
        if self.pre_img is not None:
            warp = pbcvt.GMC(curr_img, self.pre_img, self.down_scale)
        else:
            warp = np.eye(3,3)
        DSTrack.multi_gmc(strack_pool, warp[:2, :])
        DSTrack.multi_gmc(unconfirmed, warp[:2, :])
        
        # DCM
        activated_starcks, refind_stracks, u_track, u_detection_high = self.DCM(
                                                                                detections, 
                                                                                strack_pool, 
                                                                                activated_starcks,
                                                                                refind_stracks, 
                                                                                self.layers, 
                                                                                self.args.match_thresh, 
                                                                                is_fuse=True)  
            
            
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [DSTrack(tlbr, s, c) for
                          (tlbr, s, c) in zip(dets_second, scores_second, cls_second)]
        else:
            detections_second = []
        r_tracked_stracks = [t for t in u_track if t.state == TrackState.Tracked]   
        
        # DCM
        activated_starcks, refind_stracks, u_strack, u_detection_sec = self.DCM(
                                                                                detections_second, 
                                                                                r_tracked_stracks, 
                                                                                activated_starcks, 
                                                                                refind_stracks, 
                                                                                self.args.depth_levels_low, 
                                                                                0.3, 
                                                                                is_fuse=False) 
        for track in u_strack:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)  

        
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame 
        detections = [d for d in u_detection_high ]
        dists = iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh = self.args.confirm_thresh) 
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # get scores of lost tracks
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum
        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = np.asarray(
            [x.tlbr.tolist() + [x.track_id, x.score, x.cls, x.idx] for x in self.tracked_stracks if x.is_activated],
            dtype=np.float32)

        self.pre_img = curr_img
        return output_stracks

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracking with detections and scores using STrack algorithm."""
        return [DSTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    def multi_predict(self, tracks):
        """Predict and track multiple objects with YOLOv8 model."""
        DSTrack.multi_predict(tracks)
    
    @staticmethod
    def reset_id():
        """Resets the ID counter of DSTrack."""
        DSTrack.reset_id()