import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import torch
from torchvision import models

from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.tracker.utils.SLM import load_model
from supervision.tracker.utils.fast_reid.fast_reid_interfece import FastReIDInterface
from supervision.tracker.utils.feature_extractor import extract_image_patch, create_extractor, FeatureExtractor
from supervision.tracker.utils.gmc import GMC

from supervision.tracker.byte_tracker import matching

from supervision.tracker.byte_tracker.basetrack import TrackState, BaseTrack


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.features = deque([], maxlen=feat_history)

        self.class_ids = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)

        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, class_ids, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if class_ids == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.class_ids = c[0]
            if not found:
                self.cls_hist.append([class_ids, score])
                self.class_ids = class_ids
        else:
            self.cls_hist.append([class_ids, score])
            self.class_ids = class_ids

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=np.float32), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.update_cls(new_track.class_ids, new_track.score)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.update_cls(new_track.class_ids, new_track.score)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int32)
    patches = [image[box[1]:box[3], box[0]:box[2], :] for box in bboxes]
    # bboxes = clip_boxes(bboxes, image.shape)
    return patches


class SMILETracker(object):
    def __init__(self, track_high_thresh, track_low_thresh, new_track_thresh, track_buffer,
                 proximity_thresh, appearance_thresh, with_reid, fast_reid_config, fast_reid_weights, device,
                 cmc_method, name, ablation, match_thresh, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()
        self.with_reid = with_reid
        self.match_thresh = match_thresh
        self.mot20 = False
        self.fast_reid_config = fast_reid_config
        self.fast_reid_weights = fast_reid_weights
        self.device = torch.device(device)

        self.frame_id = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        if self.with_reid:
            # self.encoder = FastReIDInterface(fast_reid_config, fast_reid_weights, device)
            self.weight_path = self.fast_reid_weights
            self.encoder = load_model(self.weight_path)
            self.encoder = self.encoder.eval()
            if device == 'cuda':
                self.encoder = self.encoder.to(torch.device("cuda"))
                self.encoder = torch.compile(self.encoder)
            elif device == 'mps':
                self.encoder = self.encoder.to(torch.device("mps"))
                self.encoder = torch.compile(self.encoder, backend="aot_eager")
            else:
                self.encoder = self.encoder.to(torch.device("cpu"))
                self.encoder = torch.compile(self.encoder)

            # self.extractor = create_extractor(FeatureExtractor, batch_size=1, model=self.model, device=self.device)
        self.gmc = GMC(method=cmc_method, verbose=[name, ablation])

        """    def process_detections(self, img, detections):
        # Convert detections.xyxy to a numpy array if it's not already
        xyxy = np.array(
            detections.xyxy)  # Assuming detections.xyxy is in the format [x_center, y_center, width, height]

        # Convert from center format to tlwh format using vectorized operations
        boxes_tlwh = np.zeros_like(xyxy)
        boxes_tlwh[:, 0] = xyxy[:, 0] - xyxy[:, 2] / 2  # x top-left
        boxes_tlwh[:, 1] = xyxy[:, 1] - xyxy[:, 3] / 2  # y top-left
        boxes_tlwh[:, 2] = xyxy[:, 2]  # width
        boxes_tlwh[:, 3] = xyxy[:, 3]  # height

        # Convert back to list of lists if required by the extractor
        boxes_tlwh = boxes_tlwh.tolist()

        # Extract features
        features = self.extractor(img, boxes_tlwh)
        return features"""

    def update_with_detections(self, Detections, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        features_keep = []

        if len(Detections):
            bboxes = Detections.xyxy
            scores = Detections.confidence
            classes = Detections.class_id
            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []
            features_keep = []

        '''Extract embeddings '''
        if self.with_reid:

            # features_keep = self.encoder.inference(img, dets)

            # set dets features
            patches_det = extract_image_patches(img, dets)
            features = torch.zeros((len(patches_det), 128), dtype=torch.float32)

            for time in range(len(patches_det)):
                patches_det[time] = torch.tensor(patches_det[time]).to(self.device)
                features[time, :] = self.encoder.inference_forward_fast(patches_det[time].type(torch.float32))

            features_keep = features.cpu().detach().numpy()

        if len(dets) > 0:
            '''Detections'''
            if self.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                              (tlbr, s, c, f) in zip(dets, scores_keep, classes_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                              (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.with_reid:

            dists_iou = matching.iou_distance(strack_pool, detections)
            dists_emb = matching.embedding_distance(strack_pool, detections)
            dists_emb = matching.fuse_motion(self.kalman_filter, dists_emb, strack_pool, detections)

            if dists_emb.size != 0:
                dists = matching.gate(dists_iou, dists_emb)
            else:
                dists = dists_iou

        #             emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
        #             raw_emb_dists = emb_dists.copy()
        #             emb_dists[emb_dists > self.appearance_thresh] = 1.0
        #             emb_dists[ious_dists_mask] = 1.0
        #             dists = np.minimum(ious_dists, emb_dists)

        # Popular ReID method (JDE / FairMOT)
        # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
        # emb_dists = dists

        # IoU making ReID
        # dists = matching.embedding_distance(strack_pool, detections)
        # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                                 (tlbr, s, c) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
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
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        # output_stracks = [track for track in self.tracked_stracks]
        tracks = [track for track in self.tracked_stracks if track.is_activated]

        #         return output_stracks

        detections = Detections.empty()
        if len(tracks) > 0:
            detections.xyxy = np.array(
                [track.tlbr for track in tracks], dtype=np.float32
            )
            detections.class_id = np.array(
                [int(t.class_ids) for t in tracks], dtype=int
            )
            detections.tracker_id = np.array(
                [int(t.track_id) for t in tracks], dtype=int
            )
            detections.confidence = np.array(
                [t.score for t in tracks], dtype=np.float32
            )
        else:
            detections.tracker_id = np.array([], dtype=int)

        return detections, tracks


def joint_stracks(tlista, tlistb):
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


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
