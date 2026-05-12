# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from .byte_tracker import STrack
from .oc_sort import OCSORT, OCSortTrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYAH
from .utils.reid import build_encoder


class DeepOCSortTrack(OCSortTrack):
    """Track object for Deep OC-SORT with appearance features and observation-centric state management.

    Extends OCSortTrack with ReID embedding storage and exponential moving average smoothing, plus confidence-adaptive
    embedding update rates.

    Attributes:
        smooth_feat (np.ndarray | None): Smoothed feature vector via EMA.
        curr_feat (np.ndarray | None): Current frame's feature vector.
        features (deque): Feature history buffer.
        alpha_fixed_emb (float): Base EMA factor for embedding updates.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(
        self,
        xywh: np.ndarray,
        score: float,
        cls: Any,
        delta_t: int = 3,
        feat: np.ndarray | None = None,
        alpha_fixed_emb: float = 0.95,
        det_thresh: float = 0.25,
    ):
        """Initialize a :class:`DeepOCSortTrack` with optional appearance features.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format.
            score (float): Detection confidence in `[0, 1]`.
            cls (Any): Class label for the detection.
            delta_t (int): Temporal window for OCM velocity direction computation.
            feat (np.ndarray | None): Optional appearance feature vector for this detection.
            alpha_fixed_emb (float): Base EMA factor for embedding updates; higher = slower updates.
            det_thresh (float): Detection-confidence threshold below which the embedding is replaced rather than
                blended.
        """
        super().__init__(xywh, score, cls, delta_t)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha_fixed_emb = alpha_fixed_emb
        self.det_thresh = det_thresh
        if feat is not None:
            self.update_features(feat, score)

    def update_features(self, feat: np.ndarray, score: float | None = None) -> None:
        """Blend a new appearance feature into `smooth_feat` with confidence-adaptive EMA.

        When `score` exceeds `det_thresh`, the EMA factor `alpha = alpha_fixed_emb + (1 - alpha_fixed_emb) * (1 - trust)`
        with `trust = (score - det_thresh) / (1 - det_thresh)`, so a high-confidence detection blends in more strongly.
        At or below `det_thresh`, `alpha = 1.0` so the existing `smooth_feat` is kept and the new (low-trust) feature is
        ignored — same behavior used by the reference Deep OC-SORT implementation.

        Args:
            feat (np.ndarray): New (un-normalized) appearance feature vector.
            score (float | None): Detection confidence used to modulate the EMA factor.
        """
        norm = np.linalg.norm(feat)
        if norm < 1e-12:  # skip zero-norm features so smooth_feat isn't poisoned by NaNs
            return
        feat = feat / norm
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if score is not None and score > self.det_thresh:
                trust = (score - self.det_thresh) / max(1 - self.det_thresh, 1e-9)
                alpha = self.alpha_fixed_emb + (1 - self.alpha_fixed_emb) * (1 - trust)
            else:
                alpha = 1.0
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        self.smooth_feat = self.smooth_feat / np.linalg.norm(self.smooth_feat)

    def update(self, new_track: STrack, frame_id: int) -> None:
        """Update track state with a matched detection and refresh appearance features.

        Args:
            new_track (STrack): Matched detection for this frame, with optional `curr_feat`.
            frame_id (int): Current frame id.
        """
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        super().update(new_track, frame_id)

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        """Re-activate a lost track and refresh appearance features.

        Args:
            new_track (STrack): Detection used to revive this track.
            frame_id (int): Current frame id.
            new_id (bool): If True, assign a fresh track id instead of reusing the old one.
        """
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        super().re_activate(new_track, frame_id, new_id)

    @staticmethod
    def multi_gmc(stracks: list[DeepOCSortTrack], H: np.ndarray = np.eye(2, 3)) -> None:
        """Apply global motion compensation correctly for the XYAH Kalman state.

        The standard `multi_gmc` helper from `utils.stracks` would rotate `(a, h)` along with
        `(x, y)`, which is wrong for an aspect-ratio dim. This variant only rotates the
        position `(x, y)` and the velocity `(vx, vy)` blocks, leaves `(a, h)` and `(va, vh)`
        untouched, and also rotates the stored `last_observation` so OCR/ORU stay consistent.

        Args:
            stracks (list[DeepOCSortTrack]): Tracks to warp in place.
            H (np.ndarray): 2x3 affine homography mapping the previous frame to the current one.
        """
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        R = H[:2, :2]
        t = H[:2, 2]

        # Build 8x8 transform: rotate (x,y) and (vx,vy), identity for (a,h) and (va,vh)
        R8x8 = np.eye(8, dtype=float)
        R8x8[:2, :2] = R  # rotate position (x, y)
        R8x8[4:6, 4:6] = R  # rotate velocity (vx, vy)
        # indices 2,3 (a,h) and 6,7 (va,vh) remain identity

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            mean = R8x8.dot(mean)
            mean[:2] += t
            cov = R8x8.dot(cov).dot(R8x8.transpose())
            stracks[i].mean = mean
            stracks[i].covariance = cov

            # Also transform stored observations for OCR/ORU consistency
            if stracks[i].last_observation[0] >= 0:
                obs = stracks[i].last_observation
                # Transform xyxy observation centers
                cx, cy = (obs[0] + obs[2]) / 2, (obs[1] + obs[3]) / 2
                w, h = obs[2] - obs[0], obs[3] - obs[1]
                new_c = R @ np.array([cx, cy]) + t
                stracks[i].last_observation = np.array(
                    [
                        new_c[0] - w / 2,
                        new_c[1] - h / 2,
                        new_c[0] + w / 2,
                        new_c[1] + h / 2,
                    ],
                    dtype=np.float32,
                )


class DeepOCSORT(OCSORT):
    """Deep OC-SORT: OC-SORT enhanced with appearance features, GMC, and adaptive weighting.

    Fixes over naive integration:
    - GMC correctly handles XYAH state (rotates only x,y positions, not aspect ratio/height)
    - Cost combination uses min(IoU, appearance) following BOTSORT's proven approach
    - OCR recovery pass also uses appearance features
    - ByteTrack-style low-confidence second pass disabled by default
    """

    def __init__(self, args: Any, frame_rate: int = 30):
        """Initialize Deep OC-SORT tracker.

        Args:
            args (Namespace | IterableSimpleNamespace): Parsed tracker config providing the OC-SORT keys plus
                `gmc_method`, `proximity_thresh`, `appearance_thresh`, `alpha_fixed_emb`, `with_reid`, and `model`.
            frame_rate (int): Source video frame rate.
        """
        super().__init__(args, frame_rate)

        # GMC for camera motion compensation
        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))

        # Appearance parameters
        self.proximity_thresh = getattr(args, "proximity_thresh", 0.5)
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.75)
        self.alpha_fixed_emb = getattr(args, "alpha_fixed_emb", 0.95)

        self.encoder = build_encoder(getattr(args, "with_reid", False), getattr(args, "model", "auto"))

    def init_track(self, results, img: np.ndarray | None = None) -> list[DeepOCSortTrack]:
        """Build :class:`DeepOCSortTrack` instances, attaching ReID features when enabled.

        When `with_reid=True` and `model="auto"`, `img` should already be a list of native
        backbone features (one per detection); for any other `model`, `img` is the source frame
        and the configured external ReID encoder is invoked on detection crops.

        Args:
            results (Any): Object exposing `xywh` (or `xywhr`), `conf`, and `cls`.
            img (np.ndarray | None): Either the BGR frame or pre-extracted feature tensors, depending on the ReID
                configuration.

        Returns:
            (list[DeepOCSortTrack]): One track per detection, empty if no detections.
        """
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)

        if self.encoder is not None and img is not None:
            features = self.encoder(img, bboxes)
            return [
                DeepOCSortTrack(
                    xywh,
                    s,
                    c,
                    self.delta_t,
                    feat=f,
                    alpha_fixed_emb=self.alpha_fixed_emb,
                    det_thresh=self.args.track_high_thresh,
                )
                for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features)
            ]
        return [
            DeepOCSortTrack(xywh, s, c, self.delta_t, alpha_fixed_emb=self.alpha_fixed_emb)
            for (xywh, s, c) in zip(bboxes, results.conf, results.cls)
        ]

    def _input_for(self, img: np.ndarray | None, feats: np.ndarray | None, mask) -> Any:
        """Return what `init_track` should receive.

        For `model="auto"` (native-features mode) the encoder iterates a per-detection feature
        tensor, so we must hand it `feats[mask]`. If the upstream pipeline didn't populate
        `feats` (e.g. user-supplied detections), return None so :meth:`init_track` falls back
        to the no-encoding path instead of feeding a BGR frame into the auto encoder. For
        external ReID models, `init_track` always wants the BGR frame.
        """
        use_native = self.encoder is not None and getattr(self.args, "model", "auto") == "auto"
        if use_native:
            return feats[mask] if (feats is not None and len(feats)) else None
        return img

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high) -> None:
        """Apply GMC warp to Kalman state before first-stage association."""
        if img is None:
            return
        try:
            warp = self.gmc.apply(img, results_high.xyxy if len(results_high) else np.empty((0, 4)))
        except Exception:
            warp = np.eye(2, 3)
        DeepOCSortTrack.multi_gmc(strack_pool, warp)
        DeepOCSortTrack.multi_gmc(unconfirmed, warp)

    def _fuse_appearance(self, dists, tracks, detections, iou_dists=None):
        """Min-fuse appearance distance into the motion cost (BoT-SORT-style)."""
        if not (self.encoder is not None) or not tracks or not detections:
            return dists
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
        if iou_dists is not None:
            emb_dists[iou_dists > (1 - self.proximity_thresh)] = 1.0
        return np.minimum(dists, emb_dists)

    def reset(self) -> None:
        """Reset the Deep OC-SORT tracker, also clearing the GMC warp state."""
        super().reset()
        self.gmc.reset_params()
