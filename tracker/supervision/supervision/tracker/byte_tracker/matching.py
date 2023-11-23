from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from supervision.detection.utils import box_iou_batch

from supervision.tracker.byte_tracker.kalman_filter import chi2inv95


def indices_to_matches(
    cost_matrix: np.ndarray, indices: np.ndarray, thresh: float
) -> Tuple[np.ndarray, tuple, tuple]:
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> [np.ndarray, Tuple[int], Tuple[int, int]]:
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    indices = np.column_stack((row_ind, col_ind))

    return indices_to_matches(cost_matrix, indices, thresh)


def iou_distance(atracks: List, btracks: List) -> np.ndarray:
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if _ious.size != 0:
        _ious = box_iou_batch(np.asarray(atlbrs), np.asarray(btlbrs))
    cost_matrix = 1 - _ious

    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: List) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)

    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix