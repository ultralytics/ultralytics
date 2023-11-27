# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import warnings

import faiss
import numpy as np

try:
    from .rank_cylib.roc_cy import evaluate_roc_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython roc evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def evaluate_roc_py(distmat, q_pids, g_pids, q_camids, g_camids):
    r"""Evaluation with ROC curve.
    Key: for each query identity, its gallery images from the same camera view are discarded.

    Args:
        distmat (np.ndarray): cosine distance matrix
    """
    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    pos = []
    neg = []
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]

        sort_idx = order[keep]

        q_dist = distmat[q_idx]
        ind_pos = np.where(raw_cmc == 1)[0]
        pos.extend(q_dist[sort_idx[ind_pos]])

        ind_neg = np.where(raw_cmc == 0)[0]
        neg.extend(q_dist[sort_idx[ind_neg]])

    scores = np.hstack((pos, neg))

    labels = np.hstack((np.zeros(len(pos)), np.ones(len(neg))))
    return scores, labels


def evaluate_roc(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_cython=True
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_roc_cy(distmat, q_pids, g_pids, q_camids, g_camids)
    else:
        return evaluate_roc_py(distmat, q_pids, g_pids, q_camids, g_camids)
