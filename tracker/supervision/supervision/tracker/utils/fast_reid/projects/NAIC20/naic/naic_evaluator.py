# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from supervision.tracker.utils.fast_reid.fastreid.evaluation import ReidEvaluator
from supervision.tracker.utils.fast_reid.fastreid.evaluation.query_expansion import aqe
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from supervision.tracker.utils.fast_reid.fastreid.utils.compute_dist import build_dist

logger = logging.getLogger("fastreid.naic_submission")


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


class NaicEvaluator(ReidEvaluator):
    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs.cpu())

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        if self.cfg.TEST.SAVE_DISTMAT:
            np.save(os.path.join(self.cfg.OUTPUT_DIR, "distmat.npy"), dist)

        results = defaultdict(list)

        topk_indices = partition_arg_topK(dist, K=200, axis=1)[:, :200]
        for i in range(topk_indices.shape[0]):
            results[query_pids[i]].extend(gallery_pids[topk_indices[i]])

        with open(os.path.join(self.cfg.OUTPUT_DIR, "submit.json"), 'w') as f:
            json.dump(results, f)

        return {}
