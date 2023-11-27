# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from supervision.tracker.utils.fast_reid.fastreid.evaluation.evaluator import DatasetEvaluator
from supervision.tracker.utils.fast_reid.fastreid.evaluation.rank import evaluate_rank
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from .dsr_distance import compute_dsr_dist

logger = logging.getLogger('fastreid.partialreid.dsr_evaluation')


class DsrEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.spatial_features = []
        self.scores = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.spatial_features = []
        self.scores = []
        self.pids = []
        self.camids = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(F.normalize(outputs[0]).cpu())
        outputs1 = F.normalize(outputs[1].data).cpu()
        self.spatial_features.append(outputs1)
        self.scores.append(outputs[2].cpu())

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            spatial_features = comm.gather(self.spatial_features)
            spatial_features = sum(spatial_features, [])

            scores = comm.gather(self.scores)
            scores = sum(scores, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            spatial_features = self.spatial_features
            scores = self.scores
            pids = self.pids
            camids = self.camids

        features = torch.cat(features, dim=0)
        spatial_features = torch.cat(spatial_features, dim=0).numpy()
        scores = torch.cat(scores, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = 1 - torch.mm(query_features, gallery_features.t()).numpy()
        self._results = OrderedDict()

        query_features = query_features.numpy()
        gallery_features = gallery_features.numpy()
        if self.cfg.TEST.DSR.ENABLED:
            logger.info("Testing with DSR setting")
            dsr_dist = compute_dsr_dist(spatial_features[:self._num_query], spatial_features[self._num_query:], dist,
                                        scores[:self._num_query])

            max_value = 0
            k = 0
            for i in range(0, 101):
                lamb = 0.01 * i
                dist1 = (1 - lamb) * dist + lamb * dsr_dist
                cmc, all_AP, all_INP = evaluate_rank(dist1, query_pids, gallery_pids, query_camids, gallery_camids)
                if (cmc[0] > max_value):
                    k = lamb
                    max_value = cmc[0]
            dist1 = (1 - k) * dist + k * dsr_dist
            cmc, all_AP, all_INP = evaluate_rank(dist1, query_pids, gallery_pids, query_camids, gallery_camids)
        else:
            cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100

        return copy.deepcopy(self._results)
