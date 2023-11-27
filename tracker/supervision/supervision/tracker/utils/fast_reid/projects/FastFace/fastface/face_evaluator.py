# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import copy
import io
import logging
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from supervision.tracker.utils.fast_reid.fastreid.evaluation import DatasetEvaluator
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from supervision.tracker.utils.fast_reid.fastreid.utils.file_io import PathManager
from .verification import evaluate

logger = logging.getLogger("fastreid.face_evaluator")


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


class FaceEvaluator(DatasetEvaluator):
    def __init__(self, cfg, labels, dataset_name, output_dir=None):
        self.cfg = cfg
        self.labels = labels
        self.dataset_name = dataset_name
        self._output_dir = output_dir

        self.features = []

    def reset(self):
        self.features = []

    def process(self, inputs, outputs):
        self.features.append(outputs.cpu())

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features

        features = torch.cat(features, dim=0)
        features = F.normalize(features, p=2, dim=1).numpy()

        self._results = OrderedDict()
        tpr, fpr, accuracy, best_thresholds = evaluate(features, self.labels)

        self._results["Accuracy"] = accuracy.mean() * 100
        self._results["Threshold"] = best_thresholds.mean()
        self._results["metric"] = accuracy.mean() * 100

        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)

        PathManager.mkdirs(self._output_dir)
        roc_curve.save(os.path.join(self._output_dir, self.dataset_name + "_roc.png"))

        return copy.deepcopy(self._results)
