# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import copy
import itertools
import logging
from collections import OrderedDict

import torch

from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ClasEvaluator(DatasetEvaluator):
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._cpu_device = torch.device('cpu')

        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        pred_logits = outputs.to(self._cpu_device, torch.float32)
        labels = inputs["targets"].to(self._cpu_device)

        # measure accuracy
        acc1, = accuracy(pred_logits, labels, topk=(1,))
        num_correct_acc1 = acc1 * labels.size(0) / 100

        self._predictions.append({"num_correct": num_correct_acc1, "num_samples": labels.size(0)})

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process(): return {}

        else:
            predictions = self._predictions

        total_correct_num = 0
        total_samples = 0
        for prediction in predictions:
            total_correct_num += prediction["num_correct"]
            total_samples += prediction["num_samples"]

        acc1 = total_correct_num / total_samples * 100

        self._results = OrderedDict()
        self._results["Acc@1"] = acc1
        self._results["metric"] = acc1

        return copy.deepcopy(self._results)
