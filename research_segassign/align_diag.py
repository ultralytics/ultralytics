"""Measure how far box IoU is from the task metric (mask IoU / OKS) on matched detections, and how well the score ranks
by each. The pose run is the reference: its OKS-aware assigner gained +2.2 keypoint AP.

Usage: python align_diag.py <task: segment|pose> <weights> <data.yaml> <out.npz> [device]
"""

import sys

import numpy as np
import torch

from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou, kpt_iou, mask_iou

task, weights, data, out = sys.argv[1:5]
device = sys.argv[5] if len(sys.argv) > 5 else "0"
rows = []  # (score, box_iou, task_metric)


def spearmanr(a, b):
    """Spearman rank correlation without scipy (ties broken by order, fine at this sample size)."""
    return (np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1],)
base = SegmentationValidator if task == "segment" else PoseValidator


class V(base):
    def _process_batch(self, preds, batch):
        tp = super()._process_batch(preds, batch)
        if batch["cls"].shape[0] and preds["cls"].shape[0]:
            biou = box_iou(batch["bboxes"], preds["bboxes"])  # (G, P)
            if task == "segment":
                tiou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())
            else:
                area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
                tiou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            biou = biou * (batch["cls"][:, None] == preds["cls"][None])  # same class only
            best, g = biou.max(0)  # best gt per prediction
            keep = best >= 0.5
            t = tiou[g, torch.arange(len(g), device=g.device)]
            rows.append(torch.stack((preds["conf"][keep], best[keep], t[keep]), 1).cpu())
        return tp


args = dict(model=weights, data=data, batch=32, device=device, plots=False, verbose=False, task=task)
v = V(args=args)
v()
r = torch.cat(rows).double().numpy()
np.savez(out, r=r)
s, b, t = r.T
print(f"{task} {weights}: pairs={len(r)} (box IoU>=0.5)")
print(f"  spearman(boxIoU, task)  = {spearmanr(b, t)[0]:.3f}")
print(f"  spearman(score, boxIoU) = {spearmanr(s, b)[0]:.3f}")
print(f"  spearman(score, task)   = {spearmanr(s, t)[0]:.3f}")
print(f"  mean boxIoU={b.mean():.3f} task={t.mean():.3f} |boxIoU-task| mean={np.abs(b - t).mean():.3f}")
print(f"  box>=0.75 but task<0.5: {((b >= 0.75) & (t < 0.5)).mean() * 100:.1f}%   "
      f"box<0.75 but task>=0.75: {((b < 0.75) & (t >= 0.75)).mean() * 100:.1f}%")
