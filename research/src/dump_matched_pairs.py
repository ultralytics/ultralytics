# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dump every IoU-matched (ground truth, prediction) person pair that Pose3DValidator scores.

H10 needs the raw per-person keypoints behind MPJPE, not the averaged metric, so it can decompose the error and
refit the predicted skeletons under body-model constraints offline. This hooks `_accumulate_3d` and records
exactly the tensors it scores, so the offline numbers reproduce the validator's to the printed digit.

Usage:
    python research/src/dump_matched_pairs.py <weights.pt> <data.yaml> <out.npz> [--device 0]
"""

import argparse

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.pose3d.val import Pose3DValidator
from ultralytics.utils.metrics import box_iou

STORE = {"g": [], "p": [], "focal": [], "cx": [], "cy": [], "file": []}
_orig = Pose3DValidator._accumulate_3d


def _hooked(self, preds, batch):
    """Record the matched pairs exactly as `_accumulate_3d` selects them, then defer to it."""
    gt_kpts, pred_kpts = batch["keypoints"], preds["keypoints"]
    if gt_kpts.shape[0] and pred_kpts.shape[0]:
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        best_iou, best_pred = iou.max(1)
        keep = best_iou >= 0.5
        if keep.any():
            gt_idx = torch.nonzero(keep).flatten()
            order = torch.argsort(best_iou[keep], descending=True)
            _, first = np.unique(best_pred[keep][order].cpu().numpy(), return_index=True)
            gt_idx, pred_idx = gt_idx[order][first], best_pred[keep][order][first]
            h, w = batch["imgsz"] if "imgsz" in batch else (self.args.imgsz, self.args.imgsz)
            n = len(gt_idx)
            STORE["g"].append(gt_kpts[gt_idx].float().cpu().numpy())
            STORE["p"].append(pred_kpts[pred_idx].float().cpu().numpy())
            STORE["focal"] += [self.focal_ratio * max(float(h), float(w))] * n
            STORE["cx"] += [float(w) / 2] * n
            STORE["cy"] += [float(h) / 2] * n
            STORE["file"] += [batch.get("im_file", "")] * n
    return _orig(self, preds, batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    Pose3DValidator._accumulate_3d = _hooked
    r = YOLO(a.weights).val(data=a.data, imgsz=640, batch=32, device=a.device, plots=False, verbose=False)
    d = {k.split("/")[-1]: v for k, v in r.results_dict.items()}
    print("VALIDATOR", {k: round(float(v), 5) for k, v in d.items()}, flush=True)
    np.savez(
        a.out,
        g=np.concatenate(STORE["g"]),
        p=np.concatenate(STORE["p"]),
        focal=np.array(STORE["focal"]),
        cx=np.array(STORE["cx"]),
        cy=np.array(STORE["cy"]),
        im_file=np.array([str(f) for f in STORE["file"]]),
        metrics=np.array([d.get(k, np.nan) for k in ("MPJPE", "PA-MPJPE", "AbsRel(Z)", "delta1(Z)")]),
    )
    print("saved", a.out, len(STORE["focal"]), "pairs")
