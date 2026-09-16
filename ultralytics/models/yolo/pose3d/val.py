# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils.metrics import Pose3DMetrics, box_iou
from ultralytics.utils.pose3d import decode_z, keypoints_to_camera, procrustes_align

DELTA1_THRESHOLD_M = 0.1  # a joint counts as correct in depth if within 10 cm of GT


class Pose3DValidator(PoseValidator):
    """Validator for the 3D pose task: everything `PoseValidator` reports, plus MPJPE, PA-MPJPE and depth accuracy.

    The 3D numbers are computed only over detections that are matched to a ground-truth person at box IoU >= 0.5,
    so they measure pose quality and not detection quality; `PoseValidator`'s recall covers the latter, and both
    must be read together — a model can look excellent at MPJPE by only keeping easy people.

    Lifting pixels to metres needs intrinsics the label format does not carry, so a pinhole camera with
    `focal = focal_ratio * max(H, W)` and the principal point at the image centre is assumed, `focal_ratio` coming
    from the dataset YAML (default 1.1, the pseudo-focal convention used by CLIFF and HMR2.0). Every MPJPE here is
    therefore conditional on that assumption; PA-MPJPE, which rescales, is the less sensitive of the two.

    Examples:
        >>> from ultralytics.models.yolo.pose3d import Pose3DValidator
        >>> validator = Pose3DValidator(args=dict(model="yolo26n-pose3d.pt", data="coco8-pose3d.yaml"))
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize the 3D pose validator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "pose3d"
        self.metrics = Pose3DMetrics()
        self.focal_ratio = 1.1

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and read the assumed focal ratio from the dataset."""
        super().init_metrics(model)
        self.focal_ratio = float(self.data.get("focal_ratio", 1.1))
        self.metrics.clear_3d()

    def get_desc(self) -> str:
        """Return the results-table header, extended with the four 3D columns."""
        return ("%22s" + "%11s" * 14) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "MPJPE",
            "PA-MPJPE",
            "AbsRel(Z)",
            "d1(Z)",
        )

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run the 2D pose matching, then accumulate 3D errors over IoU-matched person pairs."""
        tp = super()._process_batch(preds, batch)
        self._accumulate_3d(preds, batch)
        return tp

    def _accumulate_3d(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        """Match predictions to ground truth by box IoU and accumulate per-person 3D errors."""
        gt_kpts, pred_kpts = batch["keypoints"], preds["keypoints"]
        if gt_kpts.shape[0] == 0 or pred_kpts.shape[0] == 0:
            return
        iou = box_iou(batch["bboxes"], preds["bboxes"])  # (n_gt, n_pred)
        best_iou, best_pred = iou.max(1)
        keep = best_iou >= 0.5
        if not keep.any():
            return
        # One prediction may not serve two ground truths; keep the higher-IoU claim on each.
        gt_idx = torch.nonzero(keep).flatten()
        _, first = np.unique(best_pred[keep].cpu().numpy(), return_index=True)
        gt_idx, pred_idx = gt_idx[first], best_pred[keep][first]

        g, p = gt_kpts[gt_idx].float(), pred_kpts[pred_idx].float()
        h, w = batch["imgsz"] if "imgsz" in batch else (self.args.imgsz, self.args.imgsz)
        focal = self.focal_ratio * max(float(h), float(w))
        g3d = keypoints_to_camera(g, focal, float(w) / 2, float(h) / 2)
        p3d = keypoints_to_camera(p, focal, float(w) / 2, float(h) / 2)

        vis = g[..., :-1, 2] != 0  # root excluded; it is a derived joint and always "visible"
        if not vis.any():
            return
        # MPJPE is root-relative by definition, so both poses are translated to their own root first.
        gr, pr = g3d[:, :-1] - g3d[:, -1:], p3d[:, :-1] - p3d[:, -1:]
        err = (pr - gr).norm(dim=-1) * 1000.0  # mm
        mpjpe = (err * vis).sum(1) / vis.sum(1).clamp_min(1)

        pa = procrustes_align(pr, gr)
        pa_err = (pa - gr).norm(dim=-1) * 1000.0
        pampjpe = (pa_err * vis).sum(1) / vis.sum(1).clamp_min(1)

        gz_rel, gz_root = decode_z(g[..., :-1, 3], g[..., -1, 3])
        pz_rel, pz_root = decode_z(p[..., :-1, 3], p[..., -1, 3])
        absrel = ((pz_root - gz_root).abs() / gz_root.clamp_min(1e-6)).flatten()
        within = ((pz_rel - gz_rel).abs() < DELTA1_THRESHOLD_M) & vis
        delta1 = within.sum(1) / vis.sum(1).clamp_min(1)

        self.metrics.update_3d(
            mpjpe.cpu().tolist(), pampjpe.cpu().tolist(), absrel.cpu().tolist(), delta1.float().cpu().tolist()
        )
