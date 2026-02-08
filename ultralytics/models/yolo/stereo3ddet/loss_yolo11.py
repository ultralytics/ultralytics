from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class Stereo3DDetLossYOLO11(v8DetectionLoss):
    """Multi-scale loss for stereo 3D detection using YOLO11-style bbox assignment.

    Handles P3+P4+P5 feature maps from the head. Aux targets are in pixel units
    and aux predictions are [B, C, HW_total] (3D tensors, not 4D).

    Expected head training output:
        (det_feats_list, aux_dict) where:
        - det_feats_list: list of 3 tensors [B, no, Hi, Wi]
        - aux_dict: dict[str, Tensor] each [B, C, HW_total]

    Expected batch keys:
        - img: Tensor [B,6,H,W]
        - batch_idx, cls, bboxes: standard YOLO detection targets
        - aux_targets: dict[str, Tensor] each [B, max_n, C] in pixel units
    """

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        reg_max: int | None = None,
        loss_weights: Dict[str, float] | None = None,
    ):
        super().__init__(model, tal_topk=tal_topk)

        if reg_max is not None:
            self.reg_max = reg_max
            self.no = self.nc + self.reg_max * 4
            self.use_dfl = self.reg_max > 1
            from ultralytics.utils.loss import BboxLoss
            self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
            self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

        self.aux_w = loss_weights or {}

    def _aux_loss(
        self,
        pred_map: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        loss_type: str = "l1",
    ) -> torch.Tensor:
        """Compute auxiliary loss on positives using gathered GT via target_gt_idx.

        Args:
            pred_map: [B, C, HW_total] — 3D flattened aux predictions.
            aux_gt: [B, max_n, C] — padded per-image GT.
            gt_idx: [B, HW_total] — assignment indices from TAL.
            fg_mask: [B, HW_total] — boolean foreground mask.
            loss_type: "l1" or "smooth_l1".
        """
        bs, c, n = pred_map.shape
        pred_flat = pred_map.permute(0, 2, 1)  # [B, HW_total, C]

        if aux_gt.shape[1] == 0:
            return pred_map.sum() * 0.0

        if gt_idx.dtype != torch.int64:
            gt_idx = gt_idx.to(torch.int64)
        gathered = aux_gt.gather(1, gt_idx.unsqueeze(-1).expand(-1, -1, c))  # [B, HW_total, C]

        pred_pos = pred_flat[fg_mask]  # [npos, C]
        tgt_pos = gathered[fg_mask]  # [npos, C]

        if pred_pos.numel() == 0:
            return pred_map.sum() * 0.0

        if loss_type == "l1":
            return F.l1_loss(pred_pos, tgt_pos, reduction="mean")
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_pos, tgt_pos, reduction="mean")
        raise ValueError(f"Unknown aux loss_type={loss_type}")

    def _compute_aux_losses(
        self,
        aux_preds: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for all 3D heads."""
        aux_losses: Dict[str, torch.Tensor] = {}
        aux_targets = batch.get("aux_targets", {})

        if isinstance(aux_targets, dict) and aux_targets:
            for k in ("lr_distance", "right_width", "dimensions", "orientation", "vertices", "vertex_offset", "vertex_dist"):
                if k not in aux_preds or k not in aux_targets:
                    continue
                aux_gt = aux_targets[k].to(self.device)
                loss_type = "smooth_l1" if k in {"lr_distance", "right_width", "dimensions"} else "l1"
                aux_losses[k] = self._aux_loss(aux_preds[k], aux_gt, target_gt_idx, fg_mask, loss_type=loss_type)

        return aux_losses

    def __call__(self, preds, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack preds — two formats:
        #   Training: (det_feats_list, aux_dict) tuple
        #   Eval/val: {"det": (inference, feats), "dimensions": ..., ...} dict
        if isinstance(preds, dict):
            det_feats = preds["det"]
            if isinstance(det_feats, tuple):
                det_feats = det_feats[1]  # extract feat list from (inference, feats)
            aux_preds = {k: v for k, v in preds.items() if k != "det"}
        else:
            det_feats, aux_preds = preds
            if isinstance(det_feats, tuple):
                det_feats = det_feats[1]

        bs = det_feats[0].shape[0]
        img = batch["img"]

        # Multi-scale: concatenate det feats across scales
        pred_raw = torch.cat([xi.view(bs, self.no, -1) for xi in det_feats], 2)  # [B, no, HW_total]
        pred_distri, pred_scores = pred_raw.split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [B, HW, nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [B, HW, reg_max*4]

        dtype = pred_scores.dtype
        imgsz = torch.tensor([img.shape[2], img.shape[3]], device=self.device, dtype=dtype)  # (h, w) pixels

        # Multi-scale anchors from all feature maps
        anchor_points, stride_tensor = make_anchors(det_feats, self.stride, 0.5)

        # Targets
        batch_idx = batch["batch_idx"].view(-1, 1).to(self.device)
        cls = batch["cls"].view(-1, 1).to(self.device)
        bboxes = batch["bboxes"].to(self.device)
        targets = torch.cat((batch_idx, cls, bboxes), 1)
        targets = self.preprocess(targets, bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss_box = torch.tensor(0.0, device=self.device)
        loss_dfl = torch.tensor(0.0, device=self.device)
        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss_box *= self.hyp.box
        loss_cls *= self.hyp.cls
        loss_dfl *= self.hyp.dfl

        # Aux losses
        aux_losses = self._compute_aux_losses(aux_preds, batch, target_gt_idx, fg_mask)

        aux_total = torch.tensor(0.0, device=self.device)
        for k, v in aux_losses.items():
            aux_total = aux_total + float(self.aux_w.get(k, 1.0)) * v

        total = loss_box + loss_cls + loss_dfl + aux_total

        items = [
            loss_box,
            loss_cls,
            loss_dfl,
            aux_losses.get("lr_distance", torch.tensor(0.0, device=self.device)),
            aux_losses.get("right_width", torch.tensor(0.0, device=self.device)),
            aux_losses.get("dimensions", torch.tensor(0.0, device=self.device)),
            aux_losses.get("orientation", torch.tensor(0.0, device=self.device)),
            aux_losses.get("vertices", torch.tensor(0.0, device=self.device)),
            aux_losses.get("vertex_offset", torch.tensor(0.0, device=self.device)),
            aux_losses.get("vertex_dist", torch.tensor(0.0, device=self.device)),
        ]
        loss_items = torch.stack(items)

        return total * bs, loss_items.detach()
