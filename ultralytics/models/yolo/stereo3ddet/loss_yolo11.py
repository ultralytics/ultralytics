from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class Stereo3DDetLossYOLO11P3(v8DetectionLoss):
    """P3-only loss for stereo3ddet using YOLO11-style bbox assignment.

    This reuses:
      - TaskAlignedAssigner -> fg_mask + target_gt_idx
      - BboxLoss (IoU + DFL)
    And trains stereo/3D branches only on fg_mask positives, gathered via target_gt_idx.

    Expected batch keys (detection-format):
      - img: Tensor [B,6,H,W]
      - batch_idx: Tensor [N,1] or [N] (image index per GT)
      - cls: Tensor [N,1] or [N] (class id per GT)
      - bboxes: Tensor [N,4] normalized xywh in input image space (letterboxed)

    Expected aux target keys:
      - aux_targets: dict[str, Tensor] each [B, max_n, C] in feature-map units (P3 grid units)
    """

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        reg_max: int | None = None,
        loss_weights: Dict[str, float] | None = None,
    ):
        # Initialize parent with model and tal_topk
        super().__init__(model, tal_topk=tal_topk)
        
        # Override reg_max if specified
        if reg_max is not None:
            self.reg_max = reg_max
            self.no = self.nc + self.reg_max * 4
            self.use_dfl = self.reg_max > 1
            # Recreate bbox_loss with new reg_max
            from ultralytics.utils.loss import BboxLoss
            self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
            self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

        # Loss weights for aux heads (defaults = 1.0; can be overridden via YAML later)
        self.aux_w = loss_weights or {}

    def _infer_stride(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Infer scalar stride from input image and P3 feature map sizes."""
        _, _, ih, iw = img.shape
        _, _, fh, fw = feat.shape
        sh = ih / fh
        sw = iw / fw
        s = (sh + sw) / 2.0
        # sanity: require near-square stride
        if abs(sh - sw) > 1e-3:
            # allow slight mismatch due to rounding, but warn via exception to catch config issues early
            raise ValueError(f"Non-uniform stride inferred: sh={sh:.4f}, sw={sw:.4f} (ih={ih},iw={iw},fh={fh},fw={fw})")
        return torch.tensor([s], device=img.device, dtype=feat.dtype)

    def _aux_loss(
        self,
        pred_map: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        loss_type: str = "l1",
    ) -> torch.Tensor:
        """Compute auxiliary loss on positives using gathered GT via target_gt_idx."""
        bs, c, h, w = pred_map.shape
        pred_flat = pred_map.permute(0, 2, 3, 1).reshape(bs, -1, c)  # [B, HW, C]
        # aux_gt: [B, max_n, C]
        # Handle empty aux_gt (no ground truths in batch)
        if aux_gt.shape[1] == 0:
            return pred_map.sum() * 0.0
        # Convert type gt_idx to int64 if not
        if gt_idx.dtype != torch.int64:
            gt_idx = gt_idx.to(torch.int64)
        gathered = aux_gt.gather(1, gt_idx.unsqueeze(-1).expand(-1, -1, c))  # [B, HW, C]

        pred_pos = pred_flat[fg_mask]  # [npos, C]
        tgt_pos = gathered[fg_mask]  # [npos, C]

        if pred_pos.numel() == 0:
            # keep grad path valid
            return pred_map.sum() * 0.0

        if loss_type == "l1":
            return F.l1_loss(pred_pos, tgt_pos, reduction="mean")
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_pos, tgt_pos, reduction="mean")
        raise ValueError(f"Unknown aux loss_type={loss_type}")

    def _compute_aux_losses(
        self,
        preds: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for 3D detection tasks.
        
        Args:
            preds: Dictionary of predictions including auxiliary heads
            batch: Batch dictionary containing aux_targets
            target_gt_idx: Ground truth indices for each anchor
            fg_mask: Foreground mask indicating positive anchors
            
        Returns:
            Dictionary of auxiliary losses
        """
        aux_losses: Dict[str, torch.Tensor] = {}
        aux_targets = batch.get("aux_targets", {})
        
        # Only compute aux when we have targets in batch (train/val).
        if isinstance(aux_targets, dict) and aux_targets:
            for k in ("lr_distance", "right_width", "dimensions", "orientation", "vertices", "vertex_offset", "vertex_dist"):
                if k not in preds or k not in aux_targets:
                    continue
                aux_gt = aux_targets[k].to(self.device)
                # Use smooth_l1 for regression-like heads, l1 for bounded targets.
                loss_type = "smooth_l1" if k in {"lr_distance", "right_width", "dimensions"} else "l1"
                aux_losses[k] = self._aux_loss(preds[k], aux_gt, target_gt_idx, fg_mask, loss_type=loss_type)
        
        return aux_losses

    def __call__(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Detect feats are in preds["det"] as list with one tensor [B, no, H, W].
        det_feats = preds["det"]
        if isinstance(det_feats, tuple):
            det_feats = det_feats[1]
        if not isinstance(det_feats, list) or len(det_feats) != 1:
            raise TypeError(f"Expected preds['det'] to be a list with 1 feature map, got {type(det_feats)} len={getattr(det_feats, '__len__', None)}")
        feat = det_feats[0]

        img = batch["img"]
        bs = feat.shape[0]
        stride = self._infer_stride(img, feat)  # [1]

        # Split detection outputs into distributions + class logits.
        pred_distri, pred_scores = feat.view(bs, self.no, -1).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [B, HW, nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [B, HW, reg_max*4]

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feat.shape[2:], device=self.device, dtype=dtype) * stride[0]  # (h,w) in pixels
        anchor_points, stride_tensor = make_anchors([feat], stride, 0.5)  # anchor_points: [HW,2], stride_tensor: [HW,1]

        # Targets: expects flattened per-object tensors.
        batch_idx = batch["batch_idx"].view(-1, 1).to(self.device)
        cls = batch["cls"].view(-1, 1).to(self.device)
        bboxes = batch["bboxes"].to(self.device)
        targets = torch.cat((batch_idx, cls, bboxes), 1)  # [N, 6] (idx, cls, xywh)
        targets = self.preprocess(targets, bs, scale_tensor=imgsz[[1, 0, 1, 0]])  # [B, nmax, 5] (cls, xyxy)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predicted bboxes (feature units) then scale to pixels for assigner.
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # [B, HW, 4] xyxy in feature units
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  # pixels
            anchor_points * stride_tensor,  # pixels
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Detection losses (box, cls, dfl) in standard style.
        loss_box = torch.tensor(0.0, device=self.device)
        loss_dfl = torch.tensor(0.0, device=self.device)
        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,  # back to feature units
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss_box *= self.hyp.box
        loss_cls *= self.hyp.cls
        loss_dfl *= self.hyp.dfl

        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(preds, batch, target_gt_idx, fg_mask)

        # Weight aux losses (default 1.0)
        aux_total = torch.tensor(0.0, device=self.device)
        for k, v in aux_losses.items():
            aux_total = aux_total + float(self.aux_w.get(k, 1.0)) * v

        total = loss_box + loss_cls + loss_dfl + aux_total

        # loss_items: keep a stable order for logging
        # [box, cls, dfl, lr_distance, right_width, dimensions, orientation, vertices, vertex_offset, vertex_dist]
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



