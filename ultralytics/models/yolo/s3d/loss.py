from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss, v8DetectionLoss


class Stereo3DDetLoss(v8DetectionLoss):
    """Multi-scale loss for stereo 3D detection using YOLO-style bbox assignment.

    Overrides loss() to add auxiliary 3D losses (lr_distance, depth, dimensions,
    orientation) on top of the standard detection losses (box, cls, dfl).

    Expected preds dict keys (from head's forward_head):
        - boxes, scores, feats: standard Detect outputs
        - lr_distance, depth, dimensions, orientation: aux branch outputs [B, C, HW_total]

    Expected batch keys:
        - img, batch_idx, cls, bboxes: standard YOLO detection targets
        - aux_targets: dict[str, Tensor] each [B, max_n, C] in pixel units
    """

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        loss_weights: dict[str, float] | None = None,
        use_bbox_loss: bool = True,
        cls_label_smoothing: float = 0.0,
        pseudo_labels: dict | None = None,
    ):
        super().__init__(model, tal_topk=tal_topk)
        self.aux_w = loss_weights or {}
        self.use_bbox_loss = use_bbox_loss
        self.cls_label_smoothing = cls_label_smoothing

        # Depth bin classification (DFL-style)
        from ultralytics.models.yolo.s3d.head import DEPTH_BINS, DEPTH_MAX, DEPTH_MIN

        self.depth_dfl_loss = DFLoss(reg_max=DEPTH_BINS)
        self.depth_log_min = math.log(DEPTH_MIN)
        self.depth_log_range = math.log(DEPTH_MAX) - math.log(DEPTH_MIN)

        # Pseudo-label curriculum from dataset YAML
        self.epoch_frac = 0.0  # 0.0 = start, 1.0 = end of training
        pl = pseudo_labels or {}
        self._pseudo_stereo_w = float(pl.get("weight", 0.0))
        self._pseudo_mono_w = float(pl.get("mono_weight", 0.0))
        self._pseudo_cutoff = float(pl.get("cutoff", 0.9))

    def _pseudo_aux_weights(self, is_pseudo_fg: torch.Tensor) -> torch.Tensor:
        """Compute per-anchor aux loss weight based on pseudo-label flag and epoch.

        Args:
            is_pseudo_fg: [npos, 1] — 0=real, 1=stereo-pseudo, 2=mono-pseudo.

        Returns:
            [npos, 1] weight tensor: 1.0 for real, reduced for pseudo, 0 after cutoff.
        """
        w = torch.ones_like(is_pseudo_fg)

        # Schedule: linear decay in final phase, then hard cutoff
        if self.epoch_frac >= self._pseudo_cutoff:
            # After cutoff: pseudo labels contribute 0 to aux losses
            pseudo_mask = is_pseudo_fg > 0
            w[pseudo_mask] = 0.0
        else:
            # Before cutoff: reduced weight for pseudo labels
            # Linear ramp-down: full weight at epoch 0, half at cutoff
            schedule = 1.0 - 0.5 * (self.epoch_frac / self._pseudo_cutoff)
            w[is_pseudo_fg == 1] = self._pseudo_stereo_w * schedule
            w[is_pseudo_fg == 2] = self._pseudo_mono_w * schedule

        return w

    def _aux_loss(
        self,
        pred_map: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        aux_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute auxiliary loss on positives using gathered GT via target_gt_idx.

        Args:
            pred_map: [B, C, HW_total] — 3D flattened aux predictions.
            aux_gt: [B, max_n, C] — padded per-image GT.
            gt_idx: [B, HW_total] — assignment indices from TAL.
            fg_mask: [B, HW_total] — boolean foreground mask.
            aux_weights: [npos, 1] — per-anchor weight (pseudo-label curriculum).
        """
        c = pred_map.shape[1]
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

        if aux_weights is not None:
            # Weighted mean: per-anchor loss × weight, normalized by weight sum
            raw = F.smooth_l1_loss(pred_pos, tgt_pos, reduction="none")  # [npos, C]
            return (raw.mean(-1, keepdim=True) * aux_weights).sum() / aux_weights.sum().clamp(min=1.0)

        return F.smooth_l1_loss(pred_pos, tgt_pos, reduction="mean")

    def _compute_aux_losses(
        self,
        aux_preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary losses for all 3D heads with pseudo-label weighting."""
        aux_losses: dict[str, torch.Tensor] = {}
        aux_targets = batch.get("aux_targets", {})

        if not isinstance(aux_targets, dict) or not aux_targets:
            return aux_losses

        # Compute per-anchor pseudo-label weights for aux losses
        aux_weights = None
        is_pseudo_gt = aux_targets.get("is_pseudo")
        if is_pseudo_gt is not None and is_pseudo_gt.shape[1] > 0:
            is_pseudo_gt = is_pseudo_gt.to(self.device)
            if target_gt_idx.dtype != torch.int64:
                target_gt_idx = target_gt_idx.to(torch.int64)
            gathered = is_pseudo_gt.gather(1, target_gt_idx.unsqueeze(-1))  # [B, HW, 1]
            is_pseudo_fg = gathered[fg_mask]  # [npos, 1]
            if is_pseudo_fg.any():
                aux_weights = self._pseudo_aux_weights(is_pseudo_fg)

        for k in ("lr_distance", "depth", "dimensions", "orientation"):
            if k not in aux_targets:
                continue
            aux_gt = aux_targets[k].to(self.device)
            if k == "depth" and "depth_bins" in aux_preds:
                aux_losses[k] = self._depth_bin_loss(
                    aux_preds["depth_bins"],
                    aux_gt,
                    target_gt_idx,
                    fg_mask,
                    aux_weights,
                )
            elif k in aux_preds:
                aux_losses[k] = self._aux_loss(aux_preds[k], aux_gt, target_gt_idx, fg_mask, aux_weights)

        return aux_losses

    def _depth_bin_loss(
        self,
        pred_bins: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        aux_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute DFL-style depth bin classification loss.

        Args:
            pred_bins: [B, n_bins, HW_total] — raw logits from depth branch.
            aux_gt: [B, max_n, 1] — log-depth GT values.
            gt_idx: [B, HW_total] — TAL assignment indices.
            fg_mask: [B, HW_total] — boolean foreground mask.
            aux_weights: [npos, 1] — per-anchor weight (pseudo-label curriculum).
        """
        n_bins = pred_bins.shape[1]
        if aux_gt.shape[1] == 0 or not fg_mask.any():
            return pred_bins.sum() * 0.0

        # Gather GT log-depth per anchor
        if gt_idx.dtype != torch.int64:
            gt_idx = gt_idx.to(torch.int64)
        gathered = aux_gt.gather(1, gt_idx.unsqueeze(-1))  # [B, HW_total, 1]

        # Convert log-depth → fractional bin index
        bin_idx = (gathered - self.depth_log_min) / self.depth_log_range * (n_bins - 1)

        # Select foreground
        pred_fg = pred_bins.permute(0, 2, 1)[fg_mask]  # [npos, n_bins]
        tgt_fg = bin_idx.squeeze(-1)[fg_mask]  # [npos]

        if pred_fg.numel() == 0:
            return pred_bins.sum() * 0.0

        raw = self.depth_dfl_loss(pred_fg, tgt_fg.unsqueeze(-1))  # [npos, 1]
        if aux_weights is not None:
            return (raw * aux_weights).sum() / aux_weights.sum().clamp(min=1.0)
        return raw.mean()

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate stereo 3D detection loss: det losses + aux 3D losses.

        Args:
            preds: Dict with boxes, scores, feats, lr_distance, depth, dimensions, orientation.
            batch: Batch dict with img, batch_idx, cls, bboxes, aux_targets.
        """
        # Separate aux preds from detection preds
        aux_keys = {"lr_distance", "depth", "depth_bins", "dimensions", "orientation"}
        aux_preds = {k: v for k, v in preds.items() if k in aux_keys}

        loss = torch.zeros(6, device=self.device)  # box, cls, lr_dist, depth, dims, orient

        # Get detection losses + TAL assignment results
        (fg_mask, target_gt_idx, _, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)

        if self.use_bbox_loss:
            loss[0] = det_loss[0]  # box (already scaled by hyp.box)
        loss[1] = det_loss[1]  # cls (already scaled by hyp.cls)
        # det_loss[2] is dfl, which is 0 since reg_max=1

        # Aux losses
        aux_losses = self._compute_aux_losses(aux_preds, batch, target_gt_idx, fg_mask)
        for i, k in enumerate(["lr_distance", "depth", "dimensions", "orientation"], 2):
            if k in aux_losses:
                loss[i] = aux_losses[k] * float(self.aux_w.get(k, 1.0))

        batch_size = preds["boxes"].shape[0]
        return loss * batch_size, loss.detach()
