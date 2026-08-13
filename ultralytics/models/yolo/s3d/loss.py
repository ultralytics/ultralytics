# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss, v8DetectionLoss


# Order of the loss vector Stereo3DDetLoss returns. The trainer derives its own loss_names from the
# criterion's dict on the first batch, but Stereo3DDetTrainer needs them before that to print the progress
# header, so both read this one tuple rather than repeating the literal.
LOSS_NAMES = ("box", "cls", "lr_dist", "depth", "dims", "orient", "proj_center", "photo")


def laplacian_nll(
    pred: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Laplacian negative log-likelihood: |pred-target|*exp(-logvar) + logvar.

    Args:
        pred: Predicted value tensor.
        target: Target value tensor.
        logvar: Predicted log-variance tensor.
        reduction: "mean" collapses to a scalar; "none" returns the elementwise loss.
    """
    loss = torch.abs(pred - target) * torch.exp(-logvar) + logvar
    return loss.mean() if reduction == "mean" else loss


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-pixel SSIM dissimilarity (1-SSIM)/2 with 3x3 average pooling (monodepth-style)."""
    c1, c2 = 0.01**2, 0.03**2
    pool = F.avg_pool2d
    mx, my = pool(x, 3, 1, 1), pool(y, 3, 1, 1)
    sx = pool(x * x, 3, 1, 1) - mx * mx
    sy = pool(y * y, 3, 1, 1) - my * my
    sxy = pool(x * y, 3, 1, 1) - mx * my
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))
    return ((1 - ssim) / 2).clamp(0, 1)


def photometric_disp_loss(disp: torch.Tensor, imgs: torch.Tensor, smooth_w: float = 0.1) -> torch.Tensor:
    """Dense self-supervised left-right photometric consistency on a P3-grid disparity map.

    Warps the right view by the predicted dense disparity (u_R = u_L - d) and penalizes photometric error (0.85*SSIM +
    0.15*L1, monodepth weighting) against the left view, plus an edge-aware disparity smoothness term. The P3 disparity
    map is bilinearly upsampled and the warp runs at FULL image resolution: low-resolution warping is too blurry to
    supervise the 1-2 px matching precision that stereo depth needs. Supervises every pixel — a dense correspondence
    signal that a monocular predictor can only satisfy by producing true disparity everywhere.

    Args:
        disp: [B, 1, H/8, W/8] width-normalized disparity (linear, >=0) on the P3 grid.
        imgs: [B, 6, H, W] letterboxed stereo pair in [0, 1] (left = ch 0-2, right = ch 3-5).
        smooth_w: Weight of the edge-aware smoothness term.

    Returns:
        (torch.Tensor): Scalar loss (photometric mean over valid pixels + smooth_w * smoothness).
    """
    _B, _, H, W = imgs.shape
    disp_full = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=True)
    left, right = imgs[:, :3], imgs[:, 3:6]

    # Broadcast the sampling grid directly (no meshgrid — torch 1.8 compatible)
    gx = torch.linspace(-1, 1, W, device=imgs.device).view(1, 1, W) - 2.0 * disp_full[:, 0]  # u - d, delta=2*d_px/W
    gy = torch.linspace(-1, 1, H, device=imgs.device).view(1, H, 1).expand_as(gx)
    grid = torch.stack([gx, gy], dim=-1)  # [B, H, W, 2]
    warped = F.grid_sample(right, grid, align_corners=True, padding_mode="border")
    valid = ((gx >= -1) & (gx <= 1)).unsqueeze(1).float()  # [B, 1, H, W]

    photo = 0.85 * _ssim(warped, left).mean(1, keepdim=True) + 0.15 * (warped - left).abs().mean(1, keepdim=True)
    photo = (photo * valid).sum() / valid.sum().clamp(min=1.0)

    # Edge-aware smoothness on the mean-normalized P3 disparity (monodepth), edges from 1/8-scale left
    left8 = F.avg_pool2d(left, 8)
    d_n = disp / disp.mean().clamp(min=1e-6)
    dx = (d_n[..., :, 1:] - d_n[..., :, :-1]).abs() * torch.exp(
        -(left8[..., :, 1:] - left8[..., :, :-1]).abs().mean(1, keepdim=True)
    )
    dy = (d_n[..., 1:, :] - d_n[..., :-1, :]).abs() * torch.exp(
        -(left8[..., 1:, :] - left8[..., :-1, :]).abs().mean(1, keepdim=True)
    )
    return photo + smooth_w * (dx.mean() + dy.mean())


def photometric_lr_loss(lr_map: torch.Tensor, imgs: torch.Tensor, smooth_w: float = 0.1) -> torch.Tensor:
    """Photometric consistency on the lr_distance head output (log width-normalized disparity).

    Requires well-textured imagery: on largely textureless scenes the photometric gradient is ambiguous and, combined
    with a weak supervised lr_distance weight, lets background disparities drift unbounded. Keep this loss disabled (the
    default) for such datasets.

    Args:
        lr_map: [B, 1, HW] lr_distance head output (log of width-normalized disparity); the first (H/8)*(W/8) entries
            (P3) are used.
        imgs: [B, 6, H, W] letterboxed stereo pair in [0, 1] (left = ch 0-2, right = ch 3-5).
        smooth_w: Weight of the edge-aware smoothness term.

    Returns:
        (torch.Tensor): Scalar loss (photometric mean over valid pixels + smooth_w * smoothness).
    """
    B, _, H, W = imgs.shape
    h8, w8 = H // 8, W // 8
    # Width-normalized disparity is physically in (0, 1], so log-disparity is bounded above by 0; the upper
    # clamp also keeps exp() finite in fp16 under AMP (exp overflows half precision above ~11).
    disp = lr_map[:, :, : h8 * w8].clamp(min=-10.0, max=0.0).exp().view(B, 1, h8, w8)
    return photometric_disp_loss(disp, imgs, smooth_w)


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
        photometric_loss: bool = False,
    ):
        super().__init__(model, tal_topk=tal_topk)
        self.loss_names = LOSS_NAMES
        self.photometric_loss = photometric_loss
        self.aux_w = loss_weights or {}
        self.use_bbox_loss = use_bbox_loss

        # Depth bin classification (DFL-style). Every parameter comes from the head's own grid: sizing
        # DFLoss from the module-level DEPTH_BINS instead would clamp targets to the default bin count
        # and gather the wrong adjacent pair the moment a head is built with a different one.
        bins = model.model[-1].depth_dfl.bin_values
        self.depth_dfl_loss = DFLoss(reg_max=len(bins))
        self.depth_log_min = bins[0].item()
        self.depth_log_range = (bins[-1] - bins[0]).item()

    def _aux_loss(
        self,
        pred_map: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary loss on positives using gathered GT via target_gt_idx.

        Args:
            pred_map: [B, C, HW_total] — 3D flattened aux predictions.
            aux_gt: [B, max_n, C] — padded per-image GT.
            gt_idx: [B, HW_total] — assignment indices from TAL.
            fg_mask: [B, HW_total] — boolean foreground mask.
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

        return F.smooth_l1_loss(pred_pos, tgt_pos, reduction="mean")

    def _lr_nll_loss(
        self,
        pred_val: torch.Tensor,
        pred_logvar: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Laplacian-NLL loss for lr_distance with per-anchor predicted log-variance.

        Args:
            pred_val: [B, C, HW_total] — lr_distance value predictions.
            pred_logvar: [B, C, HW_total] — lr_distance log-variance predictions.
            aux_gt: [B, max_n, C] — padded per-image GT.
            gt_idx: [B, HW_total] — assignment indices from TAL.
            fg_mask: [B, HW_total] — boolean foreground mask.
        """
        c = pred_val.shape[1]
        val_flat = pred_val.permute(0, 2, 1)  # [B, HW_total, C]
        logvar_flat = pred_logvar.permute(0, 2, 1)  # [B, HW_total, C]

        if aux_gt.shape[1] == 0:
            return pred_val.sum() * 0.0

        if gt_idx.dtype != torch.int64:
            gt_idx = gt_idx.to(torch.int64)
        gathered = aux_gt.gather(1, gt_idx.unsqueeze(-1).expand(-1, -1, c))  # [B, HW_total, C]

        val_pos = val_flat[fg_mask]  # [npos, C]
        logvar_pos = logvar_flat[fg_mask]  # [npos, C]
        tgt_pos = gathered[fg_mask]  # [npos, C]

        if val_pos.numel() == 0:
            return pred_val.sum() * 0.0

        return laplacian_nll(val_pos, tgt_pos, logvar_pos)

    def _compute_aux_losses(
        self,
        aux_preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary losses for all 3D heads."""
        aux_losses: dict[str, torch.Tensor] = {}
        aux_targets = batch.get("aux_targets", {})

        if not isinstance(aux_targets, dict) or not aux_targets:
            return aux_losses

        for k in ("lr_distance", "depth", "dimensions", "orientation"):
            if k not in aux_targets:
                continue
            aux_gt = aux_targets[k].to(self.device)
            if k == "depth" and "depth_bins" in aux_preds:
                aux_losses[k] = self._depth_bin_loss(aux_preds["depth_bins"], aux_gt, target_gt_idx, fg_mask)
            elif k == "orientation" and k in aux_preds:
                aux_losses[k] = self._orientation_multibin_loss(aux_preds[k], aux_gt, target_gt_idx, fg_mask)
            elif k == "lr_distance" and "lr_logvar" in aux_preds:
                aux_losses[k] = self._lr_nll_loss(
                    aux_preds["lr_distance"], aux_preds["lr_logvar"], aux_gt, target_gt_idx, fg_mask
                )
            elif k in aux_preds:
                aux_losses[k] = self._aux_loss(aux_preds[k], aux_gt, target_gt_idx, fg_mask)

        if "proj_offset" in aux_targets and "proj_offset" in aux_preds:
            aux_losses["proj_center"] = self._aux_loss(
                aux_preds["proj_offset"], aux_targets["proj_offset"].to(self.device), target_gt_idx, fg_mask
            )

        return aux_losses

    def _orientation_multibin_loss(
        self,
        pred_map: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """MultiBin orientation loss: bin classification (CE) + GT-bin residual (SmoothL1).

        Channel layout (see orientation.py): [conf_0..conf_{N-1}, sin_0, cos_0, ...].
        Target conf is one-hot of the nearest bin; only that bin's (sin,cos) residual
        is supervised.

        Args:
            pred_map: [B, ORIENT_CHANNELS, HW_total] raw head outputs (conf are logits).
            aux_gt: [B, max_n, ORIENT_CHANNELS] padded MultiBin targets.
            gt_idx: [B, HW_total] TAL assignment indices.
            fg_mask: [B, HW_total] foreground mask.
        """
        from .orientation import NUM_ORIENT_BINS

        c = pred_map.shape[1]
        if aux_gt.shape[1] == 0 or not fg_mask.any():
            return pred_map.sum() * 0.0
        if gt_idx.dtype != torch.int64:
            gt_idx = gt_idx.to(torch.int64)

        gathered = aux_gt.gather(1, gt_idx.unsqueeze(-1).expand(-1, -1, c))  # [B, HW, C]
        pred_pos = pred_map.permute(0, 2, 1)[fg_mask]  # [npos, C]
        tgt_pos = gathered[fg_mask]  # [npos, C]
        if pred_pos.numel() == 0:
            return pred_map.sum() * 0.0

        nb = NUM_ORIENT_BINS
        npos = pred_pos.shape[0]
        ar = torch.arange(npos, device=pred_pos.device)
        bin_tgt = tgt_pos[:, :nb].argmax(dim=1)  # [npos] GT bin from one-hot conf

        # Bin classification (cross-entropy over confidence logits).
        ce = F.cross_entropy(pred_pos[:, :nb], bin_tgt, reduction="none")  # [npos]

        # Residual (sin,cos) regression for the GT bin only.
        pred_res = pred_pos[:, nb:].view(npos, nb, 2)[ar, bin_tgt]  # [npos, 2]
        tgt_res = tgt_pos[:, nb:].view(npos, nb, 2)[ar, bin_tgt]  # [npos, 2]
        res = F.smooth_l1_loss(pred_res, tgt_res, reduction="none").mean(-1)  # [npos]

        return (ce + res).mean()

    def _depth_bin_loss(
        self,
        pred_bins: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DFL-style depth bin classification loss.

        Args:
            pred_bins: [B, n_bins, HW_total] — raw logits from depth branch.
            aux_gt: [B, max_n, 1] — log-depth GT values.
            gt_idx: [B, HW_total] — TAL assignment indices.
            fg_mask: [B, HW_total] — boolean foreground mask.
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

        return self.depth_dfl_loss(pred_fg, tgt_fg.unsqueeze(-1)).mean()

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate stereo 3D detection loss: det losses + aux 3D losses.

        Args:
            preds: Dict with boxes, scores, feats, lr_distance, depth, dimensions, orientation.
            batch: Batch dict with img, batch_idx, cls, bboxes, aux_targets.
        """
        # Separate aux preds from detection preds
        aux_keys = {
            "lr_distance",
            "lr_logvar",
            "depth",
            "depth_bins",
            "dimensions",
            "orientation",
            "proj_offset",
        }
        aux_preds = {k: v for k, v in preds.items() if k in aux_keys}

        loss = torch.zeros(len(LOSS_NAMES), device=self.device)  # see LOSS_NAMES for the slot order

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
        if "proj_center" in aux_losses:
            loss[6] = aux_losses["proj_center"] * float(self.aux_w.get("proj_center", 1.0))
        if self.photometric_loss and "lr_distance" in preds:
            loss[7] = photometric_lr_loss(preds["lr_distance"], batch["img"]) * float(
                self.aux_w.get("photometric", 1.0)
            )

        batch_size = preds["boxes"].shape[0]
        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))
