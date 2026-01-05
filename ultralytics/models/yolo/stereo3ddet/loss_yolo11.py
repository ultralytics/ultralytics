from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors


@dataclass
class Stereo3DDetYolo11LossOutput:
    total: torch.Tensor
    loss_items: torch.Tensor
    loss_dict: Dict[str, torch.Tensor]


class Stereo3DDetLossYOLO11P3(nn.Module):
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
        super().__init__()
        device = next(model.parameters()).device

        m = model.model[-1]
        # Detect() module lives inside our head as `detect`.
        detect = getattr(m, "detect", None)
        if detect is None:
            raise AttributeError("Stereo3DDetLossYOLO11P3 expected model.model[-1].detect (Detect module)")

        self.device = device
        self.nc = detect.nc
        self.reg_max = int(detect.reg_max if reg_max is None else reg_max)
        self.no = self.nc + self.reg_max * 4
        self.use_dfl = self.reg_max > 1

        # Hyperparameters (same container as standard losses)
        h = model.args
        self.hyp = h

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

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

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Match v8DetectionLoss.preprocess(): batchify + scale + xywh->xyxy."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode DFL distances to xyxy in feature units."""
        from ultralytics.utils.tal import dist2bbox

        if self.use_dfl:
            b, a, c = pred_dist.shape
            proj = torch.arange(self.reg_max, dtype=torch.float, device=pred_dist.device)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

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

    def forward(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Stereo3DDetYolo11LossOutput:
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

        loss_box *= getattr(self.hyp, "box", 7.5)
        loss_cls *= getattr(self.hyp, "cls", 0.5)
        loss_dfl *= getattr(self.hyp, "dfl", 1.5)

        # Auxiliary losses (masked by fg_mask, targets gathered by target_gt_idx).
        aux_targets = batch.get("aux_targets", {})
        aux_losses: Dict[str, torch.Tensor] = {}

        # Only compute aux when we have targets in batch (train/val).
        if isinstance(aux_targets, dict) and aux_targets:
            for k in ("lr_distance", "right_width", "dimensions", "orientation", "vertices", "vertex_offset", "vertex_dist"):
                if k not in preds or k not in aux_targets:
                    continue
                aux_gt = aux_targets[k].to(self.device)
                # Use smooth_l1 for regression-like heads, l1 for bounded targets.
                loss_type = "smooth_l1" if k in {"lr_distance", "right_width", "dimensions"} else "l1"
                aux_losses[k] = self._aux_loss(preds[k], aux_gt, target_gt_idx, fg_mask, loss_type=loss_type)

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
            aux_losses.get("lr_distance", total * 0.0),
            aux_losses.get("right_width", total * 0.0),
            aux_losses.get("dimensions", total * 0.0),
            aux_losses.get("orientation", total * 0.0),
            aux_losses.get("vertices", total * 0.0),
            aux_losses.get("vertex_offset", total * 0.0),
            aux_losses.get("vertex_dist", total * 0.0),
        ]
        loss_items = torch.stack(items)
        loss_dict = {
            "box": loss_box.detach(),
            "cls": loss_cls.detach(),
            "dfl": loss_dfl.detach(),
            **{k: v.detach() for k, v in aux_losses.items()},
        }
        return Stereo3DDetYolo11LossOutput(total=total, loss_items=loss_items, loss_dict=loss_dict)



