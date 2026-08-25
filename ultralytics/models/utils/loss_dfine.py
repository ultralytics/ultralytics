# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.dfine_utils import bbox2distance
from ultralytics.utils.loss import BCEDiceLoss, FocalLoss, KeypointLoss, MALoss, RankLoss, StableDINOLoss, VarifocalLoss
from ultralytics.utils.metrics import OKS_SIGMA, probiou
from ultralytics.utils.ops import crop_mask

from .box_ops import aligned_box_iou, aligned_giou, aligned_giou_new, box_cxcywh_to_xyxy
from .ops import HungarianMatcher


def _global_num_gts(num_gts: int, device: torch.device) -> float:
    """Compute global mean ground-truth count across distributed workers."""
    t = torch.tensor([num_gts], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return max(t.item(), 1.0)


class DfineLoss(nn.Module):
    """Standalone DFine/DEIM-style loss with local terms and union-set support."""

    supports_dfine = True

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 32,
        local_temperature: float = 5.0,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = True,
        use_mal: bool = False,
        use_stable_dino: bool = False,
        use_union_set: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
        stable_alpha: float = 0.25,
        stable_gamma: float = 2.0,
        stable_quality_beta: float = 2.0,
        stable_normalize_targets: bool = True,
        stable_gain: float = 1.0,
        stable_dn_gain: float = 1.0,
        matcher: dict[str, Any] | None = None,
        debug_new_giou_loss: bool = False,
        focaler_d: float = 0.0,
        focaler_u: float = 1.0,
    ):
        super().__init__()
        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "fgl": 0.15, "ddf": 1.5}
        if matcher is None:
            matcher = {}
        else:
            matcher = dict(matcher)

        self.nc = nc
        self.reg_max = reg_max
        self.local_temperature = local_temperature

        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.use_union_set = use_union_set
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.debug_new_giou_loss = debug_new_giou_loss

        self.matcher = HungarianMatcher(**matcher)
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None
        if use_mal and use_stable_dino:
            raise ValueError("MAL and Stable-DINO are alternative classification losses; enable only one")
        self.mal = MALoss(gamma, alpha) if use_mal else None
        self.stable_dino = (
            StableDINOLoss(
                alpha=stable_alpha,
                gamma=stable_gamma,
                quality_beta=stable_quality_beta,
                normalize_targets=stable_normalize_targets,
            )
            if use_stable_dino
            else None
        )
        self.stable_gain = stable_gain
        self.stable_dn_gain = stable_dn_gain

        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.fgl_gain = self.loss_gain.get("fgl", 0.0)
        self.ddf_gain = self.loss_gain.get("ddf", 0.0)
        self.rank_gain = self.loss_gain.get("rank", 0.0)
        self.rank = RankLoss() if self.rank_gain > 0 else None

        # Focaler-IoU (Zhang & Zhang, 2024): linear-interval remap of raw IoU used as an additive
        # modulation of the GIoU loss. Identity remap (0.0, 1.0) is a no-op; only non-default
        # (d, u) engages the modulation. Ge et al. (2025) report a +1.4 mAP@50:95 gain on RT-DETR-R18.
        self.focaler_d = focaler_d
        self.focaler_u = focaler_u
        self.use_focaler = focaler_u > focaler_d and (focaler_d > 0.0 or focaler_u < 1.0)

        # Cache FGL targets per forward pass (DEIM-style) for normal and DN branches.
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None
        self.device = None
        self.matcher_epoch = 0
        self.training_progress = 0.0
        self.main_indices = None  # final-layer Hungarian matches of the last forward pass
        self.aux_indices = None  # per-aux-layer Hungarian matches of the last forward pass (index 0 = encoder row)

    def _clear_local_cache(self) -> None:
        """Clear per-forward local-loss caches."""
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None
        self.main_indices = None
        self.aux_indices = None

    def _aligned_giou_loss(self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Compute the configured aligned GIoU loss vector for matched xywh boxes."""
        giou_fn = aligned_giou_new if self.debug_new_giou_loss else aligned_giou
        loss = 1.0 - giou_fn(pred_bboxes, gt_bboxes, xywh=True)
        if self.use_focaler:
            iou = aligned_box_iou(pred_bboxes, gt_bboxes, xywh=True)
            iou_focaler = ((iou - self.focaler_d) / (self.focaler_u - self.focaler_d)).clamp(0.0, 1.0)
            loss = loss + iou - iou_focaler
        return loss

    def _match(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        pred_angles: torch.Tensor | None = None,
        gt_bboxes_obb: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Wrapper over matcher to inject epoch and normalized progress for matcher scheduling.

        `pred_angles`/`gt_bboxes_obb` enable rotated-IoU (probiou) matching costs for OBB models; when unset the
        matcher falls back to axis-aligned GIoU costs.
        """
        return self.matcher(
            pred_bboxes,
            pred_scores,
            gt_bboxes,
            gt_cls,
            gt_groups,
            epoch=self.matcher_epoch,
            training_progress=self.training_progress,
            pred_angles=pred_angles,
            gt_bboxes_obb=gt_bboxes_obb,
        )

    @staticmethod
    def _global_num_matches(match_indices: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
        """Compute global mean matched-pair count across distributed workers."""
        t = torch.tensor([sum(len(src) for src, _ in match_indices)], device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t = t / dist.get_world_size()
        return max(t.item(), 1.0)

    @staticmethod
    def _get_index(
        match_indices: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Build source/target indices from Hungarian matches."""
        batch_idx = torch.cat([torch.full_like(src, i).to(device=device) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src.to(device=device) for (src, _) in match_indices])
        dst_idx = torch.cat([dst.to(device=device) for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    @staticmethod
    def _merge_union_match_indices(
        indices_main: list[tuple[torch.Tensor, torch.Tensor]],
        indices_aux_list: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Merge matches from multiple layers and keep one GT per query by frequency."""
        merged = [(src.clone(), dst.clone()) for src, dst in indices_main]
        for indices_aux in indices_aux_list:
            merged = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(merged, indices_aux)
            ]

        results = []
        for src, dst in merged:
            if src.numel() == 0:
                results.append((src.long(), dst.long()))
                continue
            ind = torch.cat([src[:, None], dst[:, None]], dim=1)
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            row_to_col = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in row_to_col:
                    row_to_col[row_idx] = col_idx
            final_rows = torch.tensor(list(row_to_col.keys()), device=ind.device, dtype=torch.long)
            final_cols = torch.tensor(list(row_to_col.values()), device=ind.device, dtype=torch.long)
            results.append((final_rows, final_cols))
        return results

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, gt_groups: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build denoising match indices."""
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long, device=dn_pos_idx[i].device) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros([0], dtype=torch.long, device=dn_pos_idx[0].device if dn_pos_idx else None),
                        torch.zeros([0], dtype=torch.long, device=dn_pos_idx[0].device if dn_pos_idx else None),
                    )
                )
        return dn_match_indices

    def _get_loss_class(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        gt_scores: torch.Tensor,
        local_num_gts: int,
        global_num_gts: float,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]

        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        class_gain = self.loss_gain["class"]
        if self.stable_dino is not None:
            loss_cls = self.stable_dino(pred_scores, gt_scores, one_hot)
            loss_cls /= max(global_num_gts, 1.0) / nq
            class_gain *= self.stable_dn_gain if "_dn" in postfix else self.stable_gain
        elif self.mal is not None:
            loss_cls = self.mal(pred_scores, gt_scores, one_hot)
            loss_cls /= max(global_num_gts, 1.0) / nq
        elif self.fl:
            if local_num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(global_num_gts, 1.0) / nq
        else:
            loss_cls = F.binary_cross_entropy_with_logits(pred_scores, gt_scores, reduction="none").mean(1).sum()
        return {name_class: loss_cls.squeeze() * class_gain}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, norm_boxes: float, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        if len(gt_bboxes) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return {name_bbox: zero, name_giou: zero}

        loss_bbox = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / norm_boxes
        loss_giou = self._aligned_giou_loss(pred_bboxes, gt_bboxes)
        loss_giou = self.loss_gain["giou"] * (loss_giou.sum() / norm_boxes)
        return {name_bbox: loss_bbox.squeeze(), name_giou: loss_giou.squeeze()}

    def _get_loss_rank(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        gt_scores: torch.Tensor,
        local_num_gts: int,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Rank & Sort regularizer on the cls head; skipped on the DN branch and when no matches exist."""
        if self.rank is None or not local_num_gts or "_dn" in postfix:
            return {}
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        target_scores = gt_scores.view(bs, nq, 1) * one_hot[..., :-1]
        return {f"loss_rank{postfix}": self.rank_gain * self.rank(pred_scores, target_scores)}

    def _compute_layer_losses(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        cls_indices: list[tuple[torch.Tensor, torch.Tensor]],
        box_indices: list[tuple[torch.Tensor, torch.Tensor]],
        cls_norm: float,
        box_norm: float,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        (cls_batch_idx, cls_src_idx), cls_gt_idx = self._get_index(cls_indices, pred_scores.device)
        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[(cls_batch_idx, cls_src_idx)] = gt_cls[cls_gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if cls_gt_idx.numel():
            pred_assigned_cls = pred_bboxes[(cls_batch_idx, cls_src_idx)]
            gt_assigned_cls = gt_bboxes[cls_gt_idx]
            gt_scores[(cls_batch_idx, cls_src_idx)] = aligned_box_iou(
                pred_assigned_cls.detach(), gt_assigned_cls, xywh=True
            )

        (box_batch_idx, box_src_idx), box_gt_idx = self._get_index(box_indices, pred_scores.device)
        pred_assigned_box = pred_bboxes[(box_batch_idx, box_src_idx)]
        gt_assigned_box = gt_bboxes[box_gt_idx]

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, int(cls_gt_idx.numel()), cls_norm, postfix),
            **self._get_loss_bbox(pred_assigned_box, gt_assigned_box, box_norm, postfix),
            **self._get_loss_rank(pred_scores, targets, gt_scores, int(cls_gt_idx.numel()), postfix),
        }

    def _compute_aux_losses(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        cls_indices_list: list[list[tuple[torch.Tensor, torch.Tensor]]] | list[tuple[torch.Tensor, torch.Tensor]],
        box_indices_list: list[list[tuple[torch.Tensor, torch.Tensor]]] | list[tuple[torch.Tensor, torch.Tensor]],
        cls_norm: float,
        box_norm: float,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        loss = torch.zeros(3, device=pred_bboxes.device)
        loss_rank_aux = torch.zeros((), device=pred_bboxes.device)
        rank_key = f"loss_rank{postfix}"
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            cls_indices = cls_indices_list[i] if isinstance(cls_indices_list[0], list) else cls_indices_list
            box_indices = box_indices_list[i] if isinstance(box_indices_list[0], list) else box_indices_list
            layer_loss = self._compute_layer_losses(
                aux_bboxes,
                aux_scores,
                gt_cls,
                gt_bboxes,
                cls_indices,
                box_indices,
                cls_norm,
                box_norm,
                postfix=postfix,
            )
            loss[0] += layer_loss[f"loss_class{postfix}"]
            loss[1] += layer_loss[f"loss_bbox{postfix}"]
            loss[2] += layer_loss[f"loss_giou{postfix}"]
            if rank_key in layer_loss:
                loss_rank_aux = loss_rank_aux + layer_loss[rank_key]
        out = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        if self.rank is not None and "_dn" not in postfix:
            out[f"loss_rank_aux{postfix}"] = loss_rank_aux
        return out

    @staticmethod
    def _unimodal_distribution_focal_loss(
        pred: torch.Tensor,
        label: torch.Tensor,
        weight_right: torch.Tensor,
        weight_left: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        dis_left = label.long()
        dis_right = dis_left + 1
        pred_f = pred.float()
        loss = F.cross_entropy(pred_f, dis_left, reduction="none").to(pred.dtype) * weight_left.reshape(-1)
        loss = loss + F.cross_entropy(pred_f, dis_right, reduction="none").to(pred.dtype) * weight_right.reshape(-1)
        if weight is not None:
            loss = loss * weight.float()
        return loss.sum() / avg_factor if avg_factor is not None else loss.sum()

    def _ddf_loss(
        self,
        pred_corners: torch.Tensor,
        teacher_corners: torch.Tensor | None,
        teacher_logits: torch.Tensor | None,
        ious: torch.Tensor,
        idx: tuple[torch.Tensor, torch.Tensor],
        is_dn: bool,
        pred_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_corners is None or teacher_logits is None:
            return torch.tensor(0.0, device=pred_corners.device)
        pred_all = pred_corners.reshape(-1, self.reg_max + 1)
        teacher_all = teacher_corners.reshape(-1, self.reg_max + 1)
        if torch.equal(pred_all, teacher_all):
            return torch.tensor(0.0, device=pred_corners.device)

        weight_targets_local = teacher_logits.sigmoid().max(dim=-1)[0]
        mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
        mask[idx] = True
        mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

        weight_targets_local[idx] = ious.to(weight_targets_local.dtype)
        weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
        pred_all_f = pred_all.float()
        teacher_all_f = teacher_all.float()
        loss_match_local = weight_targets_local * (self.local_temperature**2) * (
            self.kl_loss(
                F.log_softmax(pred_all_f / self.local_temperature, dim=1),
                F.softmax(teacher_all_f.detach() / self.local_temperature, dim=1),
            ).sum(-1).to(pred_all.dtype)
        )
        if not is_dn:
            batch_scale = 8 / pred_bboxes.shape[0]
            self.num_pos = (mask.sum() * batch_scale) ** 0.5
            self.num_neg = ((~mask).sum() * batch_scale) ** 0.5
        loss_pos = loss_match_local[mask].mean() if mask.any() else 0.0
        loss_neg = loss_match_local[~mask].mean() if (~mask).any() else 0.0
        denom = max(self.num_pos + self.num_neg, 1.0)
        return (loss_pos * self.num_pos + loss_neg * self.num_neg) / denom

    def _loss_local_single(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_corners: torch.Tensor | None,
        ref_points: torch.Tensor | None,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        norm_boxes: float,
        up: torch.Tensor | None,
        reg_scale: torch.Tensor | None,
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        teacher_corners: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        postfix: str = "",
        is_dn: bool = False,
    ) -> dict[str, torch.Tensor]:
        name_fgl = f"loss_fgl{postfix}"
        name_ddf = f"loss_ddf{postfix}"
        if pred_corners is None or ref_points is None or up is None or reg_scale is None:
            zero = torch.tensor(0.0, device=pred_bboxes.device)
            return {name_fgl: zero, name_ddf: zero}
        if match_indices is None:
            match_indices = self._match(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

        idx, gt_idx = self._get_index(match_indices, pred_bboxes.device)
        if gt_idx.numel() == 0:
            zero = torch.tensor(0.0, device=pred_bboxes.device)
            return {name_fgl: zero, name_ddf: zero}

        target_boxes = gt_bboxes[gt_idx]
        cache_name = "fgl_targets_dn" if is_dn else "fgl_targets"
        target_cache = getattr(self, cache_name)
        if target_cache is None:
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            target_cache = bbox2distance(ref_points[idx].detach(), target_boxes_xyxy, self.reg_max, reg_scale, up)
            setattr(self, cache_name, target_cache)
        target_corners, weight_right, weight_left = target_cache
        pred_corners_sel = pred_corners[idx].reshape(-1, self.reg_max + 1)

        ious = aligned_box_iou(pred_bboxes[idx], target_boxes, xywh=True)
        weight_targets = ious.unsqueeze(-1).repeat(1, 4).reshape(-1).detach()
        loss_fgl = self._unimodal_distribution_focal_loss(
            pred_corners_sel,
            target_corners.reshape(-1),
            weight_right.reshape(-1),
            weight_left.reshape(-1),
            weight=weight_targets,
            avg_factor=max(norm_boxes, 1.0),
        )
        loss_ddf = self._ddf_loss(pred_corners, teacher_corners, teacher_logits, ious, idx, is_dn, pred_bboxes)

        return {
            name_fgl: loss_fgl * self.fgl_gain,
            name_ddf: loss_ddf * self.ddf_gain,
        }

    def _get_local_bundle(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        norm_boxes: float,
        dfine_meta: dict[str, Any] | None,
        main_indices: list[tuple[torch.Tensor, torch.Tensor]],
        aux_indices: list[list[tuple[torch.Tensor, torch.Tensor]]] | list[tuple[torch.Tensor, torch.Tensor]],
        postfix: str = "",
        is_dn: bool = False,
        include_local_aux: bool = True,
    ) -> dict[str, torch.Tensor]:
        if dfine_meta is None:
            return {}
        pred_corners_all = dfine_meta.get("pred_corners")
        ref_points_all = dfine_meta.get("ref_points")
        if pred_corners_all is None or ref_points_all is None:
            return {}

        if pred_bboxes.shape[0] == pred_corners_all.shape[0] + 1:
            pred_bboxes = pred_bboxes[1:]
            pred_scores = pred_scores[1:]
        elif pred_bboxes.shape[0] != pred_corners_all.shape[0]:
            raise ValueError(
                f"Mismatch: pred_bboxes has {pred_bboxes.shape[0]} layers, pred_corners has {pred_corners_all.shape[0]}."
            )

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        losses = self._loss_local_single(
            pred_bboxes[-1],
            pred_scores[-1],
            pred_corners_all[-1],
            ref_points_all[-1],
            gt_bboxes,
            gt_cls,
            gt_groups,
            norm_boxes,
            dfine_meta.get("up"),
            dfine_meta.get("reg_scale"),
            match_indices=main_indices,
            teacher_corners=None,
            teacher_logits=None,
            postfix=postfix,
            is_dn=is_dn,
        )

        if include_local_aux and self.aux_loss and pred_bboxes.shape[0] > 1:
            teacher_corners = pred_corners_all[-1].detach()
            teacher_logits = pred_scores[-1]
            loss_fgl_aux = torch.tensor(0.0, device=pred_bboxes.device)
            loss_ddf_aux = torch.tensor(0.0, device=pred_bboxes.device)
            for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes[:-1], pred_scores[:-1])):
                layer_indices = aux_indices[i] if isinstance(aux_indices[0], list) else aux_indices
                layer_loss = self._loss_local_single(
                    aux_bboxes,
                    aux_scores,
                    pred_corners_all[i],
                    ref_points_all[i],
                    gt_bboxes,
                    gt_cls,
                    gt_groups,
                    norm_boxes,
                    dfine_meta.get("up"),
                    dfine_meta.get("reg_scale"),
                    match_indices=layer_indices,
                    teacher_corners=teacher_corners,
                    teacher_logits=teacher_logits.detach(),
                    postfix=postfix,
                    is_dn=is_dn,
                )
                loss_fgl_aux += layer_loss[f"loss_fgl{postfix}"]
                loss_ddf_aux += layer_loss[f"loss_ddf{postfix}"]

            losses[f"loss_fgl_aux{postfix}"] = loss_fgl_aux
            losses[f"loss_ddf{postfix}"] = losses.get(
                f"loss_ddf{postfix}", torch.tensor(0.0, device=pred_bboxes.device)
            ) + loss_ddf_aux
        return losses

    def _prepare_aux_indices(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        shared_if_enabled: bool = True,
        match_angles: list[torch.Tensor | None] | None = None,
        match_gt_obb: torch.Tensor | None = None,
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        if pred_bboxes.shape[0] <= 1:
            return []
        if match_angles is not None and len(match_angles) != pred_bboxes.shape[0]:
            match_angles = None  # mismatched stack (e.g. validation): fall back to axis-aligned costs
        if self.use_uni_match and shared_if_enabled:
            ind = self.uni_match_ind
            shared = self._match(
                pred_bboxes[ind],
                pred_scores[ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                pred_angles=match_angles[ind] if match_angles is not None else None,
                gt_bboxes_obb=match_gt_obb,
            )
            return [shared for _ in range(pred_bboxes.shape[0] - 1)]
        return [
            self._match(
                b,
                s,
                gt_bboxes,
                gt_cls,
                gt_groups,
                pred_angles=match_angles[i] if match_angles is not None else None,
                gt_bboxes_obb=match_gt_obb,
            )
            for i, (b, s) in enumerate(zip(pred_bboxes[:-1], pred_scores[:-1]))
        ]

    def _prepare_pre_indices(
        self,
        dfine_meta: dict[str, Any] | None,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        if dfine_meta is None:
            return None, None, None
        pre_bboxes = dfine_meta.get("pre_bboxes")
        pre_logits = dfine_meta.get("pre_logits")
        if pre_bboxes is None or pre_logits is None:
            return None, None, None
        pre_bboxes = pre_bboxes.contiguous()
        pre_logits = pre_logits.contiguous()
        pre_indices = self._match(pre_bboxes, pre_logits, gt_bboxes, gt_cls, gt_groups)
        return pre_bboxes, pre_logits, pre_indices

    def _compute_pre_losses(
        self,
        pre_bboxes: torch.Tensor | None,
        pre_logits: torch.Tensor | None,
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        cls_indices: list[tuple[torch.Tensor, torch.Tensor]] | None,
        box_indices: list[tuple[torch.Tensor, torch.Tensor]] | None,
        cls_norm: float,
        box_norm: float,
        postfix: str,
    ) -> dict[str, torch.Tensor]:
        if pre_bboxes is None or pre_logits is None or cls_indices is None:
            return {}
        return self._compute_layer_losses(
            pre_bboxes,
            pre_logits,
            gt_cls,
            gt_bboxes,
            cls_indices,
            box_indices if box_indices is not None else cls_indices,
            cls_norm,
            box_norm,
            postfix=postfix,
        )

    @staticmethod
    def _sanitize_losses(losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        dfine_meta: dict[str, Any] | None = None,
        matcher_epoch: int = 0,
        training_progress: float = 0.0,
        match_angles: list[torch.Tensor | None] | None = None,
        match_gt_obb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_bboxes, pred_scores = preds
        self.device = pred_scores.device
        self._clear_local_cache()
        self.matcher_epoch = int(matcher_epoch)
        self.training_progress = min(max(float(training_progress), 0.0), 1.0)

        if self.training and torch.is_grad_enabled():
            global_num_gts = _global_num_gts(len(batch["bboxes"]), pred_scores.device)
        else:
            global_num_gts = max(len(batch["bboxes"]), 1.0)

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        if match_angles is not None and len(match_angles) != pred_bboxes.shape[0]:
            match_angles = None  # mismatched stack (e.g. validation): fall back to axis-aligned matching costs
        main_indices = self._match(
            pred_bboxes[-1],
            pred_scores[-1],
            gt_bboxes,
            gt_cls,
            gt_groups,
            pred_angles=match_angles[-1] if match_angles is not None else None,
            gt_bboxes_obb=match_gt_obb,
        )
        self.main_indices = main_indices  # exposed for subclasses (e.g. instance-mask losses)
        aux_indices = self._prepare_aux_indices(
            pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, match_angles=match_angles, match_gt_obb=match_gt_obb
        )
        self.aux_indices = aux_indices  # exposed for subclasses; aux_indices[i+1] matches decoder layer i
        pre_bboxes, pre_logits, pre_indices = self._prepare_pre_indices(dfine_meta, gt_bboxes, gt_cls, gt_groups)

        box_union_indices = None
        norm_boxes = global_num_gts
        if self.use_union_set:
            merge_list = []
            if self.aux_loss and aux_indices:
                merge_list.extend(aux_indices)
            if pre_indices is not None:
                merge_list.append(pre_indices)
            box_union_indices = self._merge_union_match_indices(main_indices, merge_list)
            if self.training and torch.is_grad_enabled():
                norm_boxes = self._global_num_matches(box_union_indices, pred_scores.device)
            else:
                norm_boxes = max(sum(len(src) for src, _ in box_union_indices), 1.0)

        main_box_indices = box_union_indices if box_union_indices is not None else main_indices

        total_loss = self._compute_layer_losses(
            pred_bboxes[-1],
            pred_scores[-1],
            gt_cls,
            gt_bboxes,
            main_indices,
            main_box_indices,
            global_num_gts,
            norm_boxes,
        )

        if self.aux_loss and pred_bboxes.shape[0] > 1:
            aux_box_indices = box_union_indices if box_union_indices is not None else aux_indices
            total_loss.update(
                self._compute_aux_losses(
                    pred_bboxes[:-1],
                    pred_scores[:-1],
                    gt_cls,
                    gt_bboxes,
                    aux_indices,
                    aux_box_indices,
                    global_num_gts,
                    norm_boxes,
                )
            )

        total_loss.update(
            self._compute_pre_losses(
                pre_bboxes,
                pre_logits,
                gt_cls,
                gt_bboxes,
                pre_indices,
                box_union_indices if box_union_indices is not None else pre_indices,
                global_num_gts,
                norm_boxes,
                postfix="_pre",
            )
        )

        total_loss.update(
            self._get_local_bundle(
                pred_bboxes,
                pred_scores,
                batch,
                norm_boxes,
                dfine_meta,
                main_indices=main_box_indices,
                aux_indices=box_union_indices if box_union_indices is not None else aux_indices,
            )
        )

        if dn_meta is not None and dn_bboxes is not None and dn_scores is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            dn_match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])
            dn_norm = max(global_num_gts * dn_num_group, 1.0)

            total_loss.update(
                self._compute_layer_losses(
                    dn_bboxes[-1],
                    dn_scores[-1],
                    gt_cls,
                    gt_bboxes,
                    dn_match_indices,
                    dn_match_indices,
                    dn_norm,
                    dn_norm,
                    postfix="_dn",
                )
            )
            if self.aux_loss and dn_bboxes.shape[0] > 1:
                dn_aux_indices = [dn_match_indices for _ in range(dn_bboxes.shape[0] - 1)]
                total_loss.update(
                    self._compute_aux_losses(
                        dn_bboxes[:-1],
                        dn_scores[:-1],
                        gt_cls,
                        gt_bboxes,
                        dn_aux_indices,
                        dn_aux_indices,
                        dn_norm,
                        dn_norm,
                        postfix="_dn",
                    )
                )

            dn_dfine_meta = None
            if dfine_meta is not None:
                dn_dfine_meta = {
                    "pred_corners": dfine_meta.get("dn_pred_corners"),
                    "ref_points": dfine_meta.get("dn_ref_points"),
                    "pre_bboxes": dfine_meta.get("dn_pre_bboxes"),
                    "pre_logits": dfine_meta.get("dn_pre_logits"),
                    "up": dfine_meta.get("up"),
                    "reg_scale": dfine_meta.get("reg_scale"),
                }
                if dn_dfine_meta["pred_corners"] is None or dn_dfine_meta["ref_points"] is None:
                    dn_dfine_meta = None

            total_loss.update(
                self._get_local_bundle(
                    dn_bboxes,
                    dn_scores,
                    batch,
                    dn_norm,
                    dn_dfine_meta,
                    main_indices=dn_match_indices,
                    aux_indices=dn_match_indices,
                    postfix="_dn",
                    is_dn=True,
                    include_local_aux=True,
                )
            )

            dn_pre_bboxes, dn_pre_logits, _ = self._prepare_pre_indices(dn_dfine_meta, gt_bboxes, gt_cls, gt_groups)
            total_loss.update(
                self._compute_pre_losses(
                    dn_pre_bboxes,
                    dn_pre_logits,
                    gt_cls,
                    gt_bboxes,
                    dn_match_indices,
                    dn_match_indices,
                    dn_norm,
                    dn_norm,
                    postfix="_dn_pre",
                )
            )

        return self._sanitize_losses(total_loss)


class DeimSegmentationLoss(DfineLoss):
    """DfineLoss extended with instance-mask and semantic-segmentation losses for DeimSegmentDecoder.

    The mask loss follows the YOLO segmentation convention: per-image matched query coefficients are combined with
    the shared proto via einsum and supervised with a box-cropped BCE against the GT masks. Mask supervision uses
    the one-to-one Hungarian matches of every decoder layer: the final layer drives `loss_mask` and, when
    `task_aux_loss` is enabled, all earlier decoder layers are supervised with their own per-layer matches
    aggregated into `loss_mask_aux` (no denoising mask losses; the box/cls aux losses are governed separately by
    DfineLoss's `aux_loss`). The semantic segmentation aux loss supervises the training-only semseg head of Proto26
    with a BCEDice loss, mirroring v8SegmentationLoss.
    """

    supports_seg = True

    def __init__(self, *args, overlap_mask: bool = True, task_aux_loss: bool = False, **kwargs):
        """Initialize the DEIM segmentation loss.

        Args:
            overlap_mask (bool): Whether GT masks use the overlap index-map convention (bs, H, W) with instance
                ranks starting at 1, as opposed to a per-instance stack (N, H, W).
            task_aux_loss (bool): Whether to supervise the mask heads of earlier decoder layers with their
                per-layer Hungarian matches (`loss_mask_aux`), in addition to the final layer.
            *args (Any): Positional arguments forwarded to DfineLoss.
            **kwargs (Any): Keyword arguments forwarded to DfineLoss. The `mask` entry of `loss_gain` (default 5.0)
                weights the instance-mask loss; the `bbox` gain weights the semseg aux loss.
        """
        super().__init__(*args, **kwargs)
        self.overlap = overlap_mask
        self.task_aux_loss = task_aux_loss
        self.mask_gain = self.loss_gain.get("mask", 5.0)
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def _get_loss_mask(
        self,
        masks_coeff: torch.Tensor,
        proto: torch.Tensor,
        gt_masks: torch.Tensor,
        batch: dict[str, Any],
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Compute the instance-mask loss from one-to-one matches (port of v8SegmentationLoss).

        Args:
            masks_coeff (torch.Tensor): Mask coefficients for one decoder layer with shape (bs, nq, nm).
            proto (torch.Tensor): Mask prototypes with shape (bs, nm, H, W).
            gt_masks (torch.Tensor): GT masks, an overlap index-map with shape (bs, H, W) when `self.overlap` is
                True, otherwise a per-instance stack with shape (N, H, W).
            batch (dict[str, Any]): Targets dict with `bboxes` (normalized cxcywh), `batch_idx`, and `gt_groups`.
            match_indices (list[tuple[torch.Tensor, torch.Tensor]], optional): Matches to supervise; defaults to
                the final-layer o2o matches (`self.main_indices`).

        Returns:
            (torch.Tensor): Weighted instance-mask loss.
        """
        indices = self.main_indices if match_indices is None else match_indices
        bs, _, mask_h, mask_w = proto.shape
        gt_masks = gt_masks.to(proto.device)
        if gt_masks.shape[-2:] != (mask_h, mask_w):  # match proto resolution to GT masks
            proto = F.interpolate(proto, gt_masks.shape[-2:], mode="bilinear", align_corners=False)
            mask_h, mask_w = proto.shape[-2:]
        gt_xyxy = box_cxcywh_to_xyxy(batch["bboxes"])  # normalized xyxy
        total = sum(len(src) for src, _ in indices)
        if not total:
            # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
            return (masks_coeff.sum() + proto.sum()) * 0.0
        loss = masks_coeff.new_zeros(())
        offset = 0
        for i, (src_idx, dst_idx) in enumerate(indices):
            # Matcher indices are CPU tensors; move them to the model device
            src_idx = src_idx.to(proto.device)
            dst_idx = dst_idx.to(proto.device)
            if len(src_idx):
                if self.overlap:
                    local_idx = dst_idx - offset  # per-image instance rank in the overlap index map
                    gt_mask = (gt_masks[i] == (local_idx + 1).view(-1, 1, 1)).float()
                else:
                    gt_mask = gt_masks[dst_idx].float()
                mxyxy = gt_xyxy[dst_idx] * gt_xyxy.new_tensor([mask_w, mask_h, mask_w, mask_h])
                marea = batch["bboxes"][dst_idx][:, 2:4].prod(1)  # normalized box area (w * h of cxcywh)
                pred_mask = torch.einsum("in,nhw->ihw", masks_coeff[i][src_idx], proto[i])
                loss_i = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
                loss += (crop_mask(loss_i, mxyxy).mean(dim=(1, 2)) / marea).sum()
            else:
                # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
                loss += (proto[i].sum() + masks_coeff[i].sum()) * 0.0
            offset += batch["gt_groups"][i]
        return loss / total * self.mask_gain

    def _get_loss_semseg(
        self,
        semseg: torch.Tensor,
        sem_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the semantic-segmentation aux loss (port of v8SegmentationLoss).

        Args:
            semseg (torch.Tensor): Semseg aux head logits with shape (bs, nc, H, W).
            sem_masks (torch.Tensor): Per-pixel class indices with shape (bs, H, W), background as 0.
            gt_masks (torch.Tensor): GT instance masks used to zero out background, same convention as
                `_get_loss_mask`.
            batch (dict[str, Any]): Targets dict, used for `batch_idx` when `self.overlap` is False.

        Returns:
            (torch.Tensor): Weighted semseg aux loss.
        """
        bs = semseg.shape[0]
        sem_masks = sem_masks.to(semseg.device)
        target = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # (bs, nc, H, W)
        if self.overlap:
            mask_zero = gt_masks.to(semseg.device) == 0  # (bs, H, W) overlap index map
            target[mask_zero.unsqueeze(1).expand_as(target)] = 0
        else:
            batch_idx = batch["batch_idx"]
            for i in range(bs):
                instance_masks = gt_masks[batch_idx == i]
                if len(instance_masks) == 0:
                    continue
                target[i, :, instance_masks.sum(dim=0) == 0] = 0
        return self.bcedice_loss(semseg, target) * self.loss_gain["bbox"]

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        dfine_meta: dict[str, Any] | None = None,
        matcher_epoch: int = 0,
        training_progress: float = 0.0,
        dec_masks: torch.Tensor | None = None,
        proto: torch.Tensor | None = None,
        semseg: torch.Tensor | None = None,
        gt_masks: torch.Tensor | None = None,
        sem_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the detection losses plus the instance-mask and semseg losses when mask inputs are provided.

        `dec_masks` stacks per-decoder-layer o2o mask coefficients with shape (L, bs, nq, nm); the final layer is
        supervised as `loss_mask` and, when `self.task_aux_loss` is enabled, earlier layers are supervised with
        their per-layer Hungarian matches (`self.aux_indices[i + 1]`, index 0 being the encoder row) as
        `loss_mask_aux`.
        """
        total_loss = super().forward(
            preds, batch, dn_bboxes, dn_scores, dn_meta, dfine_meta, matcher_epoch, training_progress
        )
        if dec_masks is not None and proto is not None and gt_masks is not None:
            total_loss["loss_mask"] = self._get_loss_mask(dec_masks[-1], proto, gt_masks, batch)
            if (
                self.task_aux_loss
                and dec_masks.shape[0] > 1
                and self.aux_indices
                and len(self.aux_indices) >= dec_masks.shape[0]
            ):
                loss_mask_aux = dec_masks.new_zeros(())
                for i in range(dec_masks.shape[0] - 1):
                    loss_mask_aux = loss_mask_aux + self._get_loss_mask(
                        dec_masks[i], proto, gt_masks, batch, match_indices=self.aux_indices[i + 1]
                    )
                total_loss["loss_mask_aux"] = loss_mask_aux
        if semseg is not None and sem_masks is not None and gt_masks is not None:
            total_loss["loss_semseg"] = self._get_loss_semseg(semseg, sem_masks, gt_masks, batch)
        return self._sanitize_losses(total_loss)


class DeimPoseLoss(DfineLoss):
    """DfineLoss extended with keypoint and keypoint-visibility losses for DeimPoseDecoder.

    The pose loss follows the YOLO pose convention (port of v8PoseLoss.calculate_keypoints_loss): an OKS-form
    KeypointLoss on the sigmoid-decoded keypoint xy coordinates against normalized GT keypoints, weighted by the
    matched GT box area, plus a BCE-with-logits loss on the visibility channel and a plain L1 loss on the
    sigmoid-decoded xy of visible keypoints (the OKS term saturates for small errors; the L1 term keeps a linear
    gradient on coordinates). Keypoint supervision uses the
    one-to-one Hungarian matches of every decoder layer: the final layer drives `loss_pose`/`loss_kobj` and, when
    `task_aux_loss` is enabled, all earlier decoder layers are supervised with their own per-layer matches
    aggregated into `loss_pose_aux`/`loss_kobj_aux` (no denoising pose losses; the box/cls aux losses are governed
    separately by DfineLoss's `aux_loss`); the Hungarian matcher itself stays box+cls only.
    """

    supports_pose = True

    def __init__(self, *args, kpt_shape: tuple = (17, 3), task_aux_loss: bool = False, **kwargs):
        """Initialize the DEIM pose loss.

        Args:
            kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
            task_aux_loss (bool): Whether to supervise the keypoint heads of earlier decoder layers with their
                per-layer Hungarian matches (`loss_pose_aux`/`loss_kobj_aux`/`loss_kpt_l1_aux`), in addition to
                the final layer.
            *args (Any): Positional arguments forwarded to DfineLoss.
            **kwargs (Any): Keyword arguments forwarded to DfineLoss. The `pose` (default 12.0), `kobj`
                (default 1.0), and `kpt_l1` (default 5.0) entries of `loss_gain` weight the OKS keypoint,
                visibility, and keypoint-L1 losses.
        """
        super().__init__(*args, **kwargs)
        self.kpt_shape = list(kpt_shape)
        self.task_aux_loss = task_aux_loss
        self.pose_gain = self.loss_gain.get("pose", 12.0)
        self.kobj_gain = self.loss_gain.get("kobj", 1.0)
        self.kpt_l1_gain = self.loss_gain.get("kpt_l1", 5.0)
        nkpt = self.kpt_shape[0]  # number of keypoints
        is_pose = self.kpt_shape == [17, 3]
        self.sigmas = torch.from_numpy(OKS_SIGMA) if is_pose else torch.ones(nkpt) / nkpt

    def _get_loss_pose(
        self,
        kpts: torch.Tensor,
        gt_keypoints: torch.Tensor,
        batch: dict[str, Any],
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the keypoint, visibility, and keypoint-L1 losses from one-to-one matches (port of v8PoseLoss).

        Args:
            kpts (torch.Tensor): Raw keypoint predictions for one decoder layer with shape (bs, nq, nk).
            gt_keypoints (torch.Tensor): GT keypoints with shape (N, nkpt, ndim), xy normalized to the image.
            batch (dict[str, Any]): Targets dict with `bboxes` (normalized cxcywh) used for the OKS area term.
            match_indices (list[tuple[torch.Tensor, torch.Tensor]], optional): Matches to supervise; defaults to
                the final-layer o2o matches (`self.main_indices`).

        Returns:
            (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Weighted OKS keypoint loss, weighted visibility
                loss, and weighted keypoint-L1 loss.
        """
        indices = self.main_indices if match_indices is None else match_indices
        device = kpts.device
        gt_keypoints = gt_keypoints.to(device).float()
        keypoint_loss = KeypointLoss(sigmas=self.sigmas.to(device))
        total = sum(len(src) for src, _ in indices)
        if not total:
            # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
            zero = kpts.sum() * 0.0
            return zero, zero, zero
        loss_pose = kpts.new_zeros(())
        loss_kobj = kpts.new_zeros(())
        loss_kpt_l1 = kpts.new_zeros(())
        for i, (src_idx, dst_idx) in enumerate(indices):
            # Matcher indices are CPU tensors; move them to the model device
            src_idx = src_idx.to(device)
            dst_idx = dst_idx.to(device)
            if len(src_idx):
                pred_kpt = kpts[i][src_idx].view(-1, *self.kpt_shape)  # (n, nkpt, ndim)
                pred_kpt = torch.cat([pred_kpt[..., :2].sigmoid(), pred_kpt[..., 2:]], dim=-1)
                gt_kpt = gt_keypoints[dst_idx]  # (n, nkpt, ndim), matcher dst is global into concatenated GT
                kpt_mask = (
                    gt_kpt[..., 2] != 0
                    if gt_kpt.shape[-1] == 3
                    else torch.full_like(gt_kpt[..., 0], True, dtype=torch.bool)
                )
                area = batch["bboxes"][dst_idx][:, 2:4].prod(1, keepdim=True)  # normalized box area (w * h of cxcywh)
                loss_pose += keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area) * len(src_idx)
                if pred_kpt.shape[-1] == 3:
                    loss_kobj += F.binary_cross_entropy_with_logits(pred_kpt[..., 2], kpt_mask.float()) * len(src_idx)
                if self.kpt_l1_gain > 0:
                    # Direct xy regression on visible keypoints: OKS saturates quickly for small errors, so a
                    # plain L1 (in the same sigmoid-normalized space) keeps a linear gradient on coordinates
                    l1_per_kpt = F.l1_loss(pred_kpt[..., :2], gt_kpt[..., :2], reduction="none").mean(-1)
                    l1_inst = (l1_per_kpt * kpt_mask).sum(1) / kpt_mask.sum(1).clamp(min=1)  # per-instance mean
                    loss_kpt_l1 += l1_inst.mean() * len(src_idx)
            else:
                # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
                zero = kpts[i].sum() * 0.0
                loss_pose += zero
                loss_kobj += zero
                loss_kpt_l1 += zero
        return (
            loss_pose / total * self.pose_gain,
            loss_kobj / total * self.kobj_gain,
            loss_kpt_l1 / total * self.kpt_l1_gain,
        )

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        dfine_meta: dict[str, Any] | None = None,
        matcher_epoch: int = 0,
        training_progress: float = 0.0,
        dec_kpts: torch.Tensor | None = None,
        gt_keypoints: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the detection losses plus the keypoint and visibility losses when keypoint inputs are provided.

        `dec_kpts` stacks per-decoder-layer o2o keypoint predictions with shape (L, bs, nq, nk); the final layer is
        supervised as `loss_pose`/`loss_kobj`/`loss_kpt_l1` and, when `self.task_aux_loss` is enabled, earlier
        layers are supervised with their per-layer Hungarian matches (`self.aux_indices[i + 1]`, index 0 being the
        encoder row) as `loss_pose_aux`/`loss_kobj_aux`/`loss_kpt_l1_aux`.
        """
        total_loss = super().forward(
            preds, batch, dn_bboxes, dn_scores, dn_meta, dfine_meta, matcher_epoch, training_progress
        )
        if dec_kpts is not None and gt_keypoints is not None:
            (
                total_loss["loss_pose"],
                total_loss["loss_kobj"],
                total_loss["loss_kpt_l1"],
            ) = self._get_loss_pose(dec_kpts[-1], gt_keypoints, batch)
            if (
                self.task_aux_loss
                and dec_kpts.shape[0] > 1
                and self.aux_indices
                and len(self.aux_indices) >= dec_kpts.shape[0]
            ):
                loss_pose_aux = dec_kpts.new_zeros(())
                loss_kobj_aux = dec_kpts.new_zeros(())
                loss_kpt_l1_aux = dec_kpts.new_zeros(())
                for i in range(dec_kpts.shape[0] - 1):
                    pose_i, kobj_i, kpt_l1_i = self._get_loss_pose(
                        dec_kpts[i], gt_keypoints, batch, match_indices=self.aux_indices[i + 1]
                    )
                    loss_pose_aux = loss_pose_aux + pose_i
                    loss_kobj_aux = loss_kobj_aux + kobj_i
                    loss_kpt_l1_aux = loss_kpt_l1_aux + kpt_l1_i
                total_loss["loss_pose_aux"] = loss_pose_aux
                total_loss["loss_kobj_aux"] = loss_kobj_aux
                total_loss["loss_kpt_l1_aux"] = loss_kpt_l1_aux
        return self._sanitize_losses(total_loss)


class DeimOBBLoss(DfineLoss):
    """DfineLoss extended with a rotation-angle loss for DeimOBBDecoder.

    The angle loss is a wrap-invariant 1-cos(delta) term (delta wrapped mod pi to (-pi/2, pi/2]) weighted by an
    aspect-ratio factor exp(-(log(w/h))^2/lambda^2) with lambda=3, computed on the one-to-one Hungarian matches of
    every decoder layer: the final layer drives `loss_angle`/`loss_probiou` and, when `task_aux_loss` is enabled,
    all earlier decoder layers are supervised with their own per-layer matches aggregated into
    `loss_angle_aux`/`loss_probiou_aux` (no denoising angle losses; the box/cls aux losses are governed separately
    by DfineLoss's `aux_loss`). It deliberately differs from v8OBBLoss's
    sin(2*delta)^2 in two ways: (1) sin(2*delta)^2 has period pi/2 and cannot distinguish theta from theta+90deg;
    (2) 1-cos(delta) has maximal gradient at delta=90deg, whereas sin(delta)^2 is flat there, so wrong-branch
    predictions are actively repelled instead of sitting on a gradient-free plateau. Additionally, a probiou IoU
    term on the full rotated boxes couples the (w, h, theta) prediction jointly: DOTA label polygons extending
    beyond image borders are clipped by the augmentation pipeline and their recomputed minAreaRect can flip to the
    perpendicular representation, and only an angle-aware box loss keeps (w, h, theta) consistent through such
    label noise (this is what makes YOLO OBB robust to it). The Hungarian matcher also uses the predicted angles:
    its IoU cost is probiou on the full rotated boxes (the class/L1 costs stay on cls + cxcywh), while the
    axis-aligned box losses (L1, GIoU, FGL/DDF) still run on the xywh part of the xywhr GT.
    """

    supports_obb = True

    def __init__(self, *args, task_aux_loss: bool = False, **kwargs):
        """Initialize the DEIM OBB loss.

        Args:
            task_aux_loss (bool): Whether to supervise the angle heads of earlier decoder layers with their
                per-layer Hungarian matches (`loss_angle_aux`/`loss_probiou_aux`), in addition to the final layer.
            *args (Any): Positional arguments forwarded to DfineLoss.
            **kwargs (Any): Keyword arguments forwarded to DfineLoss. The `angle` entry of `loss_gain` (default 1.0)
                weights the rotation-angle loss; the `probiou` entry (default 1.0) weights the rotated-IoU loss.
        """
        super().__init__(*args, **kwargs)
        self.task_aux_loss = task_aux_loss
        self.angle_gain = self.loss_gain.get("angle", 1.0)
        self.probiou_gain = self.loss_gain.get("probiou", 1.0)

    def _get_loss_angle(
        self,
        angles: torch.Tensor,
        gt_bboxes: torch.Tensor,
        lambda_val: int = 3,
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Compute the rotation-angle loss from one-to-one matches.

        Uses 1-cos(delta) (delta wrapped mod pi) rather than v8OBBLoss's sin(2*delta)^2: the latter has period pi/2
        and scores a 90-degree angle error as zero, which only works when an angle-aware box loss (probiou/DFL)
        resolves the branch; our box losses ignore the angle, so the angle loss must make 90deg maximal, and
        1-cos(delta) additionally has maximal gradient there (sin(delta)^2 is flat at 90deg).

        Args:
            angles (torch.Tensor): Raw angle predictions for one decoder layer with shape (bs, nq, 1).
            gt_bboxes (torch.Tensor): GT rotated boxes with shape (N, 5), normalized xywhr.
            lambda_val (int): Controls the sensitivity to aspect ratio.
            match_indices (list[tuple[torch.Tensor, torch.Tensor]], optional): Matches to supervise; defaults to
                the final-layer o2o matches (`self.main_indices`).

        Returns:
            (torch.Tensor): Weighted rotation-angle loss.
        """
        indices = self.main_indices if match_indices is None else match_indices
        device = angles.device
        gt_bboxes = gt_bboxes.to(device).float()
        total = sum(len(src) for src, _ in indices)
        if not total:
            # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
            return angles.sum() * 0.0
        loss = angles.new_zeros(())
        for i, (src_idx, dst_idx) in enumerate(indices):
            # Matcher indices are CPU tensors; move them to the model device
            src_idx = src_idx.to(device)
            dst_idx = dst_idx.to(device)
            if len(src_idx):
                target = gt_bboxes[dst_idx]  # (n, 5), matcher dst is global into concatenated GT
                log_ar = torch.log((target[:, 2] + 1e-9) / (target[:, 3] + 1e-9))
                scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))
                delta_theta = angles[i][src_idx, 0] - target[:, 4]
                delta_theta = delta_theta - torch.round(delta_theta / math.pi) * math.pi
                # 1 - cos(delta), period 2pi wrapped to (-pi/2, pi/2]: a 90deg error is the loss maximum with
                # maximal gradient there (unlike sin(delta)^2, whose gradient vanishes at 90deg; see class docstring)
                loss += (scale_weight * (1.0 - torch.cos(delta_theta))).sum()
            else:
                # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
                loss += angles[i].sum() * 0.0
        return loss / total * self.angle_gain

    def _get_loss_probiou(
        self,
        pred_boxes: torch.Tensor,
        angles: torch.Tensor,
        gt_bboxes: torch.Tensor,
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Compute the rotated-IoU (probiou) loss from one-to-one matches.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes for one decoder layer with shape (bs, nq, 4), normalized
                cxcywh.
            angles (torch.Tensor): Raw angle predictions for the same decoder layer with shape (bs, nq, 1).
            gt_bboxes (torch.Tensor): GT rotated boxes with shape (N, 5), normalized xywhr.
            match_indices (list[tuple[torch.Tensor, torch.Tensor]], optional): Matches to supervise; defaults to
                the final-layer o2o matches (`self.main_indices`).

        Returns:
            (torch.Tensor): Rotated-IoU loss coupling the (w, h, theta) prediction jointly.
        """
        indices = self.main_indices if match_indices is None else match_indices
        device = angles.device
        gt_bboxes = gt_bboxes.to(device).float()
        total = sum(len(src) for src, _ in indices)
        if not total:
            # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
            return angles.sum() * 0.0
        loss = angles.new_zeros(())
        # probiou is scale-invariant; upscale normalized boxes to pixel-ish units for numerical conditioning
        # (tiny normalized w/h make the covariance determinants eps-dominated, saturating IoU at ~1 with no signal)
        scale = angles.new_tensor([1024.0, 1024.0, 1024.0, 1024.0, 1.0])
        for i, (src_idx, dst_idx) in enumerate(indices):
            # Matcher indices are CPU tensors; move them to the model device
            src_idx = src_idx.to(device)
            dst_idx = dst_idx.to(device)
            if len(src_idx):
                target = gt_bboxes[dst_idx]  # (n, 5), matcher dst is global into concatenated GT
                # Degenerate GT (polygon clipped to zero area by augmentation) makes probiou's sqrt backward NaN
                valid = (target[:, 2] > 0) & (target[:, 3] > 0)
                if valid.any():
                    src_valid = src_idx[valid]
                    pred_xywhr = torch.cat([pred_boxes[i][src_valid].float(), angles[i][src_valid]], dim=-1)
                    tgt = target[valid] * scale
                    pred_xywhr = pred_xywhr * scale
                    # Clamp w/h to >= 1px: near-zero widths/heights make probiou's sqrt backward explode (inf/NaN)
                    pred_xywhr = torch.cat(
                        [pred_xywhr[:, :2], pred_xywhr[:, 2:4].clamp(min=1.0), pred_xywhr[:, 4:]], dim=-1
                    )
                    tgt = torch.cat([tgt[:, :2], tgt[:, 2:4].clamp(min=1.0), tgt[:, 4:]], dim=-1)
                    loss += (1.0 - probiou(pred_xywhr, tgt)).sum()
                else:
                    # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
                    loss += angles[i].sum() * 0.0
            else:
                # WARNING: zero-grad sum prevents Multi-GPU DDP 'unused gradient' errors, do not remove
                loss += angles[i].sum() * 0.0
        return loss / total * self.probiou_gain

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        dfine_meta: dict[str, Any] | None = None,
        matcher_epoch: int = 0,
        training_progress: float = 0.0,
        dec_angles: torch.Tensor | None = None,
        gt_bboxes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the detection losses plus the rotation-angle and rotated-IoU losses when angle inputs are given.

        `dec_angles` stacks per-decoder-layer o2o angle predictions with shape (L, bs, nq, 1); the final layer is
        supervised as `loss_angle`/`loss_probiou` and, when `self.task_aux_loss` is enabled, earlier layers are
        supervised with their per-layer Hungarian matches (`self.aux_indices[i + 1]`, index 0 being the encoder
        row) as `loss_angle_aux`/`loss_probiou_aux`, pairing each layer's angles with its boxes in `preds[0]`
        (encoder row first, so decoder layer i sits at index i + 1).

        The same per-layer angles are also handed to the Hungarian matcher as `match_angles`, so the matching IoU
        cost runs on rotated boxes (probiou) instead of the axis-aligned GIoU used by the other tasks. The
        encoder row and the pre/dn/o2m branches have no angle head and keep axis-aligned matching costs.
        """
        match_angles = None
        if dec_angles is not None and gt_bboxes is not None:
            # Align the per-layer angles with the full prediction stack (encoder row first, then decoder layers)
            nl = preds[0].shape[0]
            nd = dec_angles.shape[0]
            match_angles = [None] * (nl - nd) + [dec_angles[i] for i in range(nd)]
        total_loss = super().forward(
            preds,
            batch,
            dn_bboxes,
            dn_scores,
            dn_meta,
            dfine_meta,
            matcher_epoch,
            training_progress,
            match_angles=match_angles,
            match_gt_obb=gt_bboxes if match_angles is not None else None,
        )
        if dec_angles is not None and gt_bboxes is not None:
            total_loss["loss_angle"] = self._get_loss_angle(dec_angles[-1], gt_bboxes)
            total_loss["loss_probiou"] = self._get_loss_probiou(preds[0][-1], dec_angles[-1], gt_bboxes)
            if (
                self.task_aux_loss
                and dec_angles.shape[0] > 1
                and self.aux_indices
                and len(self.aux_indices) >= dec_angles.shape[0]
            ):
                loss_angle_aux = dec_angles.new_zeros(())
                loss_probiou_aux = dec_angles.new_zeros(())
                for i in range(dec_angles.shape[0] - 1):
                    loss_angle_aux = loss_angle_aux + self._get_loss_angle(
                        dec_angles[i], gt_bboxes, match_indices=self.aux_indices[i + 1]
                    )
                    loss_probiou_aux = loss_probiou_aux + self._get_loss_probiou(
                        preds[0][i + 1], dec_angles[i], gt_bboxes, match_indices=self.aux_indices[i + 1]
                    )
                total_loss["loss_angle_aux"] = loss_angle_aux
                total_loss["loss_probiou_aux"] = loss_probiou_aux
        return self._sanitize_losses(total_loss)
