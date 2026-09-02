# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Standalone DEIM-style loss with FGL (Fine-Grained Localization) and DDF (Decoupled Distillation Focal) terms.

This loss is used by the ``DeimDecoder`` head. ``RTDETRDecoder`` continues to use ``RTDETRDetectionLoss``. Dispatch is
performed in ``YOLODETRDetectionModel.init_criterion``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.utils import bbox2distance
from ultralytics.utils.loss import FocalLoss, MALoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy

from .ops import HungarianMatcher

# HungarianMatcher kwargs supported by the clean-branch matcher. Other keys in the YAML matcher dict
# (change_matcher, iou_order_alpha, matcher_change_epoch) are silently dropped in the first-draft port.
_MATCHER_KEYS = {"cost_gain", "use_fl", "with_mask", "num_sample_points", "alpha", "gamma"}


def _global_num_gts(num_gts: int, device: torch.device) -> float:
    """Compute global mean ground-truth count across distributed workers.

    Args:
        num_gts (int): Number of ground truth objects on the local worker.
        device (torch.device): Device on which to build the reduction tensor.

    Returns:
        (float): Mean ground truth count across all workers, floored at 1.0.
    """
    t = torch.tensor([num_gts], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return max(t.item(), 1.0)


class DfineLoss(nn.Module):
    """Standalone DFine/DEIM-style loss with local FGL/DDF terms and optional union-set matching."""

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
        use_union_set: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
        matcher: dict[str, Any] | None = None,
    ):
        """Initialize the loss with classification, box regression, and local FGL/DDF terms.

        Args:
            nc (int): Number of object classes.
            reg_max (int): Number of discrete bins in the D-FINE distribution head.
            local_temperature (float): Softmax temperature for the DDF distillation term.
            loss_gain (dict[str, float], optional): Per-term loss weights keyed by class, bbox, giou, fgl, and ddf.
            aux_loss (bool): Apply losses to the auxiliary decoder layers.
            use_fl (bool): Use focal loss for classification.
            use_vfl (bool): Use varifocal loss on layers that have matches.
            use_mal (bool): Use matchability-aware loss, taking precedence over focal and varifocal.
            use_union_set (bool): Regress boxes against the union of main, auxiliary, and pre-head matches.
            use_uni_match (bool): Share one decoder layer's matches across all auxiliary layers.
            uni_match_ind (int): Decoder layer index whose matches are shared when use_uni_match is set.
            gamma (float): Focusing parameter for the focal, varifocal, and matchability-aware losses.
            alpha (float): Balancing parameter for the focal, varifocal, and matchability-aware losses.
            matcher (dict[str, Any], optional): HungarianMatcher keyword arguments; unsupported keys are dropped.
        """
        super().__init__()
        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "fgl": 0.15, "ddf": 1.5}
        matcher_kwargs = {k: v for k, v in (matcher or {}).items() if k in _MATCHER_KEYS}

        self.nc = nc
        self.reg_max = reg_max
        self.local_temperature = local_temperature

        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.use_union_set = use_union_set
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind

        self.matcher = HungarianMatcher(**matcher_kwargs)
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None
        self.mal = MALoss(gamma, alpha) if use_mal else None

        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.fgl_gain = self.loss_gain.get("fgl", 0.0)
        self.ddf_gain = self.loss_gain.get("ddf", 0.0)

        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None
        self.device = None

    def _clear_local_cache(self) -> None:
        """Clear per-forward local-loss caches."""
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None

    def _match(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run Hungarian matching for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted boxes in xywh format with shape (B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (T, 4).
            gt_cls (torch.Tensor): Ground truth class indices with shape (T,).
            gt_groups (list[int]): Number of ground truth boxes per batch image.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): Per-image (query index, ground truth index) match pairs.
        """
        return self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

    @staticmethod
    def _global_num_matches(match_indices: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
        """Compute global mean matched-pair count across distributed workers.

        Args:
            match_indices (list[tuple[torch.Tensor, torch.Tensor]]): Per-image match pairs from Hungarian matching.
            device (torch.device): Device on which to build the reduction tensor.

        Returns:
            (float): Mean matched-pair count across all workers, floored at 1.0.
        """
        t = torch.tensor([sum(len(src) for src, _ in match_indices)], device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t = t / dist.get_world_size()
        return max(t.item(), 1.0)

    @staticmethod
    def _get_index(
        match_indices: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (list[tuple[torch.Tensor, torch.Tensor]]): Per-image match pairs from Hungarian matching.
            device (torch.device): Device on which to place the returned indices.

        Returns:
            batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination indices into the flattened ground truth.
        """
        batch_idx = torch.cat([torch.full_like(src, i).to(device=device) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src.to(device=device) for (src, _) in match_indices])
        dst_idx = torch.cat([dst.to(device=device) for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    @staticmethod
    def _merge_union_match_indices(
        indices_main: list[tuple[torch.Tensor, torch.Tensor]],
        indices_aux_list: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Merge matches from multiple decoder layers, keeping one ground truth per query by frequency.

        Args:
            indices_main (list[tuple[torch.Tensor, torch.Tensor]]): Match pairs from the final decoder layer.
            indices_aux_list (list[list[tuple[torch.Tensor, torch.Tensor]]]): Match pairs from the remaining layers.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): Union-set match pairs with each query assigned at most once.
        """
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
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, dn_gt_idx: list[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get match indices for the denoising queries.

        Args:
            dn_pos_idx (list[torch.Tensor]): Positive denoising query indices for each image.
            dn_num_group (int): Number of denoising groups.
            dn_gt_idx (list[torch.Tensor]): Ground truth index each image's denoising queries reconstruct.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): Per-image (query index, ground truth index) match pairs.
        """
        return [
            (dn_pos_idx[i], gt_idx.to(dn_pos_idx[i].device).repeat(dn_num_group)) for i, gt_idx in enumerate(dn_gt_idx)
        ]

    def _get_loss_class(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        gt_scores: torch.Tensor,
        local_num_gts: int,
        global_num_gts: float,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss based on predictions, target values, and ground truth scores.

        Args:
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            targets (torch.Tensor): Target class indices with shape (B, N).
            gt_scores (torch.Tensor): Ground truth confidence scores with shape (B, N).
            local_num_gts (int): Number of matched pairs on the local worker.
            global_num_gts (int | float): Mean ground truth count across workers, used to normalize the loss.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing classification loss value.

        Notes:
            The function supports different classification loss types:
            - Matchability-Aware Loss (if self.mal is not None)
            - Varifocal Loss (if self.fl is not None, self.vfl is not None, and local_num_gts > 0)
            - Focal Loss (if self.fl is not None)
            - BCE Loss (default fallback)
        """
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]

        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.mal is not None:
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
        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, norm_boxes: float, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute L1 and GIoU losses for the matched bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted boxes in xywh format with shape (M, 4).
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (M, 4).
            norm_boxes (int | float): Normalizer applied to both loss terms.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing the L1 and GIoU loss values, both zero when no ground truth
                boxes are present.
        """
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        if len(gt_bboxes) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return {name_bbox: zero, name_giou: zero}

        loss_bbox = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / norm_boxes
        loss_giou = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True).squeeze(-1)
        loss_giou = self.loss_gain["giou"] * (loss_giou.sum() / norm_boxes)
        return {name_bbox: loss_bbox.squeeze(), name_giou: loss_giou.squeeze()}

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
        """Compute the classification and box losses for a single decoder layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted boxes in xywh format with shape (B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            gt_cls (torch.Tensor): Ground truth class indices with shape (T,).
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (T, 4).
            cls_indices (list[tuple[torch.Tensor, torch.Tensor]]): Match pairs used to build classification targets.
            box_indices (list[tuple[torch.Tensor, torch.Tensor]]): Match pairs used to select boxes for regression.
            cls_norm (int | float): Normalizer for the classification loss.
            box_norm (int | float): Normalizer for the box losses.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing the classification, L1, and GIoU loss values.

        Notes:
            Classification and box regression can use different match pairs, which is how union-set matching
            regresses against a wider set of queries than it classifies.
        """
        (cls_batch_idx, cls_src_idx), cls_gt_idx = self._get_index(cls_indices, pred_scores.device)
        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[(cls_batch_idx, cls_src_idx)] = gt_cls[cls_gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if cls_gt_idx.numel():
            pred_assigned_cls = pred_bboxes[(cls_batch_idx, cls_src_idx)]
            gt_assigned_cls = gt_bboxes[cls_gt_idx]
            gt_scores[(cls_batch_idx, cls_src_idx)] = bbox_iou(
                pred_assigned_cls.detach(), gt_assigned_cls, xywh=True
            ).squeeze(-1)

        (box_batch_idx, box_src_idx), box_gt_idx = self._get_index(box_indices, pred_scores.device)
        pred_assigned_box = pred_bboxes[(box_batch_idx, box_src_idx)]
        gt_assigned_box = gt_bboxes[box_gt_idx]

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, int(cls_gt_idx.numel()), cls_norm, postfix),
            **self._get_loss_bbox(pred_assigned_box, gt_assigned_box, box_norm, postfix),
        }

    def _compute_aux_losses(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        cls_indices_list,
        box_indices_list,
        cls_norm: float,
        box_norm: float,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Sum the classification and box losses over the auxiliary decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted boxes in xywh format with shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores with shape (L, B, N, C).
            gt_cls (torch.Tensor): Ground truth class indices with shape (T,).
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (T, 4).
            cls_indices_list (list): Match pairs for classification, either one list per layer or one shared list.
            box_indices_list (list): Match pairs for box regression, either one list per layer or one shared list.
            cls_norm (int | float): Normalizer for the classification loss.
            box_norm (int | float): Normalizer for the box losses.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing the summed auxiliary classification, L1, and GIoU losses.
        """
        loss = torch.zeros(3, device=pred_bboxes.device)
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            cls_indices = cls_indices_list[i] if isinstance(cls_indices_list[0], list) else cls_indices_list
            box_indices = box_indices_list[i] if isinstance(box_indices_list[0], list) else box_indices_list
            layer_loss = self._compute_layer_losses(
                aux_bboxes, aux_scores, gt_cls, gt_bboxes,
                cls_indices, box_indices, cls_norm, box_norm, postfix=postfix,
            )
            loss[0] += layer_loss[f"loss_class{postfix}"]
            loss[1] += layer_loss[f"loss_bbox{postfix}"]
            loss[2] += layer_loss[f"loss_giou{postfix}"]
        return {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }

    @staticmethod
    def _unimodal_distribution_focal_loss(
        pred: torch.Tensor,
        label: torch.Tensor,
        weight_right: torch.Tensor,
        weight_left: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Compute the unimodal distribution focal loss over a pair of neighbouring distribution bins.

        Args:
            pred (torch.Tensor): Predicted corner logits with shape (M * 4, reg_max + 1).
            label (torch.Tensor): Continuous bin targets with shape (M * 4,).
            weight_right (torch.Tensor): Interpolation weight for the upper bin with shape (M * 4,).
            weight_left (torch.Tensor): Interpolation weight for the lower bin with shape (M * 4,).
            weight (torch.Tensor, optional): Per-element loss weight with shape (M * 4,).
            avg_factor (int | float, optional): Divisor applied to the summed loss.

        Returns:
            (torch.Tensor): Scalar loss value.

        Notes:
            Cross entropy runs in float32 regardless of the input dtype and the result is cast back, keeping the
            distribution stable under autocast.
        """
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
        """Compute the decoupled distillation focal loss between student and teacher corner distributions.

        Args:
            pred_corners (torch.Tensor): Student corner logits with shape (B, N, 4 * (reg_max + 1)).
            teacher_corners (torch.Tensor, optional): Teacher corner logits with the same shape as pred_corners.
            teacher_logits (torch.Tensor, optional): Teacher class scores with shape (B, N, C).
            ious (torch.Tensor): IoU of each matched pair with shape (M,), used to weight positive locations.
            idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx) of the matched queries.
            is_dn (bool): Whether this call covers denoising queries, which reuse the cached positive/negative counts.
            pred_bboxes (torch.Tensor): Predicted boxes, used only for its batch size when rescaling those counts.

        Returns:
            (torch.Tensor): Scalar loss value, zero when no teacher is supplied or the teacher matches the student.
        """
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
        """Compute the FGL and DDF local losses for a single decoder layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted boxes in xywh format with shape (B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            pred_corners (torch.Tensor, optional): Predicted corner logits with shape (B, N, 4 * (reg_max + 1)).
            ref_points (torch.Tensor, optional): Reference points in xyxy format with shape (B, N, 4).
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (T, 4).
            gt_cls (torch.Tensor): Ground truth class indices with shape (T,).
            gt_groups (list[int]): Number of ground truth boxes per batch image.
            norm_boxes (int | float): Normalizer for the FGL loss.
            up (torch.Tensor, optional): Upper bound controlling the non-uniform bin spacing.
            reg_scale (torch.Tensor, optional): Scale controlling the non-uniform bin spacing.
            match_indices (list[tuple[torch.Tensor, torch.Tensor]], optional): Precomputed match pairs; Hungarian
                matching runs when omitted.
            teacher_corners (torch.Tensor, optional): Teacher corner logits for the DDF term.
            teacher_logits (torch.Tensor, optional): Teacher class scores for the DDF term.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.
            is_dn (bool): Whether this call covers denoising queries, which use a separate target cache.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing the gain-scaled FGL and DDF loss values, both zero when the
                distribution head outputs are absent or nothing matched.

        Notes:
            The distance targets are cached per forward pass, so every decoder layer converts the ground truth
            boxes to bin targets once rather than once per layer.
        """
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
            target_boxes_xyxy = xywh2xyxy(target_boxes, clamp_neg=True)
            target_cache = bbox2distance(ref_points[idx].detach(), target_boxes_xyxy, self.reg_max, reg_scale, up)
            setattr(self, cache_name, target_cache)
        target_corners, weight_right, weight_left = target_cache
        pred_corners_sel = pred_corners[idx].reshape(-1, self.reg_max + 1)

        ious = bbox_iou(pred_bboxes[idx], target_boxes, xywh=True).squeeze(-1)
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
        aux_indices,
        postfix: str = "",
        is_dn: bool = False,
        include_local_aux: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute the FGL and DDF local losses for the final decoder layer and, optionally, the auxiliary ones.

        Args:
            pred_bboxes (torch.Tensor): Predicted boxes in xywh format with shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores with shape (L, B, N, C).
            batch (dict[str, Any]): Batch dictionary holding cls, bboxes, and gt_groups.
            norm_boxes (int | float): Normalizer for the FGL loss.
            dfine_meta (dict[str, Any], optional): Distribution head outputs; an empty dict is returned when absent.
            main_indices (list[tuple[torch.Tensor, torch.Tensor]]): Match pairs for the final decoder layer.
            aux_indices (list): Match pairs for the auxiliary layers, either one list per layer or one shared list.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.
            is_dn (bool): Whether this call covers denoising queries.
            include_local_aux (bool): Also accumulate the auxiliary-layer FGL and DDF losses.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing the FGL and DDF loss values, empty when the distribution
                head outputs are absent.

        Notes:
            The final layer acts as the detached teacher for the auxiliary layers, so the DDF term distills the
            deepest predictions into the shallower ones within a single forward pass.
        """
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
            gt_bboxes, gt_cls, gt_groups,
            norm_boxes,
            dfine_meta.get("up"), dfine_meta.get("reg_scale"),
            match_indices=main_indices,
            postfix=postfix, is_dn=is_dn,
        )

        if include_local_aux and self.aux_loss and pred_bboxes.shape[0] > 1:
            teacher_corners = pred_corners_all[-1].detach()
            teacher_logits = pred_scores[-1]
            loss_fgl_aux = torch.tensor(0.0, device=pred_bboxes.device)
            loss_ddf_aux = torch.tensor(0.0, device=pred_bboxes.device)
            for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes[:-1], pred_scores[:-1])):
                layer_indices = aux_indices[i] if isinstance(aux_indices[0], list) else aux_indices
                layer_loss = self._loss_local_single(
                    aux_bboxes, aux_scores,
                    pred_corners_all[i], ref_points_all[i],
                    gt_bboxes, gt_cls, gt_groups,
                    norm_boxes,
                    dfine_meta.get("up"), dfine_meta.get("reg_scale"),
                    match_indices=layer_indices,
                    teacher_corners=teacher_corners,
                    teacher_logits=teacher_logits.detach(),
                    postfix=postfix, is_dn=is_dn,
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
    ):
        """Build the Hungarian match pairs for the auxiliary decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted boxes in xywh format with shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores with shape (L, B, N, C).
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (T, 4).
            gt_cls (torch.Tensor): Ground truth class indices with shape (T,).
            gt_groups (list[int]): Number of ground truth boxes per batch image.
            shared_if_enabled (bool): Allow one layer's matches to be reused when use_uni_match is set.

        Returns:
            (list): One match-pair list per auxiliary layer, empty when there is only a single decoder layer.
        """
        if pred_bboxes.shape[0] <= 1:
            return []
        if self.use_uni_match and shared_if_enabled:
            shared = self._match(
                pred_bboxes[self.uni_match_ind], pred_scores[self.uni_match_ind], gt_bboxes, gt_cls, gt_groups
            )
            return [shared for _ in range(pred_bboxes.shape[0] - 1)]
        return [self._match(b, s, gt_bboxes, gt_cls, gt_groups) for b, s in zip(pred_bboxes[:-1], pred_scores[:-1])]

    def _prepare_pre_indices(
        self,
        dfine_meta: dict[str, Any] | None,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
    ):
        """Build the Hungarian match pairs for the pre-head predictions of the encoder.

        Args:
            dfine_meta (dict[str, Any], optional): Distribution head outputs holding pre_bboxes and pre_logits.
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (T, 4).
            gt_cls (torch.Tensor): Ground truth class indices with shape (T,).
            gt_groups (list[int]): Number of ground truth boxes per batch image.

        Returns:
            pre_bboxes (torch.Tensor | None): Pre-head boxes in xywh format with shape (B, N, 4).
            pre_logits (torch.Tensor | None): Pre-head class scores with shape (B, N, C).
            pre_indices (list[tuple[torch.Tensor, torch.Tensor]] | None): Match pairs for the pre-head predictions.
        """
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
        cls_indices,
        box_indices,
        cls_norm: float,
        box_norm: float,
        postfix: str,
    ) -> dict[str, torch.Tensor]:
        """Compute the classification and box losses for the pre-head predictions of the encoder.

        Args:
            pre_bboxes (torch.Tensor, optional): Pre-head boxes in xywh format with shape (B, N, 4).
            pre_logits (torch.Tensor, optional): Pre-head class scores with shape (B, N, C).
            gt_cls (torch.Tensor): Ground truth class indices with shape (T,).
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (T, 4).
            cls_indices (list, optional): Match pairs used to build classification targets.
            box_indices (list, optional): Match pairs used to select boxes for regression, falling back to cls_indices.
            cls_norm (int | float): Normalizer for the classification loss.
            box_norm (int | float): Normalizer for the box losses.
            postfix (str): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing the classification, L1, and GIoU loss values, empty when
                the pre-head outputs are absent.
        """
        if pre_bboxes is None or pre_logits is None or cls_indices is None:
            return {}
        return self._compute_layer_losses(
            pre_bboxes, pre_logits, gt_cls, gt_bboxes,
            cls_indices,
            box_indices if box_indices is not None else cls_indices,
            cls_norm, box_norm, postfix=postfix,
        )

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        dfine_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the total loss over the main, auxiliary, pre-head, and denoising predictions.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): Predicted boxes with shape (L, B, N, 4) and class scores with
                shape (L, B, N, C), ordered from the shallowest to the deepest decoder layer.
            batch (dict[str, Any]): Batch dictionary holding cls, bboxes, and gt_groups.
            dn_bboxes (torch.Tensor, optional): Denoising boxes in xywh format with shape (L, B, N_dn, 4).
            dn_scores (torch.Tensor, optional): Denoising class scores with shape (L, B, N_dn, C).
            dn_meta (dict[str, Any], optional): Denoising metadata holding dn_pos_idx and dn_num_group.
            dfine_meta (dict[str, Any], optional): Distribution head outputs used by the FGL and DDF terms.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of individual loss values keyed by name, with a _dn suffix on the
                denoising terms and a _pre suffix on the pre-head terms.
        """
        pred_bboxes, pred_scores = preds
        self.device = pred_scores.device
        self._clear_local_cache()

        if self.training and torch.is_grad_enabled():
            global_num_gts = _global_num_gts(len(batch["bboxes"]), pred_scores.device)
        else:
            global_num_gts = max(len(batch["bboxes"]), 1.0)

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        main_indices = self._match(pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups)
        aux_indices = self._prepare_aux_indices(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)
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
            pred_bboxes[-1], pred_scores[-1], gt_cls, gt_bboxes,
            main_indices, main_box_indices, global_num_gts, norm_boxes,
        )

        if self.aux_loss and pred_bboxes.shape[0] > 1:
            aux_box_indices = box_union_indices if box_union_indices is not None else aux_indices
            total_loss.update(
                self._compute_aux_losses(
                    pred_bboxes[:-1], pred_scores[:-1], gt_cls, gt_bboxes,
                    aux_indices, aux_box_indices, global_num_gts, norm_boxes,
                )
            )

        total_loss.update(
            self._compute_pre_losses(
                pre_bboxes, pre_logits, gt_cls, gt_bboxes,
                pre_indices,
                box_union_indices if box_union_indices is not None else pre_indices,
                global_num_gts, norm_boxes, postfix="_pre",
            )
        )

        total_loss.update(
            self._get_local_bundle(
                pred_bboxes, pred_scores, batch, norm_boxes, dfine_meta,
                main_indices=main_box_indices,
                aux_indices=box_union_indices if box_union_indices is not None else aux_indices,
            )
        )

        if dn_meta is not None and dn_bboxes is not None and dn_scores is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            dn_match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, dn_meta["dn_gt_idx"])
            dn_norm = max(global_num_gts * dn_num_group, 1.0)

            total_loss.update(
                self._compute_layer_losses(
                    dn_bboxes[-1], dn_scores[-1], gt_cls, gt_bboxes,
                    dn_match_indices, dn_match_indices, dn_norm, dn_norm, postfix="_dn",
                )
            )
            if self.aux_loss and dn_bboxes.shape[0] > 1:
                dn_aux_indices = [dn_match_indices for _ in range(dn_bboxes.shape[0] - 1)]
                total_loss.update(
                    self._compute_aux_losses(
                        dn_bboxes[:-1], dn_scores[:-1], gt_cls, gt_bboxes,
                        dn_aux_indices, dn_aux_indices, dn_norm, dn_norm, postfix="_dn",
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
                    dn_bboxes, dn_scores, batch, dn_norm, dn_dfine_meta,
                    main_indices=dn_match_indices, aux_indices=dn_match_indices,
                    postfix="_dn", is_dn=True, include_local_aux=True,
                )
            )

            dn_pre_bboxes, dn_pre_logits, _ = self._prepare_pre_indices(dn_dfine_meta, gt_bboxes, gt_cls, gt_groups)
            total_loss.update(
                self._compute_pre_losses(
                    dn_pre_bboxes, dn_pre_logits, gt_cls, gt_bboxes,
                    dn_match_indices, dn_match_indices, dn_norm, dn_norm, postfix="_dn_pre",
                )
            )

        return total_loss
