# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.utils import bbox2distance
from ultralytics.utils.loss import FocalLoss, MALoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the DETR
    object detection model.

    Attributes:
        nc (int): Number of classes.
        loss_gain (dict[str, float]): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Whether to use FocalLoss.
        use_vfl (bool): Whether to use VarifocalLoss.
        use_uni_match (bool): Whether to use a fixed layer for auxiliary branch label assignment.
        uni_match_ind (int): Index of fixed layer to use if use_uni_match is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss | None): Focal Loss object if use_fl is True, otherwise None.
        vfl (VarifocalLoss | None): Varifocal Loss object if use_vfl is True, otherwise None.
        mal (MALoss | None): Matchability-Aware Loss object if use_mal is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
        use_mal: bool = False,
        matcher: dict[str, Any] | None = None,
    ):
        """Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict[str, float], optional): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
            gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
            alpha (float): The balancing factor used to address class imbalance.
            use_mal (bool): Whether to use MALoss, taking precedence over focal and varifocal.
            matcher (dict[str, Any], optional): Extra HungarianMatcher keyword arguments, merged over the default cost
                gains.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(**{"cost_gain": {"class": 2, "bbox": 5, "giou": 2}, **(matcher or {})})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None
        self.mal = MALoss(gamma, alpha) if use_mal else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        gt_scores: torch.Tensor,
        num_gts: int,
        postfix: str = "",
        norm: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss based on predictions, target values, and ground truth scores.

        Args:
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            targets (torch.Tensor): Target class indices with shape (B, N).
            gt_scores (torch.Tensor): Ground truth confidence scores with shape (B, N).
            num_gts (int): Number of ground truth objects on the local worker.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.
            norm (float, optional): Normalizer for the loss, e.g. the global ground-truth count across distributed
                workers. Defaults to num_gts.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing classification loss value.

        Notes:
            The function supports different classification loss types:
            - Matchability-Aware Loss (if self.mal is not None)
            - Varifocal Loss (if self.vfl is not None and num_gts > 0)
            - Focal Loss (if self.fl is not None)
            - BCE Loss (default fallback)
        """
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot
        norm = max(num_gts if norm is None else norm, 1) / nq

        if self.mal is not None:
            loss_cls = self.mal(pred_scores, gt_scores, one_hot)
            loss_cls /= norm
        elif self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= norm
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = "", norm: float | None = None
    ) -> dict[str, torch.Tensor]:
        """Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4).
            postfix (str, optional): String to append to the loss names for identification in multi-loss scenarios.
            norm (float, optional): Normalizer applied to both loss terms. Defaults to the number of ground truth
                boxes.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing:
                - loss_bbox{postfix}: L1 loss between predicted and ground truth boxes, scaled by the bbox loss gain.
                - loss_giou{postfix}: GIoU loss between predicted and ground truth boxes, scaled by the giou loss gain.

        Notes:
            If no ground truth boxes are provided (empty list), zero-valued tensors are returned for both losses.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            loss[name_bbox] = pred_bboxes[..., :0].sum()
            loss[name_giou] = pred_bboxes[..., :0].sum()
            return loss

        norm = len(gt_bboxes) if norm is None else norm
        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / norm
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / norm
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get auxiliary losses for intermediate decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers.
            pred_scores (torch.Tensor): Predicted scores from auxiliary layers.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            match_indices (list[tuple], optional): Pre-computed matching indices.
            postfix (str, optional): String to append to loss names.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of auxiliary losses.
        """
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(
        match_indices: list[tuple], device: torch.device | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (list[tuple]): List of tuples containing matched indices.
            device (torch.device, optional): Device to move the returned indices to.

        Returns:
            batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        if device is not None:
            batch_idx, src_idx, dst_idx = batch_idx.to(device), src_idx.to(device), dst_idx.to(device)
        return (batch_idx, src_idx), dst_idx

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, dn_gt_idx: list[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get match indices for denoising.

        Args:
            dn_pos_idx (list[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            dn_gt_idx (list[torch.Tensor]): Ground truth index each image's denoising queries reconstruct.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        for i, gt_idx in enumerate(dn_gt_idx):
            gt_idx = gt_idx.to(dn_pos_idx[i].device).repeat(dn_num_group)
            assert len(dn_pos_idx[i]) == len(gt_idx), (
                f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
            )
            dn_match_indices.append((dn_pos_idx[i], gt_idx))
        return dn_match_indices

    def _get_assigned_bboxes(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: list[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            pred_assigned (torch.Tensor): Assigned predicted bounding boxes.
            gt_assigned (torch.Tensor): Assigned ground truth bounding boxes.
        """
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate losses for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            pred_scores (torch.Tensor): Predicted class scores.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
            postfix (str, optional): String to append to loss names.
            match_indices (list[tuple], optional): Pre-computed matching indices.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of losses.
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
            batch (dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
            postfix (str, optional): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict[str, torch.Tensor]): Computed losses, including main and auxiliary (if enabled).

        Notes:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """Real-Time DEtection TRansformer (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to compute detection loss with optional denoising loss.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): Tuple containing predicted bounding boxes and scores.
            batch (dict[str, Any]): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes.
            dn_scores (torch.Tensor, optional): Denoising scores.
            dn_meta (dict[str, Any], optional): Metadata for denoising.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing total loss and denoising loss if applicable.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, dn_meta["dn_gt_idx"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss})

        return total_loss


def _dist_mean_count(count: int, device: torch.device) -> float:
    """Compute the mean of a per-worker count across distributed workers.

    Args:
        count (int): Local count, e.g. ground-truth objects or matched pairs on this worker.
        device (torch.device): Device on which to build the reduction tensor.

    Returns:
        (float): Mean count across all workers, floored at 1.0.
    """
    t = torch.tensor([count], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return max(t.item(), 1.0)


class DEIMLoss(DETRLoss):
    """DEIM detection loss for DEIMDecoder heads, extending DETRLoss with distribution-based local terms.

    Adds the DEIM matching recipe (matchability-aware classification loss and optional union-set box regression over
    main, auxiliary, and pre-head matches) and the fine-grained localization (FGL) and decoupled distillation focal
    (DDF) losses driven by the decoder's distribution-head outputs (``deim_meta``).

    Attributes:
        reg_max (int): Number of discrete bins in the distribution head.
        local_temperature (float): Softmax temperature for the DDF distillation term.
        use_union_set (bool): Regress boxes against the union of main, auxiliary, and pre-head matches.
        kl_loss (nn.KLDivLoss): KL divergence used by the DDF term.
        fgl_gain (float): Weight of the FGL loss term.
        ddf_gain (float): Weight of the DDF loss term.
        fgl_targets (tuple | None): Per-forward cache of the FGL bin targets for the matching queries.
        fgl_targets_dn (tuple | None): Per-forward cache of the FGL bin targets for the denoising queries.
        num_pos (torch.Tensor | None): Positive-location count cached by the DDF term.
        num_neg (torch.Tensor | None): Negative-location count cached by the DDF term.
    """

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
            reg_max (int): Number of discrete bins in the distribution head.
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
            matcher (dict[str, Any], optional): Extra HungarianMatcher keyword arguments.
        """
        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "fgl": 0.15, "ddf": 1.5}
        super().__init__(
            nc, loss_gain, aux_loss, use_fl, use_vfl, use_uni_match, uni_match_ind, gamma, alpha, use_mal, matcher
        )
        self.reg_max = reg_max
        self.local_temperature = local_temperature
        self.use_union_set = use_union_set
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.fgl_gain = self.loss_gain.get("fgl", 0.0)
        self.ddf_gain = self.loss_gain.get("ddf", 0.0)
        self._clear_local_cache()

    def _clear_local_cache(self) -> None:
        """Clear per-forward local-loss caches."""
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None

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
            **self._get_loss_class(pred_scores, targets, gt_scores, int(cls_gt_idx.numel()), postfix, norm=cls_norm),
            **self._get_loss_bbox(pred_assigned_box, gt_assigned_box, postfix, norm=box_norm),
        }

    def _compute_aux_losses(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        cls_indices_list: list | tuple,
        box_indices_list: list | tuple,
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
            cls_indices_list (list | tuple): Match pairs for classification, either one list per layer or one shared
                list.
            box_indices_list (list | tuple): Match pairs for box regression, either one list per layer or one shared
                list.
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
        loss = F.cross_entropy(pred_f, dis_left, reduction="none").to(pred.dtype) * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred_f, dis_right, reduction="none").to(pred.dtype) * weight_right.reshape(-1)
        if weight is not None:
            loss *= weight.float()
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
        loss_match_local = (
            weight_targets_local
            * (self.local_temperature**2)
            * (
                self.kl_loss(
                    F.log_softmax(pred_all_f / self.local_temperature, dim=1),
                    F.softmax(teacher_all_f.detach() / self.local_temperature, dim=1),
                )
                .sum(-1)
                .to(pred_all.dtype)
            )
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
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

        idx, gt_idx = self._get_index(match_indices, pred_bboxes.device)
        if gt_idx.numel() == 0:
            zero = torch.tensor(0.0, device=pred_bboxes.device)
            return {name_fgl: zero, name_ddf: zero}

        target_boxes = gt_bboxes[gt_idx]
        cache_name = "fgl_targets_dn" if is_dn else "fgl_targets"
        target_cache = getattr(self, cache_name)
        if target_cache is None:
            target_boxes_xyxy = xywh2xyxy(target_boxes)
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

        return {name_fgl: loss_fgl * self.fgl_gain, name_ddf: loss_ddf * self.ddf_gain}

    def _get_local_bundle(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        norm_boxes: float,
        deim_meta: dict[str, Any] | None,
        main_indices: list[tuple[torch.Tensor, torch.Tensor]],
        aux_indices: list | tuple,
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
            deim_meta (dict[str, Any], optional): Distribution head outputs; an empty dict is returned when absent.
            main_indices (list[tuple[torch.Tensor, torch.Tensor]]): Match pairs for the final decoder layer.
            aux_indices (list | tuple): Match pairs for the auxiliary layers, either one list per layer or one shared
                list.
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
        if deim_meta is None:
            return {}
        pred_corners_all = deim_meta.get("pred_corners")
        ref_points_all = deim_meta.get("ref_points")
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
            deim_meta.get("up"),
            deim_meta.get("reg_scale"),
            match_indices=main_indices,
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
                    deim_meta.get("up"),
                    deim_meta.get("reg_scale"),
                    match_indices=layer_indices,
                    teacher_corners=teacher_corners,
                    teacher_logits=teacher_logits.detach(),
                    postfix=postfix,
                    is_dn=is_dn,
                )
                loss_fgl_aux += layer_loss[f"loss_fgl{postfix}"]
                loss_ddf_aux += layer_loss[f"loss_ddf{postfix}"]

            losses[f"loss_fgl_aux{postfix}"] = loss_fgl_aux
            losses[f"loss_ddf{postfix}"] = (
                losses.get(f"loss_ddf{postfix}", torch.tensor(0.0, device=pred_bboxes.device)) + loss_ddf_aux
            )
        return losses

    def _prepare_aux_indices(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        shared_if_enabled: bool = True,
    ) -> list:
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
            shared = self.matcher(
                pred_bboxes[self.uni_match_ind], pred_scores[self.uni_match_ind], gt_bboxes, gt_cls, gt_groups
            )
            return [shared for _ in range(pred_bboxes.shape[0] - 1)]
        return [self.matcher(b, s, gt_bboxes, gt_cls, gt_groups) for b, s in zip(pred_bboxes[:-1], pred_scores[:-1])]

    def _prepare_pre_indices(
        self,
        deim_meta: dict[str, Any] | None,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list | None]:
        """Build the Hungarian match pairs for the pre-head predictions of the encoder.

        Args:
            deim_meta (dict[str, Any], optional): Distribution head outputs holding pre_bboxes and pre_logits.
            gt_bboxes (torch.Tensor): Ground truth boxes in xywh format with shape (T, 4).
            gt_cls (torch.Tensor): Ground truth class indices with shape (T,).
            gt_groups (list[int]): Number of ground truth boxes per batch image.

        Returns:
            pre_bboxes (torch.Tensor | None): Pre-head boxes in xywh format with shape (B, N, 4).
            pre_logits (torch.Tensor | None): Pre-head class scores with shape (B, N, C).
            pre_indices (list[tuple[torch.Tensor, torch.Tensor]] | None): Match pairs for the pre-head predictions.
        """
        if deim_meta is None:
            return None, None, None
        pre_bboxes = deim_meta.get("pre_bboxes")
        pre_logits = deim_meta.get("pre_logits")
        if pre_bboxes is None or pre_logits is None:
            return None, None, None
        pre_bboxes = pre_bboxes.contiguous()
        pre_logits = pre_logits.contiguous()
        pre_indices = self.matcher(pre_bboxes, pre_logits, gt_bboxes, gt_cls, gt_groups)
        return pre_bboxes, pre_logits, pre_indices

    def _compute_pre_losses(
        self,
        pre_bboxes: torch.Tensor | None,
        pre_logits: torch.Tensor | None,
        gt_cls: torch.Tensor,
        gt_bboxes: torch.Tensor,
        cls_indices: list | None,
        box_indices: list | None,
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

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        deim_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the total loss over the main, auxiliary, pre-head, and denoising predictions.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): Predicted boxes with shape (L, B, N, 4) and class scores with
                shape (L, B, N, C), ordered from the shallowest to the deepest decoder layer.
            batch (dict[str, Any]): Batch dictionary holding cls, bboxes, and gt_groups.
            dn_bboxes (torch.Tensor, optional): Denoising boxes in xywh format with shape (L, B, N_dn, 4).
            dn_scores (torch.Tensor, optional): Denoising class scores with shape (L, B, N_dn, C).
            dn_meta (dict[str, Any], optional): Denoising metadata holding dn_pos_idx and dn_num_group.
            deim_meta (dict[str, Any], optional): Distribution head outputs used by the FGL and DDF terms.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of individual loss values keyed by name, with a _dn suffix on the
                denoising terms and a _pre suffix on the pre-head terms.
        """
        pred_bboxes, pred_scores = preds
        self.device = pred_scores.device
        self._clear_local_cache()

        if self.training and torch.is_grad_enabled():
            global_num_gts = _dist_mean_count(len(batch["bboxes"]), pred_scores.device)
        else:
            global_num_gts = max(len(batch["bboxes"]), 1.0)

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        main_indices = self.matcher(pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups)
        aux_indices = self._prepare_aux_indices(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)
        pre_bboxes, pre_logits, pre_indices = self._prepare_pre_indices(deim_meta, gt_bboxes, gt_cls, gt_groups)

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
                norm_boxes = _dist_mean_count(sum(len(src) for src, _ in box_union_indices), pred_scores.device)
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
                deim_meta,
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

            dn_deim_meta = None
            if deim_meta is not None:
                dn_deim_meta = {
                    "pred_corners": deim_meta.get("dn_pred_corners"),
                    "ref_points": deim_meta.get("dn_ref_points"),
                    "pre_bboxes": deim_meta.get("dn_pre_bboxes"),
                    "pre_logits": deim_meta.get("dn_pre_logits"),
                    "up": deim_meta.get("up"),
                    "reg_scale": deim_meta.get("reg_scale"),
                }
                if dn_deim_meta["pred_corners"] is None or dn_deim_meta["ref_points"] is None:
                    dn_deim_meta = None

            total_loss.update(
                self._get_local_bundle(
                    dn_bboxes,
                    dn_scores,
                    batch,
                    dn_norm,
                    dn_deim_meta,
                    main_indices=dn_match_indices,
                    aux_indices=dn_match_indices,
                    postfix="_dn",
                    is_dn=True,
                    include_local_aux=True,
                )
            )

            dn_pre_bboxes, dn_pre_logits, _ = self._prepare_pre_indices(dn_deim_meta, gt_bboxes, gt_cls, gt_groups)
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

        return total_loss
