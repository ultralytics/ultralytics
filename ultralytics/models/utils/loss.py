# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


def _global_num_gts(num_gts: int, device: torch.device) -> float:
    """Compute the global average number of ground truths across distributed workers.

    In distributed training, this function sums local ground-truth counts across all processes and
    returns the average per process. It also enforces a minimum of 1.0 to avoid zero-division issues.

    Args:
        num_gts (int): Number of ground-truth objects on the current process.
        device (torch.device): Device to place the temporary tensor on.

    Returns:
        (float): Global average number of ground truths per process, clamped to at least 1.0.
    """
    t = torch.tensor([num_gts], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return max(t.item(), 1.0)


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
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = True,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
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
            matcher (dict[str, Any]): Configuration for HungarianMatcher.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        if matcher is None:
            matcher = {}
        self.matcher = HungarianMatcher(**matcher)
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

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
            local_num_gts (int): Number of ground truth objects on the local rank.
            global_num_gts (float): Global mean GT count across ranks for loss normalization.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing classification loss value.

        Notes:
            The function supports different classification loss types:
            - Varifocal Loss (if self.vfl is True and local_num_gts > 0)
            - Focal Loss (if self.fl is True)
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

        if self.fl:
            if local_num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(global_num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, global_num_gts: float, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4).
            global_num_gts (float): Global mean GT count across ranks for loss normalization.
            postfix (str, optional): String to append to the loss names for identification in multi-loss scenarios.

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
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / global_num_gts
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / global_num_gts
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
        global_num_gts: float,
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
            global_num_gts (float): Global mean GT count across ranks for loss normalization.
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
                global_num_gts,
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
    def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

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
        global_num_gts: float,
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
            global_num_gts (float): Global mean GT count across ranks for loss normalization.
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
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), global_num_gts, postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, global_num_gts, postfix),
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        global_num_gts: float | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
            batch (dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
            postfix (str, optional): Postfix for loss names.
            global_num_gts (float, optional): Global GT count (mean across ranks) for loss normalization.
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
        if global_num_gts is None:
            if self.training and torch.is_grad_enabled():
                global_num_gts = _global_num_gts(len(gt_bboxes), pred_scores.device)
            else:
                global_num_gts = max(len(gt_bboxes), 1)

        total_loss = self._get_loss(
            pred_bboxes[-1],
            pred_scores[-1],
            gt_bboxes,
            gt_cls,
            gt_groups,
            global_num_gts,
            postfix=postfix,
            match_indices=match_indices,
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1],
                    pred_scores[:-1],
                    gt_bboxes,
                    gt_cls,
                    gt_groups,
                    global_num_gts,
                    match_indices=match_indices,
                    postfix=postfix,
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

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
        match_indices: list[tuple] | None = None,
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
        if self.training and torch.is_grad_enabled():
            global_num_gts = _global_num_gts(len(batch["bboxes"]), pred_scores.device)
        else:
            global_num_gts = max(len(batch["bboxes"]), 1.0)

        total_loss = super().forward(
            pred_bboxes, pred_scores, batch, global_num_gts=global_num_gts, match_indices=match_indices
        )

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_global_num_gts = max(global_num_gts * dn_num_group, 1.0)
            dn_loss = super().forward(
                dn_bboxes,
                dn_scores,
                batch,
                postfix="_dn",
                match_indices=match_indices,
                global_num_gts=dn_global_num_gts,
            )
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, gt_groups: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get match indices for denoising.

        Args:
            dn_pos_idx (list[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (list[int]): List of integers representing number of ground truths per image.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), (
                    f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                )
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices


class RTDETRv4DetectionLoss(RTDETRDetectionLoss):
    """RT-DETRv4 detection loss with optional DFine-specific components."""

    supports_dfine = True

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 32,
        mal_alpha: float | None = None,
        mal_gamma: float | None = None,
        local_temperature: float = 5.0,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = True,
        use_uni_set: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
        matcher: dict[str, Any] | None = None,
    ):
        """Initialize RTDETRDetectionLoss with optional DFine loss components."""
        super().__init__(
            nc=nc,
            loss_gain=loss_gain,
            aux_loss=aux_loss,
            use_fl=use_fl,
            use_vfl=use_vfl,
            use_uni_match=use_uni_match,
            uni_match_ind=uni_match_ind,
            gamma=gamma,
            alpha=alpha,
            matcher=matcher,
        )
        self.reg_max = reg_max
        self.mal_alpha = mal_alpha
        self.mal_gamma = mal_gamma if mal_gamma is not None else gamma
        self.local_temperature = local_temperature
        self.use_uni_set = use_uni_set

        loss_gain = loss_gain or {}
        self.mal_gain = loss_gain["mal"] if "mal" in loss_gain else 0.0
        self.fgl_gain = loss_gain["fgl"] if "fgl" in loss_gain else 0.0
        self.ddf_gain = loss_gain["ddf"] if "ddf" in loss_gain else 0.0

    @staticmethod
    def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
        """Convert box format from cxcywh to xyxy."""
        x_c, y_c, w, h = x.unbind(-1)
        b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
        return torch.stack(b, dim=-1)

    @staticmethod
    def _weighting_function(reg_max: int, up: torch.Tensor, reg_scale: torch.Tensor) -> torch.Tensor:
        """Generate the non-uniform weighting sequence used by DFine."""
        upper_bound1 = torch.abs(up[0]) * torch.abs(reg_scale)
        upper_bound2 = upper_bound1 * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.cat(values, 0)

    def _bbox2distance(
        self, points: torch.Tensor, boxes_xyxy: torch.Tensor, up: torch.Tensor, reg_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert target boxes into DFine distance targets and interpolation weights."""
        reg_scale = torch.abs(reg_scale)
        px, py, pw, ph = points.unbind(-1)
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)

        left = (px - x1) * reg_scale / pw - 0.5 * reg_scale
        top = (py - y1) * reg_scale / ph - 0.5 * reg_scale
        right = (x2 - px) * reg_scale / pw - 0.5 * reg_scale
        bottom = (y2 - py) * reg_scale / ph - 0.5 * reg_scale
        dist = torch.stack([left, top, right, bottom], dim=-1)

        project = self._weighting_function(self.reg_max, up, reg_scale).to(dist.device).to(dist.dtype)
        dist_flat = dist.reshape(-1)
        idx_right = torch.bucketize(dist_flat, project).clamp(min=1, max=self.reg_max)
        idx_left = idx_right - 1
        proj_left = project[idx_left]
        proj_right = project[idx_right]
        denom = proj_right - proj_left
        weight_right = torch.where(denom > 0, (dist_flat - proj_left) / denom, torch.zeros_like(dist_flat))
        weight_right = weight_right.clamp(0.0, 1.0)
        weight_left = 1.0 - weight_right

        target_left = idx_left.reshape(dist.shape)
        weight_left = weight_left.reshape(dist.shape)
        weight_right = weight_right.reshape(dist.shape)
        return target_left, weight_right, weight_left

    @staticmethod
    def _unimodal_distribution_focal_loss(
        pred: torch.Tensor,
        label: torch.Tensor,
        weight_right: torch.Tensor,
        weight_left: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Compute distribution focal loss with unimodal targets."""
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1)
        loss = loss + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
        if weight is not None:
            loss = loss * weight.float()
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        else:
            loss = loss.sum()
        return loss

    @staticmethod
    def _get_union_indices(
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        indices_aux_list: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build a union set of match indices across decoder layers."""
        merged = indices
        for indices_aux in indices_aux_list:
            merged = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(merged.copy(), indices_aux.copy())
            ]

        results = []
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in merged]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _loss_class_from_indices(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        match_indices: list[tuple[torch.Tensor, torch.Tensor]],
        global_num_gts: float,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        idx, gt_idx = self._get_index(match_indices)
        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        if gt_idx.numel():
            targets[idx] = gt_cls[gt_idx]
        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if gt_idx.numel():
            gt_scores[idx] = bbox_iou(pred_bboxes[idx].detach(), gt_bboxes[gt_idx], xywh=True).squeeze(-1)
        return self._get_loss_class(pred_scores, targets, gt_scores, gt_idx.numel(), global_num_gts, postfix)

    def _loss_bbox_from_indices(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        match_indices: list[tuple[torch.Tensor, torch.Tensor]],
        global_num_gts: float,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        idx, gt_idx = self._get_index(match_indices)
        pred_sel = pred_bboxes[idx]
        gt_sel = gt_bboxes[gt_idx]
        return self._get_loss_bbox(pred_sel, gt_sel, global_num_gts, postfix)

    def _loss_mal(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        global_num_gts: float,
        match_indices: list[tuple] | None = None,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        name = f"loss_mal{postfix}"
        if self.mal_gain <= 0:
            return {name: torch.tensor(0.0, device=pred_scores.device)}
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)
        idx, gt_idx = self._get_index(match_indices)
        if gt_idx.numel() == 0:
            return {name: torch.tensor(0.0, device=pred_scores.device)}

        ious = bbox_iou(pred_bboxes[idx].detach(), gt_bboxes[gt_idx], xywh=True).squeeze(-1)
        bs, nq = pred_scores.shape[:2]
        target_classes = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        target_classes[idx] = gt_cls[gt_idx]
        target = F.one_hot(target_classes, num_classes=self.nc + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=pred_scores.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = pred_scores.sigmoid().detach()
        target_score = target_score.pow(self.mal_gamma)
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.mal_gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.mal_gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(pred_scores, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * pred_scores.shape[1] / max(global_num_gts, 1.0)
        return {name: loss * self.mal_gain}

    def _get_mal_losses(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        global_num_gts: float,
        match_indices: list[tuple] | None = None,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Compute MAL loss independent of DFine/local losses."""
        if self.mal_gain <= 0:
            return {}

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        return self._loss_mal(
            pred_bboxes,
            pred_scores,
            gt_bboxes,
            gt_cls,
            gt_groups,
            global_num_gts,
            match_indices=match_indices,
            postfix=postfix,
        )

    def _get_mal_aux_losses(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        global_num_gts: float,
        match_indices: list[tuple] | None = None,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Compute aggregated MAL loss over auxiliary decoder layers."""
        if self.mal_gain <= 0:
            return {}

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        if match_indices is None and self.use_uni_match and pred_bboxes.numel():
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind], pred_scores[self.uni_match_ind], gt_bboxes, gt_cls, gt_groups
            )

        loss_mal = torch.tensor(0.0, device=pred_bboxes.device)
        for aux_bboxes, aux_scores in zip(pred_bboxes, pred_scores):
            losses = self._loss_mal(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                global_num_gts,
                match_indices=match_indices,
                postfix=postfix,
            )
            loss_mal = loss_mal + losses[f"loss_mal{postfix}"]

        return {f"loss_mal_aux{postfix}": loss_mal}

    def _loss_local(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_corners: torch.Tensor | None,
        ref_points: torch.Tensor | None,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        global_num_gts: float,
        up: torch.Tensor | None,
        reg_scale: torch.Tensor | None,
        match_indices: list[tuple] | None = None,
        teacher_corners: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        postfix: str = "",
        is_dn: bool = False,
    ) -> dict[str, torch.Tensor]:
        name_fgl = f"loss_fgl{postfix}"
        name_ddf = f"loss_ddf{postfix}"
        if self.fgl_gain <= 0 and self.ddf_gain <= 0:
            return {
                name_fgl: torch.tensor(0.0, device=pred_bboxes.device),
                name_ddf: torch.tensor(0.0, device=pred_bboxes.device),
            }
        if pred_corners is None or ref_points is None or up is None or reg_scale is None:
            return {
                name_fgl: torch.tensor(0.0, device=pred_bboxes.device),
                name_ddf: torch.tensor(0.0, device=pred_bboxes.device),
            }
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)
        idx, gt_idx = self._get_index(match_indices)
        if gt_idx.numel() == 0:
            return {
                name_fgl: torch.tensor(0.0, device=pred_bboxes.device),
                name_ddf: torch.tensor(0.0, device=pred_bboxes.device),
            }

        target_boxes = gt_bboxes[gt_idx]
        target_boxes_xyxy = self._box_cxcywh_to_xyxy(target_boxes)
        target_corners, weight_right, weight_left = self._bbox2distance(
            ref_points[idx].detach(), target_boxes_xyxy, up, reg_scale
        )

        pred_corners_sel = pred_corners[idx].reshape(-1, self.reg_max + 1)
        target_corners = target_corners.reshape(-1)
        weight_right = weight_right.reshape(-1)
        weight_left = weight_left.reshape(-1)

        ious = bbox_iou(pred_bboxes[idx], target_boxes, xywh=True).squeeze(-1)
        weight_targets = ious.unsqueeze(-1).repeat(1, 4).reshape(-1).detach()
        loss_fgl = self._unimodal_distribution_focal_loss(
            pred_corners_sel,
            target_corners,
            weight_right,
            weight_left,
            weight=weight_targets,
            avg_factor=max(global_num_gts, 1.0),
        )

        loss_ddf = torch.tensor(0.0, device=pred_bboxes.device)
        if self.ddf_gain > 0 and teacher_corners is not None and teacher_logits is not None:
            pred_all = pred_corners.reshape(-1, self.reg_max + 1)
            teacher_all = teacher_corners.reshape(-1, self.reg_max + 1)
            if not torch.equal(pred_all, teacher_all):
                weight_targets_local = teacher_logits.sigmoid().max(dim=-1)[0]
                mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                weight_targets_local[idx] = ious.to(weight_targets_local.dtype)
                weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                loss_match_local = weight_targets_local * (self.local_temperature ** 2) * (
                    nn.KLDivLoss(reduction="none")(
                        F.log_softmax(pred_all / self.local_temperature, dim=1),
                        F.softmax(teacher_all.detach() / self.local_temperature, dim=1),
                    ).sum(-1)
                )
                if not is_dn:
                    batch_scale = 8 / pred_bboxes.shape[0]
                    self.num_pos = (mask.sum() * batch_scale) ** 0.5
                    self.num_neg = ((~mask).sum() * batch_scale) ** 0.5
                loss_pos = loss_match_local[mask].mean() if mask.any() else 0.0
                loss_neg = loss_match_local[~mask].mean() if (~mask).any() else 0.0
                denom = max(self.num_pos + self.num_neg, 1.0)
                loss_ddf = (loss_pos * self.num_pos + loss_neg * self.num_neg) / denom

        return {
            name_fgl: loss_fgl * self.fgl_gain,
            name_ddf: loss_ddf * self.ddf_gain,
        }

    def _get_local_losses(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        dfine_meta: dict[str, Any] | None,
        global_num_gts: float,
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        is_dn: bool = False,
    ) -> dict[str, torch.Tensor]:
        if dfine_meta is None:
            return {}

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

        pred_corners = dfine_meta["pred_corners"]
        if isinstance(pred_corners, torch.Tensor) and pred_corners.dim() == 4:
            pred_corners = pred_corners[-1]
        ref_points = dfine_meta["ref_points"]
        if isinstance(ref_points, torch.Tensor) and ref_points.dim() == 4:
            ref_points = ref_points[-1]

        losses = {}
        if self.fgl_gain > 0 or self.ddf_gain > 0:
            losses.update(
                self._loss_local(
                    pred_bboxes,
                    pred_scores,
                    pred_corners,
                    ref_points,
                    gt_bboxes,
                    gt_cls,
                    gt_groups,
                    global_num_gts,
                    dfine_meta["up"],
                    dfine_meta["reg_scale"],
                    match_indices=match_indices,
                    teacher_corners=None,
                    teacher_logits=None,
                    postfix=postfix,
                    is_dn=is_dn,
                )
            )
        return losses

    def _get_local_aux_losses(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        dfine_meta: dict[str, Any] | None,
        batch: dict[str, Any],
        global_num_gts: float,
        match_indices: list[tuple] | None = None,
        teacher_logits: torch.Tensor | None = None,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        if dfine_meta is None:
            return {}

        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        if match_indices is None and self.use_uni_match and pred_bboxes.numel():
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind], pred_scores[self.uni_match_ind], gt_bboxes, gt_cls, gt_groups
            )

        pred_corners_all = dfine_meta["pred_corners"]
        ref_points_all = dfine_meta["ref_points"]
        if pred_corners_all is None or ref_points_all is None:
            result = {}
            if self.fgl_gain > 0:
                result[f"loss_fgl_aux{postfix}"] = torch.tensor(0.0, device=pred_bboxes.device)
            if self.ddf_gain > 0:
                result[f"loss_ddf_aux{postfix}"] = torch.tensor(0.0, device=pred_bboxes.device)
            return result

        loss_fgl = torch.tensor(0.0, device=pred_bboxes.device)
        loss_ddf = torch.tensor(0.0, device=pred_bboxes.device)

        if isinstance(pred_corners_all, torch.Tensor) and pred_corners_all.dim() == 4:
            teacher_corners = pred_corners_all[-1]
        else:
            teacher_corners = pred_corners_all

        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            if isinstance(pred_corners_all, torch.Tensor) and pred_corners_all.dim() == 4:
                layer_pred_corners = pred_corners_all[i]
            else:
                layer_pred_corners = pred_corners_all
            if isinstance(ref_points_all, torch.Tensor) and ref_points_all.dim() == 4:
                layer_ref_points = ref_points_all[i]
            else:
                layer_ref_points = ref_points_all
            losses = self._loss_local(
                aux_bboxes,
                aux_scores,
                layer_pred_corners,
                layer_ref_points,
                gt_bboxes,
                gt_cls,
                gt_groups,
                global_num_gts,
                dfine_meta["up"],
                dfine_meta["reg_scale"],
                match_indices=match_indices,
                teacher_corners=teacher_corners,
                teacher_logits=teacher_logits,
                postfix=postfix,
            )
            loss_fgl = loss_fgl + losses[f"loss_fgl{postfix}"]
            loss_ddf = loss_ddf + losses[f"loss_ddf{postfix}"]

        result = {}
        if self.fgl_gain > 0:
            result[f"loss_fgl_aux{postfix}"] = loss_fgl
        if self.ddf_gain > 0:
            result[f"loss_ddf_aux{postfix}"] = loss_ddf
        return result

    def _get_mal_local_bundle(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        global_num_gts: float,
        dfine_meta: dict[str, Any] | None = None,
        match_indices: list[tuple] | None = None,
        local_match_indices: list[tuple] | None = None,
        postfix: str = "",
        is_dn: bool = False,
        include_mal: bool = True,
        include_local: bool = True,
        include_mal_aux: bool = True,
        include_local_aux: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute MAL and local (FGL/DDF) losses for main and aux layers."""
        losses = {}
        has_aux = self.aux_loss and pred_bboxes.shape[0] > 1
        if local_match_indices is None:
            local_match_indices = match_indices

        if include_mal and self.mal_gain > 0:
            losses.update(
                self._get_mal_losses(
                    pred_bboxes[-1],
                    pred_scores[-1],
                    batch,
                    global_num_gts,
                    match_indices=match_indices,
                    postfix=postfix,
                )
            )
            if include_mal_aux and has_aux:
                losses.update(
                    self._get_mal_aux_losses(
                        pred_bboxes[:-1],
                        pred_scores[:-1],
                        batch,
                        global_num_gts,
                        match_indices=match_indices,
                        postfix=postfix,
                    )
                )

        if include_local and dfine_meta is not None and (self.fgl_gain > 0 or self.ddf_gain > 0):
            pred_corners_all = dfine_meta["pred_corners"]
            if pred_bboxes.shape[0] == pred_corners_all.shape[0] + 1:
                pred_bboxes = pred_bboxes[1:]
                pred_scores = pred_scores[1:]
            elif pred_bboxes.shape[0] != pred_corners_all.shape[0]:
                raise ValueError(
                    f"Mismatch: pred_bboxes has {pred_bboxes.shape[0]} layers, "
                    f"pred_corners has {pred_corners_all.shape[0]} layers."
                )
            has_local_aux = self.aux_loss and pred_bboxes.shape[0] > 1
            # For main layer: no teacher (no self-distillation)
            losses.update(
                self._get_local_losses(
                    pred_bboxes[-1],
                    pred_scores[-1],
                    batch,
                    dfine_meta,
                    global_num_gts,
                    match_indices=local_match_indices,
                    postfix=postfix,
                    is_dn=is_dn,
                )
            )
            if include_local_aux and has_local_aux:
                # For auxiliary layers: use last layer as teacher for DDF loss
                teacher_logits = pred_scores[-1]
                aux_losses = self._get_local_aux_losses(
                    pred_bboxes[:-1],
                    pred_scores[:-1],
                    dfine_meta,
                    batch,
                    global_num_gts,
                    match_indices=local_match_indices,
                    teacher_logits=teacher_logits,
                    postfix=postfix,
                )
                ddf_aux_key = f"loss_ddf_aux{postfix}"
                if ddf_aux_key in aux_losses:
                    main_ddf_key = f"loss_ddf{postfix}"
                    losses[main_ddf_key] = (
                        losses[main_ddf_key]
                        if main_ddf_key in losses
                        else torch.tensor(0.0, device=pred_bboxes.device)
                    )
                    losses[main_ddf_key] = losses[main_ddf_key] + aux_losses.pop(ddf_aux_key)
                losses.update(aux_losses)

        return losses

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        dfine_meta: dict[str, Any] | None = None,
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
        if self.training and torch.is_grad_enabled():
            global_num_gts = _global_num_gts(len(batch["bboxes"]), pred_scores.device)
        else:
            global_num_gts = max(len(batch["bboxes"]), 1.0)

        if not self.use_uni_set:
            gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
            fixed_indices = None
            if self.use_uni_match and pred_bboxes.numel():
                fixed_indices = self.matcher(
                    pred_bboxes[self.uni_match_ind],
                    pred_scores[self.uni_match_ind],
                    gt_bboxes,
                    gt_cls,
                    gt_groups,
                )
            total_loss = super().forward(
                preds,
                batch,
                dn_bboxes=dn_bboxes,
                dn_scores=dn_scores,
                dn_meta=dn_meta,
                match_indices=fixed_indices,
            )
            total_loss.update(
                self._get_mal_local_bundle(
                    pred_bboxes,
                    pred_scores,
                    batch,
                    global_num_gts,
                    dfine_meta=dfine_meta,
                    match_indices=fixed_indices,
                    local_match_indices=fixed_indices,
                )
            )
        else:
            gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
            indices_main = self.matcher(pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups)
            indices_aux_list = []
            if self.aux_loss and pred_bboxes.shape[0] > 1:
                for aux_bboxes, aux_scores in zip(pred_bboxes[:-1], pred_scores[:-1]):
                    indices_aux = self.matcher(aux_bboxes, aux_scores, gt_bboxes, gt_cls, gt_groups)
                    indices_aux_list.append(indices_aux)

            indices_go = self._get_union_indices(indices_main, indices_aux_list)
            global_num_gts_go = _global_num_gts(sum(len(x[0]) for x in indices_go), pred_scores.device)

            fixed_indices = None
            if self.use_uni_match and pred_bboxes.numel():
                fixed_indices = self.matcher(
                    pred_bboxes[self.uni_match_ind],
                    pred_scores[self.uni_match_ind],
                    gt_bboxes,
                    gt_cls,
                    gt_groups,
                )
                class_indices_main = fixed_indices
                class_indices_aux = [fixed_indices for _ in range(pred_bboxes.shape[0] - 1)]
            else:
                class_indices_main = indices_main
                class_indices_aux = indices_aux_list

            total_loss = {}
            total_loss.update(
                self._loss_class_from_indices(
                    pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, class_indices_main, global_num_gts
                )
            )
            total_loss.update(self._loss_bbox_from_indices(pred_bboxes[-1], gt_bboxes, indices_go, global_num_gts_go))

            if self.aux_loss and pred_bboxes.shape[0] > 1:
                loss_class_aux = torch.tensor(0.0, device=pred_bboxes.device)
                loss_bbox_aux = torch.tensor(0.0, device=pred_bboxes.device)
                loss_giou_aux = torch.tensor(0.0, device=pred_bboxes.device)
                for aux_bboxes, aux_scores, indices_aux in zip(
                    pred_bboxes[:-1], pred_scores[:-1], class_indices_aux
                ):
                    loss_class = self._loss_class_from_indices(
                        aux_bboxes, aux_scores, gt_bboxes, gt_cls, indices_aux, global_num_gts
                    )
                    loss_bbox = self._loss_bbox_from_indices(aux_bboxes, gt_bboxes, indices_go, global_num_gts_go)
                    loss_class_aux = loss_class_aux + loss_class["loss_class"]
                    loss_bbox_aux = loss_bbox_aux + loss_bbox["loss_bbox"]
                    loss_giou_aux = loss_giou_aux + loss_bbox["loss_giou"]
                total_loss.update(
                    {
                        "loss_class_aux": loss_class_aux,
                        "loss_bbox_aux": loss_bbox_aux,
                        "loss_giou_aux": loss_giou_aux,
                    }
                )

            total_loss.update(
                self._get_mal_local_bundle(
                    pred_bboxes,
                    pred_scores,
                    batch,
                    global_num_gts,
                    dfine_meta=dfine_meta,
                    match_indices=fixed_indices,
                    local_match_indices=indices_go,
                )
            )

        dn_match_indices = None
        dn_global_num_gts = None
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            dn_match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])
            dn_global_num_gts = max(global_num_gts * dn_num_group, 1.0)

        if dn_meta is not None:
            if self.use_uni_set:
                dn_loss = super(RTDETRDetectionLoss, self).forward(
                    dn_bboxes,
                    dn_scores,
                    batch,
                    postfix="_dn",
                    match_indices=dn_match_indices,
                    global_num_gts=dn_global_num_gts,
                )
                total_loss.update(dn_loss)
            dn_dfine_meta = None
            if dfine_meta is not None:
                dn_dfine_meta = {
                    "pred_corners": dfine_meta["dn_pred_corners"],
                    "ref_points": dfine_meta["dn_ref_points"],
                    "pre_bboxes": dfine_meta["dn_pre_bboxes"],
                    "pre_logits": dfine_meta["dn_pre_logits"],
                    "up": dfine_meta["up"],
                    "reg_scale": dfine_meta["reg_scale"],
                }
                if dn_dfine_meta["pred_corners"] is None or dn_dfine_meta["ref_points"] is None:
                    dn_dfine_meta = None

            total_loss.update(
                self._get_mal_local_bundle(
                    dn_bboxes,
                    dn_scores,
                    batch,
                    dn_global_num_gts,
                    dfine_meta=dn_dfine_meta,
                    match_indices=dn_match_indices,
                    local_match_indices=dn_match_indices,
                    postfix="_dn",
                    is_dn=True,
                    include_local_aux=False,
                )
            )
        elif self.use_uni_set:
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=pred_scores.device) for k in total_loss.keys()})

        return total_loss
