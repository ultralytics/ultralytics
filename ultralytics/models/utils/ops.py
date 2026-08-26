# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh

from .box_ops import box_cxcywh_to_xyxy, pairwise_box_iou, pairwise_giou


def _curriculum_ratio(training_progress: float, power: float) -> float:
    """Return a clamped power-schedule ratio from normalized training progress."""
    progress = min(max(float(training_progress), 0.0), 1.0)
    return progress**power


class HungarianMatcher(nn.Module):
    """A module implementing the HungarianMatcher for optimal assignment between predictions and ground truth.

    HungarianMatcher performs optimal bipartite assignment over predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally mask predictions. This is
    used in end-to-end object detection models like DETR.

    Attributes:
        cost_gain (dict[str, float]): Dictionary of cost coefficients for 'class', 'bbox', 'giou', 'mask', and 'dice'
            components.
        use_fl (bool): Whether to use Focal Loss for classification cost calculation.
        with_mask (bool): Whether the model makes mask predictions.
        num_sample_points (int): Number of sample points used in mask cost calculation.
        alpha (float): Alpha factor in Focal Loss calculation.
        gamma (float): Gamma factor in Focal Loss calculation.

    Methods:
        forward: Compute optimal assignment between predictions and ground truths for a batch.
        _cost_mask: Compute mask cost and dice cost if masks are predicted.

    Examples:
        Initialize a HungarianMatcher with custom cost gains
        >>> matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})

        Perform matching between predictions and ground truth
        >>> pred_boxes = torch.rand(2, 100, 4)  # batch_size=2, num_queries=100
        >>> pred_scores = torch.rand(2, 100, 80)  # 80 classes
        >>> gt_boxes = torch.rand(10, 4)  # 10 ground truth boxes
        >>> gt_classes = torch.randint(0, 80, (10,))
        >>> gt_groups = [5, 5]  # 5 GT boxes per image
        >>> indices = matcher(pred_boxes, pred_scores, gt_boxes, gt_classes, gt_groups)
    """

    def __init__(
        self,
        cost_gain: dict[str, float] | None = None,
        use_fl: bool = True,
        with_mask: bool = False,
        num_sample_points: int = 12544,
        alpha: float = 0.25,
        gamma: float = 2.0,
        change_matcher: bool = False,
        iou_order_alpha: float = 1.0,
        matcher_change_epoch: int = 10000,
        use_mal: bool = False,
        mal_alpha: float | None = None,
        mal_gamma: float = 2.0,
        mal_transition_power: float = 2.0,
    ):
        """Initialize HungarianMatcher for optimal assignment of predicted and ground truth bounding boxes.

        Args:
            cost_gain (dict[str, float], optional): Dictionary of cost coefficients for different matching cost
                components. Should contain keys 'class', 'bbox', 'giou', 'mask', and 'dice'.
            use_fl (bool): Whether to use Focal Loss for classification cost calculation.
            with_mask (bool): Whether the model makes mask predictions.
            num_sample_points (int): Number of sample points used in mask cost calculation.
            alpha (float): Alpha factor in Focal Loss calculation.
            gamma (float): Gamma factor in Focal Loss calculation.
            change_matcher (bool): Enable DEIMv2-style matcher switch at a target epoch.
            iou_order_alpha (float): Exponent for IoU in switched matcher cost.
            matcher_change_epoch (int): Epoch at/after which switched matcher cost is used.
            use_mal (bool): Gradually replace Focal classification cost with MAL classification cost.
            mal_alpha (float | None): Optional negative-weight scale used by MAL. None applies no scaling.
            mal_gamma (float): IoU-target and negative-weight exponent used by MAL.
            mal_transition_power (float): Power applied to normalized progress for the Focal-to-MAL blend.
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 2, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma
        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        self.use_mal = use_mal
        self.mal_alpha = mal_alpha
        self.mal_gamma = mal_gamma
        self.mal_transition_power = mal_transition_power
        if self.use_mal and not self.use_fl:
            raise ValueError("MAL matcher curriculum requires use_fl=True for its Focal starting cost")
        if self.use_mal and self.change_matcher:
            raise ValueError("MAL matcher curriculum and change_matcher are alternative matcher schedules")
        if self.mal_transition_power <= 0:
            raise ValueError(f"mal_transition_power must be > 0, got {self.mal_transition_power}")

    @staticmethod
    def _pairwise_probiou(
        pred_bboxes: torch.Tensor, pred_angles: torch.Tensor, gt_bboxes_obb: torch.Tensor
    ) -> torch.Tensor:
        """Pairwise rotated IoU between predicted and GT oriented boxes, for angle-aware OBB matching.

        Boxes are upscaled to pixel-ish units and w/h clamped to >= 1px before the covariance computation, the
        same numerical conditioning as the probiou loss term (tiny normalized w/h saturate the determinants).
        """
        scale = pred_bboxes.new_tensor([1024.0, 1024.0, 1024.0, 1024.0, 1.0])

        def prep(xywhr):
            xywhr = xywhr.float() * scale
            return torch.cat([xywhr[:, :2], xywhr[:, 2:4].clamp(min=1.0), xywhr[:, 4:]], dim=-1)

        pred_xywhr = prep(torch.cat([pred_bboxes, pred_angles], dim=-1))
        gt_xywhr = prep(gt_bboxes_obb)
        return batch_probiou(pred_xywhr, gt_xywhr).clamp(0.0, 1.0)  # (bs*num_queries, num_gt)

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
        epoch: int = 0,
        training_progress: float = 0.0,
        pred_angles: torch.Tensor | None = None,
        gt_bboxes_obb: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute optimal assignment between predictions and ground truth using Hungarian algorithm.

        This method calculates matching costs based on classification scores, bounding box coordinates, and optionally
        mask predictions, then finds the optimal bipartite assignment between predictions and ground truth.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 4).
            pred_scores (torch.Tensor): Predicted classification scores with shape (batch_size, num_queries,
                num_classes).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (num_gts, 4).
            gt_cls (torch.Tensor): Ground truth class labels with shape (num_gts,).
            gt_groups (list[int]): Number of ground truth boxes for each image in the batch.
            masks (torch.Tensor, optional): Predicted masks with shape (batch_size, num_queries, height, width).
            gt_mask (list[torch.Tensor], optional): Ground truth masks, each with shape (num_masks, Height, Width).
            epoch (int): Current training epoch for optional DEIMv2 matcher schedule.
            training_progress (float): Normalized training progress in [0, 1] for the MAL matcher curriculum.
            pred_angles (torch.Tensor, optional): Predicted rotation angles with shape (batch_size, num_queries, 1).
                When given together with `gt_bboxes_obb`, the IoU cost uses probiou on rotated boxes (OBB models)
                instead of the axis-aligned GIoU.
            gt_bboxes_obb (torch.Tensor, optional): Ground truth rotated boxes with shape (num_gts, 5), xywhr.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): A list of size batch_size, each element is a tuple (index_i,
                index_j), where index_i is the tensor of indices of the selected predictions (in order) and index_j is
                the tensor of indices of the corresponding selected ground truth targets (in order).
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # Flatten to compute cost matrices in batch format
        pred_logits = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_logits) if self.use_fl else F.softmax(pred_logits, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)
        pred_xyxy = box_cxcywh_to_xyxy(pred_bboxes)
        gt_xyxy = box_cxcywh_to_xyxy(gt_bboxes)
        # Rotated pairwise IoU replaces the axis-aligned IoU everywhere when OBB angles are provided
        pair_riou = None
        if pred_angles is not None and gt_bboxes_obb is not None:
            pair_riou = self._pairwise_probiou(pred_bboxes, pred_angles.detach().reshape(-1, 1), gt_bboxes_obb)

        # DEIMv2-style dynamic matcher switch (late-epoch class*IoU ordering).
        if self.change_matcher and epoch >= self.matcher_change_epoch:
            class_score = pred_scores[:, gt_cls]
            iou_score = pair_riou if pair_riou is not None else pairwise_box_iou(pred_xyxy, gt_xyxy)[0]
            C = -(class_score * torch.pow(iou_score, self.iou_order_alpha))
        else:
            # Compute classification cost
            pred_scores = pred_scores[:, gt_cls]
            if self.use_fl:
                neg_cost_class = (1 - self.alpha) * (pred_scores**self.gamma) * (-(1 - pred_scores + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -pred_scores

            if self.use_mal:
                # Assignment needs the marginal cost of making pair (query i, target j) positive instead of leaving
                # that query/class entry as background. This is the pairwise equivalent of MALoss:
                # BCE(logit, IoU**gamma) - alpha * p**gamma * BCE(logit, 0).
                pair_logits = pred_logits[:, gt_cls]
                pair_iou = (
                    pair_riou if pair_riou is not None else pairwise_box_iou(pred_xyxy, gt_xyxy)[0].clamp(0.0, 1.0)
                )
                mal_target = pair_iou.pow(self.mal_gamma)
                mal_positive = F.binary_cross_entropy_with_logits(pair_logits, mal_target, reduction="none")
                mal_negative_weight = pred_scores.pow(self.mal_gamma)
                if self.mal_alpha is not None:
                    mal_negative_weight = mal_negative_weight * self.mal_alpha
                mal_negative = mal_negative_weight * F.softplus(pair_logits)
                mal_cost_class = mal_positive - mal_negative
                ratio = _curriculum_ratio(training_progress, self.mal_transition_power)
                cost_class = torch.lerp(cost_class, mal_cost_class, ratio)

            # Compute L1 cost between boxes
            cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt)

            # Compute IoU cost between boxes, (bs*num_queries, num_gt); probiou for OBB, axis-aligned GIoU
            # otherwise. The probiou cost is scaled by 2 to match GIoU's [0, 2] cost range (GIoU in [-1, 1] vs
            # probiou in [0, 1]), preserving the original class/bbox/iou cost balance.
            cost_giou = 2.0 * (1.0 - pair_riou) if pair_riou is not None else 1.0 - pairwise_giou(pred_xyxy, gt_xyxy)

            # Combine costs into final cost matrix
            C = (
                self.cost_gain["class"] * cost_class
                + self.cost_gain["bbox"] * cost_bbox
                + self.cost_gain["giou"] * cost_giou
            )

        # Add mask costs if available
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        C = C.view(bs, nq, -1).cpu()
        C = torch.nan_to_num(C, nan=1.0)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # (idx for queries, idx for gt)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
            for k, (i, j) in enumerate(indices)
        ]

    # This function is for future RT-DETR Segment models
    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
    #     # all masks share the same set of points for efficient matching
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
    #     sample_points = 2.0 * sample_points - 1.0
    #
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)
    #
    #     tgt_mask = torch.cat(gt_mask).unsqueeze(1)
    #     sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    #
    #     with torch.amp.autocast("cuda", enabled=False):
    #         # binary cross entropy cost
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
    #         cost_mask /= self.num_sample_points
    #
    #         # dice cost
    #         out_mask = F.sigmoid(out_mask)
    #         numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
    #         denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
    #         cost_dice = 1 - (numerator + 1) / (denominator + 1)
    #
    #         C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    #     return C


def get_cdn_group(
    batch: dict[str, Any],
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_dn: int = 100,
    cls_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    training: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
    """Generate contrastive denoising training group with positive and negative samples from ground truths.

    This function creates denoising queries for contrastive denoising training by adding noise to ground truth bounding
    boxes and class labels. It generates both positive and negative samples to improve model robustness.

    Args:
        batch (dict[str, Any]): Batch dictionary containing 'cls' (torch.Tensor with shape (num_gts,)), 'bboxes'
            (torch.Tensor with shape (num_gts, 4)), 'batch_idx' (torch.Tensor), and 'gt_groups' (list[int]) indicating
            number of ground truths per image.
        num_classes (int): Total number of object classes.
        num_queries (int): Number of object queries.
        class_embed (torch.Tensor): Class embedding weights to map labels to embedding space.
        num_dn (int): Number of denoising queries to generate.
        cls_noise_ratio (float): Noise ratio for class labels.
        box_noise_scale (float): Noise scale for bounding box coordinates.
        training (bool): Whether model is in training mode.

    Returns:
        padding_cls (torch.Tensor | None): Modified class embeddings for denoising with shape (bs, num_dn, embed_dim).
        padding_bbox (torch.Tensor | None): Modified bounding boxes for denoising with shape (bs, num_dn, 4).
        attn_mask (torch.Tensor | None): Attention mask for denoising with shape (tgt_size, tgt_size).
        dn_meta (dict[str, Any] | None): Meta information dictionary containing denoising parameters.

    Examples:
        Generate denoising group for training
        >>> batch = {
        ...     "cls": torch.tensor([0, 1, 2]),
        ...     "bboxes": torch.rand(3, 4),
        ...     "batch_idx": torch.tensor([0, 0, 1]),
        ...     "gt_groups": [2, 1],
        ... }
        >>> class_embed = torch.rand(80, 256)  # 80 classes, 256 embedding dim
        >>> cdn_outputs = get_cdn_group(batch, 80, 100, class_embed, training=True)
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    # Pad gt to max_num of a batch
    bs = len(gt_groups)
    gt_cls = batch["cls"]  # (bs*num, )
    gt_bbox = batch["bboxes"]  # bs*num, 4
    b_idx = batch["batch_idx"]

    # Each group has positive and negative queries
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # Positive and negative mask
    # (bs*num*num_group, ), the second total_num*num_group part as negative samples
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        # Apply class label noise to half of the samples
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # Randomly assign new class labels
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)

        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4

        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid

    num_dn = int(max_nums * 2 * num_group)  # total denoising queries
    dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)

    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # Match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # Reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )
