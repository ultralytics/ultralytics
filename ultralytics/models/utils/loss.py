# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.torch_utils import weighting_function
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh

from .ops import HungarianMatcher


# TODO: move classes and functions to ultralytics.utils.loss
class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the
    DETR object detection model.

    Attributes:
        nc (int): Number of classes.
        loss_gain (dict): Coefficients for different loss components.
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
        self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=False, uni_match_ind=0
    ):
        """
        Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """
        Compute classification loss based on predictions, target values, and ground truth scores.

        Args:
            pred_scores (torch.Tensor): Predicted class scores with shape (batch_size, num_queries, num_classes).
            targets (torch.Tensor): Target class indices with shape (batch_size, num_queries).
            gt_scores (torch.Tensor): Ground truth confidence scores with shape (batch_size, num_queries).
            num_gts (int): Number of ground truth objects.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            loss_cls (torch.Tensor): Classification loss value.

        Notes:
            The function supports different classification loss types:
            - Varifocal Loss (if self.vfl is True and num_gts > 0)
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
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        """
        Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 4).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4), where N is the total
                number of ground truth boxes.
            postfix (str): String to append to the loss names for identification in multi-loss scenarios.

        Returns:
            loss (dict): Dictionary containing:
                - loss_bbox{postfix} (torch.Tensor): L1 loss between predicted and ground truth boxes,
                  scaled by the bbox loss gain.
                - loss_giou{postfix} (torch.Tensor): GIoU loss between predicted and ground truth boxes,
                  scaled by the giou loss gain.

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

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
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
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        match_indices=None,
        postfix="",
        masks=None,
        gt_mask=None,
    ):
        """
        Get auxiliary losses for intermediate decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers.
            pred_scores (torch.Tensor): Predicted scores from auxiliary layers.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (List[int]): Number of ground truths per image.
            match_indices (List[tuple], optional): Pre-computed matching indices.
            postfix (str): String to append to loss names.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.

        Returns:
            (dict): Dictionary of auxiliary losses.
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
    def _get_index(match_indices):
        """
        Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (List[tuple]): List of tuples containing matched indices.

        Returns:
            (tuple): Tuple containing (batch_idx, src_idx) and dst_idx.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        """
        Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            match_indices (List[tuple]): List of tuples containing matched indices.

        Returns:
            (tuple): Tuple containing assigned predictions and ground truths.
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
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        masks=None,
        gt_mask=None,
        postfix="",
        match_indices=None,
    ):
        """
        Calculate losses for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            pred_scores (torch.Tensor): Predicted class scores.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (List[int]): Number of ground truths per image.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
            postfix (str): String to append to loss names.
            match_indices (List[tuple], optional): Pre-computed matching indices.

        Returns:
            (dict): Dictionary of losses.
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

    def forward(self, pred_bboxes, pred_scores, batch, postfix="", **kwargs):
        """
        Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape [l, b, query, 4].
            pred_scores (torch.Tensor): Predicted class scores, shape [l, b, query, num_classes].
            batch (dict): Batch information containing:
                cls (torch.Tensor): Ground truth classes, shape [num_gts].
                bboxes (torch.Tensor): Ground truth bounding boxes, shape [num_gts, 4].
                gt_groups (List[int]): Number of ground truths for each image in the batch.
            postfix (str): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict): Computed losses, including main and auxiliary (if enabled).

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
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        Forward pass to compute detection loss with optional denoising loss.

        Args:
            preds (tuple): Tuple containing predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes.
            dn_scores (torch.Tensor, optional): Denoising scores.
            dn_meta (dict, optional): Metadata for denoising.

        Returns:
            (dict): Dictionary containing total loss and denoising loss if applicable.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing number of ground truths per image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), "Expected the same length, "
                f"but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tensor: Decoded distances.
    """
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


def translate_gt(gt, reg_max, reg_scale, up):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return xyxy2xywh(bboxes)


class DEIMLoss(nn.Module):
    """This class computes the loss for DEIM."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        losses=["mal", "boxes", "local"],
        weight_dict={"loss_mal": 1, "loss_bbox": 5, "loss_giou": 2, "loss_fgl": 0.15, "loss_ddf": 1.5},
        matcher=HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2}),
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = "giou"
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

    def loss_labels_mal(self, outputs, targets, indices, values=None):
        assert "pred_boxes" in outputs
        pred_idx, gt_idx = DETRLoss._get_index(indices)
        if values is None:
            # TODO
            src_boxes = outputs["pred_boxes"][pred_idx]
            target_boxes = targets["bboxes"][gt_idx]
            ious = bbox_iou(src_boxes.detach(), target_boxes).squeeze(-1).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[pred_idx] = targets["cls"][gt_idx]
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[pred_idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        weight = (
            (self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target)
            if self.mal_alpha is not None
            else (pred_score.pow(self.gamma) * (1 - target) + target)
        )

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / len(gt_idx)
        return {"loss_mal": loss}

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        pred_idx, gt_idx = DETRLoss._get_index(indices)
        src_boxes = outputs["pred_boxes"][pred_idx]
        target_boxes = targets["bboxes"][gt_idx]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / len(target_boxes)

        # TODO: could use CIOU as well
        loss_giou = 1 - bbox_iou(src_boxes, target_boxes, GIoU=True)
        if self.boxes_weight_format is not None:
            boxes_weight = bbox_iou(src_boxes.detach(), target_boxes, GIoU=self.boxes_weight_format == "giou")
            loss_giou *= boxes_weight
        losses["loss_giou"] = loss_giou.mean()

        return losses

    def loss_local(self, outputs, targets, indices, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" not in outputs:
            return losses

        pred_idx, gt_idx = DETRLoss._get_index(indices)
        target_boxes = targets["bboxes"][gt_idx]

        pred_corners = outputs["pred_corners"][pred_idx].reshape(-1, (self.reg_max + 1))
        ref_points = outputs["ref_points"][pred_idx].detach()
        with torch.no_grad():
            if self.fgl_targets_dn is None and "is_dn" in outputs:
                self.fgl_targets_dn = bbox2distance(
                    ref_points, xywh2xyxy(target_boxes), self.reg_max, outputs["reg_scale"], outputs["up"]
                )
            if self.fgl_targets is None and "is_dn" not in outputs:
                self.fgl_targets = bbox2distance(
                    ref_points, xywh2xyxy(target_boxes), self.reg_max, outputs["reg_scale"], outputs["up"]
                )

        target_corners, weight_right, weight_left = self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets

        ious = bbox_iou(outputs["pred_boxes"][pred_idx], target_boxes)  # TODO: CIOU
        weight_targets = ious.repeat(1, 4).reshape(-1).detach()

        losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
            pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=len(target_boxes)
        )

        if "teacher_corners" in outputs:
            pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
            target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
            if not torch.equal(pred_corners, target_corners):
                weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[pred_idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                weight_targets_local[pred_idx] = ious.reshape_as(weight_targets_local[pred_idx]).to(
                    weight_targets_local.dtype
                )
                weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                loss_match_local = (
                    weight_targets_local
                    * (T**2)
                    * (
                        nn.KLDivLoss(reduction="none")(
                            F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1)
                        )
                    ).sum(-1)
                )
                if "is_dn" not in outputs:
                    batch_scale = 8 / outputs["pred_boxes"].shape[0]  # Avoid the influence of batch size per GPU
                    self.num_pos, self.num_neg = (
                        (mask.sum() * batch_scale) ** 0.5,
                        ((~mask).sum() * batch_scale) ** 0.5,
                    )
                loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                losses["loss_ddf"] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                    self.num_pos + self.num_neg
                )

        return losses

    def _merge_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
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

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "mal": self.loss_labels_mal,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, batch, reg_scale, up, dn_outputs=None, dn_meta=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        dec_bboxes, dec_scores, dec_corners, dec_refs, pre_bboxes, pre_logits, enc_topk_bboxes, enc_topk_scores = (
            outputs
        )
        dn_pre_scores, dn_pre_bboxes, dn_out_scores, dn_out_bboxes, dn_out_corners, dn_out_refs = dn_outputs
        outputs = {"pred_logits": dec_scores[-1], "pred_boxes": dec_bboxes[-1]}  # last layer predictions
        outputs["pred_corners"] = dec_corners[-1]
        outputs["ref_points"] = dec_refs[-1]
        outputs["up"] = up
        outputs["reg_scale"] = reg_scale

        outputs["aux_outputs"] = self._set_aux_loss2(
            dec_scores[:-1],
            dec_bboxes[:-1],
            dec_corners[:-1],
            dec_refs[:-1],
            dec_corners[-1],
            dec_scores[-1],
        )
        outputs["enc_aux_outputs"] = self._set_aux_loss([enc_topk_scores], [enc_topk_bboxes])
        outputs["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
        outputs["enc_meta"] = {"class_agnostic": False}

        if dn_meta is not None:
            outputs["dn_outputs"] = self._set_aux_loss2(
                dn_out_scores, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_corners[-1], dn_out_scores[-1]
            )
            outputs["dn_pre_outputs"] = {"pred_logits": dn_pre_scores, "pred_boxes": dn_pre_bboxes}
        dn_outputs = outputs.pop("dn_outputs", None)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(
            outputs["pred_boxes"].contiguous(),
            outputs["pred_logits"].contiguous(),
            batch["bboxes"],
            batch["cls"],
            batch["gt_groups"],
        )
        self._clear_cache()

        indices_aux_list, cached_indices, cached_indices_enc = [], [], []
        aux_outputs_list = outputs["aux_outputs"]
        if "pre_outputs" in outputs:
            aux_outputs_list = outputs["aux_outputs"] + [outputs["pre_outputs"]]
        for i, aux_outputs in enumerate(aux_outputs_list):
            indices_aux = self.matcher(
                aux_outputs["pred_boxes"].contiguous(),
                aux_outputs["pred_logits"].contiguous(),
                batch["bboxes"],
                batch["cls"],
                batch["gt_groups"],
            )
            cached_indices.append(indices_aux)
            indices_aux_list.append(indices_aux)
        for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
            indices_enc = self.matcher(
                aux_outputs["pred_boxes"].contiguous(),
                aux_outputs["pred_logits"].contiguous(),
                batch["bboxes"],
                batch["cls"],
                batch["gt_groups"],
            )
            cached_indices_enc.append(indices_enc)
            indices_aux_list.append(indices_enc)
        all_indices = self._merge_indices(indices, indices_aux_list)

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
            # TODO, indices and num_box are different from RT-DETRv2
            indices_in = all_indices if use_uni_set else indices
            meta = self.get_loss_meta_info(loss, outputs, batch, indices_in)
            l_dict = self.get_loss(loss, outputs, batch, indices_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if "local" in self.losses:  # only work for local loss
                    aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
                    indices_in = all_indices if use_uni_set else cached_indices[i]
                    meta = self.get_loss_meta_info(loss, aux_outputs, batch, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, batch, indices_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
                indices_in = all_indices if use_uni_set else cached_indices[-1]
                meta = self.get_loss_meta_info(loss, aux_outputs, batch, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, batch, indices_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(batch)
                for t in enc_targets:
                    t["cls"] = torch.zeros_like(t["cls"])
            else:
                enc_targets = batch

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == "boxes")
                    indices_in = all_indices if use_uni_set else cached_indices_enc[i]
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if dn_outputs is not None:
            assert dn_meta is not None
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            indices_dn = RTDETRDetectionLoss.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            for i, aux_outputs in enumerate(dn_outputs):
                if "local" in self.losses:  # only work for local loss
                    aux_outputs["is_dn"] = True
                    aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, batch, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, batch, indices_dn, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, batch, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, batch, indices_dn, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def _get_loss(self, outputs, batch, all_match_indices, match_indices, suffix=""):
        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
            indices_in = all_match_indices if use_uni_set else match_indices
            meta = self.get_loss_meta_info(loss, outputs, batch, indices_in)
            l_dict = self.get_loss(loss, outputs, batch, indices_in, **meta)

            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            if suffix:
                l_dict = {k + suffix: v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None or loss not in {"vfl", "mal"}:
            return {}

        pred_idx, gt_idx = DETRLoss._get_index(indices)
        src_boxes = outputs["pred_boxes"][pred_idx]
        target_boxes = targets["bboxes"][gt_idx]

        # TODO: could use CIOU as well
        iou = bbox_iou(src_boxes.detach(), target_boxes, GIoU=self.boxes_weight_format == "giou")
        return {"values": iou}

    # @staticmethod
    # def get_cdn_matched_indices(dn_meta, targets):
    #     """get_cdn_matched_indices"""
    #     dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
    #     num_gts = [len(t["labels"]) for t in targets]
    #     device = targets[0]["labels"].device
    #
    #     dn_match_indices = []
    #     for i, num_gt in enumerate(num_gts):
    #         if num_gt > 0:
    #             gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
    #             gt_idx = gt_idx.tile(dn_num_group)
    #             assert len(dn_positive_idx[i]) == len(gt_idx)
    #             dn_match_indices.append((dn_positive_idx[i], gt_idx))
    #         else:
    #             dn_match_indices.append(
    #                 (torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device))
    #             )
    #
    #     return dn_match_indices

    @staticmethod
    def _set_aux_loss(outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    @staticmethod
    def _set_aux_loss2(
        outputs_class, outputs_coord, outputs_corners, outputs_ref, teacher_corners=None, teacher_logits=None
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
        ]

    # TODO: DFL Loss
    def unimodal_distribution_focal_loss(
        self,
        pred,
        label,
        weight_right,
        weight_left,
        weight=None,
        reduction="sum",
        avg_factor=None,
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1) + F.cross_entropy(
            pred, dis_right, reduction="none"
        ) * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
