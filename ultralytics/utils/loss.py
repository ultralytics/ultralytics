# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.metrics import CITYSCAPES_WEIGHT, OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.geometry3d import (
    backproject_points_torch,
    boxes3d_corners_torch,
    decode_alpha_multibin,
    encode_alpha_multibin,
    paired_boxes3d_iou_torch,
    project_points_torch,
    wrap_angle_torch,
)
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False, device=pred_score.device.type):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://arxiv.org/abs/2006.04388."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        # Compute log_softmax once, then two gathers; cross_entropy(x, t) = -log_softmax(x).gather(t)
        logp = F.log_softmax(pred_dist, dim=1)
        return -(
            logp.gather(1, tl.view(-1, 1)).view(tl.shape) * wl + logp.gather(1, tr.view(-1, 1)).view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    floor = 0.01

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], floor=self.floor)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" if self.use_dfl else "l1_loss"

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )  # loss(box, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


def quality_focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """Return quality focal loss for a continuous localization target in ``[0, 1]``."""
    if beta < 0.0 or not math.isfinite(beta):
        raise ValueError(f"quality focal beta must be finite and non-negative, got {beta}")
    target = target.to(dtype=logits.dtype).clamp(0.0, 1.0)
    modulation = (logits.sigmoid() - target).abs().pow(beta)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none") * modulation


def weighted_smooth_l1_loss_fp32(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, normalizer: torch.Tensor | float
) -> torch.Tensor:
    """Return a weighted SmoothL1 mean with FP32 accumulation to prevent large-batch AMP overflow."""
    elementwise = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(-1, keepdim=True)
    denominator = (
        normalizer.float()
        if isinstance(normalizer, torch.Tensor)
        else torch.tensor(normalizer, device=elementwise.device, dtype=torch.float32)
    )
    loss = (elementwise * weight.float()).sum() / denominator.clamp_min(1.0)
    if not torch.isfinite(loss):

        def finite_max(value: torch.Tensor) -> float:
            finite = value.detach().float()[torch.isfinite(value.detach())]
            return finite.abs().max().item() if finite.numel() else float("nan")

        raise FloatingPointError(
            "Non-finite center3d loss: "
            f"pred_nonfinite={(~torch.isfinite(pred)).sum().item()}, pred_absmax={finite_max(pred):.6g}, "
            f"target_nonfinite={(~torch.isfinite(target)).sum().item()}, target_absmax={finite_max(target):.6g}, "
            f"weight_nonfinite={(~torch.isfinite(weight)).sum().item()}, weight_absmax={finite_max(weight):.6g}, "
            f"normalizer={float(denominator.detach()):.6g}"
        )
    return loss


class v8Detection3DLoss(v8DetectionLoss):
    """MonoCon-Lite/MonoDLE-style criterion for a lightweight YOLO monocular 3D detector.

    It keeps YOLO's 2D task-aligned assignment, but balances 3D regression per ground-truth object instead of letting
    low 2D IoU suppress the hardest 3D targets. The 3D objective combines box-relative projected center, direct
    log-depth regression, class-mean dimension residuals, 12-bin observation angle, disentangled
    camera-space corners, projected-corner auxiliary context, and a learned localization-quality score.
    """

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize v8Detection3DLoss with model parameters and task-aligned assignment settings."""
        super().__init__(model, tal_topk, tal_topk2)
        self.head = model.model[-1]
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss" if self.use_dfl else "l1_loss",
            "center3d_loss",
            "depth_loss",
            "alpha_loss",
            "dim_loss",
            "corner3d_loss",
            "keypoint3d_loss",
            "quality3d_loss",
        )
        self.d3_geometry_gain = float(self.hyp.get("d3_geometry_gain", 1.0))
        if not math.isfinite(self.d3_geometry_gain) or self.d3_geometry_gain <= 0.0:
            raise ValueError(f"d3_geometry_gain must be finite and positive, got {self.d3_geometry_gain}")
        self.depth_z = float(self.hyp.get("depth_z", 0.1))
        if not math.isfinite(self.depth_z) or self.depth_z < 0.0:
            raise ValueError(f"depth_z must be finite and non-negative, got {self.depth_z}")
        self.depth_z_tau = float(self.hyp.get("depth_z_tau", 2.0))
        if not math.isfinite(self.depth_z_tau) or self.depth_z_tau <= 0.0:
            raise ValueError(f"depth_z_tau must be finite and positive, got {self.depth_z_tau}")
        expected_nr = getattr(self.head, "geo_channels", 0) + 2 * getattr(self.head, "num_alpha_bins", 0)
        if (
            getattr(self.head, "nd", None) != 8
            or getattr(self.head, "geo_channels", None) != 7
            or getattr(self.head, "nr", None) != expected_nr
        ):
            raise ValueError(
                "Mono3D loss requires Detect3D nd=8 with seven direct-geometry channels and a valid MultiBin layout, "
                f"got nd={getattr(self.head, 'nd', None)}, geo_channels={getattr(self.head, 'geo_channels', None)}, "
                f"nr={getattr(self.head, 'nr', None)}, expected_nr={expected_nr}"
            )

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess 3D targets by converting to tensor format and scaling coordinates.

        Input includes a final per-object 3D-valid flag after the unchanged seven 3D label values.
        """
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            # Convert cx,cy,w,h to xyxy (columns 1-4 after cls)
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the standard 2D losses and geometry-balanced monocular 3D losses."""
        n_loss = len(self.loss_names)
        loss = torch.zeros(n_loss, device=self.device)

        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        pred_d3 = preds["d3_params"].permute(0, 2, 1).contiguous()
        pred_aux = preds.get("d3_aux")
        pred_aux = pred_aux.permute(0, 2, 1).contiguous() if pred_aux is not None else None
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Labels on disk remain unchanged. d3_valid is generated by the dataset from original geometry/difficulty.
        d3_valid = batch.get("d3_valid")
        if d3_valid is None:
            d3_values = batch["bboxes"][:, 4:11]
            d3_valid = (
                torch.isfinite(d3_values).all(1)
                & (d3_values[:, 0] >= 2.0)
                & (d3_values[:, 0] <= 65.0)
                & (d3_values[:, 3:6] > 0).all(1)
            ).to(batch["bboxes"].dtype)[:, None]
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
                d3_valid.view(-1, 1),
            ),
            1,
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        gt_labels = targets[:, :, 0:1]  # cls
        gt_bboxes = targets[:, :, 1:5]  # xyxy
        gt_d3 = targets[:, :, 5:12]
        gt_d3_valid = targets[:, :, 12:13].gt(0.5)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

            # Gather matched GT values for all TAL positives. 2D loss still uses every labeled object.
            bs_idx = torch.arange(batch_size, device=self.device)[:, None].expand_as(fg_mask)
            gt_d3_expanded = gt_d3[bs_idx, target_gt_idx]
            target_d3_all = gt_d3_expanded[fg_mask]
            target_valid_all = gt_d3_valid[bs_idx, target_gt_idx][fg_mask].squeeze(-1)
            target_boxes_all = target_bboxes[fg_mask]
            pred_boxes_all = (pred_bboxes.detach() * stride_tensor)[fg_mask]
            target_labels_all = gt_labels[bs_idx, target_gt_idx][fg_mask].squeeze(-1).long()
            pred_d3_all = pred_d3[fg_mask]
            pred_aux_all = pred_aux[fg_mask] if pred_aux is not None else None
            batch_indices_all = bs_idx[fg_mask]
            gt_indices_all = target_gt_idx[fg_mask]

            finite_target = torch.isfinite(target_d3_all).all(1)
            valid = target_valid_all & finite_target
            if valid.any():
                target_d3 = target_d3_all[valid].float()
                target_boxes = target_boxes_all[valid].float()
                matched_pred_boxes = pred_boxes_all[valid].float()
                target_labels_3d = target_labels_all[valid]
                raw = pred_d3_all[valid].float()
                raw_aux = pred_aux_all[valid].float() if pred_aux_all is not None else None
                batch_indices = batch_indices_all[valid]
                gt_indices = gt_indices_all[valid]

                # Normalize TAL weights within each object, then average objects equally. This prevents far/occluded
                # objects from receiving almost no 3D gradient merely because their 2D IoU target is low.
                raw_weight = target_scores[fg_mask].sum(-1, keepdim=True)[valid].float().clamp_min(1e-4)
                max_gt = max(gt_d3.shape[1], 1)
                object_key = batch_indices * max_gt + gt_indices
                totals = raw_weight.new_zeros(batch_size * max_gt)
                totals.scatter_add_(0, object_key, raw_weight[:, 0])
                weight = raw_weight / totals[object_key, None].clamp_min(1e-6)
                normalizer_3d = raw_weight.new_tensor(float(object_key.unique().numel())).clamp_min(1.0)

                gt_dist = target_d3[:, 0:1]
                gt_xc = target_d3[:, 1:2]
                gt_y_bottom = target_d3[:, 2:3]
                gt_w3d, gt_h3d, gt_l3d = target_d3[:, 3:4], target_d3[:, 4:5], target_d3[:, 5:6]
                gt_ry = target_d3[:, 6:7]
                gt_dims = torch.cat((gt_h3d, gt_w3d, gt_l3d), 1)

                far_depth = float(self.hyp.get("far_3d_depth", 60.0))
                far_weight = float(self.hyp.get("far_3d_weight", 0.2))
                weight *= torch.where(gt_dist > far_depth, far_weight, 1.0)

                p2s_aug = batch.get("p2s_aug")
                if p2s_aug is None or len(p2s_aug) != batch_size or any(p2 is None for p2 in p2s_aug):
                    raise RuntimeError("detect3d loss requires one valid augmented P2 matrix per batch image")
                p2_batch = torch.stack([torch.as_tensor(p2, device=self.device, dtype=torch.float32) for p2 in p2s_aug])
                p2_for_pos = p2_batch[batch_indices]

                gt_box_center = (target_boxes[:, :2] + target_boxes[:, 2:4]) * 0.5
                gt_box_size = (target_boxes[:, 2:4] - target_boxes[:, :2]).clamp_min(1.0)
                reference_valid = torch.isfinite(matched_pred_boxes).all(1, keepdim=True)
                reference_center = (matched_pred_boxes[:, :2] + matched_pred_boxes[:, 2:4]) * 0.5
                reference_size = matched_pred_boxes[:, 2:4] - matched_pred_boxes[:, :2]
                reference_valid &= torch.isfinite(reference_size).all(1, keepdim=True) & (reference_size > 0).all(
                    1, keepdim=True
                )
                # TAL positives normally have well-formed boxes. The GT fallback prevents a single malformed early
                # prediction from manufacturing a non-finite 3D target, while the 1 px floor matches deploy-time decode.
                box_center = torch.where(reference_valid, reference_center, gt_box_center)
                box_size = torch.where(reference_valid, reference_size, gt_box_size).clamp_min(1.0)
                gt_center = torch.cat((gt_xc, gt_y_bottom - gt_h3d * 0.5, gt_dist), 1)
                gt_pixel = project_points_torch(gt_center, p2_for_pos).float()
                target_center_offset = (gt_pixel - box_center) / box_size

                def weighted_mean(elementwise: torch.Tensor) -> torch.Tensor:
                    return (elementwise.float() * weight).sum() / normalizer_3d

                # 1) Projected center relative to the matched 2D box, invariant across FPN stride.
                center_element = F.smooth_l1_loss(raw[:, :2], target_center_offset, reduction="none").mean(1, True)
                loss[3] = weighted_mean(center_element)

                # 2) Continuous direct log-depth SmoothL1 with a bounded absolute camera-Z auxiliary.
                target_log_depth = gt_dist.clamp_min(1e-3).log()
                depth_beta = max(float(self.hyp.get("depth_beta", 0.2)), 1e-3)
                pred_log_depth = raw[:, 2]
                depth_element = F.smooth_l1_loss(
                    pred_log_depth[:, None], target_log_depth, beta=depth_beta, reduction="none"
                )
                pred_depth = pred_log_depth.clamp(math.log(0.1), math.log(200.0)).exp()[:, None]
                if self.depth_z > 0.0:
                    z_huber = F.smooth_l1_loss(pred_depth, gt_dist, beta=1.0, reduction="none")
                    z_bounded = -torch.expm1(-z_huber / self.depth_z_tau)
                    depth_element = depth_element + self.depth_z * z_bounded
                loss[4] = weighted_mean(depth_element)

                # 3) Class mean * exp(residual) dimensions.
                priors = self.head.dim_priors.to(device=self.device, dtype=torch.float32)[target_labels_3d]
                target_dim_residual = (gt_dims / priors).clamp_min(1e-4).log()
                dim_element = F.smooth_l1_loss(raw[:, 3:6], target_dim_residual, reduction="none").mean(1, True)
                loss[6] = weighted_mean(dim_element)
                pred_dims = priors * raw[:, 3:6].clamp(-4.0, 4.0).exp()

                # 4) MultiBin alpha classification + bounded within-bin residual.
                gt_alpha = wrap_angle_torch(gt_ry[:, 0] - torch.atan2(gt_xc[:, 0], gt_dist[:, 0]))
                bin_target, residual_target = encode_alpha_multibin(gt_alpha, self.head.num_alpha_bins)
                direction = raw[:, self.head.geo_channels :]
                bin_logits = direction[:, : self.head.num_alpha_bins]
                residual_logits = direction[:, self.head.num_alpha_bins :]
                residual_pred = residual_logits.gather(1, bin_target[:, None]).tanh() * (
                    torch.pi / self.head.num_alpha_bins
                )
                alpha_cls = F.cross_entropy(bin_logits, bin_target, reduction="none")[:, None]
                alpha_reg = F.smooth_l1_loss(residual_pred, residual_target[:, None], beta=0.1, reduction="none")
                loss[5] = weighted_mean(alpha_cls + alpha_reg)
                # Use the target bin for the training-only corner objective.  Argmax decoding is appropriate for
                # inference, but at initialization it would route every sample through bin 0 and send contradictory
                # corner gradients to one residual head until the classification logits happen to switch bins.
                pred_alpha_for_corner = wrap_angle_torch(
                    bin_target.to(residual_pred.dtype) * (2.0 * torch.pi / self.head.num_alpha_bins)
                    + residual_pred[:, 0]
                )

                # Decode camera center and yaw for complete-box geometry supervision.
                pred_pixel = box_center + raw[:, :2] * box_size
                pred_center = backproject_points_torch(pred_pixel, pred_depth[:, 0], p2_for_pos).float()
                gt_ray = torch.atan2(gt_center[:, 0], gt_center[:, 2])
                pred_ry_disentangled = wrap_angle_torch(pred_alpha_for_corner + gt_ray)
                gt_corners = boxes3d_corners_torch(gt_center, gt_dims, gt_ry[:, 0])

                # 5) SMOKE-style disentangled corner loss: center, dimensions, and rotation receive stable gradients.
                center_corners = boxes3d_corners_torch(pred_center, gt_dims, gt_ry[:, 0])
                dim_corners = boxes3d_corners_torch(gt_center, pred_dims, gt_ry[:, 0])
                rot_corners = boxes3d_corners_torch(gt_center, gt_dims, pred_ry_disentangled)
                corner_parts = [
                    F.smooth_l1_loss(value, gt_corners, beta=1.0, reduction="none").mean((1, 2), keepdim=False)[:, None]
                    for value in (center_corners, dim_corners, rot_corners)
                ]
                loss[7] = weighted_mean(torch.stack(corner_parts, 0).mean(0))

                # 6) MonoCon-style training-only projected corners, normalized by the 2D box size. Supervise only
                # visible corners: heavily truncated KITTI objects can project far outside a very narrow visible box,
                # and treating those off-image coordinates as ordinary regression targets destabilizes this auxiliary
                # branch without adding useful image evidence.
                if raw_aux is not None:
                    corner_pixels = project_points_torch(gt_corners, p2_for_pos).float()
                    target_aux = (corner_pixels - box_center[:, None]) / box_size[:, None]
                    visible = (
                        torch.isfinite(corner_pixels).all(2)
                        & (gt_corners[..., 2] > 1e-3)
                        & (corner_pixels[..., 0] >= 0.0)
                        & (corner_pixels[..., 0] < imgsz[1].float())
                        & (corner_pixels[..., 1] >= 0.0)
                        & (corner_pixels[..., 1] < imgsz[0].float())
                    )
                    raw_aux_corners = raw_aux.reshape(-1, 8, 2)
                    # Replace invisible/non-finite targets before SmoothL1: multiplying inf by a zero mask gives NaN.
                    safe_target_aux = torch.where(visible[..., None], target_aux, raw_aux_corners.detach())
                    aux_per_corner = F.smooth_l1_loss(raw_aux_corners, safe_target_aux, reduction="none").mean(2)
                    visible_float = visible.float()
                    aux_element = (aux_per_corner * visible_float).sum(1, keepdim=True) / visible_float.sum(
                        1, keepdim=True
                    ).clamp_min(1.0)
                    loss[8] = weighted_mean(aux_element)

                # 7) Complete 3D localization quality for AP ranking with a detached target.
                # q3d must describe the box that inference will actually decode. Unlike the differentiable corner
                # objective above, use the predicted winning bin here; otherwise a wrong orientation class could still
                # receive a high quality target through teacher-forced GT-bin geometry. This target is detached, so the
                # hard argmax does not create a gradient discontinuity in the orientation branch.
                with torch.no_grad():
                    decoded_alpha = decode_alpha_multibin(bin_logits, residual_logits)
                    decoded_ry = wrap_angle_torch(decoded_alpha + torch.atan2(pred_center[:, 0], pred_center[:, 2]))
                    quality_target = paired_boxes3d_iou_torch(
                        pred_center, pred_dims, decoded_ry, gt_center, gt_dims, gt_ry[:, 0]
                    )[:, None]
                quality_element = quality_focal_loss_with_logits(
                    raw[:, 6:7], quality_target, beta=float(self.hyp.get("quality3d_gamma", 2.0))
                )
                loss[9] = weighted_mean(quality_element)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.get("center3d", 5.0) * self.d3_geometry_gain  # center3d gain
        loss[4] *= self.hyp.get("depth", 5.0) * self.d3_geometry_gain  # depth gain
        loss[5] *= self.hyp.get("alpha", 1.0)  # MultiBin alpha gain
        loss[6] *= self.hyp.get("dim", 1.0) * self.d3_geometry_gain  # dimension residual gain
        loss[7] *= self.hyp.get("corner3d", 1.0) * self.d3_geometry_gain  # camera-corner gain
        loss[8] *= self.hyp.get("keypoint3d", 1.0) * self.d3_geometry_gain  # projected-corner gain
        loss[9] *= self.hyp.get("quality3d", 1.0)  # localization-quality gain

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_names = ("box_loss", "seg_loss", *self.loss_names[1:], "sem_loss")
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semantic
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semantic = proto
        else:
            pred_semantic = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semantic is not None:
                sem_idx = batch["sem_masks"].to(self.device).long().unsqueeze(1)  # Nx1xHxW
                if self.overlap:
                    present = masks != 0  # NxHxW
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    present = torch.zeros(batch_size, *masks.shape[-2:], dtype=torch.bool, device=self.device)
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i):
                            present[i] = instance_mask_i.sum(dim=0) != 0
                # One-hot targets zeroed at uncovered pixels, without F.one_hot's int64 NxHxWxC intermediate
                sem_masks = torch.zeros(sem_idx.shape[0], self.nc, *sem_idx.shape[2:], device=self.device)
                sem_masks.scatter_(1, sem_idx, present.unsqueeze(1).float())  # NxCxHxW

                loss[4] = self.bcedice_loss(pred_semantic, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semantic is not None:
                loss[4] += (pred_semantic * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # loss(box, seg, cls, dfl, semantic)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks, shape (BS, H, W) if `overlap` else (N_instances_in_batch, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks[i] == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_names = ("box_loss", "pose_loss", "kobj_loss", *self.loss_names[1:])
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
        batched_keypoints[batch_idx_long, within_idx] = keypoints

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLO26 pose estimation with RLE loss support."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.loss_names += ("rle_loss",)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])
        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        if not kpt_mask.any():
            return pred_kpt[..., :0].sum()

        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
        if not error.numel():
            return pred_kpt[..., :0].sum()

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return pred_kpt[..., :0].sum()

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, {"loss": loss.detach()}


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.loss_names = (*self.loss_names, "angle_loss")
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xywhr, (b, h*w, 5)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores[fg_mask].sum(-1)
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class DepthLoss26:
    """Criterion class for computing training losses for YOLO depth estimation.

    Uses scale-invariant log loss (SILog) + gradient-matching loss, following the Depth Anything approach. SILog handles
    scale ambiguity while gradient loss preserves edges.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize DepthLoss26."""
        device = next(model.parameters()).device
        self.device = device
        h = model.args  # hyperparameters
        self.silog_weight = h.dlog
        self.grad_weight = h.dgrad
        self.silog_lambda = h.dlam  # 1.0 = scale-invariant, 0.0 = log-RMSE
        self.grad_scales = 4
        self.loss_names = "dlog_loss", "dgrad_loss"

    @staticmethod
    def _grad_l1(pred_log: torch.Tensor, gt_log: torch.Tensor, valid_f: torch.Tensor) -> torch.Tensor:
        """L1 between predicted and GT log-depth spatial gradients (dx, dy), gated by the valid mask.

        Each gradient is zeroed unless both contributing pixels are valid, so edges are only
        matched where GT is defined.
        """
        pred_dx = (pred_log[:, :, :, 1:] - pred_log[:, :, :, :-1]) * valid_f[:, :, :, 1:] * valid_f[:, :, :, :-1]
        gt_dx = (gt_log[:, :, :, 1:] - gt_log[:, :, :, :-1]) * valid_f[:, :, :, 1:] * valid_f[:, :, :, :-1]
        pred_dy = (pred_log[:, :, 1:, :] - pred_log[:, :, :-1, :]) * valid_f[:, :, 1:, :] * valid_f[:, :, :-1, :]
        gt_dy = (gt_log[:, :, 1:, :] - gt_log[:, :, :-1, :]) * valid_f[:, :, 1:, :] * valid_f[:, :, :-1, :]
        return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

    def __call__(
        self, preds: dict[str, torch.Tensor] | torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate depth estimation loss.

        Args:
            preds (dict | torch.Tensor): Dict with "depth" key or raw tensor of (B, 1, H, W) predicted depth.
            batch (dict): Dict with "depth" key holding (B, H, W) ground truth depth in meters.

        Returns:
            loss_sum (torch.Tensor): Total loss scaled by batch size.
            loss_items (dict[str, torch.Tensor]): Detached silog/gradient losses keyed by loss_names.
        """
        loss = torch.zeros(2, device=self.device)
        pred_depth = preds["depth"] if isinstance(preds, dict) else preds
        gt_depth = batch["depth"].to(self.device)

        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)

        if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
            pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)

        valid = gt_depth > 0.001
        if valid.sum() < 10:
            # Keep the result attached so BaseTrainer's unconditional backward() works.
            return pred_depth.sum() * 0.0, dict(zip(self.loss_names, loss.detach()))

        pred_valid = pred_depth[valid]
        gt_valid = gt_depth[valid]

        pred_valid = pred_valid.clamp(min=0.001)

        log_diff = torch.log(pred_valid) - torch.log(gt_valid)
        # Centered variance form: non-negative by construction and fp16-stable near convergence.
        m = log_diff.mean()
        silog = torch.sqrt(((log_diff - m) ** 2).mean() + (1.0 - self.silog_lambda) * m**2 + 1e-6)
        loss[0] = silog * self.silog_weight

        # Multi-scale gradient-matching loss.
        pred_log = torch.log(pred_depth.clamp(min=0.001))
        gt_log = torch.log(gt_depth.clamp(min=0.001))
        valid_f = valid.float()
        grad_loss = self._grad_l1(pred_log, gt_log, valid_f)
        for _ in range(1, max(self.grad_scales, 1)):
            if pred_log.shape[-1] < 4 or pred_log.shape[-2] < 4:
                break
            vp = F.avg_pool2d(valid_f, 2)
            if vp.mean() < 0.5:  # skip sparse GT (LiDAR)
                break
            denom = vp.clamp(min=1e-6)
            pred_log = F.avg_pool2d(pred_log * valid_f, 2) / denom
            gt_log = F.avg_pool2d(gt_log * valid_f, 2) / denom
            valid_f = (vp > 0).float()
            grad_loss = grad_loss + self._grad_l1(pred_log, gt_log, valid_f)
        loss[1] = grad_loss * self.grad_weight

        return loss * pred_depth.shape[0], dict(zip(self.loss_names, loss.detach()))


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model: torch.nn.Module):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], {
            k: loss_one2many[1][k] + loss_one2one[1][k] for k in loss_one2many[1]
        }


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model: torch.nn.Module, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class E2EDetect3DLoss(E2ELoss):
    """End-to-end Detect3D criterion with logs aligned to the blended optimization objective."""

    def __init__(self, model: torch.nn.Module):
        """Initialize one-to-many and one-to-one Detect3D losses."""
        super().__init__(model, v8Detection3DLoss)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the blended Detect3D loss and correspondingly blended log items."""
        preds = self.one2many.parse_output(preds)
        loss_one2many = self.one2many.loss(preds["one2many"], batch)
        loss_one2one = self.one2one.loss(preds["one2one"], batch)
        items = {key: loss_one2many[1][key] * self.o2m + loss_one2one[1][key] * self.o2o for key in loss_one2many[1]}
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, items


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        self.loss_names = tuple(k[:-5] for k in self.vp_criterion.loss_names)  # strip "_loss" suffix
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, dict(zip(self.loss_names, loss.detach()))

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        return vp_loss[0][1], dict(zip(self.loss_names, vp_loss[1].values()))

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk, tal_topk2)
        self.loss_names = tuple(k[:-5] for k in self.vp_criterion.loss_names if k != "sem_loss")  # strip "_loss"
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, dict(zip(self.loss_names, loss.detach()))

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        # zip drops the trailing "sem_loss" item to match the logged columns
        return cls_loss, dict(zip(self.loss_names, vp_loss[1].values()))


class SemanticSegmentationLoss(nn.Module):
    """Loss function for semantic segmentation using cross-entropy and Dice terms.

    Attributes:
        nc (int): Number of semantic classes.
        ce (nn.CrossEntropyLoss): Cross-entropy loss with ignore_index=255.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize semantic segmentation loss.

        Args:
            model (torch.nn.Module): Model containing the SemanticSegment head.
        """
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        data_name = Path(str(getattr(model.args, "data", "") or "")).stem.lower()
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"} and self.nc == len(CITYSCAPES_WEIGHT)
        weight = getattr(model, "class_weights", None)  # cls_pw frequency weights, else hardcoded Cityscapes
        if weight is None and self.use_cityscapes_weight:
            weight = torch.from_numpy(CITYSCAPES_WEIGHT)
        weight = None if weight is None else weight.to(device=self.device, dtype=self.dtype)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss(reduction="sum")  # binary: class weighting intentionally unsupported
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction="sum").to(device=self.device, dtype=self.dtype)
            if weight is not None:
                # Non-persistent: weight is a deterministic constant, no need to serialize into ckpt state_dict.
                self.ce.register_buffer("weight", weight, persistent=False)

    def _resize_masks(self, masks, target_shape):
        """Resize masks to match prediction spatial dimensions."""
        if masks.shape[1:] != target_shape:
            return (
                F.interpolate(masks.float().unsqueeze(1), size=target_shape, mode="nearest").squeeze(1).to(torch.int32)
            )
        return masks

    def _ce_loss(self, preds, masks, valid):
        """Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path."""
        flat = masks.reshape(-1)
        if self.nc == 1:
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
            denominator = valid.sum()
        else:
            logits = preds.permute(0, 2, 3, 1).reshape(-1, self.nc)
            target = flat.long()
            denominator = valid.sum() if self.ce.weight is None else self.ce.weight[target[valid]].sum()
        return self.ce(logits, target) / denominator.clamp_min(1)

    def _dice_loss(self, preds, masks, valid):
        """Compute Dice loss excluding ignore pixels."""
        if self.nc == 1:
            return self._binary_dice_loss(preds, masks, valid)
        flat_target = masks.reshape(-1)
        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
        intersection = torch.zeros(self.nc, device=preds.device, dtype=torch.float32)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=torch.float32)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks, valid):
        """Compute Dice loss for single-class (binary) segmentation.

        Pixels with value 255 are excluded from Dice terms to match BCE valid-pixel filtering.
        """
        valid = valid.reshape_as(masks).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def forward(self, preds, batch):
        """Compute semantic segmentation loss with optional auxiliary loss.

        Args:
            preds (torch.Tensor | tuple): Main logits [B, nc, H', W'], or (main, aux) tuple.
            batch (dict): Batch dict with 'semantic_mask' [B, H, W] containing class IDs (255=ignore).

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): Total loss * batch_size and a dict of detached loss items
                (ce_loss, dice_loss, aux_loss).
        """
        # Unpack auxiliary logits when present.
        aux_logits = None
        if isinstance(preds, tuple):
            preds, aux_logits = preds

        masks = batch["semantic_mask"].to(preds.device)
        valid = masks.reshape(-1) != 255
        if preds.shape[2:] != masks.shape[1:]:
            preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

        # Main cross-entropy and Dice loss.
        ce_loss = self._ce_loss(preds, masks, valid)
        dice_loss = self._dice_loss(preds, masks, valid)
        total = ce_loss + dice_loss

        # Auxiliary cross-entropy loss. Match ce_loss dtype so adding to total succeeds under AMP.
        aux_loss = torch.tensor(0.0, device=preds.device, dtype=ce_loss.dtype)
        if aux_logits is not None:
            if aux_logits.shape[2:] != masks.shape[1:]:
                aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            aux_loss = self._ce_loss(aux_logits, masks, valid) * 0.4
            total += aux_loss

        loss_items = {"ce_loss": ce_loss.detach(), "dice_loss": dice_loss.detach(), "aux_loss": aux_loss.detach()}
        return total * preds.shape[0], loss_items
