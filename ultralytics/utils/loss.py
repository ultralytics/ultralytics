# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.

    Args:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma=2.0, alpha=0.75):
        """Initialize the VarifocalLoss class."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score, gt_score, label):
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Args:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float | list): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma=1.5, alpha=0.25):
        """Initialize FocalLoss class with no parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred, label):
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

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
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

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets by converting to tensor format and scaling coordinates."""
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

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class MultiPosCrossEntropyLoss(nn.Module):
    """Criterion class for computing cross entropy loss."""

    def __init__(self, gamma1=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma1
        self.reduction = reduction

    def forward(self, dist_list, target_list):
        """Calculate Cross entropy of pairs for Reid."""
        assert isinstance(dist_list, list) and isinstance(target_list, list)
        assert len(dist_list) == len(target_list), "List lengths must match"

        losses = []
        for dist, label in zip(dist_list, target_list):
            if dist.numel() == 0:
                continue

            pos_inds = label == 1
            neg_inds = label == 0
            pred_pos = dist * pos_inds.float()
            pred_neg = dist * neg_inds.float()

            pred_pos[neg_inds] = pred_pos[neg_inds] + float("inf")  # mask out negatives
            pred_neg[pos_inds] = pred_neg[pos_inds] + float("-inf")  # mask out positives

            _pos_expand = torch.repeat_interleave(pred_pos, dist.shape[1], dim=1)
            _neg_expand = pred_neg.repeat(1, dist.shape[1])

            x = F.pad((_neg_expand - _pos_expand), (0, 1), value=0)  # shape [N, M*M + 1]
            loss = torch.logsumexp(x, dim=1)  # shape [N]

            losses.append(loss.mean())

        if len(losses) == 0:
            return torch.tensor(0.0, device=dist_list[0].device, dtype=dist_list[0].dtype)

        return self.gamma * torch.stack(losses).mean()

    def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
        """Apply weights and reduce the loss."""
        if weight is not None:
            loss = loss * weight

        if avg_factor is None:
            if reduction == "mean":
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()
        else:
            if reduction == "mean":
                loss = loss.sum() / avg_factor
            elif reduction != "none":
                raise ValueError(f"avg_factor cannot be used with reduction={reduction}")

        return loss


class L2loss(nn.Module):
    """Criterion class for computing auxiliary L2 Loss."""

    def __init__(self, gamma2=1.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma2
        self.reduction = reduction

    def forward(self, pred_list, target_list):
        """Calculate L2 auxiliary loss using cosine distance for Reid."""
        assert isinstance(pred_list, list) and isinstance(target_list, list)
        assert len(pred_list) == len(target_list), "Lists must be of same length"

        losses = []
        for pred, target in zip(pred_list, target_list):
            assert pred.shape == target.shape, "Embedding shapes must match"
            if pred.numel() == 0:
                continue  # ignore empty samples
            loss = F.mse_loss(pred, target, reduction="mean")
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=pred_list[0].device, dtype=pred_list[0].dtype)

        return self.gamma * torch.stack(losses).mean()


class v8DetectionReidLoss:
    """Criterion class for computing training losses for YOLOv8 object detection and Reid."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
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

        self.is_train = None

        self.num_cam = 4

        self.iou_thresh1 = 0.65
        self.iou_thresh2 = 0.2
        self.nb_samples = 100
        self.gamma1 = 0.25
        self.gamma2 = 1.0

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        self.embd_loss = MultiPosCrossEntropyLoss(self.gamma1)
        self.l2aux = L2loss(self.gamma2)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets by converting to tensor format and scaling coordinates."""
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

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def sampler(self, batch, gt_bboxes, batch_idx, preds, strides, im="key"):
        B4, N = preds.shape[:2]  # B4 = Batch_size * 4
        B = B4 // self.num_cam
        preds = preds * strides
        batch_samples = []

        for b in range(B):
            sample_list = []
            for cam_id in range(self.num_cam):
                idx = b * self.num_cam + cam_id
                pred = preds[idx]

                gt_box = gt_bboxes[idx].to(self.device)
                if im == "key":
                    gt_ids = (
                        torch.cat(batch["key_frames"]["ins_ids"], dim=0)
                        .squeeze(1)
                        .long()
                        .to(self.device)[(batch_idx == idx)]
                        .to(self.device)
                    )

                    if gt_box.numel() == 0:
                        positive_idxs = torch.full((self.nb_samples, 2), -1, dtype=torch.long, device=self.device)
                        positive_list.append(positive_idxs)
                        continue

                    ious = bbox_iou(pred.unsqueeze(1), gt_box.unsqueeze(0), xywh=False, CIoU=True)[:, :, 0]  # (N, G)

                    mask = ious > self.iou_thresh1
                    ious_thresh = torch.where(mask, ious, torch.tensor(-1.0, device=ious.device))

                    max_ious, gt_indices = ious_thresh.max(dim=1)
                    pred_indices = torch.nonzero(max_ious > -1, as_tuple=False).squeeze(1)

                    if pred_indices.numel() == 0:
                        positive_idxs = torch.full((self.nb_samples, 2), -1, dtype=torch.long, device=self.device)
                    else:
                        positive_idxs = torch.stack([pred_indices, gt_ids[gt_indices[pred_indices]]], dim=1)

                        # Adjustment à nb_samples
                        n_pos = positive_idxs.shape[0]
                        if n_pos > self.nb_samples:
                            indices = torch.randperm(n_pos, device=self.device)[: self.nb_samples]
                            positive_idxs = positive_idxs[indices]
                        elif n_pos < self.nb_samples:
                            pad = torch.full((self.nb_samples - n_pos, 2), -1, dtype=torch.long, device=self.device)
                            positive_idxs = torch.cat([positive_idxs, pad], dim=0)
                    sample_list.append(positive_idxs)

                elif im == "ref":
                    gt_ids = (
                        torch.cat(batch["ref_frames"]["ins_ids"], dim=0)
                        .squeeze(1)
                        .long()
                        .to(self.device)[(batch_idx == idx)]
                        .to(self.device)
                    )

                    ious = bbox_iou(pred.unsqueeze(1), gt_box.unsqueeze(0), xywh=False, CIoU=True)[:, :, 0]

                    if gt_box.numel() == 0:
                        neg_sample = torch.randperm(pred.shape[0], device=self.device)
                        neg_sample = (
                            neg_sample[:num_select]
                            if pred.shape[0] >= num_select
                            else torch.cat(
                                [
                                    neg_sample,
                                    torch.full((num_select - pred.shape[0],), -1, device=device, dtype=torch.long),
                                ]
                            )
                        )

                        pad_triplets = torch.full(
                            (num_select, 3), -1, dtype=torch.long, device=self.device
                        )  # pour les positifs absents
                        all_samples = torch.cat([pad_triplets, neg_triplets], dim=0)  # (2*num_select, 3)
                        sample_list.append(all_samples)
                        continue

                    max_ious, gt_indices = ious.max(dim=1)
                    pos_mask = max_ious > self.iou_thresh1
                    pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)

                    neg_mask = (ious < self.iou_thresh2).all(dim=1)
                    neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

                    def sample_indices(src, k):
                        if src.numel() == 0:
                            return torch.full((k,), -1, dtype=torch.long, device=self.device)
                        if src.numel() >= k:
                            return src[torch.randperm(src.numel(), device=self.device)[:k]]
                        else:
                            pad = torch.full((k - src.numel(),), -1, dtype=torch.long, device=self.device)
                            return torch.cat([src, pad], dim=0)

                    pos_indices = sample_indices(pos_indices, self.nb_samples)
                    neg_indices = sample_indices(neg_indices, self.nb_samples)

                    pos_samples = torch.full((self.nb_samples, 2), -1, dtype=torch.long, device=self.device)
                    neg_samples = torch.full((self.nb_samples, 2), -1, dtype=torch.long, device=self.device)

                    valid_pos = pos_indices != -1
                    if valid_pos.any():
                        pos_samples[valid_pos, 0] = pos_indices[valid_pos]
                        pos_samples[valid_pos, 1] = gt_ids[gt_indices[pos_indices[valid_pos]]]

                    valid_neg = neg_indices != -1
                    neg_samples[valid_neg, 0] = neg_indices[valid_neg]

                    all_samples = torch.cat([pos_samples, neg_samples], dim=0)
                    sample_list.append(all_samples)

            batch_samples.append(torch.cat(sample_list, dim=0))

        return batch_samples

    def get_match_target(self, key_positive_samples, ref_samples, features):
        similarity_matrix_list = []
        target_matrix_list = []
        dist_matrix_list = []

        nb_cams = self.num_cam
        batch_size = len(key_positive_samples)
        self.nb_samples * nb_cams
        self.nb_samples * 2 * nb_cams  # 2x car positifs + négatifs
        for b in range(batch_size):
            key_samp = key_positive_samples[b]  # [4 * nb_samples, 2]
            ref_samp = ref_samples[b]  # [4 * 2 * nb_samples, 2]

            key_idx, key_ids = key_samp[:, 0].long(), key_samp[:, 1]
            ref_idx, ref_ids = ref_samp[:, 0].long(), ref_samp[:, 1]

            # Filtrage des indices valides
            valid_key_mask = key_idx != -1
            valid_ref_mask = ref_idx != -1

            if valid_key_mask.sum() == 0 or valid_ref_mask.sum() == 0:
                similarity_matrix_list.append(torch.zeros((0, 0), device=self.device))
                dist_matrix_list.append(torch.zeros((0, 0), device=self.device))
                target_matrix_list.append(torch.zeros((0, 0), device=self.device))
                continue

            valid_key_ids = key_ids[valid_key_mask]
            valid_ref_ids = ref_ids[valid_ref_mask]
            target_matrix = (valid_key_ids.unsqueeze(1) == valid_ref_ids.unsqueeze(0)).float()
            target_matrix_list.append(target_matrix)

            feat_key_all, feat_ref_all = [], []

            for cam in range(nb_cams):
                # Index range par caméra
                k_slice = slice(cam * self.nb_samples, (cam + 1) * self.nb_samples)
                r_slice = slice(cam * self.nb_samples * 2, (cam + 1) * self.nb_samples * 2)

                k_mask = key_idx[k_slice] != -1
                r_mask = ref_idx[r_slice] != -1

                if k_mask.sum() == 0 and r_mask.sum() == 0:
                    continue

                k_ids = key_idx[k_slice][k_mask]
                r_ids = ref_idx[r_slice][r_mask]

                # Accès aux features (clé/ref séparés dans `features`)
                feat_key = features[b * nb_cams + cam, k_ids]  # [N_k_cam, C]
                feat_ref = features[b * nb_cams + batch_size * nb_cams + cam, r_ids]  # [N_r_cam, C]
                # feat_key = features[b + cam*batch_size, k_ids]  # [N_k_cam, C]
                # feat_ref = features[b + cam*batch_size + batch_size * nb_cams, r_ids]

                feat_key_all.append(feat_key)
                feat_ref_all.append(feat_ref)

            dist_matrix = torch.mm(
                torch.cat(feat_key_all, dim=0), torch.cat(feat_ref_all, dim=0).t()
            )  # Distance matrix

            # Concaténer tous les features valides
            feat_key_all = F.normalize(torch.cat(feat_key_all, dim=0), p=2, dim=1)
            feat_ref_all = F.normalize(torch.cat(feat_ref_all, dim=0), p=2, dim=1)

            similarity_matrix = torch.mm(feat_key_all, feat_ref_all.t())  # Cosine similarity

            dist_matrix_list.append(dist_matrix)
            similarity_matrix_list.append(similarity_matrix)

        return similarity_matrix_list, dist_matrix_list, target_matrix_list

    def get_matchings(self, samples1, samples2, features):
        similarity_matrix_list = []
        target_matrix_list = []
        dist_matrix_list = []

        nb_cams = self.num_cam
        batch_size = len(samples1)
        # print(samples1)
        # print(samples2)
        for b in range(batch_size):
            # print("Batch ", b)
            samp1 = samples1[b]  # [4 * nb_samples, 2]
            samp2 = samples2[b]  # [4 * 2 * nb_samples, 2]

            samp1_idx, samp1_ids = samp1[:, 0].long(), samp1[:, 1]
            samp2_idx, samp2_ids = samp2[:, 0].long(), samp2[:, 1]

            # Filtrage des indices valides
            valid_samp1_mask = samp1_idx != -1
            valid_samp2_mask = samp2_idx != -1

            if valid_samp1_mask.sum() == 0 or valid_samp2_mask.sum() == 0:
                similarity_matrix_list.append(torch.zeros((0, 0), device=self.device))
                dist_matrix_list.append(torch.zeros((0, 0), device=self.device))
                target_matrix_list.append(torch.zeros((0, 0), device=self.device))
                continue

            valid_samp1_ids = samp1_ids[valid_samp1_mask]
            valid_samp2_ids = samp2_ids[valid_samp2_mask]
            target_matrix = (valid_samp1_ids.unsqueeze(1) == valid_samp2_ids.unsqueeze(0)).float()
            target_matrix_list.append(target_matrix)

            feat_samp1_all, feat_samp2_all = [], []

            for cam in range(nb_cams):
                # print("Cam ", cam)
                # Index range par caméra
                samp1_slice = slice(cam * self.nb_samples, (cam + 1) * self.nb_samples)
                if self.is_train:
                    samp2_slice = slice(cam * self.nb_samples * 2, (cam + 1) * self.nb_samples * 2)
                else:
                    samp2_slice = slice(cam * self.nb_samples, (cam + 1) * self.nb_samples)

                samp1_mask = samp1_idx[samp1_slice] != -1
                samp2_mask = samp2_idx[samp2_slice] != -1
                # print(k_mask)
                # print(r_mask)

                if samp1_mask.sum() == 0 and samp2_mask.sum() == 0:
                    continue

                s1_ids = samp1_idx[samp1_slice][samp1_mask]
                s2_ids = samp2_idx[samp2_slice][samp2_mask]
                # print("key idx", k_ids)
                # print("ref idx", r_ids)
                # Accès aux features (clé/ref séparés dans `features`)
                feat_s1 = features[b * nb_cams + cam, s1_ids]  # [N_k_cam, C]
                if self.is_train:
                    feat_s2 = features[b * nb_cams + batch_size * nb_cams + cam, s2_ids]  # [N_r_cam, C]

                else:
                    feat_s2 = feat_s1

                feat_samp1_all.append(feat_s1)
                feat_samp2_all.append(feat_s2)

            dist_matrix = torch.mm(
                torch.cat(feat_samp1_all, dim=0), torch.cat(feat_samp2_all, dim=0).t()
            )  # Distance matrix

            # Concaténer tous les features valides et normaliser pour la cosine
            feat_samp1_all = F.normalize(torch.cat(feat_samp1_all, dim=0), p=2, dim=1)
            feat_samp2_all = F.normalize(torch.cat(feat_samp2_all, dim=0), p=2, dim=1)

            similarity_matrix = torch.mm(feat_samp1_all, feat_samp2_all.t())  # Cosine similarity

            dist_matrix_list.append(dist_matrix)
            similarity_matrix_list.append(similarity_matrix)

        return similarity_matrix_list, dist_matrix_list, target_matrix_list

    def __call__(self, preds_key, preds_ref, features, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, reid
        key_feats = preds_key[1] if isinstance(preds_key, tuple) else preds_key
        ref_feats = preds_ref[1] if isinstance(preds_ref, tuple) else preds_ref

        key_pred_distri, key_pred_scores = (
            torch.cat([x.view(key_feats[0].shape[0], self.no, -1) for x in key_feats], dim=2).split(
                (self.reg_max * 4, self.nc), dim=1
            )
            if key_feats is not None
            else (None, None)
        )

        ref_pred_distri, ref_pred_scores = (
            torch.cat([x.view(ref_feats[0].shape[0], self.no, -1) for x in ref_feats], dim=2).split(
                (self.reg_max * 4, self.nc), dim=1
            )
            if ref_feats is not None
            else (None, None)
        )

        key_pred_scores = key_pred_scores.permute(0, 2, 1).contiguous() if key_pred_scores is not None else None
        key_pred_distri = key_pred_distri.permute(0, 2, 1).contiguous() if key_pred_distri is not None else None
        ref_pred_scores = ref_pred_scores.permute(0, 2, 1).contiguous() if ref_pred_scores is not None else None
        ref_pred_distri = ref_pred_distri.permute(0, 2, 1).contiguous() if ref_pred_distri is not None else None

        dtype = key_pred_scores.dtype
        batch_size = key_pred_scores.shape[0]
        imgsz = (
            torch.tensor(key_feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(key_feats, self.stride, 0.5)

        # Targets Key Image
        key_targets = torch.cat(
            (
                batch["key_frames"]["batch_idx"].view(-1, 1),
                batch["key_frames"]["cls"].view(-1, 1),
                batch["key_frames"]["bboxes"],
            ),
            1,
        )
        key_batch_idx = key_targets[:, 0].long()
        key_targets = self.preprocess(key_targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        key_gt_labels, key_gt_bboxes = key_targets.split((1, 4), 2)  # cls, xyxy
        key_mask_gt = key_gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes Key
        key_pred_bboxes = self.bbox_decode(anchor_points, key_pred_distri)  # xyxy, (b, h*w, 4)

        # Targets Reference Image
        if self.is_train:
            ref_targets = torch.cat(
                (
                    batch["ref_frames"]["batch_idx"].view(-1, 1),
                    batch["ref_frames"]["cls"].view(-1, 1),
                    batch["ref_frames"]["bboxes"],
                ),
                1,
            )
            ref_batch_idx = ref_targets[:, 0].long()
            ref_targets = self.preprocess(ref_targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            ref_gt_labels, ref_gt_bboxes = ref_targets.split((1, 4), 2)  # cls, xyxy
            ref_gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            # Pboxes Ref
            ref_pred_bboxes = self.bbox_decode(anchor_points, ref_pred_distri)  # xyxy, (b, h*w, 4)

        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        # For other losses -> Perform on key image
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            key_pred_scores.detach().sigmoid(),
            (key_pred_bboxes.detach() * stride_tensor).type(key_gt_bboxes.dtype),
            anchor_points * stride_tensor,
            key_gt_labels,
            key_gt_bboxes,
            key_mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(key_pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                key_pred_distri,
                key_pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        # Sampling des positives image key et positives/négatives image ref
        # key_positive_samples = self.sampler(batch, key_gt_bboxes, key_batch_idx, key_pred_bboxes, stride_tensor, 'key')
        # ref_samples = self.sampler(batch, ref_gt_bboxes, ref_batch_idx, ref_pred_bboxes, stride_tensor, 'ref')
        # Calcul similarity matrix et Target matrix
        # cosine_similarity, distance, asso_target = self.get_match_target(key_positive_samples, ref_samples, features)

        # Sampling des positives image key et positives/négatives image ref
        # Calcul similarity matrix et Target matrix
        key_samples = self.sampler(batch, key_gt_bboxes, key_batch_idx, key_pred_bboxes, stride_tensor, "key")
        if self.is_train:
            ref_samples = self.sampler(batch, ref_gt_bboxes, ref_batch_idx, ref_pred_bboxes, stride_tensor, "ref")
            cosine_similarity, distance, asso_target = self.get_matchings(key_samples, ref_samples, features)
        else:
            cosine_similarity, distance, asso_target = self.get_matchings(key_samples, key_samples, features)

        # Calcul Reid loss -> L2 aux + Cross-Entropy
        l2 = self.l2aux(cosine_similarity, asso_target)
        eloss = self.embd_loss(distance, asso_target)

        # Assemble losses
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] = l2 + eloss

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

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

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

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
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

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

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
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

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

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
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

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
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

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
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

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


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds, batch):
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats):
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds, batch):
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
