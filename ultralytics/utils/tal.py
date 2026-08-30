# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywh2xyxy, xywhr2xyxyxyxy, xyxy2xywh
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        topk2 (int): Secondary topk value for additional filtering.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        stride (list): List of stride values for different feature levels.
        stride_val (int): The stride value used for select_candidates_in_gts.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2=None,
    ):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            stride (list, optional): List of stride values for different feature levels.
            eps (float, optional): A small value to prevent division by zero.
            topk2 (int, optional): Secondary topk value for additional filtering.
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = self.stride[0] * 2 if len(self.stride) > 1 else self.stride[0]
        self.eps = eps
        self.hard_target = False  # 0/1 cls target for positives instead of the alignment-graded soft label
        self.o2f_k = 0  # o2f: number of ambiguous soft-labeled anchors per GT (0=disabled)
        self.o2f_T = 0.0  # o2f: positive degree of ambiguous anchors, annealed per epoch by E2ELoss.update()
        self.dup_sup = False  # dup suppression: stash certain/sibling pairs for the margin loss
        self._dup_pairs = None
        self.monitor = False  # collect per-batch assignment stats when enabled (see callbacks/tal_monitor.py)
        self.mon = self._reset_mon()

    @staticmethod
    def _reset_mon() -> dict:
        """Return an empty monitor accumulator for TAL assignment stats."""
        return {
            "n_gt": 0,
            "zero_pos": 0,
            "conflict": 0,
            "pos_pre": 0,
            "pos_post": 0,
            "align": 0.0,
            "iou": 0.0,
            "score": 0.0,
            "soft": 0.0,
            "tss": 0.0,
            "by_size": {
                b: {"n_gt": 0, "zero_pos": 0, "iou": 0.0, "score": 0.0, "soft": 0.0}
                for b in ("small", "medium", "large")
            },
        }

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        # Recover outside the except block: exiting it drops e.__traceback__, releasing the failed attempt's GPU
        # intermediates back to the allocator so the copy-back below can succeed
        LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
        result = self._forward(*(t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)))
        return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos_pre, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos_pre, overlaps, self.n_max_boxes, align_metric
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        if self.dup_sup and self.topk2 != self.topk:
            # use the pre-narrowing candidate pool (topk per GT) — post-narrowing mask_pos has 1 anchor/GT
            self._stash_dup_pairs(align_metric, pd_scores, gt_labels, mask_pos_pre, mask_gt)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        if not self.hard_target:  # hard_target keeps the 0/1 label get_targets already produced
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
            target_scores = target_scores * norm_align_metric

        if self.o2f_k and self.topk2 != self.topk:
            # always active when o2f is on: at T=0 ambiguous anchors get explicit 0 labels (hard negatives);
            # skipping this block would leave them as full TAL positives via topk2=1+K
            target_scores, fg_mask = self.apply_o2f_soft_labels(
                target_scores, fg_mask, align_metric, pd_scores, gt_labels, mask_gt
            )

        if self.monitor:
            self._update_mon(
                mask_pos_pre,
                mask_pos,
                align_metric,
                overlaps,
                pd_scores,
                gt_labels,
                gt_bboxes,
                mask_gt,
                pos_align_metrics,
                pos_overlaps,
                target_scores.sum(),
            )

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    @torch.no_grad()
    def _update_mon(
        self,
        mask_pos_pre,
        mask_pos,
        align_metric,
        overlaps,
        pd_scores,
        gt_labels,
        gt_bboxes,
        mask_gt,
        pos_align,
        pos_over,
        tss,
    ):
        """Accumulate per-GT assignment stats into self.mon (used by callbacks/tal_monitor.py)."""
        a = self.mon
        mg = mask_gt.bool()  # (b, n_gt, 1)
        n_gt = int(mg.sum())
        if n_gt == 0:
            return
        pre = (mask_pos_pre * mg).sum(-1)
        post = (mask_pos * mg).sum(-1)
        chosen = (align_metric * mg).argmax(-1)  # top-align positive anchor per GT
        has_pos = (post > 0) & mg.squeeze(-1)
        claims = mask_pos_pre.sum(-2, keepdim=True)  # (b, 1, anchors)
        conflict = ((mask_pos_pre * (claims > 1)) * mg).sum(-1) > 0

        bidx = torch.arange(pre.shape[0], device=pre.device).unsqueeze(1)
        gidx = torch.arange(pre.shape[1], device=pre.device)
        ch_align = align_metric[bidx, gidx, chosen]
        ch_iou = overlaps[bidx, gidx, chosen]
        nc = pd_scores.shape[-1]
        ch_score = pd_scores.gather(1, chosen.unsqueeze(-1).expand(-1, -1, nc)).gather(2, gt_labels.long()).squeeze(-1)
        ch_soft = ch_align * pos_over.squeeze(-1) / (pos_align.squeeze(-1) + self.eps)

        a["n_gt"] += n_gt
        a["zero_pos"] += int((mg.squeeze(-1) & ~has_pos).sum())
        a["conflict"] += int((conflict & mg.squeeze(-1)).sum())
        a["pos_pre"] += float(pre.sum())
        a["pos_post"] += float(post.sum())
        sel = has_pos
        a["align"] += float(ch_align[sel].sum())
        a["iou"] += float(ch_iou[sel].sum())
        a["score"] += float(ch_score[sel].sum())
        a["soft"] += float(ch_soft[sel].sum())
        a["tss"] += float(tss)

        areas = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])
        for name, (lo, hi) in {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e18)}.items():
            m = mg.squeeze(-1) & (areas >= lo) & (areas < hi)
            bs = a["by_size"][name]
            bs["n_gt"] += int(m.sum())
            bs["zero_pos"] += int((m & ~has_pos).sum())
            bs["iou"] += float(ch_iou[m & sel].sum())
            bs["score"] += float(ch_score[m & sel].sum())
            bs["soft"] += float(ch_soft[m & sel].sum())

    @torch.no_grad()
    def apply_o2f_soft_labels(self, target_scores, fg_mask, align_metric, pd_scores, gt_labels, mask_gt):
        """Assign one-to-few (o2f) soft classification labels to ambiguous anchors.

        For each GT the top-align anchor is the certain positive (unchanged); the next `o2f_k` anchors are
        ambiguous: they get a cls-only soft label t = (p_i / max p_k) * T (T annealed per epoch from
        o2f_tmax to o2f_tmin, see E2ELoss.update) and are excluded from box regression. Early in training
        they act mostly positive (representation learning); late, mostly negative (duplicate removal).
        Reference: One-to-Few Label Assignment for End-to-End Dense Detection (CVPR 2023).

        Args:
            target_scores (torch.Tensor): Soft target scores, shape (bs, num_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask, shape (bs, num_anchors).
            align_metric (torch.Tensor): Masked alignment metric, shape (bs, n_max_boxes, num_anchors).
            pd_scores (torch.Tensor): Predicted class probabilities, shape (bs, num_anchors, num_classes).
            gt_labels (torch.Tensor): Ground truth labels, shape (bs, n_max_boxes, 1).
            mask_gt (torch.Tensor): Valid GT mask, shape (bs, n_max_boxes, 1).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Updated target_scores and fg_mask.
        """
        k = self.o2f_k
        bs, n_gt = align_metric.shape[:2]
        topv, topi = align_metric.topk(k + 1, dim=-1)  # ordered certain + ambiguous per GT
        amb = topi[:, :, 1:]  # (b, n_gt, k) ambiguous anchor indices
        valid = (topv[:, :, 1:] > self.eps) & mask_gt.bool()
        if not valid.any():
            return target_scores, fg_mask
        flat_idx = topi.reshape(bs, -1)  # (b, n_gt*(k+1))
        p_sel = pd_scores.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, pd_scores.shape[-1]))
        flat_lab = gt_labels.long().view(bs, n_gt, 1, 1).expand(-1, -1, k + 1, -1).reshape(bs, -1, 1)
        p_sel = p_sel.gather(2, flat_lab).reshape(bs, n_gt, k + 1)  # p at GT class per selected anchor
        t_amb = p_sel[:, :, 1:] / (p_sel.amax(-1, keepdim=True) + self.eps) * self.o2f_T  # (b, n_gt, k)
        lab = gt_labels.long().expand(-1, -1, k)
        bidx = torch.arange(bs, device=amb.device).view(bs, 1, 1).expand(-1, n_gt, k)
        target_scores[bidx[valid], amb[valid], lab[valid]] = t_amb[valid].to(target_scores.dtype)
        fg_mask[bidx[valid], amb[valid]] = 0  # ambiguous anchors: cls-only, no box loss
        return target_scores, fg_mask

    @torch.no_grad()
    def _stash_dup_pairs(self, align_metric, pd_scores, gt_labels, mask_pos, mask_gt):
        """Stash (certain, best-sibling) anchor indices per GT for the duplicate-suppression margin loss.

        The certain anchor is the top-align positive; the sibling is the highest-scoring other anchor in the
        GT's candidate pool. Only indices are captured (assigner inputs are detached); the loss applies the
        prob-space margin on raw logits so gradients flow.

        Args:
            align_metric (torch.Tensor): Masked alignment metric, shape (bs, n_max_boxes, num_anchors).
            pd_scores (torch.Tensor): Predicted class probabilities, shape (bs, num_anchors, num_classes).
            gt_labels (torch.Tensor): Ground truth labels, shape (bs, n_max_boxes, 1).
            mask_pos (torch.Tensor): Positive mask, shape (bs, n_max_boxes, num_anchors).
            mask_gt (torch.Tensor): Valid GT mask, shape (bs, n_max_boxes, 1).
        """
        bs, n_gt, _ = align_metric.shape
        certain = (align_metric * mask_pos).argmax(-1)  # (b, n_gt) top-align positive per GT
        ind = torch.zeros([2, bs, n_gt], dtype=torch.long, device=gt_labels.device)
        ind[0] = torch.arange(end=bs, device=gt_labels.device).view(-1, 1).expand(-1, n_gt)
        ind[1] = gt_labels.squeeze(-1).clamp(min=0)
        scores = torch.zeros_like(align_metric, dtype=torch.float32)  # fp32: pd_scores may be fp16 under AMP
        scores[mask_pos.bool()] = pd_scores[ind[0], :, ind[1]][mask_pos.bool()].float()
        scores.scatter_(-1, certain.unsqueeze(-1), -1.0)  # exclude certain from sibling search
        sib = scores.argmax(-1)  # (b, n_gt) highest-scoring sibling
        valid = (mask_pos.sum(-1) > 1) & mask_gt.squeeze(-1).bool()  # GTs with >1 candidate
        self._dup_pairs = (certain, sib, valid, gt_labels.squeeze(-1).long())

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted vs ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        # A grid with no metric is not a candidate. topk over an all-zero row returns indices 0..topk-1, so a GT that
        # no anchor predicts would otherwise pile its positives onto the first grid cells; the center prior used to
        # mask that away, leaving this test dead, but a candidate region wider than the GT box does not.
        valid = topk_metrics > self.eps
        topk_mask = valid if topk_mask is None else topk_mask & valid

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = topk_mask.to(torch.int8)  # a masked-out pick adds nothing, rather than adding to index 0
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones[:, :, k : k + 1])

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int8,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        target_scores = target_scores * (fg_mask[:, :, None] > 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes, shape (b, n_boxes, 1).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # the smallest stride
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)  # (b, n_boxes, 1, 2) left-top, right-bottom
        return ((xy_centers - lt > eps) & (rb - xy_centers > eps)).all(3)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.
            align_metric (torch.Tensor): Alignment metric for selecting best matches.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # update overlaps
            # (b, n_max_boxes, topk2)
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class PoseTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to anchors on the keypoints instead of the box.

    ER-Pose (https://arxiv.org/abs/2603.08681) replaces the localization term of the task-aligned metric with OKS, so
    that positive selection and the soft classification target both track pose quality rather than box quality. The
    pose loss hands over the decoded predicted and padded ground-truth keypoints, both in pixels, before each call.

    `metric` chooses what grades a (GT, anchor) pair. OKS normalizes the keypoint error by instance area, so on a
    large person it sits near 1 across the whole candidate region and stops separating a good anchor from an
    excellent one: measured on a trained model, the top-1 to top-10 spread of the alignment metric falls to 1.16 on
    large instances against 1.23 for box CIoU, and the large-instance recall at OKS 0.9 drops with it. 'rect' grades
    the CIoU of the keypoint bounding rects instead, which is scale-invariant and holds that spread at 1.49 while
    staying keypoint-driven, so the box branch can still go away.

    Attributes:
        metric (str): Localization term of the task-aligned metric, 'oks' or 'rect'.
        sigmas (torch.Tensor): Per-keypoint OKS normalization constants, shape (num_keypoints,), 'oks' only.
        pd_kpts (torch.Tensor): Decoded predicted keypoints, shape (bs, num_total_anchors, num_keypoints, 2).
        gt_kpts (torch.Tensor): Ground truth keypoints, shape (bs, n_max_boxes, num_keypoints, 2 or 3).
        candidates (torch.Tensor): Candidate mask of the running assignment, shape (bs, n_max_boxes, h*w), 'oks' only.
        oks (torch.Tensor): OKS of every (GT, anchor) pair of the running assignment, shape (bs, n_max_boxes, h*w).
        kpt_expand (float): Candidate region as a multiple of the visible keypoints' bounding rect, 0 to fall back
            on the base class's ground-truth box center prior.
    """

    sigmas = pd_kpts = gt_kpts = candidates = oks = None
    kpt_expand = 0.0
    metric = "oks"

    @staticmethod
    def _kpt_rect(kpts: torch.Tensor, expand: float = 1.0) -> torch.Tensor:
        """Return the axis-aligned bounding rect of the visible keypoints, grown by `expand`.

        Args:
            kpts (torch.Tensor): Keypoints with shape (..., num_keypoints, 2 or 3). A third channel is read as
                visibility and invisible keypoints are left out of the rect.
            expand (float): Multiplier on the rect's half-extent.

        Returns:
            (torch.Tensor): Bounding rects in xyxy with shape (..., 4). A rect flat on one axis grows along the
                other; one with no visible keypoint collapses to a point at the origin.
        """
        xy = kpts[..., :2]
        vis = kpts[..., 2:] != 0 if kpts.shape[-1] == 3 else torch.ones_like(xy[..., :1], dtype=torch.bool)
        hi = xy.masked_fill(~vis, -torch.inf).amax(-2)
        lo = xy.masked_fill(~vis, torch.inf).amin(-2)
        half = ((hi - lo) / 2 * expand).nan_to_num_(neginf=0)  # no visible keypoint: collapses to a point
        half = torch.where(half > 0, half, half.amax(-1, keepdim=True))  # a flat rect grows along its other axis
        center = ((hi + lo) / 2).nan_to_num_()
        return torch.cat((center - half, center + half), -1)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """Take candidates from the visible keypoints' bounding rect grown by `kpt_expand`, not the GT box.

        Algorithm 1 of the paper ranks every grid per GT, which this reaches as the region grows past the image.
        Keeping the region keypoint-driven is what lets the box branch go away entirely, so nothing here falls back
        to `gt_bboxes`: the base class floors a point-sized rect to a stride, and a ground truth with no visible
        keypoint scores 0 everywhere and is dropped either way.
        """
        if not self.kpt_expand:
            return super().select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt, eps)
        rects = self._kpt_rect(self.gt_kpts, self.kpt_expand)
        return super().select_candidates_in_gts(xy_centers, rects, mask_gt, eps)

    def forward(self, *args, **kwargs):
        """Assign, then drop the keypoints so they stop holding memory through backward and the optimizer step."""
        try:
            return super().forward(*args, **kwargs)
        finally:
            self.pd_kpts = self.gt_kpts = self.candidates = self.oks = None

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Grade every (GT, anchor) pair on the keypoints, then let the base class build the alignment metric.

        'rect' substitutes keypoint bounding rects for the boxes and reuses the base class's CIoU, so the per-pair
        gather it already does is the entire working set. 'oks' has to precompute instead, because the base class
        hands `iou_calculation` boxes rather than keypoints; it accumulates one keypoint at a time so the working set
        stays at (bs, n_max_boxes, h*w) rather than the (num_pairs, num_keypoints, 3) gathers a per-pair formulation
        needs. On a crowded 640 batch that is 16x less memory when `kpt_expand` opens the candidate region up.
        """
        device = pd_bboxes.device  # the assigner OOM fallback re-runs on CPU, so follow the boxes
        pd_kpts, gt_kpts = (t.to(device) for t in (self.pd_kpts, self.gt_kpts))
        if self.metric == "rect":
            # The keypoints are held in fp32 because pixel-scale areas overflow fp16, so the rects and their CIoU are
            # fp32 too, and the base class sizes `overlaps` off them.
            gt_rects = self._kpt_rect(gt_kpts)
            # Under two visible keypoints leaves a rect with no extent, which overlaps nothing and would cost the
            # instance its supervision in select_topk_candidates. Fall back to its box, the one region still known.
            flat = (gt_rects[..., 2:] - gt_rects[..., :2]).amin(-1, keepdim=True) <= 0
            gt_rects = torch.where(flat, gt_bboxes.to(gt_rects.dtype), gt_rects)
            return super().get_box_metrics(pd_scores, self._kpt_rect(pd_kpts), gt_labels, gt_rects, mask_gt)
        sigmas = self.sigmas.to(device)
        area = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).prod(-1, keepdim=True)  # s^2, (bs, n_max_boxes, 1)
        vis = gt_kpts[..., 2] != 0 if gt_kpts.shape[-1] == 3 else torch.ones_like(gt_kpts[..., 0], dtype=torch.bool)
        oks = pd_kpts.new_zeros(self.bs, self.n_max_boxes, pd_kpts.shape[1])
        for i in range(gt_kpts.shape[2]):
            d = (pd_kpts[:, None, :, i, 0] - gt_kpts[:, :, None, i, 0]).pow(2) + (
                pd_kpts[:, None, :, i, 1] - gt_kpts[:, :, None, i, 1]
            ).pow(2)
            oks += (-d / ((2 * sigmas[i]).pow(2) * (area + self.eps) * 2)).exp() * vis[:, :, i, None]  # from cocoeval
        # A ground truth with no visible keypoint has no OKS, so it scores 0 everywhere and select_topk_candidates
        # drops it: unlike box IoU, this assignment leaves keypoint-free instances unsupervised.
        self.oks = oks / (vis.sum(-1, keepdim=True) + self.eps)
        self.candidates = mask_gt.bool()
        return super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Serve the precomputed OKS of the candidate pairs the base class is grading.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes of the candidate pairs, or their keypoint rects under
                'rect', shape (num_pairs, 4).
            pd_bboxes (torch.Tensor): Predicted boxes of the candidate pairs, or their keypoint rects under 'rect',
                shape (num_pairs, 4).

        Returns:
            (torch.Tensor): Localization scores in [0, 1] for each candidate pair, shape (num_pairs,).
        """
        if self.metric == "rect":
            return super().iou_calculation(gt_bboxes, pd_bboxes)  # CIoU of the rects get_box_metrics handed over
        return self.oks[self.candidates]


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (b, n_boxes, 1).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        gt_bboxes_clone = gt_bboxes.clone()
        wh_mask = gt_bboxes_clone[..., 2:4] < self.stride[0]
        gt_bboxes_clone[..., 2:4] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_clone.dtype, device=gt_bboxes_clone.device),
            gt_bboxes_clone[..., 2:4],
        )

        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes_clone)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5, normalize=False, p3_stride=None):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor = torch.stack((sx, sy), -1).view(-1, 2)
        if i == 0 and p3_stride is not None and stride == 8 and p3_stride != stride:
            # p3_stride: keep the dense stride-8 anchor positions but regress distances in p3_stride pixel units,
            # extending the P3 level's max regressable box size (e.g. 8 -> 16 for medium/large objects)
            anchor = anchor * (stride / p3_stride)
            stride = p3_stride
        if normalize:
            anchor /= torch.tensor([w, h], dtype=dtype, device=device)
        anchor_points.append(anchor)
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def rbox2dist(
    target_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_angle: torch.Tensor,
    dim: int = -1,
    reg_max: int | None = None,
):
    """Transform rotated bounding box (xywh) to distance (ltrb). This is the inverse of dist2rbox.

    Args:
        target_bboxes (torch.Tensor): Target rotated bounding boxes with shape (bs, h*w, 4), format [x, y, w, h].
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        target_angle (torch.Tensor): Target angle with shape (bs, h*w, 1).
        dim (int, optional): Dimension along which to split.
        reg_max (int, optional): Maximum regression value for clamping.

    Returns:
        (torch.Tensor): Rotated distance with shape (bs, h*w, 4), format [l, t, r, b].
    """
    xy, wh = target_bboxes.split(2, dim=dim)
    offset = xy - anchor_points  # (bs, h*w, 2)
    offset_x, offset_y = offset.split(1, dim=dim)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos

    w, h = wh.split(1, dim=dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf

    dist = torch.cat([target_l, target_t, target_r, target_b], dim=dim)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)

    return dist
