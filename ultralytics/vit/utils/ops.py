# TODO: license

import torch
import torch.nn.functional as F
import torcj.nn as nn
from scipy.optimize import linear_sum_assignment

from ultralytics.yolo.utils.loss import GIoULoss
from ultralytics.yolo.utils.ops import xywh2xyxy


class HungarianMatcher(nn.Layer):

    def __init__(self,
                 matcher_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 1,
                     'dice': 1},
                 use_focal_loss=False,
                 with_mask=False,
                 num_sample_points=12544,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super().__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self, boxes, logits, gt_bbox, gt_class, masks=None, gt_mask=None):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor|None): [b, query, h, w]
            gt_mask (List(Tensor)): list[[n, H, W]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = [len(a) for a in gt_class]
        if sum(num_gts) == 0:
            return [(torch.to_tensor([], dtype=torch.int64), torch.to_tensor([], dtype=torch.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        logits = logits.detach()
        out_prob = F.sigmoid(logits.flatten(0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.detach().flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.concat(gt_class).flatten()
        tgt_bbox = torch.concat(gt_bbox)

        # Compute the classification cost
        out_prob = torch.gather(out_prob, tgt_ids, axis=1)
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        # Compute the L1 cost between boxes
        cost_bbox = (out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(xywh2xyxy(out_bbox.unsqueeze(1)), xywh2xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + \
            self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        # Compute the mask cost and dice cost
        if self.with_mask:
            assert (masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`')
            # all masks share the same set of points for efficient matching
            sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
            sample_points = 2.0 * sample_points - 1.0

            out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
            out_mask = out_mask.flatten(0, 1)

            tgt_mask = torch.concat(gt_mask).unsqueeze(1)
            sample_points = torch.concat([a.tile([b, 1, 1, 1]) for a, b in zip(sample_points, num_gts) if b > 0])
            tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])

            with torch.amp.auto_cast(enable=False):
                # binary cross entropy cost
                pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask,
                                                                   torch.ones_like(out_mask),
                                                                   reduction='none')
                neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask,
                                                                   torch.zeros_like(out_mask),
                                                                   reduction='none')
                cost_mask = torch.matmul(pos_cost_mask, tgt_mask, transpose_y=True) + torch.matmul(
                    neg_cost_mask, 1 - tgt_mask, transpose_y=True)
                cost_mask /= self.num_sample_points

                # dice cost
                out_mask = F.sigmoid(out_mask)
                numerator = 2 * torch.matmul(out_mask, tgt_mask, transpose_y=True)
                denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
                cost_dice = 1 - (numerator + 1) / (denominator + 1)

                C = C + self.matcher_coeff['mask'] * cost_mask + \
                    self.matcher_coeff['dice'] * cost_dice

        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]
        sizes = [a.shape[0] for a in gt_bbox]
        indices = [linear_sum_assignment(c.split(sizes, -1)[i].numpy()) for i, c in enumerate(C)]
        return [(torch.to_tensor(i, dtype=torch.int64), torch.to_tensor(j, dtype=torch.int64)) for i, j in indices]
