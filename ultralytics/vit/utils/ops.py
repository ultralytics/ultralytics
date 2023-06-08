# TODO: license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.yolo.utils.metrics import bbox_iou
from ultralytics.yolo.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):

    def __init__(self,
                 cost_gain=None,
                 use_focal_loss=True,
                 with_mask=False,
                 num_sample_points=12544,
                 alpha=0.25,
                 gamma=2.0):
        """
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'mask': 1, 'dice': 1}
        self.cost_gain = cost_gain
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, bboxes, scores, gt_bboxes, gt_cls, gt_numgts, masks=None, gt_mask=None):
        """
        Args:
            bboxes (Tensor): [b, query, 4]
            scores (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_cls (List(Tensor)): list[[n, 1]]
            masks (Tensor|None): [b, query, h, w]
            gt_mask (List(Tensor)): list[[n, H, W]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, nq, nc = scores.shape

        if sum(gt_numgts) == 0:
            return [(torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        scores = scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(scores) if self.use_focal_loss else F.softmax(scores, dim=-1)
        # [batch_size * num_queries, 4]
        pred_bboxes = bboxes.detach().view(-1, 4)

        # Compute the classification cost
        pred_scores = pred_scores[:, gt_cls]
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (pred_scores ** self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # Compute the L1 cost between boxes
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt)

        # Compute the GIoU cost between boxes, (bs*num_queries, num_gt)
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)

        # Final cost matrix
        C = self.cost_gain['class'] * cost_class + \
            self.cost_gain['bbox'] * cost_bbox + \
            self.cost_gain['giou'] * cost_giou
        # Compute the mask cost and dice cost
        if self.with_mask:
            C += self._cost_mask(bs, gt_numgts, masks, gt_mask)

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_numgts, -1))]
        gt_numgts = [0] + gt_numgts[:-1]
        # (idx for queries, idx for gt)
        return [(torch.tensor(i, dtype=torch.int32), torch.tensor(j, dtype=torch.int32) + gt_numgts[k]) for k, (i, j) in enumerate(indices)]

    def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
        assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
        # all masks share the same set of points for efficient matching
        sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
        sample_points = 2.0 * sample_points - 1.0

        out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
        out_mask = out_mask.flatten(0, 1)

        tgt_mask = torch.cat(gt_mask).unsqueeze(1)
        sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
        tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])

        with torch.cuda.amp.autocast(False):
            # binary cross entropy cost
            pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
            neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
            cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
            cost_mask /= self.num_sample_points

            # dice cost
            out_mask = F.sigmoid(out_mask)
            numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
            denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)

            C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
        return C


def get_cdn_group(targets,
                  num_classes,
                  num_queries,
                  class_embed,
                  num_dn=100,
                  cls_noise_ratio=0.5,
                  box_noise_scale=1.0,
                  training=False):
    """get contrastive denoising training group"""
    if (not training) or num_dn <= 0:
        return None, None, None, None
    num_gts = [len(t) for t in targets['cls']]
    max_nums = max(num_gts)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(targets['cls'])
    gt_cls = torch.full([bs, max_nums], num_classes)  # bs, max_gt_num
    gt_bbox = torch.zeros([bs, max_nums, 4])
    mask_gt = torch.zeros([bs, max_nums])
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            gt_cls[i, :num_gt] = targets['cls'][i].squeeze(-1)
            gt_bbox[i, :num_gt] = targets['bboxes'][i]
            mask_gt[i, :num_gt] = 1
    # each group has positive and negative queries.
    dn_cls = gt_cls.repeat(1, 2 * num_group)  # bs, 2* max_gt_num * num_group
    dn_bbox = gt_bbox.repeat(1, 2 * num_group, 1)  # bs, 2* max_gt_num * num_group, 4
    mask_gt = mask_gt.repeat(1, 2 * num_group)
    # positive and negative mask
    neg_idx = torch.zeros([bs, max_nums * 2], dtype=torch.bool)
    neg_idx[:, max_nums:] = 1
    neg_idx = neg_idx.repeat(1, num_group)  # bs, 2* max_gt_num * num_group
    # contrastive denoising training positive index
    pos_idx = (~neg_idx) * mask_gt  # bs, 2* max_gt_num * num_group
    dn_pos_idx = torch.nonzero(pos_idx)[:, 1]
    dn_pos_idx = torch.split(dn_pos_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_dn = int(max_nums * 2 * num_group)

    if cls_noise_ratio > 0:
        dn_cls = dn_cls.view(-1)
        mask_gt = mask_gt.view(-1)
        # half of bbox prob
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask * mask_gt).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype)

        dn_cls[idx] = new_label
        dn_cls = dn_cls.view(bs, num_dn)
        mask_gt = mask_gt.view(bs, num_dn)

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)

        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 1, 2) * box_noise_scale  # bs, 2* max_gt_num * num_group, 4

        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = inverse_sigmoid(dn_bbox)

    class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])

    dn_cls_embed = class_embed[dn_cls.view(-1)].view(bs, num_dn, -1)  # bs, max_nums * 2 * num_group, 256

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * 2 * i] = True
    dn_meta = {'dn_pos_idx': dn_pos_idx, 'dn_num_group': num_group, 'dn_num_split': [num_dn, num_queries]}

    return dn_cls_embed.to(class_embed.device), dn_bbox.to(class_embed.device), attn_mask.to(
        class_embed.device), dn_meta


def get_cdn_group_(targets,
                  num_classes,
                  num_queries,
                  class_embed,
                  num_dn=100,
                  cls_noise_ratio=0.5,
                  box_noise_scale=1.0,
                  training=False):
    """get contrastive denoising training group"""
    if (not training) or num_dn <= 0:
        return None, None, None, None
    num_gts = targets["num_gts"]
    total_num = sum(num_gts)
    max_nums = max(num_gts)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(targets['num_gts'])
    gt_cls = targets['cls']  # (bs*num, )
    gt_bbox = targets['bboxes']  # bs*num, 4
    b_idx = targets['batch_idx']

    # each group has positive and negative queries.
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1) # (2*num_group*bs*num, )

    num_dn = int(max_nums * 2 * num_group)
    # positive and negative mask
    # (bs*num*num_group, ), the first part as positive sample
    pos_idx = torch.tensor(range(total_num), dtype=torch.long, device=gt_bbox.device).repeat(num_group)
    
    # the second part as positive sample
    neg_idx = pos_idx + num_group * total_num
    # total denoising queries

    if cls_noise_ratio > 0:
        # half of bbox prob
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # randomly put a new one here
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
        dn_bbox = inverse_sigmoid(dn_bbox)

    # class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])
    dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)
    map_indices = torch.tensor([], device=gt_bbox.device)
    if total_num:
        map_indices = torch.cat([torch.tensor(range(num)) for num in num_gts])
        map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)]).long()
    if len(dn_b_idx):
        padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
        padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * 2 * i] = True
    dn_meta = {'dn_pos_idx': pos_idx.cpu().split([n * num_group for n in num_gts]),
               'dn_num_group': num_group, 'dn_num_split': [num_dn, num_queries]}

    return padding_cls.to(class_embed.device), padding_bbox.to(class_embed.device), attn_mask.to(
        class_embed.device), dn_meta


def inverse_sigmoid(x, eps=1e-6):
    x = x.clip(min=0., max=1.)
    return torch.log(x / (1 - x + eps) + eps)
