# TODO: license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.yolo.utils.metrics import bbox_iou
from ultralytics.yolo.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):

    def __init__(self,
                 matcher_coeff=None,
                 use_focal_loss=False,
                 with_mask=False,
                 num_sample_points=12544,
                 alpha=0.25,
                 gamma=2.0):
        """
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super().__init__()
        if matcher_coeff is None:
            matcher_coeff = {'class': 1, 'bbox': 5, 'giou': 2, 'mask': 1, 'dice': 1}
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, boxes, logits, gt_bbox, gt_class, masks=None, gt_mask=None):
        """
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
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        logits = logits.detach()
        out_prob = F.sigmoid(logits.flatten(0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1), dim=-1)
        # [batch_size * num_queries, 4]
        out_bbox = boxes.detach().flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat(gt_class).flatten()
        tgt_bbox = torch.cat(gt_bbox)

        # Compute the classification cost
        out_prob = out_prob[:, tgt_ids]
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        # Compute the L1 cost between boxes
        cost_bbox = (out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the GIoU cost between boxes
        cost_giou = 1.0 - bbox_iou(out_bbox.unsqueeze(1), tgt_bbox.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + \
            self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        # Compute the mask cost and dice cost
        if self.with_mask:
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
                pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask,
                                                                   torch.ones_like(out_mask),
                                                                   reduction='none')
                neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask,
                                                                   torch.zeros_like(out_mask),
                                                                   reduction='none')
                cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
                cost_mask /= self.num_sample_points

                # dice cost
                out_mask = F.sigmoid(out_mask)
                numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
                denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
                cost_dice = 1 - (numerator + 1) / (denominator + 1)

                C = C + self.matcher_coeff['mask'] * cost_mask + self.matcher_coeff['dice'] * cost_dice

        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]
        sizes = [a.shape[0] for a in gt_bbox]
        indices = [linear_sum_assignment(c.split(sizes, -1)[i].cpu().numpy()) for i, c in enumerate(C)]
        return [(torch.tensor(i, dtype=torch.int64), torch.tensor(j, dtype=torch.int64)) for i, j in indices]


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t) for t in targets['cls']]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(targets['cls'])
    input_query_class = torch.full([bs, max_gt_num], num_classes)  # bs, max_gt_num
    input_query_bbox = torch.zeros([bs, max_gt_num, 4])
    pad_gt_mask = torch.zeros([bs, max_gt_num])
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets['cls'][i].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets['bboxes'][i]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.repeat(1, 2 * num_group)  # bs, 2* max_gt_num * num_group
    input_query_bbox = input_query_bbox.repeat(1, 2 * num_group, 1)  # bs, 2* max_gt_num * num_group, 4
    pad_gt_mask = pad_gt_mask.repeat(1, 2 * num_group)
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1])
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.repeat(1, num_group, 1)    # bs, 2* max_gt_num * num_group, 1
    positive_gt_mask = 1 - negative_gt_mask             # bs, 2* max_gt_num * num_group, 1
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask   # bs, 2* max_gt_num * num_group
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        input_query_class = input_query_class.view(-1)
        pad_gt_mask = pad_gt_mask.view(-1)
        # half of bbox prob
        mask = torch.rand(input_query_class.shape) < (label_noise_ratio * 0.5)
        chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=input_query_class.dtype)

        input_query_class[chosen_idx] = new_label
        input_query_class = input_query_class.view(bs, num_denoising)
        pad_gt_mask = pad_gt_mask.view(bs, num_denoising)

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(input_query_bbox)

        diff = (input_query_bbox[..., 2:] * 0.5).repeat(1, 1, 2) * box_noise_scale   # bs, 2* max_gt_num * num_group, 4

        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand(input_query_bbox.shape)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = xyxy2xywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])

    input_query_class = class_embed[input_query_class.flatten()].view(bs, num_denoising, -1)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # match query cannot see the reconstruct
    attn_mask[num_denoising:, :num_denoising] = True
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1):num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1):num_denoising] = True
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
    attn_mask = ~attn_mask
    dn_meta = {
        'dn_positive_idx': dn_positive_idx,
        'dn_num_group': num_group,
        'dn_num_split': [num_denoising, num_queries]}

    return input_query_class.to(class_embed.device), input_query_bbox.to(class_embed.device), attn_mask, dn_meta


def inverse_sigmoid(x, eps=1e-6):
    x = x.clip(min=0., max=1.)
    return torch.log(x / (1 - x + eps) + eps)
