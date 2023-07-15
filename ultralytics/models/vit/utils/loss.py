# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


class DETRLoss(nn.Module):

    def __init__(self,
                 nc=80,
                 loss_gain=None,
                 aux_loss=True,
                 use_fl=True,
                 use_vfl=False,
                 use_uni_match=False,
                 uni_match_ind=0):
        """
        DETR loss function.

        Args:
            nc (int): The number of classes.
            loss_gain (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_vfl (bool): Use VarifocalLoss or not.
            use_uni_match (bool): Whether to use a fixed layer to assign labels for auxiliary branch.
            uni_match_ind (int): The fixed indices of a layer.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={'class': 2, 'bbox': 5, 'giou': 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f'loss_class{postfix}'
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
            loss_cls = nn.BCEWithLogitsLoss(reduction='none')(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        loss = {k: v.squeeze() for k, v in loss.items()}
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = f'loss_mask{postfix}'
        name_dice = f'loss_dice{postfix}'

        loss = {}
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.tensor(0., device=self.device)
            loss[name_dice] = torch.tensor(0., device=self.device)
            return loss

        num_gts = len(gt_mask)
        src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
        src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
        # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
        loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
                                                                        torch.tensor([num_gts], dtype=torch.float32))
        loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      pred_bboxes,
                      pred_scores,
                      gt_bboxes,
                      gt_cls,
                      gt_groups,
                      match_indices=None,
                      postfix='',
                      masks=None,
                      gt_mask=None):
        """Get auxiliary losses"""
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(pred_bboxes[self.uni_match_ind],
                                         pred_scores[self.uni_match_ind],
                                         gt_bboxes,
                                         gt_cls,
                                         gt_groups,
                                         masks=masks[self.uni_match_ind] if masks is not None else None,
                                         gt_mask=gt_mask)
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(aux_bboxes,
                                   aux_scores,
                                   gt_bboxes,
                                   gt_cls,
                                   gt_groups,
                                   masks=aux_masks,
                                   gt_mask=gt_mask,
                                   postfix=postfix,
                                   match_indices=match_indices)
            loss[0] += loss_[f'loss_class{postfix}']
            loss[1] += loss_[f'loss_bbox{postfix}']
            loss[2] += loss_[f'loss_giou{postfix}']
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f'loss_class_aux{postfix}': loss[0],
            f'loss_bbox_aux{postfix}': loss[1],
            f'loss_giou_aux{postfix}': loss[2]}
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    def _get_index(self, match_indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        pred_assigned = torch.cat([
            t[I] if len(I) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (I, _) in zip(pred_bboxes, match_indices)])
        gt_assigned = torch.cat([
            t[J] if len(J) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (_, J) in zip(gt_bboxes, match_indices)])
        return pred_assigned, gt_assigned

    def _get_loss(self,
                  pred_bboxes,
                  pred_scores,
                  gt_bboxes,
                  gt_cls,
                  gt_groups,
                  masks=None,
                  gt_mask=None,
                  postfix='',
                  match_indices=None):
        """Get losses"""
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes,
                                         pred_scores,
                                         gt_bboxes,
                                         gt_cls,
                                         gt_groups,
                                         masks=masks,
                                         gt_mask=gt_mask)

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        # if masks is not None and gt_mask is not None:
        #     loss.update(self._get_loss_mask(masks, gt_mask, match_indices, postfix))
        return loss

    def forward(self, pred_bboxes, pred_scores, batch, postfix='', **kwargs):
        """
        Args:
            pred_bboxes (torch.Tensor): [l, b, query, 4]
            pred_scores (torch.Tensor): [l, b, query, num_classes]
            batch (dict): A dict includes:
                gt_cls (torch.Tensor) with shape [num_gts, ],
                gt_bboxes (torch.Tensor): [num_gts, 4],
                gt_groups (List(int)): a list of batch size length includes the number of gts of each image.
            postfix (str): postfix of loss name.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get('match_indices', None)
        gt_cls, gt_bboxes, gt_groups = batch['cls'], batch['bboxes'], batch['gt_groups']

        total_loss = self._get_loss(pred_bboxes[-1],
                                    pred_scores[-1],
                                    gt_bboxes,
                                    gt_cls,
                                    gt_groups,
                                    postfix=postfix,
                                    match_indices=match_indices)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices,
                                   postfix))

        return total_loss


class RTDETRDetectionLoss(DETRLoss):

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta['dn_pos_idx'], dn_meta['dn_num_group']
            assert len(batch['gt_groups']) == len(dn_pos_idx)

            # denoising match indices
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch['gt_groups'])

            # compute denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix='_dn', match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f'{k}_dn': torch.tensor(0., device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): A list includes positive indices of denoising.
            dn_num_group (int): The number of groups of denoising.
            gt_groups (List(int)): a list of batch size length includes the number of gts of each image.

        Returns:
            dn_match_indices (List(tuple)): Matched indices.

        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), 'Expected the same length, '
                f'but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively.'
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
