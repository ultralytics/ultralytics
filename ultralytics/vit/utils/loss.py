import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.vit.utils.ops import HungarianMatcher
from ultralytics.yolo.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.yolo.utils.metrics import bbox_iou


class DETRLoss(nn.Module):

    def __init__(self,
                 num_classes=80,
                 matcher=HungarianMatcher(matcher_coeff={
                     'class': 2,
                     'bbox': 5,
                     'giou': 2}),
                 loss_coeff=None,
                 aux_loss=True,
                 use_focal_loss=True,
                 use_vfl=False,
                 use_uni_match=False,
                 uni_match_ind=0):
        """
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super().__init__()

        if loss_coeff is None:
            loss_coeff = {'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1}
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

        if not self.use_focal_loss:
            self.loss_coeff['class'] = torch.full([num_classes + 1], loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']

    def _get_loss_class(self, logits, gt_class, match_indices, bg_index, num_gts, postfix='', iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f'loss_class{postfix}'
        varifocal_loss = VarifocalLoss()
        focal_loss = FocalLoss()
        target_label = torch.full(logits.shape[:2], bg_index, device=logits.device, dtype=torch.int64)
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects, gt_class, match_indices)
            target_label = target_label.view(-1, 1)
            target_label[index] = updates.to(dtype=torch.int64)
            target_label = target_label.view(bs, num_query_objects)
        if self.use_focal_loss:
            target_label = F.one_hot(target_label, self.num_classes + 1)[..., :-1]
            if iou_score is not None and self.use_vfl:
                target_score = torch.zeros([bs, num_query_objects], device=logits.device)
                if num_gt > 0:
                    target_score = target_score.view(-1, 1)
                    target_score[index] = iou_score
                target_score = target_score.view(bs, num_query_objects, 1) * target_label
                loss_ = self.loss_coeff['class'] * varifocal_loss(logits, target_score, target_label,
                                                                  num_gts / num_query_objects)  # RTDETR loss
                # loss_ = self.loss_coeff['class'] * nn.BCEWithLogitsLoss(reduction='none')(logits, target_score).mean(
                #     1).sum()  # YOLO CLS loss
            else:
                loss_ = self.loss_coeff['class'] * focal_loss(logits, target_label.float(), num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(logits, target_label, weight=self.loss_coeff['class'])

        return {name_class: loss_.squeeze()}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts, postfix=''):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox, match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = 1.0 - bbox_iou(src_bbox, target_bbox, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        loss = {k: v.squeeze() for k, v in loss.items()}
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts, postfix=''):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = f'loss_mask{postfix}'
        name_dice = f'loss_dice{postfix}'

        loss = {}
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.tensor(0., device=self.device)
            loss[name_dice] = torch.tensor(0., device=self.device)
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask, match_indices)
        src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
        # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
        loss[name_mask] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
                                                                         torch.tensor([num_gts], dtype=torch.float32))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
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
                      boxes,
                      logits,
                      gt_bbox,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix='',
                      masks=None,
                      gt_mask=None):
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5, device=boxes.device)
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(boxes[self.uni_match_ind],
                                         logits[self.uni_match_ind],
                                         gt_bbox,
                                         gt_class,
                                         masks=masks[self.uni_match_ind] if masks is not None else None,
                                         gt_mask=gt_mask)
        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(aux_boxes, aux_logits, gt_bbox, gt_class, masks=aux_masks, gt_mask=gt_mask)
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(src_bbox, target_bbox, xywh=True)
                else:
                    iou_score = None
            else:
                iou_score = None
            loss[0] += self._get_loss_class(
                aux_logits,
                gt_class,
                match_indices,
                bg_index,
                num_gts,
                postfix,
                iou_score,
            )[f'loss_class{postfix}']
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices, num_gts, postfix)
            loss[1] += loss_[f'loss_bbox{postfix}']
            loss[2] += loss_[f'loss_giou{postfix}']
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, num_gts, postfix)
                loss[3] += loss_[f'loss_mask{postfix}']
                loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f'loss_class_aux{postfix}': loss[0],
            f'loss_bbox_aux{postfix}': loss[1],
            f'loss_giou_aux{postfix}': loss[2]}
        if masks is not None and gt_mask is not None:
            loss[f'loss_mask_aux{postfix}'] = loss[3]
            loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = torch.cat([t[dst] for t, (_, dst) in zip(target, match_indices)])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = torch.cat([
            t[I] if len(I) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (I, _) in zip(src, match_indices)])
        target_assign = torch.cat([
            t[J] if len(J) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (_, J) in zip(target, match_indices)])
        return src_assign, target_assign

    def _get_num_gts(self, targets):
        num_gts = sum(len(a) for a in targets)
        num_gts = max(num_gts, 1)
        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             logits,
                             gt_bbox,
                             gt_class,
                             masks=None,
                             gt_mask=None,
                             postfix='',
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if sum(len(a) for a in gt_bbox) > 0:
                src_bbox, target_bbox = self._get_src_target_assign(boxes.detach(), gt_bbox, match_indices)
                iou_score = bbox_iou(src_bbox, target_bbox, xywh=True)
            else:
                iou_score = None
        else:
            iou_score = None

        loss = {}
        loss.update(self._get_loss_class(logits, gt_class, match_indices, self.num_classes, num_gts, postfix,
                                         iou_score))
        loss.update(self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts, postfix))
        if masks is not None and gt_mask is not None:
            loss.update(self._get_loss_mask(masks, gt_mask, match_indices, num_gts, postfix))
        return loss

    def forward(self, boxes, logits, gt_bbox, gt_class, masks=None, gt_mask=None, postfix='', **kwargs):
        """
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """
        self.device = boxes.device

        dn_match_indices = kwargs.get('dn_match_indices', None)
        num_gts = kwargs.get('num_gts', None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(boxes[-1],
                                               logits[-1],
                                               gt_bbox,
                                               gt_class,
                                               masks=masks[-1] if masks is not None else None,
                                               gt_mask=gt_mask,
                                               postfix=postfix,
                                               dn_match_indices=dn_match_indices,
                                               num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(boxes[:-1],
                                   logits[:-1],
                                   gt_bbox,
                                   gt_class,
                                   self.num_classes,
                                   num_gts,
                                   dn_match_indices,
                                   postfix,
                                   masks=masks[:-1] if masks is not None else None,
                                   gt_mask=gt_mask))

        return total_loss


class RTDETRDetectionLoss(DETRLoss):

    def forward(self, preds, batch, dn_out_bboxes=None, dn_out_logits=None, dn_meta=None, **kwargs):
        boxes, logits = preds
        gt_class, gt_bbox = batch['cls'], batch['bboxes']
        num_gts = self._get_num_gts(gt_class)
        total_loss = super().forward(boxes, logits, gt_bbox, gt_class, num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta['dn_positive_idx'], dn_meta['dn_num_group']
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super().forward(dn_out_bboxes,
                                      dn_out_logits,
                                      gt_bbox,
                                      gt_class,
                                      postfix='_dn',
                                      dn_match_indices=dn_match_indices,
                                      num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f'{k}_dn': torch.tensor(0., device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.int64)
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx), 'Expected the sa'
                f'me length, but got {len(dn_positive_idx[i])} and '
                f'{len(gt_idx)} respectively.'
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.int64), torch.zeros([0], dtype=torch.int64)))
        return dn_match_indices
