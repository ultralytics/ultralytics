import torch

from ultralytics.vit.utils.loss import DETRLoss
from ultralytics.yolo.engine.trainer import BaseTrainer


class RTDETRTrainer(BaseTrainer):
    pass


class RTDETRLoss(DETRLoss):

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix='',
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_meta=None,
                **kwargs):
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
            total_loss.update({k + '_dn': torch.to_tensor([0.]) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype='int64')
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype='int64'), torch.zeros([0], dtype='int64')))
        return dn_match_indices
