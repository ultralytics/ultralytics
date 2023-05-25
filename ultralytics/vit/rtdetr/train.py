from copy import copy

import torch
from val import RTDETRDataset, RTDETRValidator

from ultralytics.register import REGISTER
from ultralytics.vit.utils.loss import DETRLoss
from ultralytics.yolo.utils import DEFAULT_CFG, colorstr
from ultralytics.yolo.v8.detect import DetectionTrainer


class RTDETRTrainer(DetectionTrainer):

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build RTDETR Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def get_validator(self):
        """Returns a DetectionValidator for RTDETR model validation."""
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        bs = len(batch['img'])
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        REGISTER['batch'] = {'cls': gt_class, 'bboxes': gt_bbox}
        return batch

    def criterion(self, preds, batch):
        """Compute loss for RTDETR prediction and ground-truth."""
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = RTDETRLoss(use_vfl=True)

        dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta = preds
        # NOTE: `dn_meta` means it's eval mode, loss calculation for eval mode is not supported.
        if dn_meta is None:
            return 0, torch.zeros(3, device=dec_out_bboxes.device)
        dn_out_bboxes, dec_out_bboxes = torch.split(dec_out_bboxes, dn_meta['dn_num_split'], dim=2)
        dn_out_logits, dec_out_logits = torch.split(dec_out_logits, dn_meta['dn_num_split'], dim=2)

        out_bboxes = torch.cat([enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
        out_logits = torch.cat([enc_topk_logits.unsqueeze(0), dec_out_logits])

        loss = self.compute_loss((out_bboxes, out_logits),
                                 batch,
                                 dn_out_bboxes=dn_out_bboxes,
                                 dn_out_logits=dn_out_logits,
                                 dn_meta=dn_meta)
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']])


class RTDETRLoss(DETRLoss):

    def forward(self, preds, batch, dn_out_bboxes=None, dn_out_logits=None, dn_meta=None):
        boxes, logits = preds
        # NOTE: convert bboxes and cls to list.
        bs = boxes.shape[1]
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(boxes.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=boxes.device, dtype=torch.long))
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
            total_loss.update({k + '_dn': torch.tensor([0.]) for k in total_loss.keys()})

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


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize RTDETR model given training data and device."""
    model = 'rt-detr-l.yaml'
    data = cfg.data or 'coco8.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    # NOTE: F.grid_sample which is in rt-detr does not support deterministic=True
    args = dict(model=model, data=data, device=device, imgsz=640, exist_ok=True, batch=4, deterministic=False)
    trainer = RTDETRTrainer(overrides=args)
    trainer.train()


if __name__ == '__main__':
    train()
