from copy import copy

import torch
from val import RTDETRDataset, RTDETRValidator

from ultralytics.vit.utils.loss import DETRLoss
from ultralytics.yolo.utils import DEFAULT_CFG, colorstr
from ultralytics.yolo.utils.torch_utils import de_parallel
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
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def criterion(self, preds, batch):
        """Compute loss for RTDETR prediction and ground-truth."""
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = RTDETRLoss(de_parallel(self.model))

        # TODO: now the returned loss is a dict 
        dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta = preds
        if dn_meta is not None:
            if isinstance(dn_meta, list):
                dual_groups = len(dn_meta) - 1
                dec_out_bboxes = torch.split(dec_out_bboxes, dual_groups + 1, dim=2)
                dec_out_logits = torch.split(dec_out_logits, dual_groups + 1, dim=2)
                enc_topk_bboxes = torch.split(enc_topk_bboxes, dual_groups + 1, dim=1)
                enc_topk_logits = torch.split(enc_topk_logits, dual_groups + 1, dim=1)

                loss = {}
                for g_id in range(dual_groups + 1):
                    if dn_meta[g_id] is not None:
                        dn_out_bboxes_gid, dec_out_bboxes_gid = torch.split(dec_out_bboxes[g_id],
                                                                            dn_meta[g_id]['dn_num_split'], dim=2)
                        dn_out_logits_gid, dec_out_logits_gid = torch.split(dec_out_logits[g_id],
                                                                            dn_meta[g_id]['dn_num_split'], dim=2)
                    else:
                        dn_out_bboxes_gid, dn_out_logits_gid = None, None
                        dec_out_bboxes_gid = dec_out_bboxes[g_id]
                        dec_out_logits_gid = dec_out_logits[g_id]
                    out_bboxes_gid = torch.cat([enc_topk_bboxes[g_id].unsqueeze(0), dec_out_bboxes_gid])
                    out_logits_gid = torch.cat([enc_topk_logits[g_id].unsqueeze(0), dec_out_logits_gid])
                    loss_gid = self.compute_loss(
                        (out_bboxes_gid, out_logits_gid),
                        batch,
                        dn_out_bboxes=dn_out_bboxes_gid,
                        dn_out_logits=dn_out_logits_gid,
                        dn_meta=dn_meta[g_id])
                    # sum loss
                    for key, value in loss_gid.items():
                        loss.update({
                            key: loss.get(key, torch.zeros([1])) + value
                        })

                # average across (dual_groups + 1)
                for key, value in loss.items():
                    loss.update({key: value / (dual_groups + 1)})
                return loss
            else:
                dn_out_bboxes, dec_out_bboxes = torch.split(
                    dec_out_bboxes, dn_meta['dn_num_split'], dim=2)
                dn_out_logits, dec_out_logits = torch.split(
                    dec_out_logits, dn_meta['dn_num_split'], dim=2)
        else:
            dn_out_bboxes, dn_out_logits = None, None

        out_bboxes = torch.cat(
            [enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
        out_logits = torch.cat(
            [enc_topk_logits.unsqueeze(0), dec_out_logits])

        return self.compute_loss(
            (out_bboxes, out_logits),
            batch,
            dn_out_bboxes=dn_out_bboxes,
            dn_out_logits=dn_out_logits,
            dn_meta=dn_meta)


class RTDETRLoss(DETRLoss):

    def forward(self, preds, batch, dn_out_bboxes=None, dn_out_logits=None, dn_meta=None):
        gt_class, gt_bbox = batch['cls'], batch['bboxes']
        boxes, logits = preds
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


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize RTDETR model given training data and device."""
    model = 'rt-detr-l.yaml'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device, imgsz=16)
    trainer = RTDETRTrainer(overrides=args)
    trainer.train()


if __name__ == '__main__':
    train()
