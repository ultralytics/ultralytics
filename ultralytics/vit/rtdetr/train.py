from copy import copy

import torch
from val import RTDETRDataset, RTDETRValidator

from ultralytics.vit.utils.loss import DETRLoss
from ultralytics.yolo.utils import DEFAULT_CFG, colorstr
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.register import REGISTER


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

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        REGISTER['batch'] = RTDETRLoss.preprocess(targets, batch_size=len(batch["img"]))
        return batch

    def criterion(self, preds, batch):
        """Compute loss for RTDETR prediction and ground-truth."""
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = RTDETRLoss(use_vfl=True)

        # TODO: now the returned loss is a dict, but we need a tensor and a tensor.detach()
        dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta = preds
        dn_out_bboxes, dec_out_bboxes = torch.split(dec_out_bboxes, dn_meta['dn_num_split'], dim=2)
        dn_out_logits, dec_out_logits = torch.split(dec_out_logits, dn_meta['dn_num_split'], dim=2)

        out_bboxes = torch.cat([enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
        out_logits = torch.cat([enc_topk_logits.unsqueeze(0), dec_out_logits])

        return self.compute_loss((out_bboxes, out_logits),
                                 batch,
                                 dn_out_bboxes=dn_out_bboxes,
                                 dn_out_logits=dn_out_logits,
                                 dn_meta=dn_meta)


class RTDETRLoss(DETRLoss):

    def forward(self, preds, batch, dn_out_bboxes=None, dn_out_logits=None, dn_meta=None):
        boxes, logits = preds
        # NOTE: convert bboxes and cls to list.
        bs = boxes.shape[1]
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx==i].to(boxes.device))
            gt_class.append(batch['cls'][batch_idx==i].to(device=boxes.device, dtype=torch.long))
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
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.int64), torch.zeros([0], dtype=torch.int64)))
        return dn_match_indices

    @staticmethod
    def preprocess(targets, batch_size):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
        return {"cls": out[..., 0], "bboxes": out[..., 1:]}


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize RTDETR model given training data and device."""
    model = 'rt-detr-l.yaml'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device, imgsz=640, exist_ok=True, batch=4)
    trainer = RTDETRTrainer(overrides=args)
    trainer.train()


if __name__ == '__main__':
    train()
