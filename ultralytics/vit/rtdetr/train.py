from copy import copy

import torch

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRModel

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = RTDETRModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

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

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = RTDETRDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

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
        return batch

    # def criterion(self, preds, batch):
    #     """Compute loss for RTDETR prediction and ground-truth."""
    #     if not hasattr(self, 'compute_loss'):
    #         self.compute_loss = RTDETRLoss(use_vfl=True)
    #
    #     dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta = preds
    #     # NOTE: `dn_meta` means it's eval mode, loss calculation for eval mode is not supported.
    #     if dn_meta is None:
    #         return 0, torch.zeros(3, device=dec_out_bboxes.device)
    #     dn_out_bboxes, dec_out_bboxes = torch.split(dec_out_bboxes, dn_meta['dn_num_split'], dim=2)
    #     dn_out_logits, dec_out_logits = torch.split(dec_out_logits, dn_meta['dn_num_split'], dim=2)
    #
    #     out_bboxes = torch.cat([enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
    #     out_logits = torch.cat([enc_topk_logits.unsqueeze(0), dec_out_logits])
    #
    #     loss = self.compute_loss((out_bboxes, out_logits),
    #                              batch,
    #                              dn_out_bboxes=dn_out_bboxes,
    #                              dn_out_logits=dn_out_logits,
    #                              dn_meta=dn_meta)
    #     return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']])


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize RTDETR model given training data and device."""
    model = 'rtdetr-l.yaml'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    # NOTE: F.grid_sample which is in rt-detr does not support deterministic=True
    # NOTE: amp training causes nan outputs and end with error while doing bipartite graph matching
    args = dict(model=model,
                data=data,
                device=device,
                imgsz=640,
                exist_ok=True,
                batch=4,
                deterministic=False,
                amp=False)
    trainer = RTDETRTrainer(overrides=args)
    trainer.train()


if __name__ == '__main__':
    train()
