# Ultralytics YOLO 🚀, GPL-3.0 license
import sys
from copy import copy

import torch
import torch.nn.functional as F

from ultralytics.nn.tasks import KeypointModel
from ultralytics.yolo import v8
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils.ops import crop_mask, xyxy2xywh
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.tal import make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.v8.detect.train import Loss


# BaseTrainer python usage
class KeypointsTrain(v8.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        if overrides is None:
            overrides = {}
        overrides['task'] = 'keypoints'
        super().__init__(cfg, overrides)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = KeypointModel(cfg, ch=3, nc=self.data['nc'], nkpt=self.data['nkpt'], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'
        return v8.segment.SegmentationValidator(self.test_loader,
                                                save_dir=self.save_dir,
                                                logger=self.console,
                                                args=copy(self.args))

    def criterion(self, preds, batch):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = SegLoss(de_parallel(self.model), overlap=self.args.overlap_mask)
        return self.compute_loss(preds, batch)

    def plot_training_samples(self, batch, ni):
        images = batch['img']
        masks = batch['masks']
        cls = batch['cls'].squeeze(-1)
        bboxes = batch['bboxes']
        paths = batch['im_file']
        batch_idx = batch['batch_idx']
        plot_images(images, batch_idx, cls, bboxes, masks, paths=paths, fname=self.save_dir / f'train_batch{ni}.jpg')

    def plot_metrics(self):
        plot_results(file=self.csv, segment=True)  # save results.png


# Criterion class for computing training losses
class KeypointLoss(Loss):

    def __init__(self, model, overlap=True):  # model must be de-paralleled
        super().__init__(model)
        self.nkpt = model.model[-1].nkpt  # number of keypoints
        self.overlap = overlap

    def __call__(self, preds, batch):
        tcls, tbox, tkpt, indices, anchors = self.build_targets(p, targets)  # targets

        sigmas = torch.ones((self.n_kpt), device=self.device) / 10
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        keypoints = batch['keypoints'].to(self.device).float()

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
        # WARNING: Uncomment lines below in case of Multi-GPU DDP unused gradient errors
        #         else:
        #             loss[1] += proto.sum() * 0
        # else:
        #     loss[1] += proto.sum() * 0

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


def train(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or "yolov8n-kpt.yaml"
    data = cfg.data or "coco128-kpt.yaml"  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = KeypointsTrain(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
