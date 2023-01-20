# Ultralytics YOLO 🚀, GPL-3.0 license

from copy import copy

import torch
import torch.nn.functional as F

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.yolo import v8
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils.ops import crop_mask, xyxy2xywh
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.tal import make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.v8.detect.train import Loss


# BaseTrainer python usage
class SegmentationTrainer(v8.detect.DetectionTrainer):

    def __init__(self, config=DEFAULT_CFG, overrides=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(config, overrides)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
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
        images = batch["img"]
        masks = batch["masks"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(images, batch_idx, cls, bboxes, masks, paths=paths, fname=self.save_dir / f"train_batch{ni}.jpg")

    def plot_metrics(self):
        plot_results(file=self.csv, segment=True)  # save results.png


# Criterion class for computing training losses
class SegLoss(Loss):

    def __init__(self, model, overlap=True):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = overlap

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        masks = batch["masks"].to(self.device).float()
        if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
            masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

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
            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]] + 1
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == mask_idx.view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy,
                                                     marea)  # seg loss
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

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Mask loss for one image
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


def train(cfg=DEFAULT_CFG):
    cfg.model = cfg.model or "yolov8n-seg.pt"
    cfg.data = cfg.data or "coco128-seg.yaml"  # or yolo.ClassificationDataset("mnist")
    cfg.device = cfg.device if cfg.device is not None else ''
    # trainer = SegmentationTrainer(cfg)
    # trainer.train()
    from ultralytics import YOLO
    model = YOLO(cfg.model)
    model.train(**vars(cfg))


if __name__ == "__main__":
    train()
