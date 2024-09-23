# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
# Extract per-box (and even per-sample) losses from YOLOv8 detection model
import torch

from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors


class UnreducedBboxLoss(BboxLoss):

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        loss_iou = ((1.0 - iou) * weight)

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight

            # Keep track of fg_indices to know which boxes loss was calculated for
        else:
            # loss_dfl = torch.tensor(0.0).to(pred_dist.device)
            loss_dfl = torch.zeros_like(loss_iou)

        assert loss_iou.shape == loss_dfl.shape, f"IoU Loss shape {loss_iou.shape} != DFL Loss shape {loss_dfl.shape}"

        return loss_iou, loss_dfl


class v8UnreducedDetectionLoss(v8DetectionLoss):

    def __init__(self, model, tal_topk=10, training=False):
        super().__init__(model, tal_topk=tal_topk)

        # Override with unreduced BboxLoss
        m = model.model[-1]  # Detect() module
        self.bbox_loss = UnreducedBboxLoss(m.reg_max)
        self.training = training

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, (list, tuple)) else preds  # 3lc: list when using detectionmodel
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum(dim=2)

        # PER-SAMPLE: self.bce(pred_scores, target_scores.to(dtype)).sum(dim=(1,2)) / target_scores.sum(dim=(1,2)) TODO: Max between target scores and torch ones?
        # PER-box: self.bce(pred_scores, target_scores.to(dtype)).sum(dim=2)[0] ??? Do we care about scaling these? I'd argue no, since we really just want to compare boxes with each other

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # loss[0], loss[2] = self.bbox_loss(
            #     pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            # )
            box_loss, dfl_loss = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                                target_scores_sum, fg_mask)

        # loss[0] *= self.hyp.box  # box gain
        # loss[1] *= self.hyp.cls  # cls gain
        # loss[2] *= self.hyp.dfl  # dfl gain

        box_loss_full = torch.zeros_like(cls_loss)
        dfl_loss_full = torch.zeros_like(cls_loss)

        box_loss_full[fg_mask] = box_loss.to(cls_loss.dtype).squeeze()
        dfl_loss_full[fg_mask] = dfl_loss.to(cls_loss.dtype).squeeze()

        losses = {"cls_loss": cls_loss, "box_loss": box_loss_full, "dfl_loss": dfl_loss_full}

        if self.training:
            cls_weight = self.hyp.cls if hasattr(self.hyp, "cls") else self.hyp["cls"]
            box_weight = self.hyp.box if hasattr(self.hyp, "box") else self.hyp["box"]
            dfl_weight = self.hyp.dfl if hasattr(self.hyp, "dfl") else self.hyp["dfl"]
            losses["loss"] = cls_loss * cls_weight + box_loss_full * box_weight + dfl_loss_full * dfl_weight

        return losses

        # return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
