import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.yolo import v8
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG, BaseTrainer
from ultralytics.yolo.utils.metrics import FocalLoss, bbox_iou, smooth_BCE
from ultralytics.yolo.utils.modeling.tasks import SegmentationModel
from ultralytics.yolo.utils.ops import crop_mask, xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.torch_utils import de_parallel

from ..detect import DetectionTrainer


# BaseTrainer python usage
class SegmentationTrainer(DetectionTrainer):

    def load_model(self, model_cfg=None, weights=None):
        model = SegmentationModel(model_cfg or weights["model"].yaml,
                                  ch=3,
                                  nc=self.data["nc"],
                                  anchors=self.args.get("anchors"))
        if weights:
            model.load(weights)
        for _, v in model.named_parameters():
            v.requires_grad = True  # train all layers
        return model

    def get_validator(self):
        return v8.segment.SegmentationValidator(self.test_loader,
                                                save_dir=self.save_dir,
                                                logger=self.console,
                                                args=self.args)

    def criterion(self, preds, batch):
        head = de_parallel(self.model).model[-1]
        sort_obj_iou = False
        autobalance = False

        # init losses
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.args.cls_pw], device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.args.obj_pw], device=self.device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=self.args.label_smoothing)  # positive, negative BCE targets

        # Focal loss
        g = self.args.fl_gamma
        if self.args.fl_gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        balance = {3: [4.0, 1.0, 0.4]}.get(head.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        ssi = list(head.stride).index(16) if autobalance else 0  # stride 16 index
        BCEcls, BCEobj, gr, autobalance = BCEcls, BCEobj, 1.0, autobalance

        def single_mask_loss(gt_mask, pred, proto, xyxy, area):
            # Mask loss for one image
            pred_mask = (pred @ proto.view(head.nm, -1)).view(-1, *proto.shape[1:])  # (n,32) @ (32,80,80) -> (n,80,80)
            loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
            return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()

        def build_targets(p, targets):
            # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
            nonlocal head
            na, nt = head.na, targets.shape[0]  # number of anchors, targets
            tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []
            gain = torch.ones(8, device=self.device)  # normalized to gridspace gain
            ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1,
                                                                                 nt)  # same as .repeat_interleave(nt)
            if self.args.overlap_mask:
                batch = p[0].shape[0]
                ti = []
                for i in range(batch):
                    num = (targets[:, 0] == i).sum()  # find number of targets of each image
                    ti.append(torch.arange(num, device=self.device).float().view(1, num).repeat(na, 1) + 1)  # (na, num)
                ti = torch.cat(ti, 1)  # (na, nt)
            else:
                ti = torch.arange(nt, device=self.device).float().view(1, nt).repeat(na, 1)
            targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None], ti[..., None]), 2)  # append anchor indices

            g = 0.5  # bias
            off = torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device).float() * g  # offsets

            for i in range(head.nl):
                anchors, shape = head.anchors[i], p[i].shape
                gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

                # Match targets to anchors
                t = targets * gain  # shape(3,n,7)
                if nt:
                    # Matches
                    r = t[..., 4:6] / anchors[:, None]  # wh ratio
                    j = torch.max(r, 1 / r).max(2)[0] < self.args.anchor_t  # compare
                    # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                    t = t[j]  # filter

                    # Offsets
                    gxy = t[:, 2:4]  # grid xy
                    gxi = gain[[2, 3]] - gxy  # inverse
                    j, k = ((gxy % 1 < g) & (gxy > 1)).T
                    l, m = ((gxi % 1 < g) & (gxi > 1)).T
                    j = torch.stack((torch.ones_like(j), j, k, l, m))
                    t = t.repeat((5, 1, 1))[j]
                    offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                else:
                    t = targets[0]
                    offsets = 0

                # Define
                bc, gxy, gwh, at = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
                (a, tidx), (b, c) = at.long().T, bc.long().T  # anchors, image, class
                gij = (gxy - offsets).long()
                gi, gj = gij.T  # grid indices

                # Append
                indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
                tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
                anch.append(anchors[a])  # anchors
                tcls.append(c)  # class
                tidxs.append(tidx)
                xywhn.append(torch.cat((gxy, gwh), 1) / gain[2:6])  # xywh normalized

            return tcls, tbox, indices, anch, tidxs, xywhn

        if len(preds) == 2:  # eval
            p, proto, = preds
        else:  # len(3) train
            _, proto, p = preds

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        masks = batch["masks"]
        targets, masks = targets.to(self.device), masks.to(self.device).float()

        bs, nm, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lseg = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors, tidxs, xywhn = build_targets(p, targets)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls, pmask = pi[b, a, gj, gi].split((2, 2, 1, head.nc, nm), 1)  # subset of predictions

                # Box regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if gr < 1:
                    iou = (1.0 - gr) + gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if head.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(pcls, t)  # BCE

                # Mask regression
                if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                    masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                marea = xywhn[i][:, 2:].prod(1)  # mask width, height normalized
                mxyxy = xywh2xyxy(xywhn[i] * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device))
                for bi in b.unique():
                    j = b == bi  # matching index
                    if self.args.overlap_mask:
                        mask_gti = torch.where(masks[bi][None] == tidxs[i][j].view(-1, 1, 1), 1.0, 0.0)
                    else:
                        mask_gti = masks[tidxs[i]][j]
                    lseg += single_mask_loss(mask_gti, pmask[j], proto[bi], mxyxy[j], marea[j])
            else:
                lseg += (proto * 0).sum()

            obji = BCEobj(pi[..., 4], tobj)
            lobj += obji * balance[i]  # obj loss
            if autobalance:
                balance[i] = balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if autobalance:
            balance = [x / balance[ssi] for x in balance]
        lbox *= self.args.box
        lobj *= self.args.obj
        lcls *= self.args.cls
        lseg *= self.args.box / bs

        loss = lbox + lobj + lcls + lseg
        return loss * bs, torch.cat((lbox, lseg, lobj, lcls)).detach()

    def label_loss_items(self, loss_items=None, prefix="train"):
        # We should just use named tensors here in future
        keys = [f"{prefix}/lbox", f"{prefix}/lseg", f"{prefix}/lobj", f"{prefix}/lcls"]
        return dict(zip(keys, loss_items)) if loss_items is not None else keys

    def progress_string(self):
        return ('\n' + '%11s' * 7) % \
               ('Epoch', 'GPU_mem', 'box_loss', 'seg_loss', 'obj_loss', 'cls_loss', 'Size')

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


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG.parent, config_name=DEFAULT_CONFIG.name)
def train(cfg):
    cfg.model = cfg.model or "models/yolov5n-seg.yaml"
    cfg.data = cfg.data or "coco128-seg.yaml"  # or yolo.ClassificationDataset("mnist")
    trainer = SegmentationTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    """
    CLI usage:
    python ultralytics/yolo/v8/segment/train.py cfg=yolov5n-seg.yaml data=coco128-segments epochs=100 img_size=640

    TODO:
    Direct cli support, i.e, yolov8 classify_train args.epochs 10
    """
    train()
