from cProfile import label
import subprocess
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import torchvision

from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.engine.trainer import CONFIG_PATH_ABS, DEFAULT_CONFIG, BaseTrainer
from ultralytics.yolo.utils.downloads import download
from ultralytics.yolo.utils.files import WorkingDirectory
from ultralytics.yolo.utils.torch_utils import LOCAL_RANK, torch_distributed_zero_first
from ultralytics.yolo.utils.modeling.tasks import SegmentationModel
from ultralytics.yolo.utils import ops

# BaseTrainer python usage
class SegmentationTrainer(BaseTrainer):

    def get_dataset(self, dataset):
        # temporary solution. Replace with new ultralytics.yolo.ClassificationDataset module
        data = Path("datasets") / dataset
        with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(Path.cwd()):
            data_dir = data if data.is_dir() else (Path.cwd() / data)
            if not data_dir.is_dir():
                self.console.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
                t = time.time()
                if str(data) == 'imagenet':
                    subprocess.run(f"bash {v8.ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
                else:
                    url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip'
                    download(url, dir=data_dir.parent)
                # TODO: add colorstr
                s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {'bold', data_dir}\n"
                self.console.info(s)
        train_set = data_dir 
        test_set = data_dir

        return train_set, test_set

    def get_dataloader(self, dataset_path, batch_size, rank=0):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        loader = build_dataloader(img_path=dataset_path,
                                img_size=self.args.img_size,
                                batch_size=batch_size,
                                single_cls=self.args.single_cls,
                                cache=self.args.cache,
                                image_weights=self.args.image_weights,
                                stride=gs,
                                rect=self.args.rect,
                                rank=rank,
                                workers=self.args.workers,
                                shuffle=self.args.shuffle,
                                use_segments=True,
                                            )[0]
        return loader
    def preprocess_batch(self, batch):
        batch["img"] =  torch.stack(batch["img"]).to(self.device, non_blocking=True).float() / 255
        return batch

    def load_cfg(self, cfg):
        return SegmentationModel(cfg, nc=80)

    def get_validator(self):
        return v8.classify.ClassificationValidator(self.test_loader, self.device, logger=self.console)

    def criterion(self, preds, batch):
        def build_targets(self, p, targets):
            # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
            na, nt = self.na, targets.shape[0]  # number of anchors, targets
            tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []
            gain = torch.ones(8, device=self.device)  # normalized to gridspace gain
            ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
            if self.overlap:
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

            for i in range(self.nl):
                anchors, shape = self.anchors[i], p[i].shape
                gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

                # Match targets to anchors
                t = targets * gain  # shape(3,n,7)
                if nt:
                    # Matches
                    r = t[..., 4:6] / anchors[:, None]  # wh ratio
                    j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
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
        
        p, proto = preds
        targets, masks = batch["bboxes"], batch["masks"]
        targets, masks = targets.to(self.device), masks.to(self.device).float()

        bs, nm, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lseg = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors, tidxs, xywhn = build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls, pmask = pi[b, a, gj, gi].split((2, 2, 1, self.nc, nm), 1)  # subset of predictions

                # Box regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = ops.bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Mask regression
                if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                    masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                marea = xywhn[i][:, 2:].prod(1)  # mask width, height normalized
                mxyxy = ops.xywh2xyxy(xywhn[i] * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device))
                for bi in b.unique():
                    j = b == bi  # matching index
                    if self.overlap:
                        mask_gti = torch.where(masks[bi][None] == tidxs[i][j].view(-1, 1, 1), 1.0, 0.0)
                    else:
                        mask_gti = masks[tidxs[i]][j]
                    lseg += self.single_mask_loss(mask_gti, pmask[j], proto[bi], mxyxy[j], marea[j])

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lseg *= self.hyp["box"] / bs

        loss = lbox + lobj + lcls + lseg
        return loss * bs, torch.cat((lbox, lseg, lobj, lcls)).detach()


@hydra.main(version_base=None, config_path=CONFIG_PATH_ABS, config_name=str(DEFAULT_CONFIG).split(".")[0])
def train(cfg):
    cfg.cfg = "yolov5n-seg.yaml"
    cfg.data = cfg.data or "coco128-segments"  # or yolo.ClassificationDataset("mnist")
    trainer = SegmentationTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    """
    CLI usage:
    python ../path/to/train.py args.epochs=10 args.project="name" hyps.lr0=0.1

    TODO:
    Direct cli support, i.e, yolov8 classify_train args.epochs 10
    """
    train()
