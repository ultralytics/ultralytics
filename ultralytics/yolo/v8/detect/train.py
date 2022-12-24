import hydra
import torch
import torch.nn as nn

from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG, BaseTrainer
from ultralytics.yolo.utils.metrics import FocalLoss, bbox_iou, smooth_BCE
from ultralytics.yolo.utils.modeling.tasks import DetectionModel
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.torch_utils import de_parallel


# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def get_dataloader(self, dataset_path, batch_size, mode="train", rank=0):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, rank=rank, mode=mode)[0]

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        self.args.box *= 3 / nl  # scale to layers
        self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        self.args.obj *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
        self.model.names = self.data["names"]

    def load_model(self, model_cfg=None, weights=None):
        model = DetectionModel(model_cfg or weights["model"].yaml,
                               ch=3,
                               nc=self.data["nc"],
                               anchors=self.args.get("anchors"))
        if weights:
            model.load(weights)
        for _, v in model.named_parameters():
            v.requires_grad = True  # train all layers
        return model

    def get_validator(self):
        self.loss_names = 'box_loss', 'obj_loss', 'cls_loss'
        return v8.detect.DetectionValidator(self.test_loader,
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

        def build_targets(p, targets):
            # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
            nonlocal head
            na, nt = head.na, targets.shape[0]  # number of anchors, targets
            tcls, tbox, indices, anch = [], [], [], []
            gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
            ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
            targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

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
                bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
                a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
                gij = (gxy - offsets).long()
                gi, gj = gij.T  # grid indices

                # Append
                indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
                tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
                anch.append(anchors[a])  # anchors
                tcls.append(c)  # class

            return tcls, tbox, indices, anch

        if len(preds) == 2:  # eval
            _, p = preds
        else:  # len(3) train
            p = preds

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = targets.to(self.device)

        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = build_targets(p, targets)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            bs = tobj.shape[0]
            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, head.nc), 1)  # subset of predictions

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

            obji = BCEobj(pi[..., 4], tobj)
            lobj += obji * balance[i]  # obj loss
            if autobalance:
                balance[i] = balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if autobalance:
            balance = [x / balance[ssi] for x in balance]
        lbox *= self.args.box
        lobj *= self.args.obj
        lcls *= self.args.cls

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()

    def label_loss_items(self, loss_items=None, prefix="train"):
        # We should just use named tensors here in future
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        return dict(zip(keys, loss_items)) if loss_items is not None else keys

    def progress_string(self):
        return ('\n' + '%11s' * 6) % \
               ('Epoch', 'GPU_mem', *self.loss_names, 'Size')

    def plot_training_samples(self, batch, ni):
        images = batch["img"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(images, batch_idx, cls, bboxes, paths=paths, fname=self.save_dir / f"train_batch{ni}.jpg")

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG.parent, config_name=DEFAULT_CONFIG.name)
def train(cfg):
    cfg.model = cfg.model or "models/yolov5n.yaml"
    cfg.data = cfg.data or "coco128.yaml"  # or yolo.ClassificationDataset("mnist")
    trainer = DetectionTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    """
    CLI usage:
    python ultralytics/yolo/v8/detect/train.py model=yolov5n.yaml data=coco128 epochs=100 imgsz=640

    TODO:
    yolo task=detect mode=train model=yolov5n.yaml data=coco128.yaml epochs=100
    """
    train()
