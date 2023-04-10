# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from custom_models.yolo_custom.utils.bboxes_utils import intersection_over_union
from custom_models.yolo_custom.data.dataLoader import Training_Dataset
import config
from custom_models.yolo_custom.models.yolov5m import YOLOV5m


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, save_logs=False, filename=None, resume=False):
        device = next(model.parameters()).device  # get model device

        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))

        # check them here (https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml)
        # and here (https://github.com/ultralytics/yolov5/blob/master/utils/loss.py#L170)
        # also notice that these values depend on other model attributes (https://github.com/ultralytics/yolov5/blob/master/train.py#L232)
        self.lambda_class = 0.5 * (model.head.nc / 80 * 3 / model.head.nl)
        self.lambda_obj = 1 * ((config.IMAGE_SIZE / 640) ** 2 * 3 / model.head.nl)
        self.lambda_box = 0.05 * (3 / model.head.nl)

        self.anchor_t = 4.0

        self.balance = [4.0, 1.0, 0.4]  # explanation.. https://github.com/ultralytics/yolov5/issues/2026
        self.na = model.head.naxs  # number of anchors
        self.nc = model.head.nc  # number of classes
        self.nl = model.head.nl  # number of layers
        self.anchors = model.head.anchors
        self.device = device

        self.save_logs = save_logs
        self.filename = filename

        if self.save_logs:
            if not resume:
                folder = os.path.join("train_eval_metrics", filename)
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                with open(os.path.join(folder, "loss.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "batch_idx", "box_loss", "object_loss", "class_loss"])
                    print("--------------------------------------------------------------------------------------")
                    print(f'Training Logs will be saved in {os.path.join("train_eval_metrics", filename, "loss.csv")}')
                    print("--------------------------------------------------------------------------------------")
                    f.close()

    def __call__(self, p, targets, pred_size, batch_idx=None, epoch=None):  # predictions, targets
        # pred_size is not used but needs to be declared due to train_loop design
                          
        targets = targets.to(config.DEVICE, non_blocking=True)
                          
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = intersection_over_union(pbox, tbox[i], GIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.zeros_like(pcls, device=self.device)  # targets
                    t[range(n), tcls[i]] = 1
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.lambda_box
        lobj *= self.lambda_obj
        lcls *= self.lambda_class

        if self.save_logs:
            freq = 100
            if batch_idx % freq == 0:
                with open(os.path.join("train_eval_metrics", self.filename, "loss.csv"), "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, batch_idx, lbox.item(),
                                     lobj.item(), lcls.item()])

                    f.close()

        bs = tobj.shape[0]  # batch size
        # print((lbox + lobj + lcls) * bs)
        return (lbox + lobj + lcls) * bs

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h),
        # where "image" is the idx of the image in the batch: i.e. if batch_size is 4,
        # and all the images contain at least 1 target, the value of "image" will be 0,1,2,3
        na, nt = self.na, targets.shape[0]  # number of anchors (x scale), targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to grid_space gain

        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # torch.arange(na, device=self.device) i.e --> tensor([0, 1, 2]), shape 3
        # torch.arange(na, device=self.device).float().view(na, 1) --> tensor([[0.], [1.], [2.]]), shape (3, 1)
        # torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt) --> shape (3, 84), in words,
        # ai[0,:] repets for n_detections (here 84) times 0, ai[1,:] repets for n_detections (here 84) times 1, etc.
        # ai.shape --> (na,nt) --> i.e. (3, 84)

        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # shape of "targets" before this line is (n_gt_detections, 6) where 6 are (image,class,x,y,w,h)

        # targets.repeat(na, 1, 1) has shape (num anchors (x scale ?), n_gt_detections, 6)  i.e. (3, 84, 6)
        # and torch.equal(targets.repeat(na, 1, 1)[0,:], targets.repeat(na, 1, 1)[1,:]) and
        # torch.equal(targets.repeat(na, 1, 1)[0,:], targets.repeat(na, 1, 1)[2,:]) return true

        # ai[..., None]  has shape (na, nt, 1), i.e. (3, 84, 1)

        # torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2) we can the two tensors along the last dim (2) and
        # we get a (na, nt, 7) tensor, i.e. (3, 84, 7)

        g = 0.5  # bias
        # off.shape --> (5,2)
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

        # for each detection layer...
        for i in range(self.nl):
            # we get the anchors and the predictions shape of the i-eme detection layer
            anchors, shape = self.anchors[i], p[i].shape
            # anchors shape (na x scale,2), i.e (3,2) -- shape.shape i.e. (8, 3, 80, 80, 85)

            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # we set these values of the gain variable as the third,
            # second, third and second values of the shape matrix i.e. (80, 80, 80, 80)

            # "gain" will now look like this: tensor([ 1.,  1., 80., 80., 80., 80.,  1.])

            t = targets * gain  # shape(3,n,7)

            # Match targets to anchors
            # targets.shape (3, 82, 7), gain.shape (7), t.shape (3, 82, 7)
            # where the 7 in dim=2 are (image,class,x,y,w,h, detection_layer_idx) and
            # targets * gain multiply x,y,w,h  times w,h,w,h of the detection_layer

            # if there are predictions (n predicted bboxes > 0)
            if nt:
                # Matches

                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # t.shape is (na, n_gt_boxes, 7)
                # anchors[:, None].shape is (na x scale, 1, 2), i.e. (3,1,2)
                # so this operation is the ratio between the w, h (that after t = targets * gain are
                # representing w,h in terms of "grid blocks") and anchors

                # r.shape is (na, n_gt_boxes, 2) and "r" represents for each detection the number of times
                # a given anchor should be multiplied to explain its height/width

                j = torch.max(r, 1 / r).max(2)[0] < self.anchor_t  # compare

                # torch.max(r, 1 / r) has shape (na, n_gt_boxes, 2) and returns the max between
                # each element of the last dimension

                # torch.max(r, 1 / r).max(2) returns torch.return_type.max variable and to
                # access the output tensor you have to subset the first element with torch.max(r, 1 / r).max(2)[0]

                # torch.max(r, 1 / r).max(2)[0] has shape (na, n_gt_boxes) i.e. (3, 84).
                # The last dimension has "disappeared" because .max(2) just picks the maximum
                # element over the last dimension

                # torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t'] has shape (na, n_gt_boxes) i.e. (3, 84)
                # and is a boolean. What is self.hyp['anchor_t']? Check here below:
                # https://github.com/ultralytics/yolov5/issues/1310#issuecomment-723035010

                # So according to Glen, "j are target-anchor match candidates."

                t = t[j]  # filter

                # t.shape (na (xs?), n_gt_detections, 7) i.e. (3, 84, 7)
                # j.shape (na (xs?), n_gt_detections) i.e. (3, 84)

                # t[j] (?!) shape (93, 7), which means that some gt_detections might require
                # more set of anchors because they are borderline???

                # Offsets
                gxy = t[:, 2:4]  # grid xy

                # t[:, 2:4]. shape i.e (93, 2)

                gxi = gain[[2, 3]] - gxy  # inverse
                # gain.shape is 2 i.e. 80, 80 --- gxy shape i.e. (93, 2)
                # gxi.shape == gxy.shape
                # rescale/shift the x, y coordinates, why? and to get what?

                # in the following lines we are taking care of what specified here:
                # https://github.com/ultralytics/yolov5/issues/6863
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # (gxy % 1 < g) returns a boolean, true if the x/y grid-coordinate divided by 1 is less than g (0.5)
                # false otherwise
                # (gxy > 1) return a boolean, true if the x/y grid-coord is > 1, false otherwise

                # ((gxy % 1 < g) & (gxy > 1)) has shape i.e. (93, 2) and is a boolean, returning true if the conditions
                # specified above are both true, false otherwise

                # ((gxy % 1 < g) & (gxy > 1)).T has shape i.e. (2, 93) and we assign each line to the variables "j", "k"
                # "j" and "k" represent the condition w.r.t. the x and y variable
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # exact same process but with the inverse x,y matrix

                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # j.shape i.e. (5, 93), first row are all ones, and the other rows are the booleans defined above
                t = t.repeat((5, 1, 1))[j]
                # before this line t.shape was i.e. (93, 7),
                # t.repeat((5, 1, 1)) has shape (5, 93, 7)
                # j, define the line above, has shape i.e. (5, 93)

                # t.repeat((5, 1, 1))[j] has shape i.e. (276, 7)

                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # torch.zeros_like(gxy)[None].shape i.e. (1, 93, 2)
                # off[:, None] i.e (5, 1, 2)
                # (torch.zeros_like(gxy)[None] + off[:, None]) shape i.e. (5, 93, 2)

                # (torch.zeros_like(gxy)[None] + off[:, None])[j] shape i.e. (276, 2)

            else:
                # if the batch has no gt_detections
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            # t.shape is i.e. (276, 7)
            # tensor.chunk --> https://pytorch.org/docs/stable/generated/torch.chunk.html
            # t.chunk(4, 1)[0].shape i.e. (276,2),  t.chunk(4, 1)[1].shape i.e. (276,2),
            # t.chunk(4, 1)[2].shape i.e. (276,2) ,t.chunk(4, 1)[3].shape i.e. (276, 1)
            # in other words, it splits the tensor into 4 tensors over the dim=1

            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            # a.long().view(-1).shape i.e. (276)
            # bc.long().T.shape i.e. (2, 276), hence each row assigned to (b, c) accordingly
            gij = (gxy - offsets).long()
            # gxy and offsets of shape i.e. (276, 2)

            gi, gj = gij.T  # grid indices
            # gij.T.shape i.e. (2, 276) hence each row assigned to gi, gj accordingly

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # b.shape i.e. (276)
            # a.shape i.e. (276)
            # torch.tensor.clamp_ is inplace version of clamp (to save up ram usage)
            # each element of the i.e. (276) vector is clipped to 0 if value is less than 0 and
            # clipped to i.e. shape[2] - 1 if value is greater than it.
            # gj.clamp_(0, shape[2] - 1).shape i.e. (276)
            # gi.clamp_(0, shape[3] - 1).shape i.e. (276)

            # (b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)) is a tuple of len(4)

            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # gxy - gij have both shape i.e. (276, 2)
            # gwh.shape is i.e. (276, 2)

            # torch.cat((gxy - gij, gwh), 1).shape is i.e. (276, 4)
            anch.append(anchors[a])  # anchors
            # anchors.shape is i.e. (3, 2)
            # a.shape is i.e. (276)
            # anchors[a].shape is i.e. (276, 2)

            tcls.append(c)  # class

            # c.shape is i.e. 276
        # len(tcls) == len(tbox) == len(indices) == len(anch) == nt (usally 3)

        return tcls, tbox, indices, anch


if __name__ == "__main__":
    check_loss = True
    batch_size = 8
    image_height = 640
    image_width = 640
    nc = len(config.COCO)
    S = [8, 16, 32]

    anchors = config.ANCHORS
    first_out = 48

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out * 4, first_out * 8, first_out * 16), inference=False).to(config.DEVICE)

    model.load_state_dict(state_dict=torch.load("yolov5m_coco.pt"), strict=True)
    loss_fn = ComputeLoss(model, save_logs=False, filename="none")

    """dataset = MY_AUG_MS_COCO_2017(num_classes=nc, anchors=config.ANCHORS,
                                  root_directory=config.ROOT_DIR, transform=None,
                                  train=True, S=S, rect_training=True, default_size=640, bs=4,
                                  bboxes_format="coco")"""

    dataset = Training_Dataset(num_classes=nc, root_directory=config.ROOT_DIR, transform=None,
                           train=True, rect_training=True, default_size=640, bs=8,
                           bboxes_format="coco", ultralytics_loss=True)

    collate_fn = dataset.collate_fn_ultra if dataset.ultralytics_loss else dataset.collate_fn

    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False if dataset.rect_training else True, collate_fn=collate_fn)

    for images, bboxes in loader:
        images = images.float() / 255
        preds = model(images)
        loss = loss_fn(preds, bboxes, pred_size=images.shape[2:4], batch_idx=None, epoch=None)
        print(loss)




