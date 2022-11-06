import os
from numpy import save
from pathlib import Path

import torch
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.metrics import ConfusionMatrix, Metrics
from ultralytics.yolo.utils.modeling import yaml_load
from ultralytics.yolo.utils.torch_utils import de_parallel


class SegmentationValidator(BaseValidator):
    def __init__(self, dataloader, pbar=None, logger=None, args=None):
        super().__init__(dataloader, pbar, logger, args)
        if self.args.save_json:
            check_requirements(['pycocotools'])
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster
        self.data_dict = yaml_load(self.args.data) if self.args.data else None
        self.is_coco = False
        self.class_map = None
        self.targets = None

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 225
        batch["bboxes"] = batch["bboxes"].to(self.device)
        batch["masks"] = batch["masks"].to(self.device).float()
        self.nb, _, self.height, self.width = batch["img"].shape  # batch size, channels, height, width
        self.targets  = torch.cat((batch["batch_idx"].view(-1,1), batch["cls"].view(-1,1), batch["bboxes"]), 1)
        self.lb = [self.targets[self.targets[:, 0] == i, 1:] for i in range(self.nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def init_metrics(self, model):
        head = de_parallel(model).model[-1]
        if self.data_dict:
            self.is_coco = isinstance(self.data_dict.get('val'), str) and self.data_dict['val'].endswith(f'coco{os.sep}val2017.txt')
            self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))

        self.nc = head.nc
        self.nm = head.nm
        self.names = model.names
        if isinstance(self.names, (list, tuple)):  # old format
            self.names = dict(enumerate(self.names))

        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.seen = 0
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.metrics = Metrics()
        self.loss = torch.zeros(4, device=self.device)
        self.jdict = []
        self.stats = []


    def get_desc(self):
        return ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)", "Mask(P", "R",
                                  "mAP50", "mAP50-95)")

    def preprocess_preds(self, preds):
        preds[0] = ops.non_max_suppression(preds[0],
                                        self.args.conf_thres,
                                        self.args.iou_thres,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det,
                                        nm=self.nm)
        return preds

    def update_metrics(self, preds, batch):
        # Metrics
        plot_masks = []  # masks for plotting
        for si, (pred, proto) in enumerate(preds):
            labels = self.targets[self.targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            import pdb;pdb.set_trace()
            path, shape = Path(paths[si]), shapes[si][0]
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Masks
            midx = [si] if overlap else targets[:, 0] == si
            gt_masks = masks[midx]
            pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=im[si].shape[1:])

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct_bboxes = process_batch(predn, labelsn, iouv)
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls)

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if plots and batch_i < 3:
                plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                pred_masks = scale_image(im[si].shape[1:],
                                         pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1])
                save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            plot_images_and_masks(im, targets, masks, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)
            plot_images_and_masks(im, output_to_target(preds, max_det=15), plot_masks, paths,
                                  save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred


    def get_stats(self):
        acc = torch.stack((self.correct[:, 0], self.correct.max(1).values), dim=1)  # (top1, top5) accuracy
        top1, top5 = acc.mean(0).tolist()
        return {"top1": top1, "top5": top5, "fitness": top5}
