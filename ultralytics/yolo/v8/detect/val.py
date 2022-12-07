import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_file, check_requirements
from ultralytics.yolo.utils.files import yaml_load
from ultralytics.yolo.utils.metrics import ConfusionMatrix, Metric, ap_per_class, box_iou, fitness_detection
from ultralytics.yolo.utils.plotting import output_to_target, plot_images
from ultralytics.yolo.utils.torch_utils import de_parallel


class DetectionValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, logger=None, args=None):
        super().__init__(dataloader, save_dir, pbar, logger, args)
        if self.args.save_json:
            check_requirements(['pycocotools'])
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster
        self.data_dict = yaml_load(check_file(self.args.data)) if self.args.data else None
        self.is_coco = False
        self.class_map = None
        self.targets = None

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        self.nb, _, self.height, self.width = batch["img"].shape  # batch size, channels, height, width
        self.targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        self.targets = self.targets.to(self.device)
        height, width = batch["img"].shape[2:]
        self.targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)  # to pixels
        self.lb = [self.targets[self.targets[:, 0] == i, 1:]
                   for i in range(self.nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def init_metrics(self, model):
        if self.training:
            head = de_parallel(model).model[-1]
        else:
            head = de_parallel(model).model.model[-1]

        if self.data:
            self.is_coco = isinstance(self.data.get('val'),
                                      str) and self.data['val'].endswith(f'coco{os.sep}val2017.txt')
            self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.nc = head.nc
        self.names = model.names
        if isinstance(self.names, (list, tuple)):  # old format
            self.names = dict(enumerate(self.names))

        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.seen = 0
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.metrics = Metric()
        self.loss = torch.zeros(3, device=self.device)
        self.jdict = []
        self.stats = []

    def get_desc(self):
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf_thres,
                                        self.args.iou_thres,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det)
        return preds

    def update_metrics(self, preds, batch):
        # Metrics
        for si, (pred) in enumerate(preds):
            labels = self.targets[self.targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch["ori_shape"][si]
            # path = batch["shape"][si][0]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch["img"][si].shape[1:], predn[:, :4], shape)  # native-space pred

            # Evaluate
            if nl:
                tbox = ops.xywh2xyxy(labels[:, 1:5])  # target boxes
                ops.scale_boxes(batch["img"][si].shape[1:], tbox, shape)  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn, self.iouv)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls)

            # TODO: Save/log
            '''
            if self.args.save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if self.args.save_json:
                pred_masks = scale_image(im[si].shape[1:],
                                         pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1])
                save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
            '''

    def get_stats(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            results = ap_per_class(*stats, plot=self.args.plots, save_dir=self.save_dir, names=self.names)
            self.metrics.update(results[2:])
        self.nt_per_class = np.bincount(stats[3].astype(int), minlength=self.nc)  # number of targets per class
        metrics = {"fitness": fitness_detection(np.array(self.metrics.mean_results()).reshape(1, -1))}
        metrics |= zip(self.metric_keys, self.metrics.mean_results())
        return metrics

    def print_results(self):
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        self.logger.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            self.logger.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        # Print results per class
        if (self.args.verbose or (self.nc < 50 and not self.training)) and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                self.logger.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))

    def _process_batch(self, detections, labels, iouv):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    def get_dataloader(self, dataset_path, batch_size):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, mode="val")[0]

    # TODO: align with train loss metrics
    @property
    def metric_keys(self):
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP_0.5(B)", "metrics/mAP_0.5:0.95(B)"]

    def plot_val_samples(self, batch, ni):
        images = batch["img"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(images,
                    batch_idx,
                    cls,
                    bboxes,
                    paths=paths,
                    fname=self.save_dir / f"val_batch{ni}_labels.jpg",
                    names=self.names)

    def plot_predictions(self, batch, preds, ni):
        images = batch["img"]
        paths = batch["im_file"]
        plot_images(images, *output_to_target(preds, max_det=15), paths, self.save_dir / f'val_batch{ni}_pred.jpg',
                    self.names)  # pred


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG.parent, config_name=DEFAULT_CONFIG.name)
def val(cfg):
    cfg.data = cfg.data or "coco128.yaml"
    validator = DetectionValidator(args=cfg)
    validator(model=cfg.model)


if __name__ == "__main__":
    val()
