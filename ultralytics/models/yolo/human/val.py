# Ultralytics YOLO 🚀, AGPL-3.0 license
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.dataset import HumanDataset
from ultralytics.engine.results import Human, Results
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr
from ultralytics.utils.metrics import HumanMetrics, box_iou


class HumanValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a human model.

    Example:
        ```python
        from ultralytics.models.yolo.human import HumanValidator

        args = dict(model='yolov8n-human.pt', data='coco8.yaml')
        validator = HumanValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "human"
        self.metrics = HumanMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def build_dataset(self, img_path, mode="val", batch=None):
        cfg = self.args
        return HumanDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,
            rect=cfg.rect or mode == "val",  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            classes=cfg.classes,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["attributes"] = batch["attributes"].to(self.device).float()
        return batch

    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
        attributes = batch["attributes"][idx]
        return {"attributes": attributes, **super()._prepare_batch(si, batch)}

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"], iou = self._process_batch(predn, bbox, cls)
                self._process_attributes(predn[:, 6:], pbatch["attributes"], iou)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou), iou

    def _process_attributes(self, pred_attrs, gt_attrs, iou, iou_thres=0.5):
        """
        Process Human Attributes and compute the accuracy.

        Args:
            predn_attrs (torch.Tensor): The predictions of attributes with shape [M, 11].
            gt_attrs (torch.Tensor): The grounding truth of attributes with shape [N, 5].
            iou (torch.Tensor): The iou values between gt boxes and predicted human boxes with shape [N, M],
                and it's used to choose the true positive to evaluate attributes.
            iou_thres (float): The iou threshold to determine true positive samples, default: 0.5.

        Returns:
            The accuracy for each human attribute.
        """
        values, indices = iou.max(1)
        indices = indices[values >= iou_thres]  # indices for tp
        gt_attrs = gt_attrs[values >= iou_thres]
        pred_attrs = Human(pred_attrs[indices])
        weight = gt_attrs[:, 0]
        height = gt_attrs[:, 1]
        gender = gt_attrs[:, 2]
        age = gt_attrs[:, 3]
        ethnicity = gt_attrs[:, 4]
        acc_w = 1 - (pred_attrs.weight - weight).abs() / weight
        acc_h = 1 - (pred_attrs.height - height).abs() / height
        acc_g = (pred_attrs.cls_gender == gender).float()
        acc_a = 1 - (pred_attrs.age - age).abs() / age
        acc_e = (pred_attrs.cls_ethnicity == ethnicity).float()

        self.metrics.attrs_stats["weight"].append(acc_w.clip(0, 1))
        self.metrics.attrs_stats["height"].append(acc_h.clip(0, 1))
        self.metrics.attrs_stats["gender"].append(acc_g)
        self.metrics.attrs_stats["age"].append(acc_a.clip(0, 1))
        self.metrics.attrs_stats["ethnicity"].append(acc_e)

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        im = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        result = Results(im, path=None, names=self.names, boxes=predn[:, :6], human=predn[:, 6:])
        result.save_txt(file, save_conf=save_conf)

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 11) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "acc(W)",  # weight
            "acc(H)",  # height
            "acc(G)",  # gender
            "acc(A)",  # age
            "acc(E)",  # ethnicity
        )
