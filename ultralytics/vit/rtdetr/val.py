# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.yolo.data import YOLODataset
from ultralytics.yolo.data.augment import Compose, Format, LetterBox
from ultralytics.yolo.utils import colorstr, ops
from ultralytics.yolo.v8.detect import DetectionValidator

__all__ = ['RTDETRValidator']


# TODO: Temporarily, RT-DETR does not need padding.
class RTDETRDataset(YOLODataset):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)

    def build_transforms(self, hyp=None):
        """Temporarily, only for evaluation."""
        transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


class RTDETRValidator(DetectionValidator):

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        bboxes, scores = preds[:2]  # (1, bs, 300, 4), (1, bs, 300, nc)
        bboxes, scores = bboxes.squeeze_(0), scores.squeeze_(0)  # (bs, 300, 4)
        bs = len(bboxes)
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here.
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            outputs[i] = pred  # [idx]

        return outputs

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            predn[..., [0, 2]] *= shape[1]  # native-space pred
            predn[..., [1, 3]] *= shape[0]  # native-space pred

            # Evaluate
            if nl:
                tbox = ops.xywh2xyxy(bbox)  # target boxes
                tbox[..., [0, 2]] *= shape[1]  # native-space pred
                tbox[..., [1, 3]] *= shape[0]  # native-space pred
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)
