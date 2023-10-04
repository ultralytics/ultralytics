# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, MultiTaskMetrics, box_iou, kpt_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class MultiTaskValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseValidator

        args = dict(model='yolov8n-pose.pt', data='coco8-pose.yaml')
        validator = PoseValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'PoseValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = 'multi-task'
        self.metrics = MultiTaskMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')

        self.plot_masks = None

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
        batch = super().preprocess(batch)
        batch['keypoints'] = batch['keypoints'].to(self.device).float()
        batch['masks'] = batch['masks'].to(self.device).float()
        return batch

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        return ('%22s' + '%11s' * 14) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Pose(P',
                                         'R', 'mAP50', 'mAP50-95)', 'Mask(P', 'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        """Apply non-maximum suppression and return detections with high confidence scores."""
        p = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        # second output is len 4 if pt, but only 1 if exported
        proto = preds[1][-1] if len(preds[1]) == 4 else preds[1]
        return p, proto

    def init_metrics(self, model):
        """Initiate pose estimation metrics for YOLO model."""
        super().init_metrics(model)
        self.kpt_shape = self.data['kpt_shape']
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

        self.plot_masks = []
        if self.args.save_json:
            check_requirements('pycocotools>=2.0.6')
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            kpts = batch['keypoints'][idx]
            img_size = batch['img'][si].shape[1:]
            ratio_pad = batch['ratio_pad'][si]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            nk = kpts.shape[1]  # number of keypoints
            kpts_dim = kpts.shape[2]
            shape = batch['ori_shape'][si]
            correct_kpts = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            correct_masks = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, correct_kpts, correct_masks, *torch.zeros(
                        (2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Masks
            midx = [si] if self.args.overlap_mask else idx
            gt_masks = batch['masks'][midx]
            pred_masks = self.process(proto, pred[:, 6 + nk * kpts_dim:], pred[:, :4], shape=img_size)

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # predicted bboxes in native-space
            ops.scale_boxes(img_size, predn[:, :4], shape, ratio_pad=ratio_pad)
            # parse keypoints
            pred_kpts = predn[:, 6:6 + nk * kpts_dim].view(npr, nk, kpts_dim)
            # keypoints in native-space
            ops.scale_coords(img_size, pred_kpts, shape, ratio_pad=ratio_pad)

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                # target bboxes
                tbox = ops.xywh2xyxy(bbox) * torch.tensor((width, height, width, height), device=self.device)
                # target bboxes in native-space
                ops.scale_boxes(img_size, tbox, shape, ratio_pad=ratio_pad)
                tkpts = kpts.clone()
                tkpts[..., 0] *= width
                tkpts[..., 1] *= height
                # target keypoints in native-space
                tkpts = ops.scale_coords(img_size, tkpts, shape, ratio_pad=ratio_pad)
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn[:, :6], labelsn)
                correct_kpts = self._process_batch(predn[:, :6], labelsn, pred_kpts=pred_kpts, gt_kpts=tkpts)
                correct_masks = self._process_batch(predn[:, :6],
                                                    labelsn,
                                                    pred_masks=pred_masks,
                                                    gt_masks=gt_masks,
                                                    overlap=self.args.overlap_mask)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)

            self.stats.append((correct_bboxes, correct_kpts, correct_masks, pred[:, 4], pred[:, 5], cls.squeeze(-1)))

            if self.args.plots and self.batch_i < 3:
                pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
                self.plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save
            # if self.args.save_json:
            # self.pred_to_json(predn, batch['im_file'][si])

            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def _process_batch(
        self,
        detections,
        labels,
        pred_kpts=None,
        gt_kpts=None,
        pred_masks=None,
        gt_masks=None,
        overlap=False,
    ):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.
            pred_kpts (torch.Tensor, optional): Tensor of shape [N, 51] representing predicted keypoints.
                51 corresponds to 17 keypoints each with 3 values.
            gt_kpts (torch.Tensor, optional): Tensor of shape [N, 51] representing ground truth keypoints.

            # TODO: masks

        Returns:
            torch.Tensor: Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        if pred_masks is not None and gt_masks is not None:
            if overlap:
                nl = len(labels)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode='bilinear', align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        elif pred_kpts is not None and gt_kpts is not None:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(labels[:, 1:])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
        else:  # boxes
            iou = box_iou(labels[:, 1:], detections[:, :4])

        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def plot_val_samples(self, batch, batch_number):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        plot_images(
            batch['img'],
            batch['batch_idx'],
            batch['cls'].squeeze(-1),
            batch['bboxes'],
            masks=batch['masks'],
            kpts=batch['keypoints'],
            paths=batch['im_file'],
            fname=self.save_dir / f'val_batch{batch_number}_labels.jpg',
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        max_det = 15

        preds = preds[0]
        pred_kpts = [p[:max_det, 6:6 + np.prod(self.kpt_shape)].view(-1, *self.kpt_shape) for p in preds]
        pred_kpts = torch.cat(pred_kpts, 0)

        if len(self.plot_masks):
            masks = torch.cat(self.plot_masks, dim=0)
        else:
            masks = self.plot_masks

        # TODO: find out what causing slow plotting speed
        # not set to self.args.max_det due to slow plotting speed

        plot_images(
            batch['img'],
            *output_to_target(preds, max_det=max_det),
            masks=masks,
            kpts=pred_kpts,
            paths=batch['im_file'],
            fname=self.save_dir / f'val_batch{ni}_pred.jpg',
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
        self.plot_masks.clear()
