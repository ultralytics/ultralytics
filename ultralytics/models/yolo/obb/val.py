# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images


class OBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model='yolov8n-obb.pt', data='coco8-seg.yaml')
        validator = OBBValidator(args=args)
        validator(model=args['model'])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = 'obb'
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       nc=self.nc,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det,
                                       rotated=True)

    def _process_batch(self, detections, labels):
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
        iou = batch_probiou(labels[:, 1:], torch.cat([detections[:, :4], detections[:, -2:-1]], dim=-1))
        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

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
                    # TODO: Not supported confusion matrix for obb yet.
                    # if self.args.plots:
                    #     self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # TODO: optimize this inefficient way of scaling rotated boxes
            ops.scale_rotated_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                                    ratio_pad=batch['ratio_pad'][si])  # native-space pred

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = bbox.view(-1, 8) * torch.tensor(
                    (height, width), device=self.device)[[1, 0, 1, 0, 1, 0, 1, 0]]  # target boxes
                tbox = ops.xyxyxyxy2xywhr(tbox)
                ops.scale_rotated_boxes(batch['img'][si].shape[1:], tbox, shape,
                                        ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: Not supported confusion matrix for obb yet.
                # if self.args.plots:
                #     self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch['img'],
                    *output_to_rotated_target(preds, max_det=self.args.max_det),
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)  # pred

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xywhr2xyxyxyxy(torch.cat([predn[:, :4], predn[:, -1:]], dim=-1))
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5),
                'file_name': stem})

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        return stats
