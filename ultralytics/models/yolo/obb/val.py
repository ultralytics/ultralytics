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
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -2:-1]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        idx = batch['batch_idx'] == si
        cls = batch['cls'][idx].squeeze(-1)
        bbox = batch['bboxes'][idx]
        ori_shape = batch['ori_shape'][si]
        imgsz = batch['img'].shape[2:]
        ratio_pad = batch['ratio_pad'][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            ops.scale_rotated_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        prepared_batch = dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)
        return prepared_batch

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        ops.scale_rotated_boxes(pbatch['imgsz'], predn[:, :4], pbatch['ori_shape'],
                                ratio_pad=pbatch['ratio_pad'])  # native-space pred
        return predn

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
        box = ops.xywhr2xyxyxyxy(torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)).view(-1, 8)
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
