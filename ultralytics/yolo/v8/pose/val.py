from ultralytics.yolo.v8.detect import DetectionValidator
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils.metrics import PoseMetrics, box_iou, pose_iou
import torch
import numpy as np


class PoseValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
        super().__init__(dataloader, save_dir, pbar, args)
        self.args.task = 'pose'
        self.metrics = PoseMetrics(save_dir=self.save_dir)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch['keypoints'] = batch['keypoints'].to(self.device).float()
        return batch

    def get_desc(self):
        return ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Pose(P',
                                         'R', 'mAP50', 'mAP50-95)')

    def _process_batch(self, detections, labels, pred_kpts=None, gt_kpts=None, kpts=False):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
            pred_kpts (array[N, 17, 3])
            gt_kpts (array[N, 17, 3])
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        if kpts:
            iou = mask_iou(gt_kpts.view(gt_kpts.shape[0], -1), pred_kpts.view(pred_kpts.shape[0], -1))
        else:  # boxes
            iou = box_iou(labels[:, 1:], detections[:, :4])

        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)



def val(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n-pose.pt'
    data = cfg.data or 'coco128-pose.yaml'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = PoseValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()

