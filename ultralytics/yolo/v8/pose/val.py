from ultralytics.yolo.v8.detect import DetectionValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, NUM_THREADS, ops
from ultralytics.yolo.utils.metrics import PoseMetrics, box_iou, mask_iou


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

    def postprocess(self, preds):
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        return super().update_metrics(preds, batch)



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

