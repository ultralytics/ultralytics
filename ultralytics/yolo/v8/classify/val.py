# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER
from ultralytics.yolo.utils.metrics import ClassifyMetrics


class ClassificationValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
        super().__init__(dataloader, save_dir, pbar, args)
        self.args.task = 'classify'
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', 'top5_acc')

    def init_metrics(self, model):
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = batch['img'].half() if self.args.half else batch['img'].float()
        batch['cls'] = batch['cls'].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        n5 = min(len(self.model.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5])
        self.targets.append(batch['cls'])

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        # self.metrics.confusion_matrix = self.confusion_matrix  # TODO: classification ConfusionMatrix

    def get_stats(self):
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def get_dataloader(self, dataset_path, batch_size):
        return build_classification_dataloader(path=dataset_path,
                                               imgsz=self.args.imgsz,
                                               batch_size=batch_size,
                                               augment=False,
                                               shuffle=False,
                                               workers=self.args.workers)

    def print_results(self):
        pf = '%22s' + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.metrics.top1, self.metrics.top5))


def val(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n-cls.pt'  # or "resnet18"
    data = cfg.data or 'mnist160'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = ClassificationValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()
