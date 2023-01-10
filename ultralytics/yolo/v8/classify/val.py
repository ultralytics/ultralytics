# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra

from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CONFIG
from ultralytics.yolo.utils.metrics import ClassifyMetrics


class ClassificationValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, logger=None, args=None):
        super().__init__(dataloader, save_dir, pbar, logger, args)
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', 'top5_acc')

    def init_metrics(self, model):
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        self.pred.append(preds.argsort(1, descending=True)[:, :5])
        self.targets.append(batch["cls"])

    def get_stats(self):
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def get_dataloader(self, dataset_path, batch_size):
        return build_classification_dataloader(path=dataset_path,
                                               imgsz=self.args.imgsz,
                                               batch_size=batch_size,
                                               workers=self.args.workers)

    def print_results(self):
        pf = '%22s' + '%11.3g' * len(self.metrics.keys)  # print format
        self.logger.info(pf % ("all", self.metrics.top1, self.metrics.top5))


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def val(cfg):
    cfg.data = cfg.data or "imagenette160"
    cfg.model = cfg.model or "resnet18"
    validator = ClassificationValidator(args=cfg)
    validator(model=cfg.model)


if __name__ == "__main__":
    val()
