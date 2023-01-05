import hydra
import torch

from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CONFIG
from ultralytics.yolo.utils.metrics import ClassifyMetrics


class ClassificationValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, logger=None, args=None):
        super().__init__(dataloader, save_dir, pbar, logger, args)
        self.metrics = ClassifyMetrics()

    def init_metrics(self, model):
        self.correct = torch.tensor([], device=next(model.parameters()).device)

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        targets = batch["cls"]
        correct_in_batch = (targets[:, None] == preds).float()
        self.correct = torch.cat((self.correct, correct_in_batch))

    def get_stats(self):
        self.metrics.process(self.correct)
        return self.metrics.results_dict

    def get_dataloader(self, dataset_path, batch_size):
        return build_classification_dataloader(path=dataset_path, imgsz=self.args.imgsz, batch_size=batch_size)


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def val(cfg):
    cfg.data = cfg.data or "imagenette160"
    cfg.model = cfg.model or "resnet18"
    validator = ClassificationValidator(args=cfg)
    validator(model=cfg.model)


if __name__ == "__main__":
    val()
