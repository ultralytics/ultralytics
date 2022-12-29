import hydra
import torch

from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CONFIG


class ClassificationValidator(BaseValidator):

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
        acc = torch.stack((self.correct[:, 0], self.correct.max(1).values), dim=1)  # (top1, top5) accuracy
        top1, top5 = acc.mean(0).tolist()
        return {"top1": top1, "top5": top5, "fitness": top5}

    def get_dataloader(self, dataset_path, batch_size):
        return build_classification_dataloader(path=dataset_path, imgsz=self.args.imgsz, batch_size=batch_size)

    @property
    def metric_keys(self):
        return ["top1", "top5"]


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def val(cfg):
    cfg.data = cfg.data or "imagenette160"
    cfg.model = cfg.model or "resnet18"
    validator = ClassificationValidator(args=cfg)
    validator(model=cfg.model)


if __name__ == "__main__":
    val()
