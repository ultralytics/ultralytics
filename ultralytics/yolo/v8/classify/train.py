import subprocess
import time
from pathlib import Path

import hydra
import torch

from ultralytics.yolo import v8
from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG, BaseTrainer
from ultralytics.yolo.utils import colorstr
from ultralytics.yolo.utils.downloads import download
from ultralytics.yolo.utils.files import WorkingDirectory
from ultralytics.yolo.utils.torch_utils import LOCAL_RANK, torch_distributed_zero_first


# BaseTrainer python usage
class ClassificationTrainer(BaseTrainer):

    def get_dataloader(self, dataset_path, batch_size=None, rank=0):
        return build_classification_dataloader(path=dataset_path,
                                               imgsz=self.args.img_size,
                                               batch_size=self.args.batch_size,
                                               rank=rank)

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def get_validator(self):
        return v8.classify.ClassificationValidator(self.test_loader, self.device, logger=self.console)

    def criterion(self, preds, batch):
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"])
        return loss, loss


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG.parent, config_name=DEFAULT_CONFIG.name)
def train(cfg):
    cfg.model = cfg.model or "resnet18"
    cfg.data = cfg.data or "imagenette160"  # or yolo.ClassificationDataset("mnist")
    trainer = ClassificationTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    """
    CLI usage:
    python ultralytics/yolo/v8/classify/train.py model=resnet18 data=imagenette160 epochs=1 img_size=224

    TODO:
    Direct cli support, i.e, yolov8 classify_train args.epochs 10
    """
    train()
