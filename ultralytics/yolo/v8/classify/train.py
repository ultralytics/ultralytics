from pathlib import Path

import hydra
import torch
import torchvision

from ultralytics.nn.tasks import ClassificationModel, attempt_load_weights
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CONFIG


class ClassificationTrainer(BaseTrainer):

    def __init__(self, config=DEFAULT_CONFIG, overrides={}):
        overrides["task"] = "classify"
        super().__init__(config, overrides)

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None):
        model = ClassificationModel(cfg, nc=self.data["nc"])
        if weights:
            model.load(weights)

        return model

    def setup_model(self):
        """
        load/create/download model for any task
        """
        # classification models require special handling

        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model = self.model
        pretrained = False
        # Load a YOLO model locally, from torchvision, or from Ultralytics assets
        if model.endswith(".pt"):
            model = model.split(".")[0]
            pretrained = True
        else:
            self.model = self.get_model(cfg=model)

        # order: check local file -> torchvision assets -> ultralytics asset
        if Path(f"{model}.pt").is_file():  # local file
            self.model = attempt_load_weights(f"{model}.pt", device='cpu')
        elif model in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1' if pretrained else None)
        else:
            self.model = attempt_load_weights(f"{model}.pt", device='cpu')

        return  # dont return ckpt. Classification doesn't support resume

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        return build_classification_dataloader(path=dataset_path,
                                               imgsz=self.args.imgsz,
                                               batch_size=batch_size,
                                               rank=rank)

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def get_validator(self):
        return v8.classify.ClassificationValidator(self.test_loader, self.save_dir, logger=self.console)

    def criterion(self, preds, batch):
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"])
        return loss, loss

    def check_resume(self):
        pass

    def resume_training(self, ckpt):
        pass

    def final_eval(self):
        pass


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def train(cfg):
    cfg.model = cfg.model or "resnet18"
    cfg.data = cfg.data or "imagenette160"  # or yolo.ClassificationDataset("mnist")
    trainer = ClassificationTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    """
    CLI usage:
    python ultralytics/yolo/v8/classify/train.py model=resnet18 data=imagenette160 epochs=1 imgsz=224

    TODO:
    Direct cli support, i.e, yolov8 classify_train args.epochs 10
    """
    train()
