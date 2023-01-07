import hydra
import torch
import torchvision

from ultralytics.nn.tasks import ClassificationModel, attempt_load_weights
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CONFIG


class ClassificationTrainer(BaseTrainer):

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        super().__init__(config, overrides)

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None):
        model = ClassificationModel(cfg, nc=self.data["nc"])
        if weights:
            model.load(weights)

        # Update defaults
        if self.args.imgsz == 640:
            self.args.imgsz = 224

        return model

    def setup_model(self):
        """
        load/create/download model for any task
        """
        # classification models require special handling

        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model = str(self.model)
        # Load a YOLO model locally, from torchvision, or from Ultralytics assets
        if model.endswith(".pt"):
            self.model = attempt_load_weights(model, device='cpu')
        elif model.endswith(".yaml"):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            pretrained = True
            self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1' if pretrained else None)
        else:
            FileNotFoundError(f'ERROR: model={model} not found locally or online. Please check model name.')

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

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def get_validator(self):
        return v8.classify.ClassificationValidator(self.test_loader, self.save_dir, logger=self.console)

    def criterion(self, preds, batch):
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"])
        return loss, loss

    def resume_training(self, ckpt):
        pass

    def final_eval(self):
        pass


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def train(cfg):
    cfg.model = cfg.model or "yolov8n-cls.yaml"  # or "resnet18"
    cfg.data = cfg.data or "imagenette160"  # or yolo.ClassificationDataset("mnist")
    # trainer = ClassificationTrainer(cfg)
    # trainer.train()
    from ultralytics import YOLO
    model = YOLO(cfg.model)
    model.train(**cfg)


if __name__ == "__main__":
    """
    CLI usage:
    python ultralytics/yolo/v8/classify/train.py model=resnet18 data=imagenette160 epochs=1 imgsz=224

    TODO:
    Direct cli support, i.e, yolov8 classify_train args.epochs 10
    """
    train()
