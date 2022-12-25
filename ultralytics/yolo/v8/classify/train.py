import hydra
import torch

from ultralytics.yolo import v8
from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG, BaseTrainer
from ultralytics.yolo.utils.modeling import get_model
from ultralytics.yolo.utils.modeling.tasks import ClassificationModel


class ClassificationTrainer(BaseTrainer):

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def load_model(self, model_cfg=None, weights=None):
        # TODO: why treat clf models as unique. We should have clf yamls?
        if isinstance(weights, dict):  # yolo ckpt
            weights = weights["model"]
        if weights and not weights.__class__.__name__.startswith("yolo"):  # torchvision
            model = weights
        else:
            model = ClassificationModel(model_cfg, weights, self.data["nc"])
        ClassificationModel.reshape_outputs(model, self.data["nc"])
        for m in model.modules():
            if not weights and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout is not None:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def load_ckpt(self, ckpt):
        return get_model(ckpt)

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
        return v8.classify.ClassificationValidator(self.test_loader, self.device, logger=self.console)

    def criterion(self, preds, batch):
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"])
        return loss, loss

    def check_resume(self):
        pass

    def resume_training(self, ckpt):
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
