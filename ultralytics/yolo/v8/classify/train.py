# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import torchvision

from ultralytics.nn.tasks import ClassificationModel, attempt_load_one_weight
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CONFIG
from ultralytics.yolo.utils.torch_utils import strip_optimizer


class ClassificationTrainer(BaseTrainer):

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        super().__init__(config, overrides)

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ClassificationModel(cfg, nc=self.data["nc"])

        pretrained = False
        for m in model.modules():
            if not pretrained and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training

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
            self.model, _ = attempt_load_one_weight(model, device='cpu')
        elif model.endswith(".yaml"):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            pretrained = True
            self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1' if pretrained else None)
        else:
            FileNotFoundError(f'ERROR: model={model} not found locally or online. Please check model name.')

        return  # dont return ckpt. Classification doesn't support resume

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        return build_classification_dataloader(path=dataset_path,
                                               imgsz=self.args.imgsz,
                                               batch_size=batch_size if mode == "train" else (batch_size * 2),
                                               augment=mode == "train",
                                               rank=rank,
                                               workers=self.args.workers)

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        return ('\n' + '%11s' * (4 + len(self.loss_names))) % \
            ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def get_validator(self):
        self.loss_names = ['loss']
        return v8.classify.ClassificationValidator(self.test_loader, self.save_dir, logger=self.console)

    def criterion(self, preds, batch):
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction='sum') / self.args.nbs
        loss_items = loss.detach()
        return loss, loss_items

    # def label_loss_items(self, loss_items=None, prefix="train"):
    #     """
    #     Returns a loss dict with labelled training loss items tensor
    #     """
    #     # Not needed for classification but necessary for segmentation & detection
    #     keys = [f"{prefix}/{x}" for x in self.loss_names]
    #     if loss_items is not None:
    #         loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
    #         return dict(zip(keys, loss_items))
    #     else:
    #         return keys

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(loss_items), 5)]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def resume_training(self, ckpt):
        pass

    def final_eval(self):
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                # TODO: validate best.pt after training completes
                # if f is self.best:
                #     self.console.info(f'\nValidating {f}...')
                #     self.validator.args.save_json = True
                #     self.metrics = self.validator(model=f)
                #     self.metrics.pop('fitness', None)
                #     self.run_callbacks('on_fit_epoch_end')


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def train(cfg):
    cfg.model = cfg.model or "yolov8n-cls.yaml"  # or "resnet18"
    cfg.data = cfg.data or "mnist160"  # or yolo.ClassificationDataset("mnist")
    cfg.lr0 = 0.1
    cfg.weight_decay = 5e-5
    cfg.label_smoothing = 0.1
    cfg.warmup_epochs = 0.0
    trainer = ClassificationTrainer(cfg)
    trainer.train()
    # from ultralytics import YOLO
    # model = YOLO(cfg.model)
    # model.train(**cfg)


if __name__ == "__main__":
    """
    CLI usage:
    python ultralytics/yolo/v8/classify/train.py model=resnet18 data=imagenette160 epochs=1 imgsz=224

    TODO:
    Direct cli support, i.e, yolov8 classify_train args.epochs 10
    """
    train()
