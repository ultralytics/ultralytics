import subprocess
import time
from pathlib import Path

import hydra
import torch
import torchvision
from val import ClassificationValidator

from ultralytics.yolo import BaseTrainer, utils, v8
from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.trainer import CONFIG_PATH_ABS, DEFAULT_CONFIG


# BaseTrainer python usage
class ClassificationTrainer(BaseTrainer):

    def get_dataset(self):
        # temporary solution. Replace with new ultralytics.yolo.ClassificationDataset module
        data = Path("datasets") / self.data
        with utils.torch_distributed_zero_first(utils.LOCAL_RANK), utils.WorkingDirectory(Path.cwd()):
            data_dir = data if data.is_dir() else (Path.cwd() / data)
            if not data_dir.is_dir():
                self.console.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
                t = time.time()
                if str(data) == 'imagenet':
                    subprocess.run(f"bash {v8.ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
                else:
                    url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{self.data}.zip'
                    utils.download(url, dir=data_dir.parent)
                # TODO: add colorstr
                s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {'bold', data_dir}\n"
                self.console.info(s)
        train_set = data_dir / "train"
        test_set = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test or data/val

        return train_set, test_set

    def get_dataloader(self, dataset, batch_size=None, rank=0):
        return build_classification_dataloader(path=dataset, batch_size=self.train.batch_size, rank=rank)

    def get_model(self):
        # temp. minimal. only supports torchvision models
        if self.model in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
            model = torchvision.models.__dict__[self.model](weights='IMAGENET1K_V1' if self.train.pretrained else None)
        else:
            raise ModuleNotFoundError(f'--model {self.model} not found.')
        for m in model.modules():
            if not self.train.pretrained and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in model.parameters():
            p.requires_grad = True  # for training

        return model

    def get_validator(self):
        return ClassificationValidator(self.test_loader, self.device, logger=self.console)  # validator

    def criterion(self, preds, targets):
        return torch.nn.functional.cross_entropy(preds, targets)


@hydra.main(version_base=None, config_path=CONFIG_PATH_ABS, config_name=str(DEFAULT_CONFIG).split(".")[0])
def train(cfg):
    cfg.model = cfg.model or "squeezenet1_0"
    cfg.data = cfg.data or "imagenette160"  # or yolo.ClassificationDataset("mnist")
    trainer = ClassificationTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    """
    CLI usage:
    python ../path/to/train.py train.epochs=10 train.project="name" hyps.lr0=0.1

    TODO:
    Direct cli support, i.e, yolov8 classify_train train.epochs 10
    """
    train()
