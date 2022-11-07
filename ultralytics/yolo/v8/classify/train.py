import subprocess
import time
from pathlib import Path

import hydra
import torch

from ultralytics.yolo import v8
from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG, BaseTrainer
from ultralytics.yolo.utils.downloads import download
from ultralytics.yolo.utils.files import WorkingDirectory
from ultralytics.yolo.utils.loggers import colorstr
from ultralytics.yolo.utils.torch_utils import LOCAL_RANK, torch_distributed_zero_first


# BaseTrainer python usage
class ClassificationTrainer(BaseTrainer):

    def get_dataset(self, dataset):
        # temporary solution. Replace with new ultralytics.yolo.ClassificationDataset module
        data = Path("datasets") / dataset
        with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(Path.cwd()):
            data_dir = data if data.is_dir() else (Path.cwd() / data)
            if not data_dir.is_dir():
                self.console.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
                t = time.time()
                if str(data) == 'imagenet':
                    subprocess.run(f"bash {v8.ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
                else:
                    url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip'
                    download(url, dir=data_dir.parent)
                s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
                self.console.info(s)
        train_set = data_dir / "train"
        test_set = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test or data/val

        return train_set, test_set

    def get_dataloader(self, dataset_path, batch_size=None, rank=0):
        return build_classification_dataloader(path=dataset_path, batch_size=self.args.batch_size, rank=rank)

    def get_validator(self):
        return v8.classify.ClassificationValidator(self.test_loader, self.device, logger=self.console)

    def criterion(self, preds, targets):
        return torch.nn.functional.cross_entropy(preds, targets)


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
