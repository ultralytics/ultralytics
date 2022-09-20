import time
from pathlib import Path
import subprocess
import torch
import torchvision
import torchvision.transforms as T
import torch.hub as hub

from ultralytics.yolo import BaseTrainer, utils, v8
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG, CONFIG_PATH_ABS
import hydra
from omegaconf import DictConfig, OmegaConf


# BaseTrainer python usage 
class Trainer(BaseTrainer):
    def get_dataset(self):
        # temporary solution. Replace with new ultralytics.yolo.ClassificationDataset module
        with utils.torch_distributed_zero_first(utils.LOCAL_RANK), utils.WorkingDirectory(Path.cwd()):
            data_dir = self.dataset if self.dataset.is_dir() else (Path.cwd() / self.dataset)
            if not data_dir.is_dir():
                self.console.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
                t = time.time()
                if str(self.dataset) == 'imagenet':
                    subprocess.run(f"bash {v8.ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
                else:
                    url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{self.dataset}.zip'
                    utils.download(url, dir=data_dir.parent)
                # TODO: add colorstr
                s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {'bold', data_dir}\n"
                self.console.info(s)
        
        train_set = data_dir / "train"
        test_set =  data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test or data/val        
        transform =  T.Compose([T.ToTensor()])
        train_set = torchvision.datasets.ImageFolder(train_set, transform=transform)
        test_set = torchvision.datasets.ImageFolder(test_set, transform=transform)

        return train_set, test_set
    
    def get_dataloader(self, dataset, batch_size=None):
        loader = torchvision.datasets.DataLoader(dataset=dataset,
                                                batch_size=batch_size or self.batch_size)
        
        return loader
    
    def get_model(self):
        # temp. minimal. only supports torchvision models 
        if self.train.model in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
            model = torchvision.models.__dict__[self.train.model](weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ModuleNotFoundError(f'--model {opt.model} not found. Available models are: \n' + '\n'.join(m))
        for m in model.modules():
            if not self.train.pretrained and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in model.parameters():
            p.requires_grad = True  # for training
        model = model.to(self.device)
    

"""
CLI usage:
python ../path/to/train.py train.epochs=10 train.project="name" hyps.lr0=0.1

TODO:
Direct cli support, i.e, yolov8 classify_train train.epochs 10 
"""

@hydra.main(config_path=CONFIG_PATH_ABS, config_name=str(DEFAULT_CONFIG).split(".")[0])
def train(cfg):
    model = torch.nn.Sequential(torch.nn.Linear(10,100))
    dataset = "mnist" # or yolo.ClassificationDataset("mnist")
    criterion = None # yolo.Loss object
    trainer = Trainer(model, "mnist", criterion, cfg)
    #trainer.train()

if __name__ == "__main__":
    train()

