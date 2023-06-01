from copy import copy

import torch

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.yolo.v8.detect import DetectionTrainer

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = RTDETRDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build RTDETR Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)


    def get_validator(self):
        """Returns a DetectionValidator for RTDETR model validation."""
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        bs = len(batch['img'])
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch

    @staticmethod
    def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        """
        Builds an optimizer with the specified parameters and parameter groups.

        Args:
            model (nn.Module): model to optimize
            name (str): name of the optimizer to use
            lr (float): learning rate
            momentum (float): momentum
            decay (float): weight decay

        Returns:
            optimizer (torch.optim.Optimizer): the built optimizer
        """
        import torch.nn as nn
        from ultralytics.yolo.utils import LOGGER
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr)  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

        optimizer.add_param_group({'params': g[0]})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                    f'{len(g[1])} weight, {len(g[0])} weight, {len(g[2])} bias')
        return optimizer


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize RTDETR model given training data and device."""
    model = 'rtdetr-l.yaml'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    # NOTE: F.grid_sample which is in rt-detr does not support deterministic=True
    # NOTE: amp training causes nan outputs and end with error while doing bipartite graph matching
    args = dict(model=model,
                data=data,
                device=device,
                imgsz=640,
                exist_ok=True,
                batch=4,
                deterministic=False,
                amp=False)
    trainer = RTDETRTrainer(overrides=args)
    trainer.train()


if __name__ == '__main__':
    train()
