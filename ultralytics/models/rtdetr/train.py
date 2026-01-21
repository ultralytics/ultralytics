# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
import random
import math

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr
from torch import nn, optim
from ultralytics.utils import LOGGER
import torch

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

    This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of
    RT-DETR. The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable
    inference speed.

    Attributes:
        loss_names (tuple): Names of the loss components used for training.
        data (dict): Dataset configuration containing class count and other parameters.
        args (dict): Training arguments and hyperparameters.
        save_dir (Path): Directory to save training results.
        test_loader (DataLoader): DataLoader for validation/testing data.

    Methods:
        get_model: Initialize and return an RT-DETR model for object detection tasks.
        build_dataset: Build and return an RT-DETR dataset for training or validation.
        get_validator: Return a DetectionValidator suitable for RT-DETR model validation.

    Examples:
        >>> from ultralytics.models.rtdetr.train import RTDETRTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = RTDETRTrainer(overrides=args)
        >>> trainer.train()

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.
    """

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        """Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        freeze_bn = str(getattr(self.args, "freeze_bn", "none")).lower().replace("+", "_")
        if freeze_bn in {"backbone", "backbone_neck"}:
            from ultralytics.nn.modules.utils import freeze_batch_norm2d

            nb = len(model.yaml["backbone"])
            freeze_to = nb
            if freeze_bn == "backbone_neck":
                head = model.yaml.get("head", [])
                head_cut = next((i for i, layer in enumerate(head) if layer[2] == "RTDETRDecoder"), len(head))
                freeze_to = nb + head_cut

            for i, m in enumerate(model.model[:freeze_to]):
                frozen = freeze_batch_norm2d(m)
                if frozen is not m:
                    model.model[i] = frozen
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            multi_scale_range_low = 1 - self.args.multi_scale
            multi_scale_range_high = 1 + self.args.multi_scale
            sz = (
                random.randrange(
                    int(self.args.imgsz * multi_scale_range_low),
                    int(self.args.imgsz * multi_scale_range_high + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected based on the
                number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'.
            backbone_lr_ratio (float, optional): The learning rate ratio for the backbone parameters.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        backbone_lr_ratio = self.args.backbone_lr_ratio
        g = [], [], [], [], [], []  # optimizer parameter groups, 6 groups now
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        backbone_len = self.backbone_len

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name

                # Check if this is a backbone layer
                is_backbone = False
                parts = fullname.split(".")

                # Handle DDP wrapping: "module.model.0.conv.weight" vs "model.0.conv.weight"
                if parts[0] == "module":
                    parts = parts[1:]  # Remove "module" prefix for DDP

                if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    is_backbone = layer_idx < backbone_len

                if is_backbone:
                    # Backbone parameters (groups 3, 4, 5)
                    if "bias" in fullname:
                        g[5].append(param)  # backbone bias
                    elif isinstance(module, bn) or "logit_scale" in fullname:
                        g[4].append(param)  # backbone bn weight
                    else:
                        g[3].append(param)  # backbone weight with decay
                else:
                    # Head parameters (groups 0, 1, 2)
                    if "bias" in fullname:
                        g[2].append(param)  # head bias
                    elif isinstance(module, bn) or "logit_scale" in fullname:
                        g[1].append(param)  # head bn weight
                    else:
                        g[0].append(param)  # head weight with decay

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        # Head param groups (normal lr)
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # head weights
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})    # head bn

        backbone_lr = lr * backbone_lr_ratio
        # Backbone param groups (smaller lr)
        optimizer.add_param_group({"params": g[5], "lr": backbone_lr, "weight_decay": 0.0})   # backbone bias
        optimizer.add_param_group({"params": g[3], "lr": backbone_lr, "weight_decay": decay}) # backbone weights
        optimizer.add_param_group({"params": g[4], "lr": backbone_lr, "weight_decay": 0.0})   # backbone bn

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups:\n"
            f"  Head:     {len(g[1])} bn, {len(g[0])} weight(decay={decay}), {len(g[2])} bias (lr={lr})\n"
            f"  Backbone: {len(g[4])} bn, {len(g[3])} weight(decay={decay}), {len(g[5])} bias (lr={backbone_lr})"
        )
        return optimizer

    def get_validator(self):
        """Return a DetectionValidator suitable for RT-DETR model validation."""
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        loss_gain = self.model.yaml.get("loss", {}).get("loss_gain", {}) if hasattr(self.model, "yaml") else {}
        if "fgl" in loss_gain:
            loss_names.append("fgl_loss")
        if "ddf" in loss_gain:
            loss_names.append("ddf_loss")
        if "mal" in loss_gain:
            loss_names.append("mal_loss")
        self.loss_names = tuple(loss_names)
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
