# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

from torch import nn, optim

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.optim.muon import MuSGD
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

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

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer with a separate learning rate for backbone parameters.

        Splits each parameter group into head (lr=lr) and backbone (lr=lr * backbone_lr_ratio) subgroups.
        Backbone layers are identified as model.{idx} entries with idx < len(model.yaml['backbone']).

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): Optimizer name; if 'auto', selected based on iteration count.
            lr (float, optional): Base learning rate for head parameters.
            momentum (float, optional): Momentum factor.
            decay (float, optional): Weight decay (L2 regularization).
            iterations (float, optional): Number of iterations; selects optimizer when name is 'auto'.

        Returns:
            (torch.optim.Optimizer): Constructed optimizer with backbone-specific LR groups.
        """
        backbone_lr_ratio = self.args.backbone_lr_ratio
        if backbone_lr_ratio <= 0:
            raise ValueError(f"Invalid backbone_lr_ratio={backbone_lr_ratio}. Expected > 0.")
        # Normalize optimizer name once so case-insensitive aliases (e.g. "Auto", "musgd") behave correctly
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower(), name)
        g = [{}, {}, {}, {}, {}, {}, {}, {}]  # 8 groups: head 0-3, backbone 4-7 (indices 3, 7 are muon)
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'lr0' and 'momentum' automatically for AdamW... "
            )
            nc = self.data.get("nc", 10)
            name, lr, momentum = "AdamW", round(0.002 * 5 / (4 + nc), 6), 0.9
            self.args.warmup_bias_lr = 0.0

        use_muon = name == "MuSGD"
        unwrapped = unwrap_model(model)
        backbone_len = len(unwrapped.yaml.get("backbone", []))
        for module_name, module in unwrapped.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                parts = fullname.split(".")
                is_backbone = (
                    len(parts) > 1 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) < backbone_len
                )
                is_norm_like = isinstance(module, bn) or "logit_scale" in fullname
                base = 4 if is_backbone else 0  # 0-3 head slots, 4-7 backbone slots
                if param.ndim >= 2 and use_muon:
                    g[base + 3][fullname] = param  # muon weights
                elif "bias" in fullname:
                    g[base + 2][fullname] = param
                elif is_norm_like:
                    g[base + 1][fullname] = param
                else:
                    g[base + 0][fullname] = param

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name in {"SGD", "MuSGD"}:
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        backbone_lr = lr * backbone_lr_ratio
        head_groups = [
            {"params": list(g[0].values()), **optim_args, "weight_decay": decay, "param_group": "head_weight"},
            {"params": list(g[1].values()), **optim_args, "weight_decay": 0.0, "param_group": "head_bn"},
            {"params": list(g[2].values()), **optim_args, "param_group": "head_bias"},
        ]
        back_groups = [
            {
                "params": list(g[4].values()),
                **optim_args,
                "lr": backbone_lr,
                "weight_decay": decay,
                "param_group": "backbone_weight",
            },
            {
                "params": list(g[5].values()),
                **optim_args,
                "lr": backbone_lr,
                "weight_decay": 0.0,
                "param_group": "backbone_bn",
            },
            {
                "params": list(g[6].values()),
                **optim_args,
                "lr": backbone_lr,
                "weight_decay": 0.0,
                "param_group": "backbone_bias",
            },
        ]
        if use_muon:
            head_groups.append(
                {
                    "params": list(g[3].values()),
                    **optim_args,
                    "weight_decay": decay,
                    "use_muon": True,
                    "param_group": "head_muon",
                }
            )
            back_groups.append(
                {
                    "params": list(g[7].values()),
                    **optim_args,
                    "lr": backbone_lr,
                    "weight_decay": decay,
                    "use_muon": True,
                    "param_group": "backbone_muon",
                }
            )

        params = [pg for pg in head_groups + back_groups if pg["params"]]
        optimizer = MuSGD(params=params, muon=0.2, sgd=1.0) if use_muon else getattr(optim, name)(params=params)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, backbone_lr={backbone_lr}, "
            f"momentum={momentum}) with parameter groups\n"
            f"  Head:     {len(g[1])} bn, {len(g[0])} weight(decay={decay}), {len(g[2])} bias"
            + (f", {len(g[3])} muon" if use_muon else "")
            + f" (lr={lr})\n"
            f"  Backbone: {len(g[5])} bn, {len(g[4])} weight(decay={decay}), {len(g[6])} bias"
            + (f", {len(g[7])} muon" if use_muon else "")
            + f" (lr={backbone_lr})"
        )
        return optimizer

    def get_validator(self):
        """Return an RTDETRValidator suitable for RT-DETR model validation."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
