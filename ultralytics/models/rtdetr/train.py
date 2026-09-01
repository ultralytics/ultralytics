# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from copy import copy

from torch import nn, optim

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel, YOLODETRDetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .val import DEIMDataset, RTDETRDataset, RTDETRValidator, compute_policy_epochs


class RTDETRTrainer(DetectionTrainer):
    """Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

    This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of
    RT-DETR. The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable
    inference speed.

    Attributes:
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.
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
        model = self.set_model_names_for_load(
            RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
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
            fraction=1.0 if self.data.get("complete") else get_split_fraction(self.args.fraction, mode),
        )

    def get_validator(self):
        """Return an RTDETRValidator suitable for RT-DETR model validation."""
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


class DEIMTrainer(RTDETRTrainer):
    """RT-DETR trainer for DeimDecoder models with augmentation decay + flat-cosine LR.

    ``backbone_lr_ratio`` defaults to 0.1 and discounts the backbone param groups' LR in ``build_optimizer``.
    """

    _epoch_callback_registered = False

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the DEIM trainer with a 0.1 default backbone learning-rate ratio."""
        super().__init__(cfg, {"backbone_lr_ratio": 0.1, **(overrides or {})}, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build YOLODETRDetectionModel and load weights; cls-head rows remap by class name inside model.load().

        Args:
            cfg (str | dict, optional): Model configuration.
            weights (str | Path, optional): Pretrained weights to load.
            verbose (bool): Log the model summary.

        Returns:
            (YOLODETRDetectionModel): Model ready for training.
        """
        model = self.set_model_names_for_load(
            YOLODETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build DEIMDataset for train (with decay schedule); use it for val as well (no augmentation applied).

        Args:
            img_path (str): Path to the image directory.
            mode (str): Dataset mode, either train or val; only train applies the augmentation decay schedule.
            batch (int, optional): Batch size, used for rect mode.

        Returns:
            (DEIMDataset): Dataset for the requested mode.
        """
        return DEIMDataset(
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
            fraction=1.0 if self.data.get("complete") else get_split_fraction(self.args.fraction, mode),
        )

    def _setup_scheduler(self):
        """Set up the flat-cosine LR schedule used by DEIM training."""
        _, mid, _ = compute_policy_epochs(self.args)
        flat_epoch = int(mid)
        gamma = float(self.args.lrf)
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"flatcosine got invalid lrf={gamma}. Expected 0.0 <= lrf <= 1.0.")
        decay_epochs = max(self.epochs - flat_epoch, 1)

        def _flat_cosine(epoch: int) -> float:
            """Hold the learning rate flat until flat_epoch, then decay it to lrf on a cosine curve."""
            if epoch < flat_epoch:
                return 1.0
            progress = min(max((epoch - flat_epoch) / decay_epochs, 0.0), 1.0)
            return gamma + 0.5 * (1.0 - gamma) * (1.0 + math.cos(math.pi * progress))

        self.lf = _flat_cosine
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _on_train_epoch_start(self, trainer=None):
        """Propagate epoch to dataset transforms and stop multi-scale at the no-aug boundary.

        Args:
            trainer (DEIMTrainer, optional): Trainer passed by the callback, defaulting to self.
        """
        trainer = trainer or self
        epoch = int(trainer.epoch)
        dataset = trainer.train_loader.dataset
        dataset.set_epoch(epoch)
        trainer.train_loader.reset()
        stop_epoch = int(dataset.policy_epochs[-1])
        if epoch == stop_epoch and trainer.args.multi_scale > 0:
            trainer.args.multi_scale = 0.0
            LOGGER.info(f"DEIM no-aug stage at epoch {epoch}: disabling multi-scale")

    def train(self, *args, **kwargs):
        """Disable close_mosaic (decay schedule replaces it) and register the epoch callback.

        Args:
            *args (Any): Positional arguments forwarded to RTDETRTrainer.train.
            **kwargs (Any): Keyword arguments forwarded to RTDETRTrainer.train.

        Returns:
            (Any): Result of the parent train call.
        """
        if self.args.close_mosaic:
            self.args.close_mosaic = 0
        if not self._epoch_callback_registered:
            self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
            self._epoch_callback_registered = True
        return super().train(*args, **kwargs)

    def get_validator(self):
        """Return an RTDETRValidator with loss_names extended for the DEIM head.

        Returns:
            (RTDETRValidator): Validator whose loss names match the head in use.
        """
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        head_name = type(unwrap_model(self.model).model[-1]).__name__
        if head_name == "DeimDecoder":
            loss_names += ["fgl_loss", "ddf_loss"]
        self.loss_names = tuple(loss_names)
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer with 6 param groups split head/backbone; 'auto' resolves to AdamW with DEIM LR defaults.

        Args:
            model (nn.Module): Model whose parameters are grouped.
            name (str): Optimizer name; auto resolves to AdamW with the DEIM learning rate defaults.
            lr (float): Learning rate for the head groups; the backbone groups scale it by backbone_lr_ratio.
            momentum (float): Momentum or beta1, depending on the optimizer.
            decay (float): Weight decay applied to the weight groups only.
            iterations (float): Total training iterations, used by the auto resolver.

        Returns:
            (torch.optim.Optimizer): Optimizer with six parameter groups.

        Notes:
            Groups 0-2 hold the head weights, norms, and biases; groups 3-5 hold the backbone equivalents. Norm
            and bias groups get no weight decay.
        """
        backbone_lr_ratio = float(self.args.backbone_lr_ratio)
        if backbone_lr_ratio <= 0:
            raise ValueError(f"Invalid backbone_lr_ratio={backbone_lr_ratio}. Expected > 0.")
        model = unwrap_model(model)  # so .yaml access and parameter names work identically under DDP and single-GPU
        g = [{}, {}, {}, {}, {}, {}]  # head: [0 weight, 1 bn, 2 bias]; backbone: [3 weight, 4 bn, 5 bias]
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if name == "auto":
            name, lr, momentum = "AdamW", 5e-4, 0.9
            self.args.warmup_momentum, self.args.warmup_bias_lr = momentum, 0.0  # no bias/momentum warmup for Adam
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, ignoring 'lr0={self.args.lr0}' and "
                f"'momentum={self.args.momentum}' and using DEIM defaults '{name}', 'lr0={lr}', 'momentum={momentum}'..."
            )
        backbone_len = len(model.yaml["backbone"])

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                parts = fullname.split(".")
                is_backbone = (
                    len(parts) > 1 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) < backbone_len
                )
                is_norm_like_param = (
                    isinstance(module, bn) or module.__class__.__name__ == "DEIMRMSNorm" or "logit_scale" in fullname
                )
                if is_backbone:
                    if "bias" in fullname:
                        g[5][fullname] = param  # backbone bias
                    elif is_norm_like_param:
                        g[4][fullname] = param  # backbone bn
                    else:
                        g[3][fullname] = param  # backbone weight (decay)
                else:
                    if "bias" in fullname:
                        g[2][fullname] = param  # head bias
                    elif is_norm_like_param:
                        g[1][fullname] = param  # head bn
                    else:
                        g[0][fullname] = param  # head weight (decay)

        g = [list(x.values()) for x in g]
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD"}
        if name not in optimizers:
            raise NotImplementedError(f"Optimizer '{name}' not supported by DEIMTrainer.")

        backbone_lr = lr * backbone_lr_ratio
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        else:  # SGD
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)

        # Head groups (lr)
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # head weights
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # head bn
        # Backbone groups (backbone_lr)
        optimizer.add_param_group({"params": g[5], "lr": backbone_lr, "weight_decay": 0.0})  # backbone bias
        optimizer.add_param_group({"params": g[3], "lr": backbone_lr, "weight_decay": decay})  # backbone weights
        optimizer.add_param_group({"params": g[4], "lr": backbone_lr, "weight_decay": 0.0})  # backbone bn

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups:\n"
            f"  Head:     {len(g[1])} bn, {len(g[0])} weight(decay={decay}), {len(g[2])} bias (lr={lr})\n"
            f"  Backbone: {len(g[4])} bn, {len(g[3])} weight(decay={decay}), {len(g[5])} bias (lr={backbone_lr})"
        )
        return optimizer
