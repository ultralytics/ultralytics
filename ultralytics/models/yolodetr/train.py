# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-DETR trainer: RT-DETR base + DEIM-style augmentation decay + optional flat-cosine LR schedule.

DEIM-specific knobs are *not* added to default.yaml. Instead, they are class-level defaults on the
trainer and can be set per-run by passing them as kwargs to ``model.train(...)`` (see
``examples/train_yolodetr.py``). The trainer intercepts those kwargs in ``__init__`` before
``get_cfg`` runs, so unknown-key warnings are avoided.
"""

from __future__ import annotations

import math
from copy import copy

from torch import nn, optim

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms
from ultralytics.data.utils import get_split_fraction
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator
from ultralytics.nn.tasks import YOLODETRDetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

__all__ = ("YOLODETRDataset", "YOLODETRTrainer", "YOLODETRValidator")

_NO_AUG_EPOCH = 4  # DEIM trains the final epochs without augmentation
_YOLODETR_DEFAULTS = {
    "backbone_lr_ratio": 0.1,
}


def compute_deim_scheduled_prob(base_prob: float, epoch: int, stop_epoch: int) -> float:
    """Linearly decay an augmentation probability to 0 by the no-aug stage boundary.

    Args:
        base_prob (float): Probability configured in the hyperparameters.
        epoch (int): Current epoch.
        stop_epoch (int): Epoch at which the probability reaches 0.

    Returns:
        (float): Decayed probability for this epoch.
    """
    base_prob = float(base_prob)
    if base_prob <= 0.0 or stop_epoch <= 0 or epoch >= stop_epoch:
        return 0.0
    return base_prob * max(0.0, 1.0 - (float(epoch) / float(stop_epoch)))


def compute_policy_epochs(hyp) -> tuple[int, int, int]:
    """Compute DEIM stage boundaries from ``epochs`` and the fixed four-epoch no-augmentation tail.

    Args:
        hyp (SimpleNamespace | IterableSimpleNamespace): Hyperparameters carrying the total epoch count.

    Returns:
        start (int): End of stage 1, where the flat learning rate begins.
        mid (int): End of stage 2 and start of stage 3, where the cosine decay begins.
        stop (int): End of stage 3 and start of the no-augmentation tail.
    """
    epochs = max(1, int(hyp.epochs))
    stop = epochs - min(_NO_AUG_EPOCH, epochs)
    start = min(4, max(0, stop - 1))
    mid = start + (stop - start) // 2
    if not (0 <= start <= mid <= stop <= epochs):
        raise ValueError(
            f"compute_policy_epochs produced invalid boundaries: "
            f"start={start}, mid={mid}, stop={stop}, epochs={epochs}."
        )
    return start, mid, stop


class YOLODETRDataset(RTDETRDataset):
    """RT-DETR dataset variant that linearly decays YOLO augmentation probabilities over epochs.

    All augmentation probabilities (mosaic, mixup, copy_paste) decay from their base hyp value to 0 linearly across
    ``[0, stop_epoch]``, where ``stop_epoch`` leaves the final four epochs for the DEIM no-aug tail. Past stop_epoch
    every augmentation is hard-zeroed.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Stash base hyp values then defer to the parent for normal dataset construction.

        Args:
            *args (Any): Positional arguments forwarded to RTDETRDataset.
            data (dict, optional): Dataset dictionary.
            **kwargs (Any): Keyword arguments forwarded to RTDETRDataset; hyp is required.
        """
        hyp = kwargs["hyp"]
        self.base_hyp = copy(hyp)
        self.policy_epochs = compute_policy_epochs(hyp)
        super().__init__(*args, data=data, **kwargs)
        if self.augment:
            self.set_epoch(0)

    def _build_v8_epoch_hyp(self, epoch: int):
        """Clone the base hyp and apply linear decay; zero everything past the no-aug boundary.

        Args:
            epoch (int): Current epoch.

        Returns:
            (SimpleNamespace | IterableSimpleNamespace): Copy of the base hyperparameters with the augmentation
                probabilities decayed for this epoch.
        """
        hyp = copy(self.base_hyp)
        _, _, stop = self.policy_epochs
        if epoch >= stop:
            for key in (
                "mosaic",
                "mixup",
                "copy_paste",
                "cutmix",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "hsv_h",
                "hsv_s",
                "hsv_v",
            ):
                setattr(hyp, key, 0.0)
            hyp.augmentations = []
        else:
            hyp.mosaic = compute_deim_scheduled_prob(self.base_hyp.mosaic, epoch, stop)
            hyp.mixup = compute_deim_scheduled_prob(self.base_hyp.mixup, epoch, stop)
            hyp.copy_paste = compute_deim_scheduled_prob(self.base_hyp.copy_paste, epoch, stop)
        return hyp

    def build_transforms(self, hyp=None):
        """Build v8 transforms with current (possibly decayed) hyp values.

        Args:
            hyp (SimpleNamespace | IterableSimpleNamespace, optional): Hyperparameters for this epoch.

        Returns:
            (Compose): Transform pipeline ending in the Format transform.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if not self.rect else 0.0
            hyp.mixup = hyp.mixup if not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if not self.rect else 0.0
            # Keep v8 MixUp inputs same-sized; current v8 Mosaic no longer carries the old mosaic_border crop hint.
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            # Matches YOLODataset/RTDETRDataset: a no-op resize on the already-square val image whose only
            # effect is rewriting ratio_pad into the ((gain_h, gain_w), (pad_w, pad_h)) form scale_boxes needs.
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def set_epoch(self, epoch: int) -> None:
        """Rebuild transforms with decayed hyp probabilities for the current epoch.

        Args:
            epoch (int): Current epoch.
        """
        self.epoch = epoch
        if self.augment:
            self.transforms = self.build_transforms(hyp=self._build_v8_epoch_hyp(epoch))


class YOLODETRValidator(RTDETRValidator):
    """RT-DETR validator that ignores YOLODETR trainer-only arguments."""

    _YOLODETR_ARGS = tuple(_YOLODETR_DEFAULTS)

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize validator after removing YOLODETR-only args from the standard CFG namespace.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader for validation.
            save_dir (Path, optional): Directory for saving results.
            args (SimpleNamespace, optional): Validator arguments.
            _callbacks (list, optional): Callbacks registered on the validator.
        """
        super().__init__(dataloader, save_dir=save_dir, args=self._sanitize_args(args), _callbacks=_callbacks)

    @classmethod
    def _sanitize_args(cls, args):
        """Return a copy of args without YOLODETR-only trainer knobs.

        Args:
            args (SimpleNamespace, optional): Validator arguments.

        Returns:
            (SimpleNamespace | None): Copy without the trainer-only keys, or None when args is None.
        """
        if args is None:
            return None
        args = copy(args)
        for key in cls._YOLODETR_ARGS:
            if hasattr(args, key):
                delattr(args, key)
        return args


class YOLODETRTrainer(RTDETRTrainer):
    """RT-DETR trainer for YOLODETR models with augmentation decay + optional flat-cosine LR.

    DEIM hyperparameter defaults live on this class and are overridable via ``model.train(...)``
    kwargs. ``default.yaml`` is intentionally not extended.

    Supported kwargs (defaults shown):
        backbone_lr_ratio (float): Multiplier applied to backbone LR. Default 0.1.
    """

    _DEIM_DEFAULTS = _YOLODETR_DEFAULTS
    _epoch_callback_registered = False

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Pop DEIM kwargs from overrides before get_cfg, then write them onto self.args.

        Args:
            cfg (str | dict, optional): Base configuration.
            overrides (dict, optional): Configuration overrides, which may carry DEIM-specific keys.
            _callbacks (list, optional): Callbacks registered on the trainer.
        """
        overrides = dict(overrides or {})
        deim_overrides = {k: overrides.pop(k) for k in list(overrides) if k in self._DEIM_DEFAULTS}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        for k, default in self._DEIM_DEFAULTS.items():
            # A resume restores the checkpoint value onto self.args, so only fall back to the default when unset.
            setattr(self.args, k, deim_overrides.get(k, getattr(self.args, k, default)))

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
        """Build YOLODETRDataset for train (with decay schedule); use RT-DETR's dataset for val.

        Args:
            img_path (str): Path to the image directory.
            mode (str): Dataset mode, either train or val; only train applies the augmentation decay schedule.
            batch (int, optional): Batch size, used for rect mode.

        Returns:
            (YOLODETRDataset): Dataset for the requested mode.
        """
        return YOLODETRDataset(
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
        """Set up the flat-cosine LR schedule used by YOLODETR training."""
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
            trainer (YOLODETRTrainer, optional): Trainer passed by the callback, defaulting to self.
        """
        trainer = trainer or self
        epoch = int(trainer.epoch)
        dataset = trainer.train_loader.dataset
        dataset.set_epoch(epoch)
        trainer.train_loader.reset()
        stop_epoch = int(dataset.policy_epochs[-1])
        if epoch == stop_epoch and trainer.args.multi_scale > 0:
            trainer.args.multi_scale = 0.0
            LOGGER.info(f"YOLODETR no-aug stage at epoch {epoch}: disabling multi-scale")

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
            (YOLODETRValidator): Validator whose loss names match the head in use.
        """
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        head_name = type(unwrap_model(self.model).model[-1]).__name__
        if head_name == "DeimDecoder":
            loss_names += ["fgl_loss", "ddf_loss"]
        self.loss_names = tuple(loss_names)
        return YOLODETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

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
            raise NotImplementedError(f"Optimizer '{name}' not supported by YOLODETRTrainer.")

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
