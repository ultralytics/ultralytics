# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-DETR trainer: RT-DETR base + DEIM-style augmentation decay + optional flat-cosine LR schedule.

DEIM-specific knobs are *not* added to default.yaml. Instead, they are class-level defaults on the
trainer and can be set per-run by passing them as kwargs to ``model.train(...)`` (see
``examples/train_yolodetr.py``). The trainer intercepts those kwargs in ``__init__`` before
``get_cfg`` runs, so unknown-key warnings are avoided.
"""

from __future__ import annotations

import math
import random
from copy import copy

import torch
from torch import nn, optim

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator
from ultralytics.nn.modules.head import DeimDecoder
from ultralytics.nn.tasks import YOLODETRDetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

__all__ = ("YOLODETRTrainer", "YOLODETRDataset", "YOLODETRValidator")

_YOLODETR_DEFAULTS = {
    "no_aug_epoch": 4,
    "backbone_lr_ratio": 0.02,
    "base_size_repeat": 3,
}


def compute_deim_scheduled_prob(base_prob: float, epoch: int, stop_epoch: int) -> float:
    """Linearly decay an augmentation probability to 0 by the no-aug stage boundary."""
    base_prob = float(base_prob)
    if base_prob <= 0.0 or stop_epoch <= 0 or epoch >= stop_epoch:
        return 0.0
    return base_prob * max(0.0, 1.0 - (float(epoch) / float(stop_epoch)))


def compute_policy_epochs(hyp) -> tuple[int, int, int]:
    """Compute DEIM stage boundaries from ``epochs`` and optional ``no_aug_epoch``.

    Returns:
        (tuple[int, int, int]): (stage1_end, stage2_end / stage3_start, stage3_end / stage4_start).
    """
    epochs = max(1, int(hyp.epochs))
    no_aug_epoch = getattr(hyp, "no_aug_epoch", None)
    if no_aug_epoch is None:
        no_aug_epoch = 3 if epochs >= 100 else (2 if epochs >= 60 else 0)
    no_aug_epoch = int(no_aug_epoch)
    if no_aug_epoch < 0 or no_aug_epoch > epochs:
        raise ValueError(f"compute_policy_epochs got invalid no_aug_epoch={no_aug_epoch} for epochs={epochs}.")
    stop = epochs - no_aug_epoch
    start = min(4, max(0, stop - 1))
    mid = start + stop // 2
    if not (0 <= start <= mid <= stop <= epochs):
        raise ValueError(
            f"compute_policy_epochs produced invalid boundaries: "
            f"start={start}, mid={mid}, stop={stop}, epochs={epochs}."
        )
    return start, mid, stop


class YOLODETRDataset(RTDETRDataset):
    """RT-DETR dataset variant that linearly decays YOLO augmentation probabilities over epochs.

    All augmentation probabilities (mosaic, mixup, copy_paste) decay from their base hyp value to 0 linearly across
    ``[0, stop_epoch]`` where ``stop_epoch = epochs - no_aug_epoch``. Past stop_epoch every augmentation is hard-zeroed
    for the DEIM no-aug tail.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Stash base hyp values then defer to the parent for normal dataset construction."""
        hyp = kwargs["hyp"]
        self.base_hyp = copy(hyp)
        self.policy_epochs = compute_policy_epochs(hyp)
        super().__init__(*args, data=data, **kwargs)
        if self.augment:
            self.set_epoch(0)

    def _build_v8_epoch_hyp(self, epoch: int):
        """Clone the base hyp and apply linear decay; zero everything past the no-aug boundary."""
        hyp = copy(self.base_hyp)
        _, _, stop = self.policy_epochs
        if epoch >= stop:
            for key in (
                "mosaic", "mixup", "copy_paste", "cutmix",
                "degrees", "translate", "scale", "shear", "perspective",
                "hsv_h", "hsv_s", "hsv_v",
            ):
                setattr(hyp, key, 0.0)
            hyp.augmentations = []
        else:
            hyp.mosaic = compute_deim_scheduled_prob(self.base_hyp.mosaic, epoch, stop)
            hyp.mixup = compute_deim_scheduled_prob(self.base_hyp.mixup, epoch, stop)
            hyp.copy_paste = compute_deim_scheduled_prob(self.base_hyp.copy_paste, epoch, stop)
        return hyp

    def build_transforms(self, hyp=None):
        """Build v8 transforms with current (possibly decayed) hyp values."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if not self.rect else 0.0
            hyp.mixup = hyp.mixup if not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if not self.rect else 0.0
            # Keep v8 MixUp inputs same-sized; current v8 Mosaic no longer carries the old mosaic_border crop hint.
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([])
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
        """Rebuild transforms with decayed hyp probabilities for the current epoch."""
        self.epoch = epoch
        if self.augment:
            self.transforms = self.build_transforms(hyp=self._build_v8_epoch_hyp(epoch))


class YOLODETRValidator(RTDETRValidator):
    """RT-DETR validator that ignores YOLODETR trainer-only arguments."""

    _YOLODETR_ARGS = tuple(_YOLODETR_DEFAULTS)

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize validator after removing YOLODETR-only args from the standard CFG namespace."""
        super().__init__(dataloader, save_dir=save_dir, args=self._sanitize_args(args), _callbacks=_callbacks)

    @classmethod
    def _sanitize_args(cls, args):
        """Return a copy of args without YOLODETR-only trainer knobs."""
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
        no_aug_epoch (int): Length of the trailing no-augmentation tail. Default 4.
        backbone_lr_ratio (float): Multiplier applied to backbone LR. Default 0.02.
        base_size_repeat (int): Extra weight given to the base imgsz when sampling multi-scale sizes. Default 3.
    """

    _DEIM_DEFAULTS = _YOLODETR_DEFAULTS
    _epoch_callback_registered = False

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Pop DEIM kwargs from overrides before get_cfg, then write them onto self.args."""
        overrides = dict(overrides or {})
        deim_overrides = {k: overrides.pop(k) for k in list(overrides) if k in self._DEIM_DEFAULTS}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        for k, default in self._DEIM_DEFAULTS.items():
            setattr(self.args, k, deim_overrides.get(k, default))
        self.args.no_aug_epoch = min(int(self.args.no_aug_epoch), int(self.args.epochs))

    def _sample_multiscale_size(self) -> int:
        """Sample a multi-scale size, biasing the base imgsz by ``base_size_repeat`` extra picks."""
        low = max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale)))
        high = int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride)
        low = (low // self.stride) * self.stride
        high = (high // self.stride) * self.stride
        if high <= low:
            return low
        base_size_repeat = int(self.args.base_size_repeat or 0)
        if base_size_repeat <= 0:
            return random.randrange(low, high) // self.stride * self.stride
        scales = list(range(low, high + 1, self.stride))
        base = max(low, min(high, (int(self.args.imgsz) // self.stride) * self.stride))
        scales.extend([base] * base_size_repeat)
        return random.choice(scales)

    def preprocess_batch(self, batch: dict) -> dict:
        """Normalize images and apply ``base_size_repeat``-weighted multi-scale sampling."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = self._sample_multiscale_size()
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def setup_model(self):
        """Build model, then force amp=False if a DeimDecoder is present (FP16 numerical stability)."""
        ckpt = super().setup_model()
        if self.args.amp and any(isinstance(m, DeimDecoder) for m in self.model.modules()):
            self.args.amp = False
            if RANK in {-1, 0}:
                LOGGER.info("YOLODETRTrainer: DeimDecoder detected, forcing amp=False for numerical stability")
        return ckpt

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build YOLODETRDetectionModel and optionally load weights with class-row remapping."""
        model = YOLODETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            self._load_with_class_transfer(model, weights, verbose=RANK in {-1, 0})
        return model

    def _load_with_class_transfer(self, model, weights, verbose=True):
        """Load source weights into model, remapping class-row tensors when name lists differ.

        Handles Obj365 -> COCO style transfer:
          - drops ``denoising_class_embed.*`` tensors when shape mismatch (random reinit)
          - remaps ``score_head`` / ``class_embed`` rows by class-name matching (incl. aliases)
        """
        from ultralytics.utils.class_map import (
            is_default_numeric_names, names_to_list, remap_class_row_state_dict, resolve_names,
        )
        from ultralytics.utils.torch_utils import intersect_dicts

        src_args = getattr(weights, "args", None)
        src_data_yaml = src_args.get("data") if isinstance(src_args, dict) else getattr(src_args, "data", None)
        src_names = resolve_names(getattr(weights, "names", None), src_data_yaml)
        dst_names = resolve_names(self.data.get("names"), getattr(self.args, "data", None))

        csd = weights.float().state_dict()
        state_dict = model.state_dict()

        dn_discard = [
            k for k in csd
            if "denoising_class_embed" in k and k in state_dict and csd[k].shape != state_dict[k].shape
        ]
        for k in dn_discard:
            del csd[k]
            if verbose:
                LOGGER.info(f"Discarded '{k}' due to class-count mismatch (will be randomly initialized)")

        src_is_default = is_default_numeric_names(src_names)
        dst_is_default = is_default_numeric_names(dst_names)
        src_name_list = names_to_list(src_names)
        dst_name_list = names_to_list(dst_names)
        names_match = bool(src_name_list) and src_name_list == dst_name_list
        if (
            src_names is not None and dst_names is not None
            and not src_is_default and not dst_is_default and not names_match
        ):
            csd, remapped, missing = remap_class_row_state_dict(
                csd, state_dict, src_names=src_names, dst_names=dst_names
            )
            if verbose and remapped:
                LOGGER.info(f"Remapped {len(remapped)} class tensors using source->target class-name map")
            if verbose and missing:
                LOGGER.info(f"{len(missing)} target classes were not mapped and kept target initialization")

        updated_csd = intersect_dicts(csd, state_dict)
        model.load_state_dict(updated_csd, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(updated_csd)}/{len(model.state_dict())} items from pretrained weights")

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build YOLODETRDataset for train (with decay schedule); use RT-DETR's dataset for val."""
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
            fraction=self.args.fraction if mode == "train" else 1.0,
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
            if epoch < flat_epoch:
                return 1.0
            progress = min(max((epoch - flat_epoch) / decay_epochs, 0.0), 1.0)
            return gamma + 0.5 * (1.0 - gamma) * (1.0 + math.cos(math.pi * progress))

        self.lf = _flat_cosine
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _on_train_epoch_start(self, trainer=None):
        """Propagate epoch to dataset transforms and stop multi-scale at the no-aug boundary."""
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
        """Disable close_mosaic (decay schedule replaces it) and register the epoch callback."""
        if self.args.close_mosaic:
            self.args.close_mosaic = 0
        if not self._epoch_callback_registered:
            self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
            self._epoch_callback_registered = True
        return super().train(*args, **kwargs)

    def get_validator(self):
        """Return an RTDETRValidator with loss_names extended for the DEIM head."""
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        head_name = type(unwrap_model(self.model).model[-1]).__name__
        if head_name == "DeimDecoder":
            loss_names += ["fgl_loss", "ddf_loss"]
        self.loss_names = tuple(loss_names)
        return YOLODETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer with 6 parameter groups split between head and backbone (backbone_lr_ratio)."""
        backbone_lr_ratio = float(self.args.backbone_lr_ratio)
        if backbone_lr_ratio <= 0:
            raise ValueError(f"Invalid backbone_lr_ratio={backbone_lr_ratio}. Expected > 0.")
        model = unwrap_model(model)  # so .yaml access and parameter names work identically under DDP and single-GPU
        g = [{}, {}, {}, {}, {}, {}]  # head: [0 weight, 1 bn, 2 bias]; backbone: [3 weight, 4 bn, 5 bias]
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if name == "auto":
            name = "AdamW"
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
