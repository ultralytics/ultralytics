# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import optim

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.nn.tasks import load_checkpoint, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, OBBMetrics, PoseMetrics, SegmentMetrics, batch_probiou, kpt_iou, mask_iou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import one_cycle, strip_optimizer, unwrap_model

from .detr_augment import (
    compute_deim_scheduled_prob,
    compute_policy_epochs,
    resolve_deim_aug_scheduler,
    rtdetr_deim_transforms,
)
from .train import RTDETRTrainer
from .val import RTDETRDataset, RTDETRValidator

__all__ = (
    "RTDETRDEIMDataset",
    "RTDETRDEIMValidator",
    "RTDETRDEIMTrainer",
    "RTDETRDEIMTrainerV2",
    "RTDETRDEIMSegmentValidator",
    "RTDETRDEIMSegmentTrainer",
    "RTDETRDEIMPoseValidator",
    "RTDETRDEIMPoseTrainer",
    "RTDETRDEIMOBBValidator",
    "RTDETRDEIMOBBTrainer",
)


class _RTDETRDEIMBatchAugment:
    """Batch-level DEIM augmentations (MixUp + CopyBlend) with selectable scheduler mode."""

    _COPYBLEND_AREA_THRESHOLD = 100.0
    _COPYBLEND_NUM_OBJECTS = 3
    _COPYBLEND_RANDOM_NUM_OBJECTS = False
    _COPYBLEND_TYPE = "blend"
    _COPYBLEND_WITH_EXPAND = True
    _COPYBLEND_EXPAND_RATIOS = (0.1, 0.25)

    def __init__(
        self,
        mixup_prob: float,
        mixup_epochs: tuple[int, int],
        copyblend_prob: float,
        copyblend_epochs: tuple[int, int],
        scheduler_mode: str = "legacy",
        decay_min_prob: float = 0.0,
    ) -> None:
        self.base_mixup_prob = float(mixup_prob)
        self.mixup_epochs = mixup_epochs
        self.base_copyblend_prob = float(copyblend_prob)
        self.copyblend_epochs = copyblend_epochs
        self.scheduler_mode = str(scheduler_mode)
        self.decay_min_prob = float(decay_min_prob)
        self.mixup_prob = self.base_mixup_prob
        self.copyblend_prob = self.base_copyblend_prob
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for DEIM batch augmentation scheduling."""
        self.epoch = epoch
        _, mixup_stop = self.mixup_epochs
        _, copyblend_stop = self.copyblend_epochs
        if self.scheduler_mode == "decay":
            self.mixup_prob = compute_deim_scheduled_prob(self.base_mixup_prob, epoch, mixup_stop, self.decay_min_prob)
            self.copyblend_prob = compute_deim_scheduled_prob(self.base_copyblend_prob, epoch, copyblend_stop, self.decay_min_prob)
        else:
            self.mixup_prob = self.base_mixup_prob
            self.copyblend_prob = self.base_copyblend_prob

    def __call__(self, batch: list[dict]) -> dict:
        new_batch = YOLODataset.collate_fn(batch)
        mixup_start, mixup_stop = self.mixup_epochs
        copyblend_start, copyblend_stop = self.copyblend_epochs
        # Preserve the original precedence: try MixUp first, then CopyBlend.
        if mixup_start <= self.epoch < mixup_stop and random.random() < self.mixup_prob:
            return self._apply_mixup(new_batch)
        if (
            copyblend_start <= self.epoch < copyblend_stop
            and random.random() < self.copyblend_prob
        ):
            return self._apply_copyblend(new_batch)
        return new_batch

    @staticmethod
    def _boxes_area_xywhn(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Compute absolute area from normalized xywh boxes."""
        if boxes.numel() == 0:
            return boxes.new_zeros((0,), dtype=torch.float32)
        return boxes[:, 2].to(torch.float32) * float(w) * boxes[:, 3].to(torch.float32) * float(h)

    @staticmethod
    def _stack_or_empty(tensors: list[torch.Tensor], shape: tuple[int, ...], *, like: torch.Tensor) -> torch.Tensor:
        """Stack or return empty tensor with desired shape/dtype/device."""
        if tensors:
            return torch.cat(tensors, dim=0)
        return torch.empty(shape, device=like.device, dtype=like.dtype)

    def _batch_to_targets(self, batch: dict) -> list[dict[str, torch.Tensor]]:
        """Convert flattened Ultralytics target format into DEIMv2-style per-image targets."""
        images = batch["img"]
        bs, _, h, w = images.shape
        bboxes = batch["bboxes"]
        cls = batch["cls"]
        batch_idx = batch["batch_idx"].view(-1).to(dtype=torch.long)
        labels_flat = cls.view(-1)
        mixup_flat = batch.get("mixup")
        mixup_flat = mixup_flat.view(-1) if isinstance(mixup_flat, torch.Tensor) else None

        targets = []
        for i in range(bs):
            mask = batch_idx == i
            boxes_i = bboxes[mask]
            labels_i = labels_flat[mask]
            area_i = self._boxes_area_xywhn(boxes_i, w=w, h=h)
            target = {"boxes": boxes_i.clone(), "labels": labels_i.clone(), "area": area_i}
            if mixup_flat is not None and mixup_flat.numel() == bboxes.shape[0]:
                target["mixup"] = mixup_flat[mask].clone()
            targets.append(target)
        return targets

    def _targets_to_batch(self, batch: dict, targets: list[dict[str, torch.Tensor]]) -> dict:
        """Convert DEIMv2-style per-image targets back to flattened Ultralytics format."""
        ref_boxes = batch["bboxes"]
        ref_cls = batch["cls"]
        ref_batch_idx = batch["batch_idx"]

        boxes_list, cls_list, batch_idx_list, mixup_list = [], [], [], []
        has_mixup = any("mixup" in t for t in targets)

        for i, target in enumerate(targets):
            boxes = target["boxes"]
            n = int(boxes.shape[0])
            if n == 0:
                continue

            labels = target["labels"]
            labels = labels.view(-1, 1) if ref_cls.ndim == 2 else labels.view(-1)

            boxes_list.append(boxes.to(device=ref_boxes.device, dtype=ref_boxes.dtype))
            cls_list.append(labels.to(device=ref_cls.device, dtype=ref_cls.dtype))
            batch_idx_list.append(torch.full((n,), i, device=ref_batch_idx.device, dtype=ref_batch_idx.dtype))

            if has_mixup:
                if "mixup" in target:
                    mixup_vals = target["mixup"]
                else:
                    mixup_vals = torch.ones((n,), device=ref_boxes.device, dtype=torch.float32)
                mixup_list.append(mixup_vals.to(device=ref_boxes.device, dtype=torch.float32))

        batch["bboxes"] = self._stack_or_empty(boxes_list, (0, ref_boxes.shape[1]), like=ref_boxes)
        if ref_cls.ndim == 2:
            batch["cls"] = self._stack_or_empty(cls_list, (0, ref_cls.shape[1]), like=ref_cls)
        else:
            batch["cls"] = self._stack_or_empty(cls_list, (0,), like=ref_cls)
        batch["batch_idx"] = self._stack_or_empty(batch_idx_list, (0,), like=ref_batch_idx)

        if has_mixup:
            batch["mixup"] = self._stack_or_empty(mixup_list, (0,), like=ref_boxes).to(torch.float32)
        elif "mixup" in batch:
            batch.pop("mixup")

        return batch

    def _apply_mixup(self, batch: dict) -> dict:
        images = batch["img"]
        bs = images.shape[0]
        if bs < 2:
            return batch

        targets = self._batch_to_targets(batch)
        shifted_targets = targets[-1:] + targets[:-1]
        updated_targets = []

        beta = round(random.uniform(0.45, 0.55), 6)
        images_f = images.to(torch.float32)
        shifted_images = torch.roll(images_f, shifts=1, dims=0)
        batch["img"] = shifted_images.mul(1.0 - beta).add(images_f.mul(beta))

        for target, shifted_target in zip(targets, shifted_targets):
            out = {
                "boxes": torch.cat([target["boxes"], shifted_target["boxes"]], dim=0),
                "labels": torch.cat([target["labels"], shifted_target["labels"]], dim=0),
                "area": torch.cat([target["area"], shifted_target["area"]], dim=0),
                "mixup": torch.tensor(
                    [beta] * len(target["labels"]) + [1.0 - beta] * len(shifted_target["labels"]),
                    device=batch["img"].device,
                    dtype=torch.float32,
                ),
            }
            updated_targets.append(out)

        return self._targets_to_batch(batch, updated_targets)

    def _apply_copyblend(self, batch: dict) -> dict:
        """CopyBlend implementation aligned with DEIMv2 collate behavior."""
        images = batch["img"]
        bs = images.shape[0]
        if bs < 2:
            return batch

        images_f = images.to(torch.float32)
        targets = self._batch_to_targets(batch)
        beta = round(random.uniform(0.45, 0.55), 6)
        img_height, img_width = images_f[0].shape[-2:]

        objects_pool: dict[str, list[Any]] = {
            "boxes": [],
            "labels": [],
            "areas": [],
            "image_idx": [],
            "image_height": [],
            "image_width": [],
        }

        for i in range(bs):
            source_boxes = targets[i]["boxes"]
            source_labels = targets[i]["labels"]
            source_areas = targets[i]["area"]

            valid_objects = [idx for idx in range(len(source_boxes)) if source_areas[idx] >= self._COPYBLEND_AREA_THRESHOLD]
            for idx in valid_objects:
                objects_pool["boxes"].append(source_boxes[idx])
                objects_pool["labels"].append(source_labels[idx])
                objects_pool["areas"].append(source_areas[idx])
                objects_pool["image_idx"].append(i)
                objects_pool["image_height"].append(img_height)
                objects_pool["image_width"].append(img_width)

        if len(objects_pool["boxes"]) == 0:
            return batch

        for key in ["boxes", "labels", "areas"]:
            objects_pool[key] = torch.stack(objects_pool[key]) if objects_pool[key] else torch.tensor([])

        updated_images = images_f.clone()
        updated_targets = [
            {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in target.items()} for target in targets
        ]

        for i in range(bs):
            pool_size = len(objects_pool["boxes"])
            if self._COPYBLEND_RANDOM_NUM_OBJECTS:
                num_objects = random.randint(1, min(self._COPYBLEND_NUM_OBJECTS, pool_size))
            else:
                num_objects = min(self._COPYBLEND_NUM_OBJECTS, pool_size)

            selected_indices = random.sample(range(pool_size), num_objects)
            blend_boxes, blend_labels, blend_areas, blend_mixup_ratios = [], [], [], []

            for idx in selected_indices:
                box = objects_pool["boxes"][idx]
                label = objects_pool["labels"][idx]
                area = objects_pool["areas"][idx]
                source_idx = objects_pool["image_idx"][idx]
                source_height = objects_pool["image_height"][idx]
                source_width = objects_pool["image_width"][idx]

                cx, cy, bw, bh = box.tolist()
                x1_src = int((cx - bw / 2) * source_width)
                y1_src = int((cy - bh / 2) * source_height)
                x2_src = int((cx + bw / 2) * source_width)
                y2_src = int((cy + bh / 2) * source_height)

                x1_src = max(x1_src, 0)
                y1_src = max(y1_src, 0)
                x2_src = min(x2_src, img_width)
                y2_src = min(y2_src, img_height)
                new_w_px = x2_src - x1_src
                new_h_px = y2_src - y1_src
                if new_w_px <= 0 or new_h_px <= 0:
                    continue

                x1 = random.randint(0, img_width - new_w_px) if new_w_px < img_width else 0
                y1 = random.randint(0, img_height - new_h_px) if new_h_px < img_height else 0
                x2, y2 = x1 + new_w_px, y1 + new_h_px

                new_cx = (x1 + new_w_px / 2) / img_width
                new_cy = (y1 + new_h_px / 2) / img_height
                new_w = new_w_px / img_width
                new_h = new_h_px / img_height

                blend_boxes.append(torch.tensor([new_cx, new_cy, new_w, new_h], device=box.device, dtype=box.dtype))
                blend_labels.append(label)
                blend_areas.append(area)
                blend_mixup_ratios.append(1.0 - beta)

                if self._COPYBLEND_WITH_EXPAND:
                    alpha = round(random.uniform(self._COPYBLEND_EXPAND_RATIOS[0], self._COPYBLEND_EXPAND_RATIOS[1]), 6)
                    expand_w = int(new_w_px * alpha)
                    expand_h = int(new_h_px * alpha)

                    x1_expand = x1_src - max(x1_src - expand_w, 0)
                    y1_expand = y1_src - max(y1_src - expand_h, 0)
                    x2_expand = min(x2_src + expand_w, img_width) - x2_src
                    y2_expand = min(y2_src + expand_h, img_height) - y2_src

                    new_x1_expand = x1 - max(x1 - x1_expand, 0)
                    new_y1_expand = y1 - max(y1 - y1_expand, 0)
                    new_x2_expand = min(x2 + x2_expand, img_width) - x2
                    new_y2_expand = min(y2 + y2_expand, img_height) - y2

                    x1_src, y1_src = x1_src - new_x1_expand, y1_src - new_y1_expand
                    x2_src, y2_src = x2_src + new_x2_expand, y2_src + new_y2_expand
                    x1, y1 = x1 - new_x1_expand, y1 - new_y1_expand
                    x2, y2 = x2 + new_x2_expand, y2 + new_y2_expand

                copy_patch_orig = images_f[source_idx, :, y1_src:y2_src, x1_src:x2_src]
                if self._COPYBLEND_TYPE == "blend":
                    blended_patch = updated_images[i, :, y1:y2, x1:x2] * beta + copy_patch_orig * (1 - beta)
                    updated_images[i, :, y1:y2, x1:x2] = blended_patch
                else:
                    updated_images[i, :, y1:y2, x1:x2] = copy_patch_orig

            if blend_boxes:
                blend_boxes_t = torch.stack(blend_boxes)
                blend_labels_t = torch.stack(blend_labels)
                blend_areas_t = torch.stack(blend_areas)

                updated_targets[i]["mixup"] = torch.tensor(
                    [1.0] * len(updated_targets[i]["boxes"]) + blend_mixup_ratios,
                    device=blend_boxes_t.device,
                    dtype=torch.float32,
                )
                updated_targets[i]["boxes"] = torch.cat([updated_targets[i]["boxes"], blend_boxes_t])
                updated_targets[i]["labels"] = torch.cat([updated_targets[i]["labels"], blend_labels_t])
                updated_targets[i]["area"] = torch.cat([updated_targets[i]["area"], blend_areas_t])

        batch["img"] = updated_images
        return self._targets_to_batch(batch, updated_targets)


class RTDETRDEIMDataset(RTDETRDataset):
    """RT-DETR dataset variant that uses a dedicated DEIM augmentation pipeline."""

    def __init__(self, *args, data=None, **kwargs):
        hyp = kwargs["hyp"]
        self.base_hyp = copy(hyp)
        self.deim_aug_scheduler = resolve_deim_aug_scheduler(hyp)
        self.policy_epochs, self.mixup_epochs, self.copyblend_epochs = self._compute_deim_schedule(hyp)
        self.mosaic_prob = float(hyp.mosaic)
        self.mixup_prob = float(hyp.mixup)
        self.copyblend_prob = float(hyp.copy_paste)
        self.decay_min_prob = float(hyp.aug_decay_min_prob)
        self.uses_deim_batch_augments = False
        super().__init__(*args, data=data, **kwargs)
        if self.augment:
            if self.rtdetr_augmentations and (self.mixup_prob > 0.0 or self.copyblend_prob > 0.0):
                self.collate_fn = _RTDETRDEIMBatchAugment(
                    mixup_prob=self.mixup_prob,
                    mixup_epochs=self.mixup_epochs,
                    copyblend_prob=self.copyblend_prob,
                    copyblend_epochs=self.copyblend_epochs,
                    scheduler_mode=self.deim_aug_scheduler,
                    decay_min_prob=self.decay_min_prob,
                )
                self.uses_deim_batch_augments = True
            self.set_epoch(0)

    def _compute_deim_schedule(self, hyp) -> tuple[tuple[int, int, int], tuple[int, int], tuple[int, int]]:
        """Compute DEIM stage boundaries for the selected scheduler mode."""
        policy_epochs = compute_policy_epochs(hyp)
        if self.deim_aug_scheduler == "decay":
            stop = policy_epochs[2]
            policy_epochs = (0, stop, stop)
            mixup_epochs = (0, stop)
            copyblend_epochs = (0, stop)
        else:
            mixup_epochs = policy_epochs[:2]
            copyblend_epochs = (policy_epochs[0], policy_epochs[2])
        return policy_epochs, mixup_epochs, copyblend_epochs

    def _build_v8_epoch_hyp(self, epoch: int):
        """Clone base hparams and apply optional DEIM-style decay for the v8 augmentation branch."""
        hyp = copy(self.base_hyp)
        _, _, stop = self.policy_epochs
        if self.deim_aug_scheduler == "decay":
            _, mixup_stop = self.mixup_epochs
            _, copy_paste_stop = self.copyblend_epochs
            hyp.mosaic = compute_deim_scheduled_prob(self.mosaic_prob, epoch, stop, self.decay_min_prob)
            hyp.mixup = compute_deim_scheduled_prob(self.mixup_prob, epoch, mixup_stop, self.decay_min_prob)
            hyp.copy_paste = compute_deim_scheduled_prob(self.copyblend_prob, epoch, copy_paste_stop, self.decay_min_prob)
            if epoch >= stop:
                # Match DEIM's final no-aug tail by neutralizing all remaining v8 augmentations.
                hyp.mosaic = 0.0
                hyp.mixup = 0.0
                hyp.copy_paste = 0.0
                hyp.cutmix = 0.0
                hyp.degrees = 0.0
                hyp.translate = 0.0
                hyp.scale = 0.0
                hyp.shear = 0.0
                hyp.perspective = 0.0
                hyp.hsv_h = 0.0
                hyp.hsv_s = 0.0
                hyp.hsv_v = 0.0
                hyp.augmentations = []
        return hyp

    def build_transforms(self, hyp=None):
        """Build DEIM transforms for train and standard formatting for train/val."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            if self.rtdetr_augmentations:
                transforms = rtdetr_deim_transforms(
                    self,
                    self.imgsz,
                    hyp,
                    stretch=True,
                    policy_epochs=self.policy_epochs,
                    mosaic_prob=self.mosaic_prob,
                )
            else:
                transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            transforms = Compose([])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to transforms and collate_fn for DEIM/v8 augmentation scheduling."""
        self.epoch = epoch
        if self.rtdetr_augmentations and hasattr(self.transforms, "set_epoch"):
            self.transforms.set_epoch(epoch)
        elif self.augment and self.deim_aug_scheduler == "decay":
            self.transforms = self.build_transforms(hyp=self._build_v8_epoch_hyp(epoch))

        if self.uses_deim_batch_augments:
            self.collate_fn.set_epoch(epoch)


class RTDETRDEIMValidator(RTDETRValidator):
    """Validator that builds the DEIM dataset variant."""

    def __call__(self, trainer=None, model=None):
        """Persist current train epoch so preprocess can attach it to validation batches."""
        if trainer is not None:
            self._val_epoch = int(trainer.epoch)
            self._val_training_progress = min(
                max(trainer.epoch / max(trainer.epochs - 1, 1), 0.0),
                1.0,
            )
        return super().__call__(trainer=trainer, model=model)

    def preprocess(self, batch):
        """Inject epoch into validation batches during training for DFine matcher scheduling."""
        batch = super().preprocess(batch)
        if self.training:
            if not hasattr(self, "_val_epoch"):
                raise KeyError("RTDETRDEIM validation requires epoch, but validator state is missing.")
            batch["epoch"] = int(self._val_epoch)
            batch["training_progress"] = float(self._val_training_progress)
        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        return RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )


class RTDETRDEIMTrainer(RTDETRTrainer):
    """RT-DETR trainer variant with isolated DEIM augmentation scheduling."""
    _deim_callback_registered = False

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        dataset = RTDETRDEIMDataset(
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
        return dataset

    def _setup_scheduler(self):
        """Initialize LR scheduler with optional DEIM flat-cosine schedule."""
        scheduler_arg = self.args.lr_scheduler
        if scheduler_arg is None:
            return super()._setup_scheduler()
        scheduler_name = str(scheduler_arg).lower()
        if not scheduler_name:
            return super()._setup_scheduler()

        if scheduler_name in {"linear"}:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        elif scheduler_name in {"cosine", "cos", "cos_lr"}:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        elif scheduler_name in {"flatcosine", "flat_cosine", "flatcos"}:
            # Flat phase keeps LR constant, then cosine anneals to lrf.
            if self.args.flat_epoch is None:
                _, flat_epoch, _ = compute_policy_epochs(self.args)
            else:
                flat_epoch = int(self.args.flat_epoch)
            if not (0 <= flat_epoch <= self.epochs):
                raise ValueError(
                    f"flatcosine got invalid flat_epoch={flat_epoch} for epochs={self.epochs}. "
                    "Expected 0 <= flat_epoch <= epochs."
                )
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
        else:
            LOGGER.warning(f"Unknown lr_scheduler='{scheduler_name}', falling back to default scheduler.")
            return super()._setup_scheduler()

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _on_train_epoch_start(self, trainer=None):
        """Apply DEIM epoch scheduling to transforms/collate and stop multi-scale at stage-4 start."""
        trainer = trainer or self
        epoch = trainer.epoch
        dataset = trainer.train_loader.dataset
        dataset.set_epoch(epoch)
        # InfiniteDataLoader keeps workers/iterator alive; reset so worker-side
        # dataset transforms and collate_fn pick up the updated epoch.
        trainer.train_loader.reset()
        stop_epoch = int(dataset.policy_epochs[-1])
        if epoch == stop_epoch and trainer.args.multi_scale > 0:
            trainer.args.multi_scale = 0.0
            LOGGER.info(f"DEIM stage-4 at epoch {epoch}: disabling multi-scale")

    def train(self, *args, **kwargs):
        # DEIM trainer handles augmentation schedule explicitly.
        # Disable the base trainer's hard close_mosaic hook whenever DEIM controls augmentation decay.
        if self.args.close_mosaic and (
            self.args.rtdetr_augmentations or resolve_deim_aug_scheduler(self.args) == "decay"
        ):
            self.args.close_mosaic = 0
        if not self._deim_callback_registered:
            self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
            self._deim_callback_registered = True
        return super().train(*args, **kwargs)

    def get_validator(self):
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        loss_gain = self.model_yaml.get("loss", {}).get("loss_gain", {})
        if loss_gain.get("fgl", 0) > 0:
            loss_names.append("fgl_loss")
        if loss_gain.get("ddf", 0) > 0:
            loss_names.append("ddf_loss")
        if loss_gain.get("rank", 0) > 0:
            loss_names.append("rank_loss")
        model = unwrap_model(self.model)
        if getattr(model.model[-1], "one_to_many_groups", 0) > 0:
            loss_names.extend(["giou_o2m", "cls_o2m", "l1_o2m"])
        self.loss_names = tuple(loss_names)
        return RTDETRDEIMValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


class RTDETRDEIMTrainerV2(RTDETRDEIMTrainer):
    """DEIM trainer with DEIMv2-like stage checkpointing and EMA refresh at stage switch."""

    _deim_v2_callback_registered = False
    _deim_ema_restart_decay = 0.9999  # DEIMv2 default

    def _dist_barrier(self) -> None:
        """Synchronize all ranks if running distributed training."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _ckpt_fitness(self, path: Path) -> float:
        """Read checkpoint fitness safely."""
        if not path.exists():
            return float("-inf")
        try:
            _, ckpt = load_checkpoint(path)
            return float(ckpt.get("best_fitness", float("-inf")))
        except Exception as e:
            LOGGER.warning(f"Could not read checkpoint fitness from {path}: {e}")
            return float("-inf")

    def _init_stage_state(self) -> None:
        """Initialize DEIMv2 stage-control state once training loader is ready."""
        if getattr(self, "_deim_stage_state_initialized", False):
            return

        dataset = getattr(getattr(self, "train_loader", None), "dataset", None)
        if dataset is None or not hasattr(dataset, "policy_epochs"):
            return

        self.best_stg1 = self.wdir / "best_stg1.pt"
        self.best_stg2 = self.wdir / "best_stg2.pt"
        self._deim_stop_epoch = int(dataset.policy_epochs[-1])
        self._deim_stage_switched = bool(self.start_epoch >= self._deim_stop_epoch)
        self._deim_stage1_best_fitness = self._ckpt_fitness(self.best_stg1)
        self._deim_stage2_best_fitness = self._ckpt_fitness(self.best_stg2)
        self._deim_ema_restart_decay = float(getattr(self.args, "ema_restart_decay", self._deim_ema_restart_decay))
        self._deim_stage_state_initialized = True

    def _set_ema_restart_decay(self) -> None:
        """Rebind EMA decay schedule to restart value."""
        if not self.ema:
            return
        decay = float(self._deim_ema_restart_decay)
        tau = float(getattr(self.args, "ema_tau", 2000.0))
        self.ema.decay = lambda x, d=decay, t=tau: d * (1.0 - math.exp(-x / t))

    def _reload_stage1_anchor(self) -> bool:
        """Reload stage-1 best checkpoint into model/optimizer/scaler/EMA."""
        source = self.best_stg1 if self.best_stg1.exists() else (self.best if self.best.exists() else self.last)
        if source is None or not source.exists():
            LOGGER.warning("DEIMv2 stage switch requested but no checkpoint was found for stage-1 anchor reload.")
            return False

        _, ckpt = load_checkpoint(source)
        if ckpt.get("ema") is None:
            LOGGER.warning(f"Checkpoint {source} has no EMA state; skipping DEIMv2 stage-1 anchor reload.")
            return False

        ema_state = ckpt["ema"].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            LOGGER.warning(f"Checkpoint {source} contains NaN/Inf EMA tensors; skipping DEIMv2 stage-1 reload.")
            return False

        unwrap_model(self.model).load_state_dict(ema_state)
        self._load_checkpoint_state(ckpt)
        self._set_ema_restart_decay()
        LOGGER.info(
            f"DEIMv2 stage switch: loaded stage-1 anchor {source} and refreshed EMA decay to {self._deim_ema_restart_decay:.4f}."
        )
        return True

    def _on_fit_epoch_end_v2(self, trainer=None):
        """Track stage-wise best checkpoints similar to DEIMv2."""
        trainer = trainer or self
        self._init_stage_state()
        if not getattr(self, "_deim_stage_state_initialized", False):
            return

        fitness = trainer.fitness
        if fitness is None:
            return
        fitness = float(fitness)
        epoch = int(trainer.epoch)

        if epoch < self._deim_stop_epoch:
            if fitness > self._deim_stage1_best_fitness and trainer.last.exists():
                self._deim_stage1_best_fitness = fitness
                if RANK in {-1, 0}:
                    self.best_stg1.write_bytes(trainer.last.read_bytes())
        else:
            if fitness > self._deim_stage2_best_fitness and trainer.last.exists():
                self._deim_stage2_best_fitness = fitness
                if RANK in {-1, 0}:
                    self.best_stg2.write_bytes(trainer.last.read_bytes())

    def _on_train_epoch_start(self, trainer=None):
        """Apply base DEIM policy update and perform stage-switch anchor reload once."""
        super()._on_train_epoch_start(trainer=trainer)
        trainer = trainer or self
        self._init_stage_state()
        if not getattr(self, "_deim_stage_state_initialized", False):
            return

        epoch = int(trainer.epoch)
        if self._deim_stage_switched or epoch != self._deim_stop_epoch:
            return

        if RANK in {-1, 0} and (not self.best_stg1.exists()) and self.best.exists():
            self.best_stg1.write_bytes(self.best.read_bytes())
        self._dist_barrier()
        self._deim_stage_switched = self._reload_stage1_anchor()

    def train(self, *args, **kwargs):
        """Register V2 callbacks and run training."""
        self._deim_stage_state_initialized = False
        if not self._deim_v2_callback_registered:
            self.add_callback("on_fit_epoch_end", self._on_fit_epoch_end_v2)
            self._deim_v2_callback_registered = True
        return super().train(*args, **kwargs)

    def final_eval(self):
        """Prefer stage-2 best for final evaluation when available."""
        self._init_stage_state()
        stage2 = getattr(self, "best_stg2", None)
        if stage2 is not None and stage2.exists():
            best_orig = self.best
            self.best = stage2
            try:
                super().final_eval()
            finally:
                self.best = best_orig
            if RANK in {-1, 0} and best_orig.exists():
                strip_optimizer(best_orig)
            return
        super().final_eval()


class RTDETRDEIMSegmentValidator(RTDETRDEIMValidator):
    """RT-DETR DEIM validator for instance segmentation models built on DeimSegmentDecoder.

    Ports the SegmentationValidator mask handling (proto-based mask assembly, GT mask preparation, mask IoU stats,
    and COCO segm JSON) onto the RT-DETR postprocess, where top-k selection already happens inside the decoder head
    and predictions are normalized [cx, cy, w, h, score, cls, mc...] rows.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize the validator with task 'segment' and SegmentMetrics."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess validation batch and cast GT masks to float."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and select the mask processing function based on save_json/save_txt flags."""
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        # More accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the DEIM dataset variant with segmentation masks enabled."""
        return RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task="segment",
        )

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Convert decoder outputs to pixel-space detections and assemble masks from protos and coefficients.

        Args:
            preds (torch.Tensor | list | tuple): Model predictions `((y, proto), x)` where `y` has shape
                (batch_size, num_queries, 6 + nm) with last dimension [cx, cy, w, h, score, class, mc...].

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing 'bboxes' (xyxy
                pixel format), 'conf', 'cls', and 'masks'.
        """
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        y = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        bboxes, scores, labels, coeffs = y.split((4, 1, 1, proto.shape[1]), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
        scores = scores.squeeze(-1)
        labels = labels.squeeze(-1)
        imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto

        outputs = []
        for b, s, lab, c, p in zip(bboxes, scores, labels, coeffs, proto):
            keep = s > self.args.conf  # confidence threshold (also drops NaN: NaN > x is False)
            b, s, lab, c = b[keep], s[keep], lab[keep], c[keep]
            masks = (
                self.process(p, c, b, shape=imgsz)
                if c.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=b.device,
                )
            )
            outputs.append({"bboxes": b, "conf": s, "cls": lab, "masks": masks})
        return outputs

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for validation, converting GT masks to per-instance binary masks at proto size."""
        prepared_batch = super()._prepare_batch(si, batch)
        nl = prepared_batch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared_batch["masks"] = masks
        return prepared_batch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute correct prediction matrices for boxes and masks."""
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())  # float, uint8
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_m": tp_m})  # update tp with mask IoU
        return tp

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        super().gather_stats()  # gather stats from DetectionValidator
        self._gather_image_metrics(self.metrics.seg)

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Plot batch predictions with masks and bounding boxes."""
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)  # plot bboxes

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scale masks to the original image size; boxes are already scaled in postprocessing/pred_to_json."""
        return {
            **predn,
            "masks": ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[
                0
            ].byte(),
        }

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Save one JSON result for COCO evaluation, including the RLE-encoded segmentation."""

        def to_string(counts: list[int]) -> str:
            """Convert the RLE object into a compact string representation."""
            result = []

            for i in range(len(counts)):
                x = int(counts[i])

                # Apply delta encoding for all counts after the second entry
                if i > 2:
                    x -= int(counts[i - 2])

                # Variable-length encode the value
                while True:
                    c = x & 0x1F  # Take 5 bits
                    x >>= 5

                    # If the sign bit (0x10) is set, continue if x != -1;
                    # otherwise, continue if x != 0
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20  # Set continuation bit
                    c += 48  # Shift to ASCII
                    result.append(chr(c))
                    if not more:
                        break

            return "".join(result)

        def multi_encode(pixels: torch.Tensor) -> list[int]:
            """Convert multiple binary masks using Run-Length Encoding (RLE)."""
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = torch.where(transitions)
            col_idx = col_idx + 1

            # Compute run lengths
            counts = []
            for i in range(pixels.shape[0]):
                positions = col_idx[row_idx == i]
                if len(positions):
                    count = torch.diff(positions).tolist()
                    count.insert(0, positions[0].item())
                    count.append(len(pixels[i]) - positions[-1].item())
                else:
                    count = [len(pixels[i])]

                # Ensure starting with background (0) count
                if pixels[i][0].item() == 1:
                    count = [0, *count]
                counts.append(count)

            return counts

        pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)  # N, H*W
        h, w = predn["masks"].shape[1:3]
        counts = multi_encode(pred_masks)
        rles = []
        for c in counts:
            rles.append({"size": [h, w], "counts": to_string(c)})
        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r  # segmentation

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return COCO-style instance segmentation evaluation metrics."""
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])


class RTDETRDEIMSegmentTrainer(RTDETRDEIMTrainer):
    """RT-DETR DEIM trainer variant for instance segmentation models built on DeimSegmentDecoder.

    Note:
        The DEIM batch-level augmentations (MixUp/CopyBlend collate, active with `rtdetr_augmentations=True`) do not
        handle masks; train segmentation models with `rtdetr_augmentations=False`.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize the trainer with task 'segment'."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the DEIM dataset variant with segmentation masks enabled."""
        return RTDETRDEIMDataset(
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
            task="segment",
        )

    def get_validator(self):
        """Return an RTDETRDEIMSegmentValidator with loss names extended by the mask and semseg losses."""
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        loss_gain = self.model_yaml.get("loss", {}).get("loss_gain", {})
        if loss_gain.get("fgl", 0) > 0:
            loss_names.append("fgl_loss")
        if loss_gain.get("ddf", 0) > 0:
            loss_names.append("ddf_loss")
        if loss_gain.get("rank", 0) > 0:
            loss_names.append("rank_loss")
        model = unwrap_model(self.model)
        if getattr(model.model[-1], "one_to_many_groups", 0) > 0:
            loss_names.extend(["giou_o2m", "cls_o2m", "l1_o2m"])
        loss_names.extend(["mask_loss", "sem_loss", "mask_aux_loss"])
        self.loss_names = tuple(loss_names)
        return RTDETRDEIMSegmentValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


class RTDETRDEIMPoseValidator(RTDETRDEIMValidator):
    """RT-DETR DEIM validator for pose estimation models built on DeimPoseDecoder.

    Ports the PoseValidator keypoint handling (GT keypoint preparation, OKS-based pose stats via kpt_iou, and COCO
    keypoints JSON) onto the RT-DETR postprocess, where top-k selection already happens inside the decoder head and
    predictions are normalized [cx, cy, w, h, score, cls, kpts...] rows.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize the validator with task 'pose' and PoseMetrics."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess validation batch and cast GT keypoints to float."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and keypoint sigmas for OKS calculation."""
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the DEIM dataset variant with keypoints enabled."""
        return RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task="pose",
        )

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Convert decoder outputs to pixel-space detections with per-instance keypoints.

        Args:
            preds (torch.Tensor | list | tuple): Model predictions `(y, x)` where `y` has shape (batch_size,
                num_queries, 6 + nk) with last dimension [cx, cy, w, h, score, class, kpts...] (keypoint xy
                normalized, visibility as raw logits).

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing 'bboxes' (xyxy
                pixel format), 'conf', 'cls', and 'keypoints' (M, nkpt, ndim, xy in pixels).
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]
        y = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        nk = self.kpt_shape[0] * self.kpt_shape[1]
        bboxes, scores, labels, kpts = y.split((4, 1, 1, nk), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
        scores = scores.squeeze(-1)
        labels = labels.squeeze(-1)
        kpts = kpts.view(*kpts.shape[:2], *self.kpt_shape).clone()
        kpts[..., 0] *= self.args.imgsz
        kpts[..., 1] *= self.args.imgsz

        outputs = []
        for b, s, lab, k in zip(bboxes, scores, labels, kpts):
            keep = s > self.args.conf  # confidence threshold (also drops NaN: NaN > x is False)
            outputs.append({"bboxes": b[keep], "conf": s[keep], "cls": lab[keep], "keypoints": k[keep]})
        return outputs

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for validation, scaling GT keypoints to pixel coordinates."""
        prepared_batch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = prepared_batch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        prepared_batch["keypoints"] = kpts
        return prepared_batch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute correct prediction matrices for boxes and keypoints."""
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # update tp with kpts IoU
        return tp

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        super().gather_stats()  # gather stats from DetectionValidator
        self._gather_image_metrics(self.metrics.pose)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO pose detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scale keypoints to the original image size; boxes are already scaled in postprocessing/pred_to_json."""
        kpts = predn["keypoints"].clone()
        kpts[..., 0] *= pbatch["ori_shape"][1] / self.args.imgsz
        kpts[..., 1] *= pbatch["ori_shape"][0] / self.args.imgsz
        return {**predn, "kpts": kpts}

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Save one JSON result for COCO evaluation, including the flattened keypoint list."""
        super().pred_to_json(predn, pbatch)
        kpts = predn["kpts"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return COCO-style keypoint evaluation metrics."""
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])


class RTDETRDEIMPoseTrainer(RTDETRDEIMTrainer):
    """RT-DETR DEIM trainer variant for pose estimation models built on DeimPoseDecoder.

    Note:
        The DEIM batch-level augmentations (MixUp/CopyBlend collate, active with `rtdetr_augmentations=True`) do not
        handle keypoints; train pose models with `rtdetr_augmentations=False`.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize the trainer with task 'pose'."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and ensure it contains the required `kpt_shape` key."""
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        """Build the model with `kpt_shape` synced from the dataset config (mirrors PoseModel)."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        data_kpt_shape = self.data["kpt_shape"]
        if data_kpt_shape and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

    def set_model_attributes(self):
        """Set keypoint shape and keypoint names attributes on the model."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the DEIM dataset variant with keypoints enabled."""
        return RTDETRDEIMDataset(
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
            task="pose",
        )

    def get_validator(self):
        """Return an RTDETRDEIMPoseValidator with loss names extended by the pose and kobj losses."""
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        loss_gain = self.model_yaml.get("loss", {}).get("loss_gain", {})
        if loss_gain.get("fgl", 0) > 0:
            loss_names.append("fgl_loss")
        if loss_gain.get("ddf", 0) > 0:
            loss_names.append("ddf_loss")
        if loss_gain.get("rank", 0) > 0:
            loss_names.append("rank_loss")
        model = unwrap_model(self.model)
        if getattr(model.model[-1], "one_to_many_groups", 0) > 0:
            loss_names.extend(["giou_o2m", "cls_o2m", "l1_o2m"])
        loss_names.extend(["pose_loss", "kobj_loss", "kpt_l1_loss", "pose_aux_loss", "kobj_aux_loss", "kpt_l1_aux_loss"])
        self.loss_names = tuple(loss_names)
        return RTDETRDEIMPoseValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


class RTDETRDEIMOBBValidator(RTDETRDEIMValidator):
    """RT-DETR DEIM validator for oriented bounding box models built on DeimOBBDecoder.

    Ports the OBBValidator rotated-box handling (xywhr GT preparation, batch_probiou true positives, and rbox/poly
    JSON) onto the RT-DETR postprocess, where top-k selection already happens inside the decoder head and predictions
    are normalized [cx, cy, w, h, score, cls, angle] rows.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize the validator with task 'obb' and OBBMetrics."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics()

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for OBB validation."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # check if dataset is DOTA format
        self.confusion_matrix.task = "obb"  # set confusion matrix task to 'obb'

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the DEIM dataset variant with oriented bounding boxes enabled."""
        return RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task="obb",
        )

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Convert decoder outputs to pixel-space rotated detections.

        Args:
            preds (torch.Tensor | list | tuple): Model predictions `(y, x)` where `y` has shape (batch_size,
                num_queries, 7) with last dimension [cx, cy, w, h, score, class, angle].

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing 'bboxes' (xywhr
                pixel format), 'conf', and 'cls'.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]
        y = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        bboxes, scores, labels, angles = y.split((4, 1, 1, 1), dim=-1)
        bboxes = bboxes * self.args.imgsz  # normalized xywh -> pixel xywh
        scores = scores.squeeze(-1)
        labels = labels.squeeze(-1)

        outputs = []
        for b, s, lab, a in zip(bboxes, scores, labels, angles):
            keep = s > self.args.conf  # confidence threshold (also drops NaN: NaN > x is False)
            outputs.append({"bboxes": torch.cat([b[keep], a[keep]], dim=-1), "conf": s[keep], "cls": lab[keep]})
        return outputs

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for validation, scaling the xywh part of GT xywhr boxes to pixel coordinates."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Compute the correct prediction matrix using probabilistic IoU between rotated boxes."""
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Plot predicted oriented bounding boxes on input images and save the result."""
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i
        keys = preds[0].keys()
        batched_preds = {k: torch.cat([x[k] for x in preds], dim=0) for k in keys}
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO OBB detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Save one JSON result with rotated bounding boxes in both rbox and polygon formats."""
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = predn["bboxes"]
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for r, b, s, c in zip(rbox.tolist(), poly.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "score": round(s, 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Save predictions in DOTA submission format (port of OBBValidator.eval_json).

        Writes per-class Task1 txt files for the split tiles, plus a merged version where tile predictions are
        mapped back to the original image coordinates and NMS-deduplicated with probiou (threshold 0.3, matching
        the YOLO OBB convention; results may differ slightly from the official DOTA merging script).
        """
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            with open(pred_json, encoding="utf-8") as f:
                data = json.load(f)
            # Save split results
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                p = d["poly"]

                with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__", 1)[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_probiou)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats


class RTDETRDEIMOBBTrainer(RTDETRDEIMTrainer):
    """RT-DETR DEIM trainer variant for oriented bounding box models built on DeimOBBDecoder.

    Note:
        The DEIM batch-level augmentations (MixUp/CopyBlend collate, active with `rtdetr_augmentations=True`) do not
        handle rotated boxes; train OBB models with `rtdetr_augmentations=False`.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize the trainer with task 'obb'."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the DEIM dataset variant with oriented bounding boxes enabled."""
        return RTDETRDEIMDataset(
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
            task="obb",
        )

    def get_validator(self):
        """Return an RTDETRDEIMOBBValidator with loss names extended by the angle loss."""
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        loss_gain = self.model_yaml.get("loss", {}).get("loss_gain", {})
        if loss_gain.get("fgl", 0) > 0:
            loss_names.append("fgl_loss")
        if loss_gain.get("ddf", 0) > 0:
            loss_names.append("ddf_loss")
        if loss_gain.get("rank", 0) > 0:
            loss_names.append("rank_loss")
        model = unwrap_model(self.model)
        if getattr(model.model[-1], "one_to_many_groups", 0) > 0:
            loss_names.extend(["giou_o2m", "cls_o2m", "l1_o2m"])
        loss_names.extend(["angle_loss", "probiou_loss", "angle_aux_loss", "probiou_aux_loss"])
        self.loss_names = tuple(loss_names)
        return RTDETRDEIMOBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
