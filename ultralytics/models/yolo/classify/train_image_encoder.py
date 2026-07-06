# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Trainer for encoder distillation pretraining from frozen vision foundation models.

Distill one or more teachers (EUPE, DINOv3, SAM3, SigLIP2) into a YOLO backbone using online
teacher forward each step. Supports single-teacher (EUPE Section 4: proxy -> student) and
multi-teacher (EUPE Section 3: multiple teachers -> proxy, Eq.6: L = sum_i L_i).

Dataset support: WebDataset tar shards (DataComp-12M) and image folders (COCO, ImageNet).
"""

from __future__ import annotations

import itertools
import json
from copy import copy
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from callbacks.distill_aug import classify_augmentations_distill
from ultralytics.data.augment import classify_transforms
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.image_encoder import ImageEncoderModel
from ultralytics.nn.teacher_model import TEACHER_REGISTRY, build_teacher_model, safe_key
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils import callbacks as ul_callbacks

# DataComp-12M has images up to ~268M pixels. PIL raises DecompressionBombError above 179M pixels,
# crashing DataLoader workers despite wds.warn_and_continue. Standard for web-crawled pipelines.
Image.MAX_IMAGE_PIXELS = None

# ImageNet normalization (used by EUPE, DINOv3, SigLIP2, SAM3 -- standard for ViT models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BalancedSampler(torch.utils.data.WeightedRandomSampler):
    """Per-source temperature-balanced sampler for ConcatDataset.

    For source sizes ``N_i`` and temperature ``t`` in [0, 1], per-sample weight is ``N_i ** -t``. t=0 reproduces uniform
    sampling; t=0.5 is sqrt-balanced (EUPE / DINOv3 convention); t=1 fully balances across sources regardless of size.

    DDP-safe: each rank seeds its own generator with ``epoch * num_replicas + rank`` and draws ``num_samples //
    num_replicas`` IID samples with replacement.

    Args:
        sizes (list[int]): Per-source sample counts (``[len(d) for d in concat.datasets]``).
        t (float): Temperature in [0, 1].
        total_n (int): Total epoch samples across all ranks (typically ``len(concat)``).
        num_replicas (int, optional): DDP world size.
        rank (int, optional): DDP rank.
    """

    def __init__(self, sizes: list[int], t: float, total_n: int, num_replicas: int = 1, rank: int = 0):
        weights = torch.cat([torch.full((s,), s**-t, dtype=torch.double) for s in sizes])
        super().__init__(weights, total_n // num_replicas, replacement=True)
        self.num_replicas, self.rank, self.epoch = num_replicas, rank, 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.epoch * self.num_replicas + self.rank)
        yield from torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _WebDatasetLoader(DataLoader):
    """DataLoader for WebDataset shards, compatible with ultralytics trainer.

    Wraps a wds.DataPipeline inside a standard DataLoader by converting it to an IterableDataset. Provides .reset(),
    .dataset (with __len__), .num_workers, .sampler -- all required by engine/trainer.py (lines 283, 370, 379, 403).

    Args:
        pipeline: WebDataset pipeline (wds.DataPipeline).
        num_samples (int): Estimated total samples in dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
    """

    def __init__(self, pipeline, num_samples, batch_size, num_workers, drop_last=False):
        """Initialize with WebDataset pipeline wrapped as IterableDataset."""

        class _IterDS(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter(pipeline)

            def __len__(self):
                return num_samples

        super().__init__(
            _IterDS(), batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=drop_last
        )

        import types

        object.__setattr__(self, "sampler", types.SimpleNamespace(set_epoch=lambda _: None))

    def __iter__(self):
        """Yield batches, stopping at epoch boundary (same pattern as InfiniteDataLoader)."""
        return itertools.islice(super().__iter__(), len(self))

    def reset(self):
        """No-op: WebDataset workers don't need reset (no mosaic state)."""


class _ImageOnlyDataset(Dataset):
    """Load all images from a directory for label-free distillation.

    Walk a directory recursively for image files. No class labels needed -- teacher provides the supervision signal.
    """

    def __init__(self, root, transform):
        """Initialize with directory path and transform.

        Args:
            root (str | Path): Directory containing images (flat or nested).
            transform: Callable that takes PIL image and returns a tensor.
        """
        self.samples = sorted(p for p in Path(root).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
        self.transform = transform
        if not self.samples:
            raise FileNotFoundError(f"No images found in {root}")
        LOGGER.info(f"ImageOnlyDataset: {len(self.samples)} images from {root}")

    def __len__(self):
        """Return number of images."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Load image and apply transform."""
        return self.transform(Image.open(self.samples[index]).convert("RGB"))


class ImageEncoderTrainer(ClassificationTrainer):
    """Trainer for single or multi-teacher encoder distillation pretraining.

    Single-teacher: EUPE Section 4 (proxy -> student). Multi-teacher: EUPE Section 3 / Eq.6 (sum over teachers). All
    teachers run online each step (frozen, no pre-computed embeddings) following AM-RADIO/EUPE convention.

    Attributes:
        teachers (list[str]): Parsed from args.teachers ('+' separated string).
        teacher_models (dict): Loaded frozen teacher models keyed by safe name (e.g. 'eupe_vitb16').
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize ImageEncoderTrainer.

        Args:
            cfg (dict[str, Any], optional): Default configuration dictionary.
            overrides (dict[str, Any], optional): Parameter overrides. Use 'teachers' key with '+' separated
            variants (e.g. 'eupe: vitb16' or 'eupe:vitb16+dinov3:vitl16').
            _callbacks (dict, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides.setdefault("close_mosaic", 0)  # no mosaic in distillation, avoids .reset() call
        overrides.setdefault("teachers", "eupe:vitb16")
        self.teacher_models = {}
        # Distillation reports distill_loss only, so filter Platform hooks here before BaseTrainer.__init__ runs them
        # and lets Platform send a cancel trigger for missing classification metrics.
        original_add_integration_callbacks = ul_callbacks.add_integration_callbacks

        def add_integration_callbacks_without_platform(instance):
            original_add_integration_callbacks(instance)
            for event_callbacks in instance.callbacks.values():
                event_callbacks[:] = [
                    cb for cb in event_callbacks if cb.__module__ != "ultralytics.utils.callbacks.platform"
                ]

        ul_callbacks.add_integration_callbacks = add_integration_callbacks_without_platform
        try:
            super().__init__(cfg, overrides, _callbacks)
        finally:
            ul_callbacks.add_integration_callbacks = original_add_integration_callbacks
        # self.args.teachers survives DDP serialization (in check_dict_alignment allowed_custom_keys)
        raw = self.args.teachers
        self.teachers = raw.split("+") if isinstance(raw, str) else raw
        self._safe_keys = [safe_key(n) for n in self.teachers]
        self._teacher_imgsz = max(TEACHER_REGISTRY[n]["imgsz"] for n in self.teachers)
        # Multi-scale distillation (R1): student sees a rotating set of input sizes so the frozen backbone
        # learns the higher token counts it meets at detection resolution (640 -> 20x20 P5 vs 224 -> 7x7).
        # The loader serves the largest scale (genuine detail, not upsampled 224); preprocess_batch
        # round-robins the student size per step and downsamples the teacher back to its native res.
        ss = getattr(self.args, "student_scales", None)
        self._student_scales = [int(s) for s in str(ss).split(",")] if ss else None
        self._student_scale_step = 0
        self._load_imgsz = max(self._student_scales) if self._student_scales else self._teacher_imgsz
        # High-res adaptation tail (DINOv3 / FastViT convention): the student runs at ``_hires_res`` for the
        # last ``_hires_tail_epochs`` epochs so its frozen P5 attention adapts to the higher token count met at
        # detection resolution (224 -> 7x7 vs 512 -> 16x16). The load resolution and teacher res stay put, so
        # every pre-tail epoch is identical to a no-tail run and the tail is the only changed variable.
        ht = getattr(self.args, "hires_tail", None)
        self._hires_res, self._hires_tail_epochs = map(int, ht.split(":")) if ht else (None, None)

        # Register hooks here (not in runner): dist.py:57 only serializes self.args to DDP workers.
        from callbacks import beta2_override, grad_clip, muon_w, nfs_sync, paths, wd_schedule  # runner-local package

        grad_clip_v = getattr(self.args, "grad_clip", None)
        beta2_v = getattr(self.args, "beta2", None)
        muon_w_v = getattr(self.args, "muon_w", None)
        nfs_sync_v = getattr(self.args, "nfs_sync", False)
        for v, mod in ((grad_clip_v, grad_clip), (beta2_v, beta2_override), (muon_w_v, muon_w)):
            if v is not None:
                self.add_callback("on_train_start", mod.override(float(v)))
        if nfs_sync_v:
            sync_start, sync_end = nfs_sync.setup(str(paths.NFS_MIRROR_ROOT), interval_sec=paths.SYNC_INTERVAL_SEC)
            self.add_callback("on_train_start", sync_start)
            self.add_callback("on_train_end", sync_end)

        # WD cosine schedule: DINOv3 / DINOv2 / EUPE all ramp weight_decay from a small start
        # value to a larger end value over training. Reference shapes:
        #   DINOv3 ConvNeXt-T distill ``configs/train/distillation_convnext/convnext_tiny_p16.yaml``
        #     schedules.weight_decay {start=0.04, peak=0.2, end=0.2, warmup_epochs=500}.
        #   EUPE ``configs/ssl_default_config.yaml`` optim.weight_decay=0.04, weight_decay_end=0.4.
        #   DINOv2 paper §A.3: cosine wd 0.04→0.4.
        # Activated by setting ``wd_end`` in the runner train_args (None or 0 keeps fixed wd).
        # Photometric augs (grayscale/gaussian_blur/solarize) live in
        # ``callbacks/distill_aug.py:classify_augmentations_distill`` and are read directly from
        # self.args inside ``_build_transforms`` below; no callback needed.
        # ``weight_decay`` is in CFG_FRACTION_KEYS so already coerced to float by the cfg layer;
        # ``wd_end`` arrives from the recipe dict as a Python float. No paranoia casts needed.
        wd_end_v = getattr(self.args, "wd_end", None)
        if wd_end_v is not None and wd_end_v > 0:
            self.add_callback(
                "on_train_epoch_start",
                wd_schedule.override(start=self.args.weight_decay, end=wd_end_v),
            )

        if RANK in (-1, 0):
            LOGGER.info(
                f"ImageEncoderTrainer hooks: grad_clip={grad_clip_v} beta2={beta2_v} "
                f"muon_w={muon_w_v} nfs_sync={nfs_sync_v} wd_end={wd_end_v}"
            )

    def _setup_train(self):
        """Set bf16 autocast after AMP check to avoid poisoning the yolo26n detection test."""
        super()._setup_train()
        if self.amp:
            # bf16 instead of fp16: fp16 backbone produces nan on ~5% of DataComp-12M batches
            # (max 65504 vs bf16 sharing fp32 exponent range). Follows DUNE convention.
            torch.set_autocast_dtype("cuda", torch.bfloat16)
        # kNN eval on ImageNet for frozen feature quality tracking (EUPE/RADIO protocol: k=20, T=0.07).
        # Enabled via knn_eval=<imagenet_path> in train_args (survives DDP via allowed_custom_keys).
        # Runs inside trainer so DDP subprocess inherits it (model.add_callback in run scripts is
        # lost because DDP re-creates trainer from serialized args only, utils/dist.py:79).
        self._knn_state = {}
        knn_path = getattr(self.args, "knn_eval", "")
        if knn_path and RANK in {-1, 0}:
            knn_path = Path(knn_path)
            if knn_path.is_dir():
                self._knn_state["path"] = knn_path
            else:
                LOGGER.warning(f"kNN eval skipped: {knn_path} not found")

    @staticmethod
    def _resolve_paths(data_path):
        """Resolve a path to (train, val). val is None when no held-out val split is discoverable.

        Order: WebDataset shards/ → COCO-style images/train2017 → ImageNet-style train/ →
        swap the last `train` component of the path for `val` and use if that dir exists → None.
        """
        p = Path(data_path)
        if (p / "shards").is_dir():
            return str(p), None
        if (p / "images" / "train2017").is_dir():
            v = p / "images" / "val2017"
            return str(p / "images" / "train2017"), (str(v) if v.is_dir() else None)
        if (p / "train").is_dir():
            v = p / "val"
            return str(p / "train"), (str(v) if v.is_dir() else None)
        parts = list(p.parts)
        if "train" in parts:
            i = len(parts) - 1 - parts[::-1].index("train")  # last occurrence
            v = Path(*parts[:i], "val", *parts[i + 1 :])
            if v.is_dir():
                return str(p), str(v)
        return str(p), None

    def get_dataset(self):
        """Build minimal data dict for distillation (no check_cls_dataset needed).

        Auto-detect layout per path: shards/*.tar (WebDataset), images/train2017 (COCO), train/ (ImageNet),
        or flat; sources without a held-out val split are dropped from val to prevent train→val duplication.
        Supports comma-separated paths for combining ImageFolder datasets (no WebDataset mixing).

        Returns:
            (dict): Data dict with 'train', 'val', 'nc', 'names', 'channels' keys.
        """
        paths = [p.strip() for p in str(self.args.data).split(",")]
        if len(paths) == 1:
            train_path, val_path = self._resolve_paths(paths[0])
            if val_path is None:  # no held-out val → fall back to train so val loss is still tracked
                val_path = train_path
        else:
            train_paths, val_paths = [], []
            for p in paths:
                t, v = self._resolve_paths(p)
                train_paths.append(t)
                if v is not None:
                    val_paths.append(v)
            train_path = train_paths
            if not val_paths:
                LOGGER.warning("No held-out val splits across sources; val falls back to train (noisy)")
                val_paths = train_paths
            val_path = val_paths
        return {
            "train": train_path,
            "val": val_path,
            "nc": 1000,
            "names": {i: str(i) for i in range(1000)},
            "channels": 3,
        }

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return ImageEncoderModel with per-teacher adaptor heads.

        Args:
            cfg (Any, optional): Model configuration.
            weights (Any, optional): Pre-trained model weights.
            verbose (bool, optional): Whether to display model information.

        Returns:
            (ImageEncoderModel): Model with per-teacher adaptor heads.
        """
        self._load_teachers()
        # Build teacher config dict for the model
        teachers_cfg = {}
        for name in self.teachers:
            reg = TEACHER_REGISTRY[name]
            teachers_cfg[name] = {
                "embed_dim": reg["embed_dim"],
                "num_patches": reg["num_patches"],
                "token_types": reg["token_types"],
            }
        loss_cfg = {}
        for k in ("cos_weight", "l1_weight", "cls_l1", "loss_type"):
            v = getattr(self.args, k, None)
            if v is not None:
                loss_cfg[k] = v
        model = ImageEncoderModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            teachers=teachers_cfg,
            proj_hidden_dim=getattr(self.args, "proj_hidden_dim", None),
            loss_cfg=loss_cfg or None,
            distill_path=getattr(self.args, "distill_path", "adaptor"),
            adaptor_arch=getattr(self.args, "adaptor_arch", "mlp"),
        )
        if weights:
            model.load(weights)
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        model.model[-1].linear.requires_grad_(False)
        return model

    def _load_teachers(self):
        """Load and cache all frozen teacher models."""
        normalize_input = bool(getattr(self.args, "normalize_teacher_input", False))
        for name, sk in zip(self.teachers, self._safe_keys):
            if sk not in self.teacher_models:
                LOGGER.info(f"Loading teacher '{name}' (normalize_input={normalize_input})...")
                self.teacher_models[sk] = build_teacher_model(name, self.device, normalize_input=normalize_input)
                n = sum(p.numel() for p in self.teacher_models[sk].parameters()) / 1e6
                LOGGER.info(f"  {name}: {n:.1f}M params, embed_dim={self.teacher_models[sk].embed_dim}")

    def _build_transforms(self, mode):
        """Build shared transform at the load resolution with ImageNet normalization.

        Same augmented image goes to both teacher and student, resized to their respective resolutions
        in preprocess_batch (EUPE Stage 2 / DUNE / AM-RADIO convention). ``_load_imgsz`` is the teacher
        resolution normally, or the largest student scale under multi-scale distillation.
        """
        sz = self._load_imgsz
        if mode == "train":
            return classify_augmentations_distill(
                size=sz,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                hflip=self.args.fliplr,
                vflip=self.args.flipud,
                erasing=self.args.erasing,
                auto_augment=self.args.auto_augment,
                hsv_h=self.args.hsv_h,
                hsv_s=self.args.hsv_s,
                hsv_v=self.args.hsv_v,
                grayscale=getattr(self.args, "grayscale", 0.0),
                gaussian_blur=getattr(self.args, "gaussian_blur", 0.0),
                solarize=getattr(self.args, "solarize", 0.0),
                interpolation="BICUBIC",
            )
        return classify_transforms(size=sz, mean=IMAGENET_MEAN, std=IMAGENET_STD, interpolation="BICUBIC")

    def build_dataset(self, img_path, mode: str = "train", batch=None):
        """Build dataset from WebDataset shards, image folder, or list of image folders.

        Args:
            img_path (str | list[str]): Path(s) to dataset. List triggers ConcatDataset for multi-dataset.
            mode (str, optional): Dataset mode.
            batch (Any, optional): Unused.

        Returns:
            Dataset yielding transformed image tensors.
        """
        tf = self._build_transforms(mode)
        if isinstance(img_path, list):
            datasets = [_ImageOnlyDataset(p, tf) for p in img_path]
            LOGGER.info(f"Combined dataset: {' + '.join(f'{p}({len(d)})' for p, d in zip(img_path, datasets))}")
            return torch.utils.data.ConcatDataset(datasets)
        shards = sorted(str(p) for p in Path(img_path).glob("shards/*.tar"))
        if shards:
            import webdataset as wds

            return (
                wds.WebDataset(shards, shardshuffle=mode == "train", nodesplitter=wds.split_by_node)
                .shuffle(1000 if mode == "train" else 0)
                .decode("pil", handler=wds.warn_and_continue)
                .to_tuple("jpg", handler=wds.warn_and_continue)
                .map(lambda sample: tf(sample[0]))
            )
        return _ImageOnlyDataset(img_path, tf)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return DataLoader for WebDataset or image folder.

        Args:
            dataset_path (str): Path to dataset root.
            batch_size (int, optional): Batch size.
            rank (int, optional): Process rank for DDP.
            mode (str, optional): 'train' or 'val'.

        Returns:
            (DataLoader): DataLoader yielding (student_imgs, teacher_imgs) batches.
        """
        dataset = self.build_dataset(dataset_path, mode)
        if isinstance(dataset, torch.utils.data.IterableDataset):
            # WebDataset pipeline -- read actual sample count from img2dataset stats
            num_samples = sum(
                json.load(open(f))["successes"] for f in sorted(Path(dataset_path).glob("shards/*_stats.json"))
            )
            if not num_samples:
                num_samples = len(list(Path(dataset_path).glob("shards/*.tar"))) * 6000
                LOGGER.warning(f"No stats files found, estimating {num_samples} samples from shard count")
            # In DDP, split_by_node partitions shards across ranks -- divide samples accordingly
            world_size = torch.distributed.get_world_size() if RANK != -1 else 1
            per_rank = num_samples // world_size
            # Infinite cycling so short ranks don't exhaust early; islice caps at len(self)
            dataset.with_epoch(per_rank)
            return _WebDatasetLoader(dataset, per_rank, batch_size, self.args.workers, drop_last=mode == "train")
        sample_t = float(getattr(self.args, "sample_t", 0.0) or 0.0)
        if mode == "train" and sample_t > 0 and isinstance(dataset, torch.utils.data.ConcatDataset):
            sizes = [len(d) for d in dataset.datasets]
            world = torch.distributed.get_world_size() if RANK != -1 else 1
            sampler = BalancedSampler(sizes, sample_t, len(dataset), world, max(RANK, 0))
            if RANK in (-1, 0):
                probs = [s ** (1 - sample_t) for s in sizes]
                norm = sum(probs)
                LOGGER.info(
                    f"BalancedSampler t={sample_t}: per-source p=["
                    + ", ".join(f"{p / norm:.4f}" for p in probs)
                    + f"] sizes={sizes}"
                )
        else:
            sampler = (
                torch.utils.data.distributed.DistributedSampler(dataset, shuffle=mode == "train")
                if RANK != -1
                else None
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None and mode == "train",
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=mode == "train",
        )

    def preprocess_batch(self, batch):
        """Move images to device, resize for student and teacher, run all teachers.

        One augmented image loaded at ``_load_imgsz`` is resized to the student and teacher resolutions
        via F.interpolate. Follows DUNE convention (dune/teachers/forward.py:30). Under multi-scale
        distillation (R1) the student size rotates through ``_student_scales`` per step while the teacher
        stays at its native resolution; the loss (image_encoder.py) resamples teacher patches to the
        student grid, so the student learns higher token counts on genuine detail.

        Args:
            batch (torch.Tensor): Images at the load resolution (B, 3, H, W).

        Returns:
            (dict): Batch with 'img', 'cls', per-teacher entries, and '_teacher_keys'.
        """
        imgs = batch.to(self.device, non_blocking=True)
        # Per-batch gate (not an on_train_epoch_start hook like close_mosaic): the tail only swaps the
        # interpolation target, no dataloader rebuild, so reading live self.epoch here also makes a mid-tail
        # resume re-enter the tail regime for free.
        if self._hires_res and self.epoch >= self.epochs - self._hires_tail_epochs:
            student_size = self._hires_res
        elif self._student_scales:
            student_size = self._student_scales[self._student_scale_step % len(self._student_scales)]
            self._student_scale_step += 1
        else:
            student_size = self.args.imgsz
        student_imgs = (
            torch.nn.functional.interpolate(imgs, size=student_size, mode="bilinear", antialias=True)
            if student_size != self._load_imgsz
            else imgs
        )
        teacher_imgs = (
            torch.nn.functional.interpolate(imgs, size=self._teacher_imgsz, mode="bilinear", antialias=True)
            if self._teacher_imgsz != self._load_imgsz
            else imgs
        )

        result = {
            "img": student_imgs,
            "cls": torch.zeros(imgs.shape[0], dtype=torch.long, device=self.device),
            "_teacher_keys": self._safe_keys,
        }

        for sk in self._safe_keys:
            out = self.teacher_models[sk].encode(teacher_imgs)
            result[sk] = {"cls": out.cls, "patches": out.patches}

        return result

    def get_validator(self):
        """Return ImageEncoderValidator for loss-only validation."""
        from ultralytics.models.yolo.classify.val_image_encoder import ImageEncoderValidator

        dp = getattr(self.model, "distill_path", "adaptor")
        sub = ("cls_cos", "patch_cos", "patch_l1") if dp != "feat_map" else ("feat_p3", "feat_p4", "feat_p5")
        self.loss_names = []
        for sk in self._safe_keys:
            self.loss_names.extend([f"{sk}/{s}" for s in sub])
        self.loss_names.extend(list(sub))

        # Define epoch-based x-axis so aggregate metrics align across backfilled and new runs
        try:
            import wandb

            if wandb.run:
                for prefix in ("train", "val"):
                    for s in sub:
                        wandb.define_metric(f"{prefix}/{s}", step_metric="epoch")
        except ImportError:
            pass

        validator = ImageEncoderValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        validator.teacher_models = self.teacher_models
        validator._teacher_imgsz = self._teacher_imgsz  # val loads at _load_imgsz, teacher needs its native res
        return validator

    def validate(self):
        """Run validation, then kNN eval if configured."""
        metrics, fitness = super().validate()
        if metrics is not None and "path" in self._knn_state:
            knn_top1 = self._knn_eval()
            if knn_top1 is not None:
                metrics["knn/top1"] = round(knn_top1, 4)
        return metrics, fitness

    def _knn_eval(self, every_n=5):
        """Run kNN accuracy eval on ImageNet (k=20, T=0.07). Skips non-Nth epochs."""
        epoch = self.epoch + 1
        if epoch % every_n != 0 and epoch < self.epochs:
            return None
        from ultralytics.utils.knn_eval import extract_features, knn_accuracy

        # Build dataloaders on first call, cache for reuse
        if "train_loader" not in self._knn_state:
            from types import SimpleNamespace

            from ultralytics.data import ClassificationDataset
            from ultralytics.data.build import build_dataloader

            root = self._knn_state["path"]
            args = SimpleNamespace(
                imgsz=224,
                cache=False,
                fraction=1.0,
                auto_augment="",
                erasing=0.0,
                crop_fraction=1.0,
                scale=0.92,
                fliplr=0.5,
                flipud=0.0,
                hsv_h=0.015,
                hsv_s=0.4,
                hsv_v=0.4,
            )
            train_ds = ClassificationDataset(str(root / "train"), args=args, augment=False, prefix="knn-train")
            val_ds = ClassificationDataset(str(root / "val"), args=args, augment=False, prefix="knn-val")
            self._knn_state["train_loader"] = build_dataloader(train_ds, 256, 8, shuffle=False, rank=-1)
            self._knn_state["val_loader"] = build_dataloader(val_ds, 256, 8, shuffle=False, rank=-1)
            self._knn_state["num_classes"] = len(train_ds.base.classes)

        model = self.ema.ema if self.ema else self.model
        LOGGER.info(f"kNN eval: epoch {epoch}, extracting features...")
        train_feats, train_labels = extract_features(model, self._knn_state["train_loader"], self.device)
        val_feats, val_labels = extract_features(model, self._knn_state["val_loader"], self.device)
        top1 = knn_accuracy(
            train_feats,
            train_labels,
            val_feats,
            val_labels,
            k=20,
            temp=0.07,
            num_classes=self._knn_state["num_classes"],
            device=self.device,
        )
        LOGGER.info(f"kNN eval: top-1 = {top1:.2f}% (epoch {epoch})")
        return top1

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return labeled loss items for WandB logging.

        Args:
            loss_items (torch.Tensor, optional): Loss items tensor.
            prefix (str, optional): Prefix for loss names.

        Returns:
            (dict | list): Labeled loss dict or list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(x), 5) for x in loss_items]
        # Append teacher-averaged aggregates (cls_cos, patch_cos, patch_l1)
        n = len(self._safe_keys)
        for i in range(3):
            loss_items.append(round(sum(loss_items[j * 3 + i] for j in range(n)) / n, 5))
        result = dict(zip(keys, loss_items))
        result["epoch"] = self.epoch + 1
        return result

    def plot_training_samples(self, batch, ni):
        """Skip training sample plotting for distillation (no class labels)."""
        pass

    def final_eval(self):
        """Rewrite the finished checkpoints in place as portable stock ClassificationModel weights (rank 0).

        Distillation has no accuracy metric to validate, so this replaces the base strip+val step. The in-memory model
        is an ImageEncoderModel (a ClassificationModel subclass carrying distillation-only adaptor heads), so its pickled
        form references this fork module and will not load on branches that lack it. export_backbone rebuilds a plain
        ClassificationModel from the same yaml, transfers the backbone by name, drops the heads, and reuses
        strip_optimizer for a canonical FP16 checkpoint. Runs never resume from a finished checkpoint, so overwriting in
        place is safe. Unlike the base method it does not propagate last.pt's train_results into best.pt; distillation
        metrics are logged to wandb.
        """
        if RANK in {-1, 0}:
            from ultralytics.nn.image_encoder import export_backbone

            for f in (self.best, self.last):
                if f.exists():
                    export_backbone(f, out_path=f)

    def plot_metrics(self):
        """Skip metric plotting for distillation (non-standard loss columns; use WandB)."""
