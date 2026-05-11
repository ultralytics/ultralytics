# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.data import ReidDataset, build_reid_dataloader
from ultralytics.models import yolo
from ultralytics.nn.tasks import ReidModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first
from ..classify.train import ClassificationTrainer


def _extract_clip_visual_sd(weights):
    """Return CLIP visual-tower state_dict (without `visual.` prefix) if `weights` is a
    CLIP checkpoint, else None.

    Accepts:
      - str path to TorchScript .pt (OpenAI CLIP)
      - torch.jit.RecursiveScriptModule (already-loaded TorchScript)
      - dict / state_dict with `visual.*` keys
    """
    sd = None
    try:
        # Already a loaded torchscript module
        if hasattr(weights, "state_dict") and "RecursiveScript" in type(weights).__name__:
            sd = weights.state_dict()
        elif isinstance(weights, str):
            # Try TorchScript load first; fall back to torch.load
            try:
                m = torch.jit.load(weights, map_location="cpu")
                sd = m.state_dict()
            except Exception:
                sd = torch.load(weights, map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
        elif isinstance(weights, dict):
            sd = weights
        if not isinstance(sd, dict):
            return None
        # Need at least visual.conv1.weight and visual.class_embedding to call this CLIP.
        if "visual.conv1.weight" in sd and "visual.class_embedding" in sd:
            return {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
    except Exception:
        return None
    return None


class ReidTrainer(ClassificationTrainer):
    """Trainer for person re-identification models.

    Extends BaseTrainer with ReID-specific dataset handling (Market-1501), PK batch sampling,
    and multi-loss training (cross-entropy + triplet).

    Attributes:
        model (ReidModel): The ReID model to be trained.
        data (dict): Dataset information including identity names and count.
        loss_names (list[str]): Names of loss components: ['ce_loss', 'tri_loss'].

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidTrainer
        >>> args = dict(model="yolo26n-reid.yaml", data="Market-1501.yaml", epochs=60)
        >>> trainer = ReidTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize ReidTrainer.

        Args:
            cfg (dict): Default configuration dictionary.
            overrides (dict, optional): Parameter overrides.
            _callbacks (list, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "reid"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 256
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """Set the model's identity names and configure loss from trainer args."""
        super().set_model_attributes()
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a ReidModel configured for training.

        Args:
            cfg: Model configuration.
            weights: Pre-trained weights.
            verbose (bool): Whether to display model info.

        Returns:
            (ReidModel): Configured model.
        """
        # Forward ReID loss hparams so ReidModel.init_criterion works without the trainer side-effect.
        reid_kwargs = {
            k: getattr(self.args, k)
            for k in (
                "triplet_margin",
                "label_smoothing",
                "triplet_weight",
                "ce_weight",
                "center_weight",
                "center_momentum",
                "focal_gamma",
                "supcon_temp",
                "arcface",
                "arcface_margin",
                "arcface_scale",
                "gem_p",
                "nonlocal_block",
            )
            if hasattr(self.args, k)
        }
        model = ReidModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and RANK == -1,
            **reid_kwargs,
        )
        if weights:
            # Detect a CLIP-style TorchScript or weight-only checkpoint with a `visual.*` prefix
            # (e.g. OpenAI CLIP ViT-B/16) and route through the dedicated loader. Fall back to
            # standard Ultralytics state_dict loading for everything else.
            visual_sd = _extract_clip_visual_sd(weights)
            if visual_sd is not None:
                from ultralytics.nn.modules.vit import ViTBackbone, load_clip_visual_into_sd
                for m in model.modules():
                    if isinstance(m, ViTBackbone):
                        info = load_clip_visual_into_sd(m, visual_sd, strict=False)
                        from ultralytics.utils import LOGGER
                        LOGGER.info(f"CLIP ViT visual weights loaded: {info['loaded']} keys; missing={len(info.get('missing', []))}, unexpected={len(info.get('unexpected', []))}")
                        break
            else:
                model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        return model

    def setup_model(self):
        """Load or create model for ReID tasks.

        Intercept CLIP-style TorchScript weights here so the upstream loader (which expects
        a yolo-style checkpoint with `.model`) never sees them. Build the model from cfg,
        then route the CLIP path through `_extract_clip_visual_sd` + `load_clip_visual_into_sd`.
        """
        pretrained = getattr(self.args, "pretrained", None)
        is_clip_path = isinstance(pretrained, str) and pretrained.lower().endswith((".pt", ".pth")) and (
            "vit" in pretrained.lower().split("/")[-1] or "clip" in pretrained.lower().split("/")[-1]
        )
        if is_clip_path:
            # Build model from cfg only; load CLIP visual weights ourselves.
            cfg_path = self.model if isinstance(self.model, str) else getattr(self.args, "model", None)
            self.model = self.get_model(cfg=cfg_path, weights=pretrained, verbose=RANK == -1)
            ReidModel.reshape_outputs(self.model, self.data["nc"])
            return None
        ckpt = super().setup_model()
        ReidModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Create a ReidDataset instance.

        Args:
            img_path (str): Path to dataset split.
            mode (str): 'train', 'val', or 'test'.
            batch: Unused.

        Returns:
            (ReidDataset): Dataset for the specified split.
        """
        return ReidDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode, data=self.data)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return dataloader with PK sampling for training.

        Args:
            dataset_path (str): Path to dataset.
            batch_size (int): Batch size.
            rank (int): Process rank for DDP.
            mode (str): 'train' or 'val'.

        Returns:
            (DataLoader): Configured dataloader.
        """
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        if mode == "train":
            # PK sampling: P identities x K images
            # TODO
            p = getattr(self.args, "reid_p", 16)
            k = getattr(self.args, "reid_k", 4)
            loader = build_reid_dataloader(dataset, batch_size, self.args.workers, p=p, k=k, shuffle=True, rank=rank)
        else:
            from ultralytics.data import build_dataloader

            loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)

        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def get_validator(self):
        """Return a ReidValidator instance."""
        self.loss_names = ["ce_loss", "tri_loss"]
        return yolo.reid.ReidValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items.

        ReID validation uses query-gallery mAP, not loss computation, so val-prefixed
        loss items are omitted to keep results.csv and plots clean.

        Args:
            loss_items: Loss tensor items.
            prefix (str): Prefix for loss names.

        Returns:
            Loss keys or dict of loss items.
        """
        if prefix == "val":
            return [] if loss_items is None else {}
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))
