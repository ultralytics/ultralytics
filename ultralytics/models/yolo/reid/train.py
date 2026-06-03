# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.data import build_reid_dataloader
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import ReidModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first
from ..classify.train import ClassificationTrainer


def _extract_clip_visual_sd(weights):
    """Return CLIP visual-tower state_dict (without `visual.` prefix) if `weights` is a CLIP
    checkpoint, else None.

    Safe-load only: uses ``torch.jit.load`` (TorchScript) or ``torch.load(weights_only=True)``
    — never executes arbitrary pickle reduce ops in the checkpoint. A non-CLIP yolo .pt
    (which contains pickled Python objects) fails the weights_only=True load cleanly and
    returns None so the caller can fall through to the standard YOLO loader.

    Accepts:
      - str path: TorchScript or weights-only .pt (OpenAI CLIP layout)
      - torch.jit.ScriptModule / RecursiveScriptModule (already-loaded TorchScript)
      - dict / state_dict with ``visual.*`` keys
    """
    sd = None
    # Already-loaded TorchScript module (RecursiveScriptModule today; ScriptModule covers
    # freeze/optimize-for-inference outputs and any future rename).
    if hasattr(weights, "state_dict") and "Script" in type(weights).__name__:
        sd = weights.state_dict()
    elif isinstance(weights, str):
        # Try TorchScript first (OpenAI CLIP TorchScript .pt). On failure, fall back to a
        # safe weights_only=True load — that path rejects non-CLIP yolo .pt files cleanly.
        try:
            m = torch.jit.load(weights, map_location="cpu")
            sd = m.state_dict()
        except Exception:
            try:
                sd = torch.load(weights, map_location="cpu", weights_only=True)
            except Exception:
                return None
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
    elif isinstance(weights, dict):
        sd = weights
    if not isinstance(sd, dict):
        return None
    if "visual.conv1.weight" in sd and "visual.class_embedding" in sd:
        return {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
    return None


def _inject_clip_visual_sd(model, visual_sd):
    """Load an already-extracted CLIP visual state_dict into a model's ViTBackbone (in place)."""
    from ultralytics.nn.modules.vit import ViTBackbone, load_clip_visual_into_sd
    from ultralytics.utils import LOGGER

    for m in model.modules():
        if isinstance(m, ViTBackbone):
            info = load_clip_visual_into_sd(m, visual_sd, strict=False)
            LOGGER.info(
                f"CLIP ViT visual weights loaded: {info['loaded']} keys; "
                f"missing={len(info.get('missing', []))}, unexpected={len(info.get('unexpected', []))}"
            )
            return True
    return False


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

    def get_model(self, cfg=None, weights=None, verbose: bool = True, visual_sd: dict | None = None):
        """Return a ReidModel configured for training.

        Args:
            cfg: Model configuration.
            weights: Pre-trained weights (path, dict, or pre-loaded TorchScript module).
            verbose (bool): Whether to display model info.
            visual_sd (dict, optional): Pre-extracted CLIP visual state_dict. When provided
                (typically by ``setup_model``), the CLIP-detection step is skipped so the
                checkpoint is not loaded twice.

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
        if visual_sd is not None:
            _inject_clip_visual_sd(model, visual_sd)
        elif weights:
            # Caller passed weights without pre-extracting CLIP — detect and dispatch here.
            sd = _extract_clip_visual_sd(weights)
            if sd is not None:
                _inject_clip_visual_sd(model, sd)
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

        Detect CLIP-style pretrained weights eagerly by inspecting the state_dict (looking
        for ``visual.*`` keys), not by sniffing the filename — a checkpoint named
        ``my_run.pt`` may still be CLIP, and one named ``vit_other.pt`` may not. The check
        uses a safe-load path (TorchScript or ``weights_only=True``), so it does not execute
        arbitrary pickle reduce ops. When CLIP is detected, build the model from cfg and
        inject the already-extracted visual state_dict via ``get_model(visual_sd=...)`` so
        the checkpoint is not loaded a second time. Otherwise fall through to the parent
        loader which handles standard YOLO ``.pt`` checkpoints.

        Accepts ``self.args.pretrained`` as: a str path, a pre-loaded TorchScript module,
        or a state_dict dict — the helper detects each.
        """
        pretrained = getattr(self.args, "pretrained", None)
        visual_sd = _extract_clip_visual_sd(pretrained) if pretrained else None
        if visual_sd is not None:
            cfg_path = self.model if isinstance(self.model, str) else getattr(self.args, "model", None)
            # Pass visual_sd directly to skip re-extraction inside get_model.
            self.model = self.get_model(cfg=cfg_path, weights=None, verbose=RANK == -1, visual_sd=visual_sd)
            ReidModel.reshape_outputs(self.model, self.data["nc"])
            return None
        ckpt = super().setup_model()
        ReidModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Create a ReidDataset instance via the centralised build_yolo_dataset() entry point.

        Args:
            img_path (str): Path to dataset split.
            mode (str): 'train', 'val', 'test', or 'gallery'.
            batch (int, optional): Batch size (unused for ReID — kept for parent signature parity).

        Returns:
            (ReidDataset): Dataset for the specified split.
        """
        return build_yolo_dataset(self.args, img_path, batch or 0, self.data, mode=mode)

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

    def plot_training_samples(self, batch, ni):
        """Plotting training samples is a no-op for ReID — pid integers (often hundreds in
        Market-1501) are not human-meaningful class names, so a mosaic of pid-labelled crops
        adds visual noise without conveying anything useful. ``ReidValidator.plot_predictions``
        is similarly a no-op."""
        pass

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
