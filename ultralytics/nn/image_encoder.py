# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Image encoder model and loss for universal encoder distillation.

Distill one or more frozen teachers (EUPE, DINOv3, SAM3, SigLIP2) into a YOLO backbone using
CLS + patch token feature matching. Supports single-teacher and multi-teacher distillation.

Loss formulation follows EUPE (arXiv:2603.22387) Eq.5-6 and AM-RADIO (arXiv:2312.06709) Eq.2-3:
  Per teacher: L_cls = cosine(student, teacher), L_patch = 0.9*cosine + 0.1*smooth_L1
  Multi-teacher: L = sum_i (L_cls_i + L_patch_i) -- EUPE Eq.6, equal weighting
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import ClassificationModel
from ultralytics.nn.teacher_model import safe_key


def _make_adaptor(in_dim, out_dim, hidden_dim=None, arch="mlp"):
    """Create an adaptor head mapping student features to teacher embed_dim.

    Two variants:
        - "mlp" (EUPE Section 4.1): 2-layer "linear-LN-GELU-linear" without bias. Default hidden=in_dim; EUPE uses 3072
          for 86M+ students. RADIO v1 uses ReLU (RADIO/radio/adaptor_mlp.py:27); we follow EUPE's GELU.
        - "linear" (EdgeCrafter 2603.18739 Section 3.3): single token-wise Linear. Paper argues a minimal adapter keeps
          the representational burden on the backbone rather than letting a high-capacity projection absorb the
          mismatch.

    Args:
        in_dim (int): Input feature dimension from backbone.
        out_dim (int): Output dimension matching teacher embed_dim.
        hidden_dim (int, optional): MLP hidden dimension (ignored when arch="linear"). Defaults to in_dim.
        arch (str): "mlp" | "linear".
    """
    if arch == "linear":
        return nn.Linear(in_dim, out_dim, bias=False)
    h = hidden_dim if hidden_dim is not None else in_dim
    return nn.Sequential(
        nn.Linear(in_dim, h, bias=False),
        nn.LayerNorm(h),
        nn.GELU(),
        nn.Linear(h, out_dim, bias=False),
    )


class ImageEncoderLoss:
    """Multi-teacher CLS + patch token distillation loss for encoder pretraining.

    Per-teacher loss (EUPE Eq.5): L_cls_i = 1 - cos_sim(student_cls, teacher_cls) L_patch_i = alpha * (1 - cos_sim(s,
    t)) + beta * smooth_L1(s, t) Multi-teacher total (EUPE Eq.6): L = sum_i (L_cls_i + L_patch_i) -- equal weighting, no
    teacher dropping

    Alpha=0.9, beta=0.1 per EUPE Eq.5. AM-RADIO (arXiv:2312.06709, Section 3.3) states "to mostly rely on the
    empirically better cosine distance, but also match vector magnitudes".

    Skip L_cls for patches-only teachers (SAM3, ConvNeXt) per DUNE token_types convention (verified:
    dune/teachers/config.py:25,36).

    Attributes:
        cos_weight (float): Alpha in EUPE Eq.5.
        l1_weight (float): Beta in EUPE Eq.5.
    """

    def __init__(
        self,
        cos_weight=0.9,
        l1_weight=0.1,
        cls_l1=False,
        distill_path="adaptor",
        loss_type="cos_l1",
        gram_weight=0.0,
    ):
        """Initialize ImageEncoderLoss.

        Args:
            cos_weight (float): Weight for cosine similarity (both CLS and patch).
            l1_weight (float): Weight for smooth L1 (patch always, CLS if cls_l1=True).
            cls_l1 (bool): Add smooth L1 to CLS loss. Default False matches EUPE Eq.5.
            UNIC/DUNE use True with 0.5/0.5 weights (`unic/modeling/losses.py: 54`).
            distill_path (str): "adaptor" (cos+L1 on final-stage CLS+patch through adaptor MLP) or "feat_map"
                (unweighted MSE across per-scale student feature maps vs bilinearly-resized teacher final-block tokens,
                EdgeCrafter 2603.18739 Eq. L_distill = sum_l ||phi(X_L^S) - X_l^T||^2).
            loss_type (str): "cos_l1" (default, EUPE Eq.5 patch loss 0.9*cos + 0.1*smooth_L1) or "l2" (pure MSE on
                un-normalized patch features, no cosine). CLS loss stays cosine in both. Tests Mengyu's L2-wins evidence
                (same-arch YOLO det features) in our cross-arch ViT to YOLO conv regime. The third logged loss item
                (wandb "patch_l1") holds smooth_L1 under cos_l1 and MSE under l2.
            gram_weight (float): Weight for DINOv3-style Gram loss on patch-token similarities. Disabled when zero.
        """
        self.cos_weight = cos_weight
        self.l1_weight = l1_weight
        self.cls_l1 = cls_l1
        self.distill_path = distill_path
        self.loss_type = loss_type
        self.gram_weight = gram_weight
        if self.gram_weight > 0 and self.distill_path == "feat_map":
            raise ValueError("Gram loss requires distill_path='adaptor' patch tokens")

    def _teacher_loss(self, s_cls, s_patch, t_cls, t_patch):
        """Compute loss for a single teacher (EUPE Eq.5).

        Args:
            s_cls (torch.Tensor): Student CLS features (B, D).
            s_patch (torch.Tensor): Student patch features (B, N, D).
            t_cls (torch.Tensor | None): Teacher CLS features or None for patches-only.
            t_patch (torch.Tensor): Teacher patch features (B, N, D), pre-aligned to the student grid by forward().

        Returns:
            (tuple): (teacher_loss, [cls_cos, patch_cos, patch_term]).
        """
        # Force fp32 for loss: fp16 cosine_similarity eps=1e-8 rounds to 0, causing nan on
        # near-zero features (random init). Follows DUNE (dune/model/losses.py:58).
        with torch.autocast(device_type=s_cls.device.type, dtype=torch.float32, enabled=s_cls.is_cuda):
            # CLS loss (skip for patches-only teachers)
            if t_cls is not None:
                t_cls_ = t_cls.to(s_cls)
                cls_cos = 1.0 - F.cosine_similarity(s_cls, t_cls_, dim=-1).mean()
                cls_l1 = F.smooth_l1_loss(s_cls, t_cls_) if self.cls_l1 else torch.tensor(0.0, device=s_cls.device)
            else:
                cls_cos = torch.tensor(0.0, device=s_cls.device)
                cls_l1 = torch.tensor(0.0, device=s_cls.device)
            # patch_cos stays outside the branch so it is logged (train/val metric) even in l2 mode where the
            # optimized patch term is MSE instead. See loss_type docstring for the cos_l1 vs l2 formulas.
            patch_cos = 1.0 - F.cosine_similarity(s_patch, t_patch, dim=-1).mean()
            if self.loss_type == "l2":
                patch_term = F.mse_loss(s_patch, t_patch)
                patch_loss = patch_term
            else:
                patch_term = F.smooth_l1_loss(s_patch, t_patch)
                patch_loss = self.cos_weight * patch_cos + self.l1_weight * patch_term
            # cls_l1=False: cls_cos + patch_loss. cls_l1=True (UNIC): cos_w*cls_cos + l1_w*cls_l1 + patch_loss
            cls_cos_w = self.cos_weight if self.cls_l1 else 1.0
            loss = cls_cos_w * cls_cos + self.l1_weight * cls_l1 + patch_loss

        return loss, [cls_cos.detach(), patch_cos.detach(), patch_term.detach()]

    @staticmethod
    def _align_patch_tokens(s_patch, t_patch):
        """Resize teacher patch tokens to the student patch grid."""
        # Spatial alignment: if student and teacher have different patch counts, interpolate teacher
        # to match student grid. EUPE Section 3.1 upsamples to max(N_S, N_T); for teachers with
        # very large grids (SAM3: 5184 patches), we downsample teacher to student grid instead.
        # TODO: for SAM3, use pixel-shuffle upsampling in the adaptor MLP instead of downsampling
        # teacher patches. C-RADIOv4 (RADIO/radio/adaptor_mlp.py:99-107) uses einops.rearrange
        # with upsample_factor to produce higher-res student patches matching SAM's dense grid.
        t_patch = t_patch.to(s_patch)
        if t_patch.shape[1] != s_patch.shape[1]:
            h = int(s_patch.shape[1] ** 0.5)
            th = int(t_patch.shape[1] ** 0.5)
            t_patch = t_patch.transpose(1, 2).reshape(t_patch.shape[0], t_patch.shape[2], th, th)
            t_patch = F.interpolate(t_patch, size=(h, h), mode="bilinear", antialias=True)
            t_patch = t_patch.flatten(2).transpose(1, 2)
        return t_patch

    def _gram_loss(self, s_patch, t_patch):
        """Compute the DINOv3 Gram loss over the local batch via the factored covariance form.

        Uses ||X X^T||_F^2 = ||X^T X||_F^2 to avoid building the (B*N, B*N) token similarity matrix,
        so the cost scales with the feature dim D, not the token count.

        Args:
            s_patch (torch.Tensor): Student patch features (B, N, D).
            t_patch (torch.Tensor): Teacher patch features (B, N, D).

        Returns:
            (torch.Tensor): Scalar Gram loss.
        """
        s_patch = F.normalize(s_patch.float(), dim=-1).flatten(0, 1)
        t_patch = F.normalize(t_patch.float(), dim=-1).flatten(0, 1)
        s_cov = s_patch.T @ s_patch
        t_cov = t_patch.T @ t_patch
        st_cov = s_patch.T @ t_patch
        denom = s_patch.shape[0] ** 2
        return (s_cov.square().sum() + t_cov.square().sum() - 2 * st_cov.square().sum()) / denom

    def _teacher_loss_feat_map(self, s_scales, t_patch):
        """Per-scale MSE between student feat maps and bilinearly-resized teacher tokens.

        EdgeCrafter Eq: `L_distill = sum_l ||phi(X_L^S) - X_l^T||^2_2`. Unweighted sum across scales; no cosine, no L1.

        Args:
            s_scales (list[torch.Tensor]): Student feature maps per scale, each shape (B, D_teacher, H_s, W_s). Adaptor
                (1x1 Conv) already applied upstream.
            t_patch (torch.Tensor): Teacher patch tokens (B, N, D_teacher).

        Returns:
            (tuple): (teacher_loss, [mse_scale0, mse_scale1, mse_scale2]).
        """
        b, n, d = t_patch.shape
        grid = int(n**0.5)
        t_grid = t_patch.transpose(1, 2).reshape(b, d, grid, grid).to(s_scales[0])
        items = []
        loss = torch.tensor(0.0, device=s_scales[0].device)
        # Force fp32 for numerical parity with the adaptor path.
        with torch.autocast(device_type=s_scales[0].device.type, dtype=torch.float32, enabled=s_scales[0].is_cuda):
            for s in s_scales:
                h, w = s.shape[-2:]
                t = (
                    F.interpolate(t_grid, size=(h, w), mode="bilinear", antialias=True)
                    if (h, w) != (grid, grid)
                    else t_grid
                )
                mse = F.mse_loss(s, t)
                loss = loss + mse
                items.append(mse.detach())
        return loss, items

    def __call__(self, preds, batch):
        """Compute multi-teacher distillation loss (EUPE Eq.6: sum over teachers).

        Args:
            preds (dict): distill_path="adaptor": {teacher_key: (student_cls, student_patches)}.
            distill_path="feat_map":
            {teacher_key: [student_scale0, student_scale1, student_scale2]} with each (B, D_teacher, H, W).
            batch (dict): {teacher_key: {"cls": Tensor|None, "patches": Tensor}} per teacher. Must also contain
                "_teacher_keys" listing the active teacher keys.

        Returns:
            (tuple): (total_loss, stacked loss_items for all teachers).
        """
        teacher_keys = batch["_teacher_keys"]
        first = next(iter(preds.values()))
        dev = first[0].device
        total_loss = torch.tensor(0.0, device=dev)
        all_items = []

        for key in teacher_keys:
            if self.distill_path == "feat_map":
                t_patch = batch[key]["patches"]
                loss_i, items_i = self._teacher_loss_feat_map(preds[key], t_patch)
            else:
                s_cls, s_patch = preds[key]
                t_cls = batch[key]["cls"]
                t_patch = self._align_patch_tokens(s_patch, batch[key]["patches"])
                loss_i, items_i = self._teacher_loss(s_cls, s_patch, t_cls, t_patch)
                if self.gram_weight > 0:
                    gram_loss = self._gram_loss(s_patch, t_patch)
                    loss_i = loss_i + self.gram_weight * gram_loss
                    items_i.append(gram_loss.detach())
            total_loss = total_loss + loss_i
            all_items.extend(items_i)

        return total_loss, torch.stack(all_items)


class ImageEncoderModel(ClassificationModel):
    """YOLO backbone with per-teacher CLS and patch adaptor heads for encoder distillation.

    Architecture follows EUPE (arXiv:2603.22387) ConvNeXt student recipe:
    - CLS via global avg pool (verified eupe/models/convnext.py:220)
    - Patches via bilinear upsample to teacher grid (verified eupe/models/convnext.py:256)
    - Shared LayerNorm on concatenated [CLS; patches] before splitting (verified eupe/models/convnext.py:224)
    - Per-teacher 2-layer MLP adaptor heads for CLS and patches (AM-RADIO pattern: head_mlp + feat_mlp,
    verified RADIO/radio/adaptor_generic.py; EUPE Section 4.1 MLP spec)

    Multi-teacher: each teacher gets its own adaptor pair with its own embed_dim and grid size. Single-teacher mode is
    the special case with one entry in the adaptors dict.

    Attributes:
        token_norm (nn.LayerNorm): Shared norm for [CLS; patches] (EUPE ConvNeXt pattern).
        adaptors (nn.ModuleDict): Per-teacher adaptor heads {safe_name: ModuleDict{"cls": MLP, "patch": MLP}}.
    """

    # P3/P4/P5 on the cls yaml (strides 8/16/32); layers 0-8 are what cls->det intersect_dicts transfers.
    FEAT_MAP_TAPS = (3, 5, 8)

    def __init__(
        self,
        cfg="yolo26s-cls.yaml",
        ch=3,
        nc=1000,
        verbose=True,
        teachers=None,
        proj_hidden_dim=None,
        loss_cfg=None,
        distill_path="adaptor",
        adaptor_arch="mlp",
    ):
        """Initialize ImageEncoderModel with per-teacher adaptor heads.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int): Number of classes (unused during distillation).
            verbose (bool): Whether to display model information.
            teachers (dict): Per-teacher config. Keys are teacher names (e.g. "eupe:vitb16"), values are dicts with
                "embed_dim", "num_patches", "token_types". If None, defaults to a single EUPE-ViT-B teacher for
                backward compat.
            proj_hidden_dim (int, optional): Adaptor MLP hidden dimension. None = use backbone dim (c_). EUPE uses 3072
                for 86M+ students.
            loss_cfg (dict, optional): Loss config with keys cos_weight, l1_weight, cls_l1, loss_type, distill_path.
                None = EUPE defaults.
            distill_path (str): "adaptor" (default, cos+L1 on final-stage CLS+patch through adaptor MLP) or "feat_map"
                (EdgeCrafter-style MSE at student L3/L5/L8 vs teacher final-block tokens bilinearly resized per scale;
                uses 1x1 Conv2d adaptors).
            adaptor_arch (str): "mlp" (default, 2-layer EUPE) or "linear" (single token-wise Linear, EdgeCrafter Section
                3.3). Applies to the "adaptor" path only; "feat_map" path forces a 1x1 Conv2d per scale.
        """
        super().__init__(cfg, ch, nc, verbose)
        self._loss_cfg = loss_cfg or {}
        self._loss_cfg.setdefault("distill_path", distill_path)
        self.distill_path = distill_path
        if teachers is None:
            teachers = {"eupe:vitb16": {"embed_dim": 768, "num_patches": 256, "token_types": ("cls", "patches")}}

        c_ = self.model[-1].linear.in_features
        self.token_norm = nn.LayerNorm(c_)

        self.adaptors = nn.ModuleDict()
        for name, tcfg in teachers.items():
            safe = safe_key(name)
            heads = nn.ModuleDict()
            if "cls" in tcfg["token_types"]:
                heads["cls"] = _make_adaptor(c_, tcfg["embed_dim"], proj_hidden_dim, arch=adaptor_arch)
            heads["patch"] = _make_adaptor(c_, tcfg["embed_dim"], proj_hidden_dim, arch=adaptor_arch)
            self.adaptors[safe] = heads

        if distill_path == "feat_map":
            tap_channels = self._infer_tap_channels(ch)
            self.feat_adaptors = nn.ModuleDict()
            for name, tcfg in teachers.items():
                safe = safe_key(name)
                per_scale = nn.ModuleList(
                    [nn.Conv2d(c, tcfg["embed_dim"], kernel_size=1, bias=False) for c in tap_channels]
                )
                self.feat_adaptors[safe] = per_scale

    def _infer_tap_channels(self, ch):
        """Dummy-forward through backbone to read channel counts at FEAT_MAP_TAPS."""
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, ch, 224, 224)
            tap_channels = {}
            for i, m in enumerate(self.model[:-1]):
                x = m(x)
                if i in self.FEAT_MAP_TAPS:
                    tap_channels[i] = x.shape[1]
        self.train()
        return [tap_channels[i] for i in self.FEAT_MAP_TAPS]

    def loss(self, batch, preds=None):
        """Compute multi-teacher distillation loss from backbone features.

        Args:
            batch (dict): Batch with 'img' and per-teacher entries {key: {"cls": T|None, "patches": T}}.
            preds: Unused (computed internally).

        Returns:
            (tuple): (loss, loss_items) from ImageEncoderLoss.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        x = batch["img"]

        # EdgeCrafter-style feat-map path: tap cls-phase transferable layers (P3/P4/P5), project per-teacher per-scale
        # through 1x1 Conv2d, and compute MSE vs bilinearly-resized teacher tokens. Exits early without running the
        # final Classify head (not needed for MSE-on-feature-maps).
        if self.distill_path == "feat_map":
            taps = []
            for i, m in enumerate(self.model[:-1]):
                x = m(x)
                if i in self.FEAT_MAP_TAPS:
                    taps.append(x)
            teacher_preds = {}
            for key in self.feat_adaptors:
                per_scale = self.feat_adaptors[key]
                teacher_preds[key] = [proj(t) for proj, t in zip(per_scale, taps)]
            return self.criterion(teacher_preds, batch)

        for m in self.model[:-1]:
            x = m(x)
        head = self.model[-1]
        features = head.conv(x)  # (B, 1280, H, W) shared features

        # CLS via global avg pool (EUPE ConvNeXt: x_pool = x.mean([-2, -1]))
        cls_feats = head.pool(features).flatten(1)  # (B, 1280)

        # Per-teacher: interpolate to the live teacher grid, normalize, project through adaptors.
        # EUPE convnext.py:224 does norm(cat([cls, patches])), but LayerNorm(c_) normalizes
        # over channel dim independently per position, so separate application is equivalent.
        cls_normed = self.token_norm(cls_feats)
        teacher_preds = {}
        for key in self.adaptors:
            n = batch[key]["patches"].shape[1]
            h = int(n**0.5)
            if h * h != n:
                raise ValueError(f"Teacher patches for {key} must form a square grid, got {n} tokens")
            # Patches via bilinear upsample (EUPE convnext.py:256, bilinear+antialias)
            patch_feats = F.interpolate(features, size=(h, h), mode="bilinear", antialias=True)
            patch_feats = patch_feats.flatten(2).transpose(1, 2)  # (B, N, 1280)
            patch_normed = self.token_norm(patch_feats)

            heads = self.adaptors[key]
            s_cls = heads["cls"](cls_normed) if "cls" in heads else cls_normed
            s_patch = heads["patch"](patch_normed)
            teacher_preds[key] = (s_cls, s_patch)

        result = self.criterion(teacher_preds, batch)

        # Nan instrumentation: log diagnostic info on first nan occurrence per training run
        if result[1].isnan().any() and not getattr(self, "_nan_logged", False):
            self._nan_logged = True
            import logging

            log = logging.getLogger("ultralytics")
            log.warning("NAN DETECTED in loss items. Diagnostic dump:")
            log.warning(f"  loss_items: {result[1].tolist()}")
            log.warning(
                f"  features: nan={features.isnan().any()}, inf={features.isinf().any()}, "
                f"max={features.abs().max():.4f}, dtype={features.dtype}"
            )
            log.warning(f"  cls_normed: nan={cls_normed.isnan().any()}, max={cls_normed.abs().max():.4f}")
            for key in self.adaptors:
                s_cls, s_patch = teacher_preds[key]
                t_data = batch[key]
                log.warning(
                    f"  {key}/s_cls: nan={s_cls.isnan().any()}, inf={s_cls.isinf().any()}, "
                    f"max={s_cls.abs().max():.4f}, dtype={s_cls.dtype}"
                )
                log.warning(
                    f"  {key}/s_patch: nan={s_patch.isnan().any()}, inf={s_patch.isinf().any()}, "
                    f"max={s_patch.abs().max():.4f}"
                )
                t_cls = t_data["cls"]
                if t_cls is not None:
                    log.warning(f"  {key}/t_cls: nan={t_cls.isnan().any()}, max={t_cls.abs().max():.4f}")
                else:
                    log.warning(f"  {key}/t_cls: N/A (patches-only teacher)")
                log.warning(
                    f"  {key}/t_patch: nan={t_data['patches'].isnan().any()}, max={t_data['patches'].abs().max():.4f}"
                )
            # Check adaptor weight magnitudes
            for key in self.adaptors:
                for name, p in self.adaptors[key].named_parameters():
                    if p.isnan().any() or p.isinf().any():
                        log.warning(f"  PARAM {key}/{name}: nan={p.isnan().any()}, inf={p.isinf().any()}")
                    if p.abs().max() > 1000:
                        log.warning(f"  PARAM {key}/{name}: max={p.abs().max():.1f} (large)")

        return result

    def init_criterion(self):
        """Initialize distillation loss."""
        return ImageEncoderLoss(**self._loss_cfg)


def export_backbone(ckpt_path, out_path=None):
    """Export a portable stock-classifier checkpoint from a phase1 distillation checkpoint.

    Phase1 saves the full ImageEncoderModel (a ClassificationModel subclass carrying distillation-only adaptor heads),
    so its pickled top object references this fork module and will not unpickle on branches that lack it. This rebuilds
    a plain ClassificationModel from the same yaml, transfers the backbone weights by name (BaseModel.load ->
    intersect_dicts, shape-safe), and reuses strip_optimizer to emit a canonical FP16 checkpoint that stock Ultralytics
    loads anywhere. Returns None as a no-op when the checkpoint is already a stock model (idempotent on re-runs).

    Args:
        ckpt_path (str | Path): Phase1 checkpoint (best.pt or last.pt).
        out_path (str | Path, optional): Output path. Defaults to '<ckpt_dir>/<stem>_backbone.pt'.

    Returns:
        (Path | None): The portable checkpoint path, or None if the checkpoint was already stock.
    """
    from ultralytics.utils.patches import torch_load
    from ultralytics.utils.torch_utils import strip_optimizer

    ckpt_path = Path(ckpt_path)
    ckpt = torch_load(ckpt_path, map_location="cpu")
    enc = ckpt.get("ema") or ckpt["model"]
    if not isinstance(enc, ImageEncoderModel):  # already a stock checkpoint, nothing to convert
        return None
    out_path = out_path or ckpt_path.with_name(f"{ckpt_path.stem}_backbone.pt")
    model = ClassificationModel(cfg=enc.yaml, verbose=False)
    model.load(enc)  # intersect_dicts transfers the backbone by name and drops the distill heads
    model.half()  # emit FP16 directly so the intermediate save is not a full FP32 copy
    torch.save({"model": model}, out_path)
    strip_optimizer(out_path)  # reuse the Ultralytics cleaner: freeze, metadata, drop optimizer
    return out_path


if __name__ == "__main__":
    import sys

    # Backfill: convert existing phase1 checkpoints (saved before the in-place final_eval) to stock in place.
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/shared-datasets/fatih-runs/classify/yolo-next-encoder")
    for f in sorted(root.glob("phase1-*/weights/best.pt")) + sorted(root.glob("phase1-*/weights/last.pt")):
        if export_backbone(f, out_path=f):
            print("converted", f)
