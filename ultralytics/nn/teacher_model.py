# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Frozen teacher models for universal encoder distillation.

Teacher abstraction inspired by:
- EUPE/UNIC/DUNE: forward_features() -> {"x_norm_clstoken", "x_norm_patchtokens"} dict convention
- RADIO (NVlabs/RADIO, adaptor_base.py): typed AdaptorInput/RadioOutput NamedTuples for error catching
- DUNE (naver/dune, teachers/config.py): per-teacher token_types -- SAM3 produces patches only
- MobileCLIP (ultralytics/nn/image_model.py): TorchScript .ts pattern for zero-dependency inference
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# Pipeline normalization that arrives at ``encode()``. The classification trainer at
# ``ultralytics/models/yolo/classify/train_image_encoder.py:38-39`` applies these stats via
# ``classify_augmentations_distill`` / ``classify_transforms`` before any teacher sees the tensor. ``_prep`` undoes this
# before re-normalizing with each teacher's training-time stats.
PIPELINE_IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
PIPELINE_IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def safe_key(variant: str) -> str:
    """Convert teacher variant name to safe key (nn.ModuleDict keys can't contain ':')."""
    return variant.replace(":", "_")


@dataclass
class TeacherOutput:
    """Typed output from any teacher model, used by ImageEncoderLoss.

    Typed dataclass instead of raw dict, following RADIO's AdaptorInput/RadioOutput pattern
    (RADIO/radio/adaptor_base.py) to catch mismatches at construction, not key lookup.

    Attributes:
        cls (torch.Tensor | None): CLS/summary features (B, D). None for patches-only teachers (SAM3) where
        CLS is not meaningful -- following DUNE convention (dune/teachers/config.py: 25,36).
        patches (torch.Tensor): Spatial/patch features (B, N, D). Always present.
    """

    cls: torch.Tensor | None
    patches: torch.Tensor


class TeacherModel(nn.Module):
    """Abstract base for frozen teacher models in encoder distillation.

    All subclasses produce TeacherOutput with CLS and/or patch tokens. The token_types attribute indicates which outputs
    are meaningful for loss computation -- following DUNE's per-teacher token_types config (verified:
    dune/teachers/config.py:16,25,36).

    Separate from ImageModel (image_model.py) because output contract differs: ImageModel returns a single vector;
    TeacherModel returns CLS + patch features for spatial distillation.

    Attributes:
        embed_dim (int): Teacher embedding dimension.
        num_patches (int): Number of patch tokens at default resolution.
        token_types (tuple[str, ...]): Which outputs are meaningful: ("cls", "patches") or ("patches",).
    """

    embed_dim: int = 0
    num_patches: int = 0
    token_types: tuple[str, ...] = ("cls", "patches")

    # Teacher's training-time normalization stats. Override per subclass: ImageNet for EUPE/DINOv3, SigLIP-style (0.5,
    # 0.5, 0.5) for SigLIP2/MoonViT/SAM3. Defaults match ``PIPELINE_IMAGE_MEAN/STD`` so ``_prep`` is mathematically
    # identity when un-overridden, and is bit-identity when ``_normalize_input=False`` (early return).
    IMAGE_MEAN: tuple[float, float, float] = PIPELINE_IMAGE_MEAN
    IMAGE_STD: tuple[float, float, float] = PIPELINE_IMAGE_STD

    def __init__(self):
        """Initialize the TeacherModel base class."""
        super().__init__()
        self._normalize_input = False

    def _freeze(self, cfg, device):
        """Freeze model and set teacher attributes from config dict.

        Args:
            cfg (dict): Config with 'embed_dim', 'num_patches', 'token_types' keys.
            device (torch.device, optional): Device to move model to.
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if device is not None:
            self.model = self.model.to(device)
        self.embed_dim = cfg["embed_dim"]
        self.num_patches = cfg["num_patches"]
        self.token_types = cfg["token_types"]

    def _prep(self, image: torch.Tensor) -> torch.Tensor:
        """Convert pipeline-normalized input to the teacher's training-time distribution.

        ``train_image_encoder.py:_build_transforms`` already applies ``PIPELINE_IMAGE_MEAN/STD`` (ImageNet stats) before
        this. When ``_normalize_input=True``, undo that and re-normalize with the teacher's ``IMAGE_MEAN/STD``. For
        teachers that share ImageNet stats (EUPE/DINOv3) this collapses to a no-op (correct: no work to do). For
        (0.5, 0.5, 0.5)-stat teachers (SigLIP2/MoonViT/SAM3) this converts ImageNet-normalized → SigLIP-normalized.

        When ``_normalize_input=False`` the input tensor is returned verbatim, preserving bit-identical behavior with
        the legacy pipeline (regression guard). Stats live on the class; tensors are created on the image's device per
        call to avoid buffer device-tracking (``_freeze`` moves only ``self.model``, not ``self``).

        Args:
            image (torch.Tensor): Pipeline-normalized image tensor (B, 3, H, W), in ImageNet stats.

        Returns:
            (torch.Tensor): Re-normalized tensor (or the input unchanged when ``_normalize_input=False``).
        """
        if not self._normalize_input:
            return image
        pipe_mean = image.new_tensor(PIPELINE_IMAGE_MEAN).view(1, 3, 1, 1)
        pipe_std = image.new_tensor(PIPELINE_IMAGE_STD).view(1, 3, 1, 1)
        mean = image.new_tensor(self.IMAGE_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(self.IMAGE_STD).view(1, 3, 1, 1)
        return ((image * pipe_std + pipe_mean) - mean) / std

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> TeacherOutput:
        """Encode images into CLS and patch token features.

        Use torch.no_grad() instead of inference_mode so output tensors can participate in
        autograd loss computation (e.g. smooth_l1_loss). UNIC (unic/main_unic.py:373) and
        DUNE (dune/model/dune.py) both use torch.no_grad() for teacher forward.

        Args:
            image (torch.Tensor): Preprocessed image tensor (B, 3, H, W).

        Returns:
            (TeacherOutput): Typed output with cls and patches tensors.
        """
        raise NotImplementedError


class EUPETeacher(TeacherModel):
    """EUPE teacher for encoder distillation (https://arxiv.org/abs/2603.22387).

    EUPE uses a 3-stage pipeline (Section 3): Stage 1 distills PEcore-G (1.9B) + PElang-G (1.7B) + DINOv3-H+ (840M) into
    a 1.9B proxy; Stage 2 distills the proxy into efficient students at 256x256; Stage 3 finetunes with multi-resolution
    {256, 384, 512}. The released models are Stage 2/3 students.

    Supports ViT (vitb16, vits16) and ConvNeXt (convnextb) variants. ConvNeXt produces CLS via global average pooling
    (verified: eupe/models/convnext.py:220, x_pool = x.mean([-2, -1])).

    Attributes:
        model: The EUPE backbone (DinoVisionTransformer or ConvNeXt).
    """

    # EUPE preprocessing uses ImageNet stats (eupe/models/vit.py, default DinoVisionTransformer normalization).
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    EUPE_REPO = "/home/fatih/dev/eupe"
    CONFIGS = {
        "vitb16": {
            "hub_name": "eupe_vitb16",
            "hf_repo": "facebook/EUPE-ViT-B",
            "hf_file": "EUPE-ViT-B.pt",
            "embed_dim": 768,
            "num_patches": 256,  # 16x16 grid at 256x256, patch_size=16
            "imgsz": 256,
            "token_types": ("cls", "patches"),
        },
        "vits16": {
            "hub_name": "eupe_vits16",
            "hf_repo": "facebook/EUPE-ViT-S",
            "hf_file": "EUPE-ViT-S.pt",
            "embed_dim": 384,
            "num_patches": 256,
            "imgsz": 256,
            "token_types": ("cls", "patches"),
        },
        "convnextb": {
            "hub_name": "eupe_convnext_base",
            "hf_repo": "facebook/EUPE-ConvNeXt-B",
            "hf_file": "EUPE-ConvNeXt-B.pt",
            "embed_dim": 1024,
            "num_patches": 64,  # 8x8 grid at 256x256 with 32x downsample
            "imgsz": 256,
            # ConvNeXt CLS is global avg pool (eupe/models/convnext.py:220: x_pool = x.mean([-2, -1]))
            "token_types": ("cls", "patches"),
        },
    }

    def __init__(self, variant: str = "vitb16", device: torch.device = None):
        """Initialize EUPE teacher model.

        Args:
            variant (str): Model variant ('vitb16', 'vits16', or 'convnextb').
            device (torch.device, optional): Device to load the model on.
        """
        super().__init__()
        if variant not in self.CONFIGS:
            raise ValueError(f"Unknown EUPE variant '{variant}'. Supported: {list(self.CONFIGS)}")
        from huggingface_hub import hf_hub_download

        cfg = self.CONFIGS[variant]
        weights = hf_hub_download(cfg["hf_repo"], cfg["hf_file"])
        self.model = torch.hub.load(self.EUPE_REPO, cfg["hub_name"], source="local", weights=weights)
        self._freeze(cfg, device)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> TeacherOutput:
        """Encode images via EUPE forward_features.

        Both ViT and ConvNeXt return the same dict keys (x_norm_clstoken, x_norm_patchtokens) --
        this is EUPE's architectural polymorphism (eupe/models/convnext.py:227-229).

        Args:
            image (torch.Tensor): Preprocessed image tensor (B, 3, H, W).

        Returns:
            (TeacherOutput): CLS and patch features.
        """
        image = self._prep(image)
        out = self.model.forward_features(image)
        cls = out["x_norm_clstoken"] if "cls" in self.token_types else None
        return TeacherOutput(cls=cls, patches=out["x_norm_patchtokens"])


class DINOv3Teacher(TeacherModel):
    """DINOv3 teacher via HuggingFace transformers (https://arxiv.org/abs/2508.10104).

    DINOv3 ViT models use 4 register tokens and RoPE positional embeddings. Output has 1 CLS + 4 register + N patch
    tokens; we extract CLS and patches, skipping registers.

    ConvNeXt variants use the same architecture as EUPE ConvNeXt (DINOv3 trains them with DINO/iBOT self-distillation).
    ConvNeXt CLS is token 0 in last_hidden_state (global average pooled).

    Attributes:
        model: The DINOv3 backbone from HuggingFace transformers.
    """

    # DINOv3 preprocessing uses ImageNet stats (Meta DINOv3 release, dinov3 reference processor).
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    CONFIGS = {
        "vitb16": {
            "hf_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "embed_dim": 768,
            "num_patches": 196,  # 14x14 at 224x224, patch_size=16
            "imgsz": 224,
            "n_registers": 4,
            "token_types": ("cls", "patches"),
        },
        "vitl16": {
            "hf_model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "embed_dim": 1024,
            "num_patches": 196,
            "imgsz": 224,
            "n_registers": 4,
            "token_types": ("cls", "patches"),
        },
        "convnextb": {
            "hf_model": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            "embed_dim": 1024,
            "num_patches": 49,  # 7x7 at 224x224 with 32x downsample
            "imgsz": 224,
            "n_registers": 0,
            # CLS is token 0 in last_hidden_state (verified: shape [1, 50, 1024] = 1 CLS + 49 patches)
            "token_types": ("cls", "patches"),
        },
        "vit7b": {
            "hf_model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            "embed_dim": 4096,
            "num_patches": 196,
            "imgsz": 224,
            "n_registers": 4,
            "token_types": ("cls", "patches"),
        },
    }

    def __init__(self, variant: str = "vitl16", device: torch.device = None):
        """Initialize DINOv3 teacher from HuggingFace.

        Args:
            variant (str): Model variant ('vitb16', 'vitl16', 'convnextb', or 'vit7b').
            device (torch.device, optional): Device to load the model on.
        """
        super().__init__()
        if variant not in self.CONFIGS:
            raise ValueError(f"Unknown DINOv3 variant '{variant}'. Supported: {list(self.CONFIGS)}")
        from transformers import AutoModel

        cfg = self.CONFIGS[variant]
        self.model = AutoModel.from_pretrained(cfg["hf_model"])
        self._freeze(cfg, device)
        self._n_registers = cfg["n_registers"]

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> TeacherOutput:
        """Encode images via DINOv3 HuggingFace model.

        ViT output has [CLS, reg0..reg3, patch0..patchN] token ordering.
        Skip CLS + registers to get patch tokens (verified: dune/teachers/config.py
        shows DINOv2 with num_register_tokens=4).

        Args:
            image (torch.Tensor): Preprocessed image tensor (B, 3, H, W).

        Returns:
            (TeacherOutput): CLS and patch features, skipping register tokens.
        """
        image = self._prep(image)
        out = self.model(pixel_values=image)
        hidden = out.last_hidden_state  # (B, 1 + n_reg + N_patches, D)
        cls = hidden[:, 0] if "cls" in self.token_types else None
        patches = hidden[:, 1 + self._n_registers :]  # skip CLS + registers
        return TeacherOutput(cls=cls, patches=patches)


class SigLIP2Teacher(TeacherModel):
    """SigLIP2 teacher via HuggingFace transformers.

    SigLIP2-Giant-Opt uses the SigLIP v1 architecture (model_type="siglip", not NaFlex "siglip2"). No explicit CLS token
    in the sequence -- uses attention-pooled summary via pooler_output. Used by C-RADIOv4 as the CLIP teacher
    (RADIOv2.5, arXiv:2412.07679, Table 1 config C onward).

    Attributes:
        model: SiglipVisionModel from HuggingFace transformers.
    """

    # SigLIP preprocessing uses (0.5, 0.5, 0.5) mean/std (HF google/siglip2-giant-opt-patch16-384 processor config).
    IMAGE_MEAN = (0.5, 0.5, 0.5)
    IMAGE_STD = (0.5, 0.5, 0.5)

    CONFIGS = {
        "g": {
            "hf_model": "google/siglip2-giant-opt-patch16-384",
            "embed_dim": 1536,
            "num_patches": 576,  # (384/16)^2
            "imgsz": 384,
            "token_types": ("cls", "patches"),
        },
    }

    def __init__(self, variant: str = "g", device: torch.device = None):
        """Initialize SigLIP2 teacher from HuggingFace.

        Args:
            variant (str): Model variant ('g' for Giant-Opt).
            device (torch.device, optional): Device to load the model on.
        """
        super().__init__()
        if variant not in self.CONFIGS:
            raise ValueError(f"Unknown SigLIP2 variant '{variant}'. Supported: {list(self.CONFIGS)}")
        from transformers import SiglipVisionModel

        cfg = self.CONFIGS[variant]
        self.model = SiglipVisionModel.from_pretrained(cfg["hf_model"])
        self._freeze(cfg, device)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> TeacherOutput:
        """Encode images via SigLIP2 vision model.

        SigLIP has no CLS token in the sequence. pooler_output is an attention-pooled summary
        (SiglipMultiheadAttentionPoolingHead) that serves as the CLS equivalent.

        Args:
            image (torch.Tensor): Preprocessed image tensor (B, 3, 384, 384).

        Returns:
            (TeacherOutput): Attention-pooled CLS and patch token features.
        """
        image = self._prep(image)
        out = self.model(pixel_values=image)
        return TeacherOutput(cls=out.pooler_output, patches=out.last_hidden_state)


class SAM3Teacher(TeacherModel):
    """SAM3.1 ViT-L teacher using ultralytics' built-in SAM3 backbone.

    SAM3 ViT (sam3/vitdet.py) uses embed_dim=1024, patch_size=14, depth=32 with RoPE and windowed
    attention. retain_cls_token=False, so no CLS token in output -- patches only. This follows
    AM-RADIO's convention of lambda_SAM=0 for summary loss (arXiv:2312.06709, Section 3.3) and DUNE's
    token_types=["patch"] for non-CLS teachers (dune/teachers/config.py:25).

    The backbone forward returns spatial feature maps in NCHW format. We reshape to (B, N, D) patch token format for
    consistency with other teachers.

    Attributes:
        model: SAM3 ViT backbone (without the FPN neck or decoder).
    """

    # SAM3 preprocessing uses (0.5, 0.5, 0.5) at [0, 1] scale -- ``models/sam/predict.py:2202-2203`` sets the predictor
    # mean/std = 127.5 at [0, 255] scale (SigLIP-style midpoint), not the ImageNet stats used for SAM1/2 at line 463.
    IMAGE_MEAN = (0.5, 0.5, 0.5)
    IMAGE_STD = (0.5, 0.5, 0.5)

    def __init__(self, variant: str = "l", device: torch.device = None):
        """Initialize SAM3 teacher from ultralytics built-in weights.

        Args:
            variant (str): Model variant ('l' for ViT-L).
            device (torch.device, optional): Device to load the model on.
        """
        super().__init__()
        from ultralytics.models.sam.sam3.vitdet import ViT
        from ultralytics.utils.downloads import attempt_download_asset

        # SAM3 ViT-L config from build_sam3.py:37-62
        self.model = ViT(
            img_size=1008,
            pretrain_img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=32,
            num_heads=16,
            mlp_ratio=4.625,
            norm_layer="LayerNorm",
            drop_path_rate=0.0,  # no drop path for frozen teacher
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,  # no CLS token in output
            ln_pre=True,
            ln_post=False,
            return_interm_layers=False,
            bias_patch_embed=False,
        )
        # Load pretrained weights from SAM3 checkpoint (ViT backbone is under detector.backbone.vision_backbone.trunk.*)
        ckpt_path = attempt_download_asset("sam3.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prefix = "detector.backbone.vision_backbone.trunk."
        trunk_sd = {k[len(prefix) :]: v for k, v in ckpt.items() if k.startswith(prefix)}
        self.model.load_state_dict(trunk_sd, strict=False)
        self._freeze({"embed_dim": 1024, "num_patches": 0, "token_types": ("patches",)}, device)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> TeacherOutput:
        """Encode images via SAM3 ViT backbone.

        SAM3 ViT forward returns list[Tensor] in NCHW format (vitdet.py:498). We take the final
        feature map and reshape to (B, N, D) patch token format.

        Args:
            image (torch.Tensor): Preprocessed image tensor (B, 3, H, W).

        Returns:
            (TeacherOutput): Patches only (cls=None, patches in (B, N, D) format).
        """
        image = self._prep(image)
        feat_maps = self.model(image)  # list of (B, C, H, W)
        feat = feat_maps[-1]  # final feature map: (B, 1024, H', W')
        patches = feat.flatten(2).transpose(1, 2)  # (B, N, 1024)
        return TeacherOutput(cls=None, patches=patches)


class MoonViTTeacher(TeacherModel):
    """MoonViT-SO-400M teacher from Kimi-VL (https://huggingface.co/moonshotai/MoonViT-SO-400M).

    NaFlex / native-resolution vision encoder, patch_size=14, with 2x2 patch-merging at the head. The HuggingFace
    forward signature is ``model(pixel_values: (N_total_patches, 3, P, P), grid_hws: (B, 2)) -> list[Tensor]`` per
    image; each output tensor has shape ``(N_merged, kh*kw, D)`` produced by ``patch_merger``
    (modeling_moonvit.py:483-509) via ``view(new_h, kh, new_w, kw, D).permute(0, 2, 1, 3, 4).view(new_h*new_w, kh*kw,
    D)``. We pre-patchify the input, forward, then un-merge with the self-inverse permute(0, 2, 1, 3, 4) to recover
    raster-ordered patches for the student's spatial loss. Patches-only (no CLS / pooler module), like SAM3.

    transformers>=5.5.0 ``PreTrainedModel.post_init`` assigns ``all_tied_weights_keys`` per-instance, and the
    meta-device weight loading hook reads it back (``_move_missing_keys_from_meta_to_device`` and the
    ``is_remote_code()`` branch of ``mark_tied_weights_as_initialized``). MoonViT's ``trust_remote_code`` modeling file
    predates the attribute and never sets it; we install a class-level default before instantiation. Standard HF models
    still set the instance attribute, which shadows this default for them.

    Attributes:
        model: MoonViT-SO-400M backbone from HuggingFace (trust_remote_code).
    """

    # MoonViT/Kimi-VL preprocessing uses (0.5, 0.5, 0.5) mean/std (LocateAnything-3B preprocessor_config.json,
    # image_processing_locateanything.py:23-24).
    IMAGE_MEAN = (0.5, 0.5, 0.5)
    IMAGE_STD = (0.5, 0.5, 0.5)

    # Encode images this many at a time. MoonViT packs every image in one model() call into a single
    # sequence and runs dense O(N^2) attention under the eager/sdpa backends, so a full batch is
    # quadratic in batch size. Chunk=4 measured 5.5x faster and 20x less peak memory than a 64-image
    # batch on Blackwell. Only raise this if flash_attention_2 (varlen block-diagonal) is enabled.
    ENCODE_CHUNK = 4

    CONFIGS = {
        "so400m": {
            "hf_model": "moonshotai/MoonViT-SO-400M",
            "revision": "a889d399ff2306053e4e28d499d3b8f97d3e5007",  # pin remote modeling code; bump after rev verification
            "embed_dim": 1152,
            "num_patches": 256,  # 16x16 raw patches at 224x224, patch_size=14 (un-merged from 8x8 cells x 4 patches)
            "imgsz": 224,
            "patch_size": 14,
            "merge_kernel": (2, 2),
            "token_types": ("patches",),
        },
    }

    def __init__(self, variant: str = "so400m", device: torch.device = None):
        """Initialize MoonViT teacher from HuggingFace.

        Args:
            variant (str): Model variant ('so400m').
            device (torch.device, optional): Device to load the model on.
        """
        super().__init__()
        if variant not in self.CONFIGS:
            raise ValueError(f"Unknown MoonViT variant '{variant}'. Supported: {list(self.CONFIGS)}")
        import transformers.modeling_utils as _mu

        if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
            _mu.PreTrainedModel.all_tied_weights_keys = {}
        from transformers import AutoModel

        cfg = self.CONFIGS[variant]
        # Force fp32: MoonViT's saved weights default to bf16, but distillation feeds fp32 images (other teachers are
        # fp32-on-disk and avoid this implicit dtype). Without this override the patch_embed Conv2d errors with
        # ``Input type (float) and bias type (c10::BFloat16) should be the same``.
        self.model = AutoModel.from_pretrained(
            cfg["hf_model"], revision=cfg["revision"], trust_remote_code=True, dtype=torch.float32
        )
        self._patch_size = cfg["patch_size"]
        self._merge_kernel = cfg["merge_kernel"]
        self._freeze(cfg, device)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> TeacherOutput:
        """Encode images via MoonViT NaFlex API.

        Pre-patchifies each ENCODE_CHUNK-image slice into (cb*Hg*Wg, 3, P, P), forwards with image_grid_hws, then
        un-merges the 2x2 patch-merging output back to raster-order (B, Hg*Wg, D). Chunking keeps each model() call to
        cb*256 tokens. The eager/sdpa backends run one dense O(N^2) attention over the whole call, so packing the full
        batch is quadratic in B. Inverse of the merge in modeling_moonvit.py:496-505:
        ``view(new_h, new_w, kh, kw, D).permute(0, 2, 1, 3, 4)`` undoes ``view(new_h, kh, new_w, kw, D).permute(0, 2,
        1, 3, 4)``. Verified mathematically: round-trip on identity tensor exactly recovers the input.

        Args:
            image (torch.Tensor): Preprocessed image tensor (B, 3, H, W). H and W must be divisible by ``patch_size *
                merge_kernel`` (28 for SO-400M). Multi-teacher launches must constrain ``_teacher_imgsz`` to a multiple
                of 28 or this raises ``ValueError``.

        Returns:
            (TeacherOutput): Patches only (cls=None), raster-ordered (B, Hg*Wg, D).
        """
        image = self._prep(image)
        B, _, H, W = image.shape
        P = self._patch_size
        kh, kw = self._merge_kernel
        if H % (P * kh) or W % (P * kw):
            raise ValueError(
                f"MoonViT requires H, W divisible by patch_size*merge_kernel ({P}*{kh}={P * kh}); got H={H}, W={W}. "
                f"Multi-teacher launches: constrain student/teacher imgsz to a multiple of {P * kh}."
            )
        Hg, Wg = H // P, W // P
        new_h, new_w = Hg // kh, Wg // kw
        D = self.embed_dim
        unmerged = []
        for start in range(0, B, self.ENCODE_CHUNK):
            chunk = image[start : start + self.ENCODE_CHUNK]
            cb = chunk.shape[0]
            patches_in = chunk.unfold(2, P, P).unfold(3, P, P)  # (cb, 3, Hg, Wg, P, P)
            patches_in = patches_in.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(cb * Hg * Wg, 3, P, P)
            grid_hws = torch.tensor([[Hg, Wg]], device=image.device, dtype=torch.long).expand(cb, 2)
            outputs = self.model(patches_in, grid_hws)  # list[Tensor], each (new_h*new_w, kh*kw, D)
            unmerged.extend(
                t.view(new_h, new_w, kh, kw, D).permute(0, 2, 1, 3, 4).contiguous().view(-1, D) for t in outputs
            )
        return TeacherOutput(cls=None, patches=torch.stack(unmerged, dim=0))


class TorchScriptTeacher(TeacherModel):
    """Load a TorchScript-traced teacher model (.ts file).

    Traced from _TraceWrapper which returns a (cls, patches) tuple. This removes the dependency on external repos (EUPE,
    DINOv3, etc.) during training, following the MobileCLIP pattern in ultralytics/nn/image_model.py (MobileCLIPImageTS
    loads .ts via torch.jit.load).

    For patches-only teachers (SAM3), the .ts returns (zeros, patches) and token_types is set to ("patches",) so the
    loss function skips the CLS component.

    Attributes:
        model (torch.jit.ScriptModule): Traced teacher model.
    """

    def __init__(
        self, ts_path: str, embed_dim: int, num_patches: int, token_types: tuple[str, ...], device: torch.device = None
    ):
        """Initialize TorchScript teacher from a .ts file.

        Args:
            ts_path (str): Path to the traced .ts file.
            embed_dim (int): Teacher embedding dimension.
            num_patches (int): Number of patch tokens.
            token_types (tuple[str, ...]): Which outputs are meaningful.
            device (torch.device, optional): Device to load the model on.
        """
        super().__init__()
        self.model = torch.jit.load(ts_path, map_location=device or "cpu")
        self.model.eval()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.token_types = token_types

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> TeacherOutput:
        """Encode images via traced TorchScript model.

        Does NOT call ``_prep``: traced ``.ts`` files bake their own preprocessing assumptions into the graph and the
        ``TorchScriptTeacher`` class doesn't carry per-trace ``IMAGE_MEAN/STD`` overrides. Honoring ``_normalize_input``
        here would silently no-op (inherited PIPELINE defaults) and mislead the user. ``.ts`` teachers are deprecated
        for distillation (see ``CLAUDE.md`` "All supported teachers"); use native Python teachers when normalization
        matters.

        Args:
            image (torch.Tensor): Preprocessed image tensor (B, 3, H, W).

        Returns:
            (TeacherOutput): CLS (None if patches-only) and patch features.
        """
        cls, patches = self.model(image)
        return TeacherOutput(
            cls=cls if "cls" in self.token_types else None,
            patches=patches,
        )


# Teacher registry built from per-class CONFIGS to avoid duplicating embed_dim/num_patches/token_types.
# SAM3 has no CONFIGS dict (hardcoded ViT-L config), so it's added manually.
TEACHER_REGISTRY = {}
for _prefix, _cls in [("eupe", EUPETeacher), ("dinov3", DINOv3Teacher), ("siglip2", SigLIP2Teacher), ("moonvit", MoonViTTeacher)]:
    for _variant, _cfg in _cls.CONFIGS.items():
        TEACHER_REGISTRY[f"{_prefix}:{_variant}"] = {
            "cls": _cls,
            "embed_dim": _cfg["embed_dim"],
            "num_patches": _cfg["num_patches"],
            "imgsz": _cfg["imgsz"],
            "token_types": _cfg["token_types"],
        }
TEACHER_REGISTRY["sam3:l"] = {
    "cls": SAM3Teacher,
    "embed_dim": 1024,
    "num_patches": 0,
    "imgsz": 1024,
    "token_types": ("patches",),
}


def resolve_teacher_key(spec: str) -> str:
    """Resolve a TEACHER_REGISTRY key from its colon ("eupe:vitb16") or safe_key ("eupe_vitb16") form.

    Returns spec unchanged when nothing matches so the caller's own lookup raises a clear error.

    Args:
        spec (str): Teacher key in registry or safe_key form.

    Returns:
        (str): The matching registry key, or spec unchanged when none matches.
    """
    return next((k for k in TEACHER_REGISTRY if k == spec or safe_key(k) == spec), spec)


def build_teacher_model(
    variant: str, device: torch.device = None, normalize_input: bool = False
) -> TeacherModel:
    """Build a frozen teacher model for encoder distillation.

    Args:
        variant (str): Teacher variant (e.g., "eupe:vitb16", "dinov3:vitl16", "sam3:l").
        device (torch.device, optional): Device to load the model on.
        normalize_input (bool, optional): When True, ``_prep`` converts the pipeline's ImageNet-normalized input to each
            teacher's training-time distribution (no-op for EUPE/DINOv3 which already match ImageNet stats; SigLIP-style
            ``2x - 1`` conversion for SigLIP2/MoonViT/SAM3). Default False returns the pipeline tensor verbatim,
            preserving bit-identical legacy behavior.

    Returns:
        (TeacherModel): Instantiated frozen teacher model.
    """
    if variant not in TEACHER_REGISTRY:
        raise ValueError(f"Unknown teacher '{variant}'. Supported: {list(TEACHER_REGISTRY)}")
    _, size = variant.split(":")
    teacher = TEACHER_REGISTRY[variant]["cls"](size, device)
    teacher._normalize_input = normalize_input
    return teacher


class TeacherDetBackbone(nn.Module):
    """Frozen foundation teacher as a single-scale detection backbone.

    Wraps build_teacher_model() and reshapes its patch tokens (B, N, D) into a (B, D, gh, gw) feature map so a detection
    neck and head can train on top while the teacher stays frozen, measuring the frozen-feature detection ceiling
    (run_enc_distill_phase2.py teacher_frozen_det mode). The embedding dimension equals the output channel count and is
    set as the layer-0 output channels by parse_model.

    The teacher MUST also be frozen via the trainer freeze=1 arg: BaseTrainer._setup_train re-enables requires_grad for
    any floating param not in the freeze list (engine/trainer.py:319), so freezing in __init__ alone is undone.

    The spec is passed in safe_key form ("eupe_vitb16"), not registry form ("eupe:vitb16"): parse_model runs string args
    through ast.literal_eval, which raises an unsuppressed SyntaxError on a colon but a suppressed ValueError on the
    underscore form, so only the underscore form survives as a string arg.

    Detection feeds [0, 1] RGB (cv2 BGR is flipped to RGB by Format when hyp.bgr=0). EUPE/DINOv3 use ImageNet stats,
    equal to the phase-1 distillation normalization, so (x - IMAGE_MEAN) / IMAGE_STD reproduces the exact distribution
    the teacher was distilled on. The teacher runs at the detection resolution (ViTDet convention: pos-embeds / RoPE
    interpolate to the det grid), not its native pretraining resolution.

    Attributes:
        teacher (TeacherModel): The wrapped frozen foundation teacher.
        embed_dim (int): Teacher embedding dimension, equal to the output channel count.
        patch_stride (int): Teacher downsample factor, used to recover the (gh, gw) token grid from the input size.
    """

    def __init__(self, spec: str):
        """Initialize the frozen teacher detection backbone.

        Args:
            spec (str): Teacher key in registry ("eupe:vitb16") or safe_key ("eupe_vitb16") form.
        """
        super().__init__()
        spec = resolve_teacher_key(spec)
        reg = TEACHER_REGISTRY[spec]
        self.teacher = build_teacher_model(spec)
        self.teacher._normalize_input = False  # forward normalizes the [0,1] detection input directly
        self.embed_dim = self.teacher.embed_dim
        self.patch_stride = reg["imgsz"] // math.isqrt(reg["num_patches"])
        self.register_buffer("mean", torch.tensor(self.teacher.IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.teacher.IMAGE_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> TeacherDetBackbone:
        """Set training mode but keep the frozen teacher in eval (no BatchNorm/dropout updates)."""
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [0, 1] RGB detection images into a single (B, embed_dim, gh, gw) feature map.

        Args:
            x (torch.Tensor): Detection-pipeline images (B, 3, H, W) in [0, 1], RGB.

        Returns:
            (torch.Tensor): Spatial feature map (B, embed_dim, gh, gw), gh=H/patch_stride, gw=W/patch_stride.
        """
        h, w = x.shape[-2:]
        patches = self.teacher.encode((x - self.mean) / self.std).patches  # (B, N, D)
        b, n, d = patches.shape
        gh, gw = h // self.patch_stride, w // self.patch_stride
        # Recover the real (gh, gw) grid instead of assuming a square sqrt(N): detection val uses rect=True (H!=W), so a
        # square-grid assumption would raise on reshape or silently scramble the spatial map and invalidate the run.
        assert gh * gw == n, f"teacher grid {gh}x{gw}={gh * gw} != {n} tokens at input {h}x{w}, stride {self.patch_stride}"
        return patches.transpose(1, 2).reshape(b, d, gh, gw)
