"""Distillation-flavoured classification augmentation pipeline.

This module provides ``classify_augmentations_distill`` — a thin wrapper around Ultralytics's
``ultralytics/data/augment.py:classify_augmentations`` that adds the DINOv3 / UNIC / DUNE
photometric stack on top of the standard RandomResizedCrop + HFlip + ColorJitter + ToTensor +
Normalize + RandomErasing pipeline.

It is kept here (in the runner-local ``callbacks/`` package) rather than in ``ultralytics/data``
to avoid touching the upstream classification training pipeline. The encoder distillation
trainer (``ultralytics/models/yolo/classify/train_image_encoder.py:_build_transforms``) calls
this enriched variant directly; legacy callers still see the bit-equivalent upstream pipeline.

Reference recipes audited (all student-side, all single-crop unless noted):

  DINOv3 / DINOv2 ``DataAugmentationDINO`` — ``dinov3/data/augmentations.py:18-227``:
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1) p=0.8,
    RandomGrayscale p=0.2, GaussianBlur asymmetric (g1 p=1.0 / g2 p=0.1, local p=0.5),
    Solarize threshold=0.5 p=0.2 (g2 only). Multi-crop (2 global + 8 local).

  EUPE ``eupe/configs/ssl_default_config.yaml`` lines 148-172 — inherits the DINOv3 stack
    identically (horizontal_flips: true, share_color_jitter: false).

  UNIC ``unic/main_unic.py:485-521`` — single global crop:
    HFlip(0.5) -> RandomApply([ColorJitter(0.4/0.4/0.2/0.1)], p=0.8) ->
    RandomApply([Grayscale(3ch)], p=0.2) -> RandomApply([GaussianBlur k=9, sigma=(0.1, 5.0)], p=0.2)
    -> RandomSolarize(threshold=0.5, p=0.2) -> Normalize(ImageNet).

  DUNE ``dune/data/transform.py:9-39`` — identical to UNIC; rrc_scale=(0.40, 1.0) instead
    of timm default (0.08, 1.0); higher input resolution 336.

  EdgeCrafter §A.1 (paper, arXiv:2603.18739) — RRC@224 + HFlip + ColorJitter + Grayscale +
    GaussianBlur + MixUp.

We mirror UNIC / DUNE single-crop because our distill loss path emits one view per image
(multi-crop would require restructuring the per-batch tensor and loss; deferred). Order of
ops mirrors UNIC ``main_unic.py:485-521`` exactly: HFlip -> ColorJitter -> Grayscale ->
GaussianBlur -> Solarize -> ToTensor -> Normalize -> RandomErasing.
"""

from __future__ import annotations

import torchvision.transforms as T

from ultralytics.data.augment import DEFAULT_MEAN, DEFAULT_STD, classify_augmentations


def classify_augmentations_distill(
    size: int = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    scale: tuple[float, float] | None = None,
    ratio: tuple[float, float] | None = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    auto_augment: str | None = None,
    hsv_h: float = 0.015,
    hsv_s: float = 0.4,
    hsv_v: float = 0.4,
    force_color_jitter: bool = False,
    color_jitter: float = 1.0,
    erasing: float = 0.0,
    grayscale: float = 0.0,
    gaussian_blur: float = 0.0,
    solarize: float = 0.0,
    interpolation: str = "BILINEAR",
):
    """Build a distillation-style classification training transform.

    Identical to ``ultralytics.data.augment.classify_augmentations`` except for the extra optional knobs ported from
    DINOv3 / UNIC / DUNE encoder-distillation recipes. Every knob defaults to its upstream-equivalent neutral value so
    callers can switch over safely.

    DINOv3 ``ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)`` maps onto Ultralytics's ``hsv_v /
    hsv_v / hsv_s / hsv_h`` binding (brightness and contrast share hsv_v) inside the upstream call.

    Args:
        size (int): Target size for the image after transformations.
        mean (tuple[float, float, float]): Mean values for each RGB channel used in normalization.
        std (tuple[float, float, float]): Standard deviation values for each RGB channel.
        scale (tuple[float, float] | None): Range of the proportion of the original image area to crop.
        ratio (tuple[float, float] | None): Range of aspect ratio for the cropped area.
        hflip (float): Probability of horizontal flip.
        vflip (float): Probability of vertical flip.
        auto_augment (str | None): Auto augmentation policy: 'randaugment', 'augmix', 'autoaugment' or None. DINOv3 /
            UNIC / DUNE distillation recipes leave this as None and rely on the photometric stack below.
        hsv_h (float): Image HSV-Hue augmentation factor; binds to T.ColorJitter(hue=hsv_h). DINOv3 uses 0.1.
        hsv_s (float): HSV-Saturation; binds to T.ColorJitter(saturation=hsv_s). DINOv3 uses 0.2.
        hsv_v (float): HSV-Value; binds to T.ColorJitter(brightness=hsv_v, contrast=hsv_v). DINOv3 uses 0.4.
        force_color_jitter (bool): Apply ColorJitter even if auto_augment is enabled.
        color_jitter (float): Probability of applying the ColorJitter above; upstream fires it on every sample. DINOv3 /
            UNIC / DUNE all wrap the same jitter in ``RandomApply(p=0.8)`` — see the module docstring for refs.
        erasing (float): Probability of RandomErasing. DINOv3 / EUPE / UNIC / DUNE / AM-RADIO do NOT use random erasing;
            off by default.
        grayscale (float): Probability of converting to 3-channel grayscale. DINOv3 / DINOv2 / UNIC use 0.2.
        Reference: ``dinov3/data/augmentations.py:DataAugmentationDINO`` and ``unic/main_unic.py:497``.
        gaussian_blur (float): Probability of applying ``T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))``.
            UNIC ``main_unic.py:516`` uses kernel=9, p=0.2; DINOv3 uses asymmetric (g1 p=1.0 /
            g2 p=0.1) across two global crops which we approximate uniformly here.
        solarize (float): Probability of ``T.RandomSolarize(threshold=128)``. DINOv3 / UNIC / DUNE use 0.2 on the second
            global crop only; we apply uniformly to the single view.
        interpolation (str): Interpolation method: 'NEAREST', 'BILINEAR' or 'BICUBIC'.

    Returns:
        (torchvision.transforms.Compose): A composition of training-time augmentations.

    Examples:
        >>> tf = classify_augmentations_distill(
        ...     size=224,
        ...     auto_augment=None,
        ...     hsv_h=0.1,
        ...     hsv_s=0.2,
        ...     hsv_v=0.4,
        ...     grayscale=0.2,
        ...     gaussian_blur=0.5,
        ...     solarize=0.2,
        ...     erasing=0.0,
        ... )
    """
    compose = classify_augmentations(
        size=size,
        mean=mean,
        std=std,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        auto_augment=auto_augment,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        force_color_jitter=force_color_jitter,
        erasing=erasing,
        interpolation=interpolation,
    )

    # Upstream appends ColorJitter bare, so it fires on every sample. Gate on ``< 1.0`` rather than
    # rewriting unconditionally: ``T.RandomApply.forward`` draws ``torch.rand(1)`` before comparing
    # against ``p``, so wrapping at p=1.0 would shift the RNG stream and change sample 1. ``next``
    # returns None when auto_augment suppressed the jitter and there is nothing to wrap.
    if color_jitter < 1.0:
        i = next((i for i, t in enumerate(compose.transforms) if isinstance(t, T.ColorJitter)), None)
        if i is not None:
            compose.transforms[i] = T.RandomApply([compose.transforms[i]], p=color_jitter)

    # DINOv3 / UNIC / DUNE photometric extras — strict UNIC ``main_unic.py:485-521`` order.
    # ``T.RandomGrayscale`` outputs 3-channel grayscale (replicates the single channel) — what
    # DINOv3 ``DataAugmentationDINO`` and UNIC produce; downstream Normalize stays valid because
    # the tensor shape is unchanged. ``T.GaussianBlur`` has no ``p`` kwarg so ``T.RandomApply`` is
    # the only option; kernel_size=9, sigma=(0.1, 2.0): UNIC ``main_unic.py:516`` uses k=9,
    # sigma=(0.1, 5.0); DINOv3 uses sigma=(0.1, 2.0) — we pick the tighter range for less
    # aggressive blur at imgsz=224. ``threshold=128`` is the uint8 midpoint = 0.5 in [0, 1] PIL
    # space; matches DINOv3 / UNIC.
    extras = []
    if grayscale > 0.0:
        extras.append(T.RandomGrayscale(p=grayscale))
    if gaussian_blur > 0.0:
        extras.append(T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=gaussian_blur))
    if solarize > 0.0:
        extras.append(T.RandomSolarize(threshold=128, p=solarize))

    # Insert before upstream's ``final_tfl = [ToTensor, Normalize, RandomErasing]`` (3 trailing
    # entries). If upstream ever changes the size of ``final_tfl``, update the slice in lockstep.
    compose.transforms[-3:-3] = extras
    return compose
