#!/usr/bin/env python
"""Phase 1: Encoder distillation pretraining on DataComp-12M."""

import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch

from callbacks import paths, wandb_config
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train_image_encoder import ImageEncoderTrainer

RECIPES = {
    "default": dict(lr0=3e-4, weight_decay=0.05, warmup_epochs=1, epochs=10, momentum=0.9, grad_clip=3.0, beta2=None),
    # EUPE Stage 2: proxy->student distillation (arXiv:2603.22387 Sec 4.1, ssl_default_config.yaml:131-147)
    # Same loss as ours (0.9cos+0.1L1, Eq.5-6). beta2=None -> uses default 0.999 matching EUPE
    "eupe": dict(lr0=2e-5, weight_decay=1e-4, warmup_epochs=1, epochs=30, momentum=0.9, grad_clip=3.0, beta2=None),
    # AM-RADIO: multi-teacher distillation (arXiv:2312.06709 Sec 4, Eq.2-3)
    # Same loss as ours (0.9cos+0.1L1). beta2=0.95 from MobileCLIP2 (training/configs/run_dfndr2b.sh)
    "radio": dict(lr0=1e-3, weight_decay=0.02, warmup_epochs=1, epochs=30, momentum=0.9, grad_clip=1.0, beta2=0.95),
    # UNIC (ECCV 2024) reproduction used for phase1-b1-unic-eupe-vitb16 (R1 ablation baseline).
    # lr0/wd/warmup matched from /data/shared-datasets/fatih-runs/.../phase1-b1-unic-eupe-vitb16/args.yaml.
    "unic": dict(lr0=6e-4, weight_decay=0.03, warmup_epochs=2, epochs=30, momentum=0.9, grad_clip=3.0, beta2=None),
    # DINOv3-aligned recipe — addresses the fastvit-s × 7-source collapse to chance-level kNN
    # observed in fvs-fm/fvs-ad. Mirrors three published recipes that converge with hybrid-ViT
    # students under multi-source distillation:
    #   DINOv3 ConvNeXt-T distill ``configs/train/distillation_convnext/convnext_tiny_p16.yaml``:
    #     lr peak=2e-4, warmup=80/500 ep (16%), wd schedule 0.04→0.2, clip_grad=3.0,
    #     adamw beta2 default 0.999.
    #   EUPE SSL ``configs/ssl_default_config.yaml`` optim+schedules: lr=1e-3, wd=0.04→0.4,
    #     beta2=0.999, multi-crop (2 global @224 + 8 local @96).
    #   UNIC ``main_unic.py:485-521`` photometric stack: ColorJitter(0.4/0.4/0.2/0.1)@0.8,
    #     Grayscale@0.2, GaussianBlur(k=9, σ=(0.1, 5.0))@0.2, RandomSolarize(thr=0.5)@0.2.
    # We adopt single-crop at 224 (multi-crop deferred until loss-path adapter exists).
    #
    # Knob → reference mapping:
    #   lr0=2e-4         → DINOv3 distillation_convnext/convnext_tiny_p16.yaml schedules.lr.peak
    #   warmup_epochs=1  → matches wave-1 radio recipe (effective 2 ep at batch=1024 after Goyal scale).
    #                      Pinned to wave-1 for clean A/B; the 5x lower lr already addresses the
    #                      ep~7 BN-coupling fragility that drove wave-1 fastvit divergence,
    #                      so longer warmup is not needed.
    #   weight_decay=0.04 + wd_end=0.2 → DINOv3 schedules.weight_decay.{start, peak} (callbacks/wd_schedule.py)
    #   grad_clip=3.0    → DINOv3 optim.clip_grad
    #   beta2=None       → falls through to AdamW default 0.999 (DINOv3 / EUPE convention)
    #   auto_augment=""  → disables Ultralytics RandAugment, falls back to plain ColorJitter +
    #                      our DINOv3-style additions (Grayscale / GaussianBlur / Solarize). Without
    #                      this, RandAugment is on by default (DEFAULT_CFG) and Ultralytics auto-
    #                      disables ColorJitter, breaking alignment with DINOv3.
    #   hsv_h=0.1, hsv_s=0.2, hsv_v=0.4 → maps to T.ColorJitter(brightness=0.4, contrast=0.4,
    #                      saturation=0.2, hue=0.1), matching DINOv3 DataAugmentationDINO exactly
    #                      (Ultralytics's classify_augmentations binds brightness=contrast=hsv_v).
    #   grayscale=0.2    → DINOv3 / UNIC / DUNE
    #   gaussian_blur=0.5 → DINOv3 averaged across two-view asymmetric (g1=1.0 / g2=0.1); UNIC uses 0.2
    #   solarize=0.2     → DINOv3 g2-only; applied uniformly to single view here
    #   erasing=0.0      → DINOv3 / EUPE / UNIC / DUNE / AM-RADIO do NOT use random erasing
    "dinov3": dict(
        lr0=2e-4,
        weight_decay=0.04,
        wd_end=0.2,
        warmup_epochs=1,
        epochs=114,
        momentum=0.9,
        grad_clip=3.0,
        beta2=None,
        auto_augment=None,
        erasing=0.0,
        hsv_h=0.1,
        hsv_s=0.2,
        hsv_v=0.4,
        grayscale=0.2,
        gaussian_blur=0.5,
        solarize=0.2,
    ),
    # dinov3 photometric augs + radio-style constant wd. Addresses dinov3 backbone
    # weight magnitude collapse (memory project_dinov3_weight_collapse): dinov3 wd
    # 0.04->0.2 schedule drove 7x layer-0 L2 shrink vs radio at matched 7-src recipe,
    # breaking phase-2 det (P=R=mAP=0). Lowers wd to 0.02 + drops the schedule (wd_end
    # omitted) so weight magnitudes track the radio sibling while keeping the dinov3
    # photometric augs that won kNN +0.9pp at the same config.
    "dinov3_lowwd": dict(
        lr0=2e-4,
        weight_decay=0.02,
        warmup_epochs=1,
        epochs=114,
        momentum=0.9,
        grad_clip=3.0,
        beta2=None,
        auto_augment=None,
        erasing=0.0,
        hsv_h=0.1,
        hsv_s=0.2,
        hsv_v=0.4,
        grayscale=0.2,
        gaussian_blur=0.5,
        solarize=0.2,
    ),
}

# Reference global step-batch the recipes' lr0 and warmup_epochs are tuned for. When
# per_gpu_batch * world_size exceeds this, lr0 and warmup_epochs scale linearly and nbs rises
# to the global batch so wd_eff stays at the recipe value.
NBS_CANONICAL = 512

DATA_7SRC_DEFAULT = ",".join(
    [
        "/data/shared-datasets/imagenet",
        "/data/shared-datasets/coco",
        "/data/shared-datasets/yoloe26_data/Objects365v1/images/train",
        "/data/shared-datasets/yoloe26_data/mixed_grounding/gqa/paired/train",
        "/data/shared-datasets/yoloe26_data/flickr/paired/train",
        "/data/shared-datasets/DOTAv1-split/paired/train",
        "/data/shared-datasets/SODA-A-split/images/train",
    ]
)


def _pop_flag(argv: list[str], flag: str, is_bool: bool = False) -> tuple[list[str], str]:
    """Pop a --flag [value] pair from argv, return (remaining_argv, value).

    Args:
        argv: argument list
        flag: flag name (e.g. "--resume")
        is_bool: if True, flag has no value argument
    """
    if flag not in argv:
        return argv, ""
    i = argv.index(flag)
    if is_bool:
        return argv[:i] + argv[i + 1 :], "true"
    return argv[:i] + argv[i + 2 :], argv[i + 1]


def _load_train_args(resume: str) -> dict:
    """Load saved training arguments from a checkpoint."""
    return torch.load(Path(resume), map_location="cpu", weights_only=False)["train_args"]


def main(argv: list[str]) -> None:
    """Launch a fresh phase 1 run or resume from a checkpoint.

    Args:
        argv: [gpu, teachers, name, recipe, model_yaml, data, epochs]
        --resume <path>: resume from checkpoint
        --cos_weight <float>: cosine loss weight (default 0.9)
        --l1_weight <float>: smooth L1 loss weight (default 0.1)
        --cls_l1: add smooth L1 to CLS token loss (default False)
        --loss_type <str>: patch loss "cos_l1" (default, 0.9cos+0.1L1) or "l2" (pure MSE on un-normalized features)
        --lr <float>: override recipe lr0 (applied before batch scaling)
        --batch <int>: per-GPU (per-rank) batch. Global batch = per-GPU * world_size. When the
            global batch exceeds NBS_CANONICAL (512), lr0 and warmup_epochs scale linearly and
            nbs is raised to the global batch so wd_eff is invariant.
        --sample_t <float>: per-source temperature for ConcatDataset sampling. 0=uniform (default,
            existing behavior), 0.5=sqrt-balanced (EUPE / DINOv3 convention), 1=fully balanced.
            Active only when the dataset is a ConcatDataset (multi-source ``data=`` arg).
        --optimizer <name>: ultralytics optimizer name (default ``AdamW``). ``MuSGD`` swaps in
            Muon-based updates for distillation ablations. Recipe ``beta2`` is ignored when non-AdamW.
        --normalize_teacher_input: presence-only flag (no value). When set, convert the pipeline's ImageNet-normalized
            input to each teacher's training-time distribution: no-op for EUPE/DINOv3 (which already match ImageNet
            stats), SigLIP-style ``2x - 1`` for SigLIP2/MoonViT/SAM3. Default off matches all existing phase1 anchors.
            On resume, inherits from the checkpoint when not re-passed.
        --student_scales <csv>: comma-joined student input sizes for multi-scale distillation, e.g. "224,448,640".
            The loader serves the largest scale (genuine detail); preprocess round-robins the student size per step
            while the teacher stays at its native res. Trains the frozen backbone on the higher token counts it meets
            at detection resolution (640 -> 20x20 P5 vs 224 -> 7x7). Unset = single-scale at ``imgsz`` (legacy).
    """
    args = argv[1:]
    args, resume = _pop_flag(args, "--resume")
    args, cos_w = _pop_flag(args, "--cos_weight")
    args, l1_w = _pop_flag(args, "--l1_weight")
    args, cls_l1_str = _pop_flag(args, "--cls_l1", is_bool=True)
    args, lr_override = _pop_flag(args, "--lr")
    args, batch_override = _pop_flag(args, "--batch")
    args, nbs_override = _pop_flag(args, "--nbs")  # pin effective (accumulated) batch; lr/warmup scale off it
    args, fork_from = _pop_flag(args, "--fork_from")  # format: <parent_run_id>:<fork_step>
    args, distill_path = _pop_flag(args, "--distill_path")
    args, adaptor_arch = _pop_flag(args, "--adaptor_arch")
    args, sample_t_str = _pop_flag(args, "--sample_t")
    args, optimizer = _pop_flag(args, "--optimizer")
    args, norm_in_str = _pop_flag(args, "--normalize_teacher_input", is_bool=True)
    args, loss_type = _pop_flag(args, "--loss_type")
    args, student_scales = _pop_flag(args, "--student_scales")  # e.g. "224,448,640" (R1 multi-scale)
    args, high_res_final_epochs = _pop_flag(args, "--high_res_final_epochs")  # "<imgsz>:<epochs>" e.g. "384:12"
    args, _hires_legacy = _pop_flag(args, "--hires_tail")  # legacy alias for --high_res_final_epochs

    cos_weight = float(cos_w) if cos_w else 0.9
    l1_weight = float(l1_w) if l1_w else 0.1
    cls_l1 = bool(cls_l1_str)
    distill_path = distill_path or "adaptor"
    adaptor_arch = adaptor_arch or "mlp"
    sample_t = float(sample_t_str) if sample_t_str else 0.0
    optimizer = optimizer or "AdamW"
    normalize_teacher_input = bool(norm_in_str)
    loss_type = loss_type or "cos_l1"
    student_scales = student_scales or None
    high_res_final_epochs = high_res_final_epochs or _hires_legacy or None

    if resume:
        resume = paths.patch_resume(resume)
    resume_args = _load_train_args(resume) if resume else {}

    # Bool flags are presence-only: ``_pop_flag(is_bool=True)`` returns ``""`` when absent, so there's no way for a
    # resume to express "stay True". Inherit from the checkpoint when the CLI didn't re-pass the flag, so the drift
    # guard below doesn't fire spuriously on every resume of a normalize-on run.
    if resume_args and not norm_in_str:
        normalize_teacher_input = bool(resume_args.get("normalize_teacher_input", False))

    gpu = args[0] if args else "0"
    teachers = args[1] if len(args) > 1 else resume_args.get("teachers", "eupe:vitb16")
    name = (
        args[2] if len(args) > 2 else resume_args.get("name", f"phase1-{teachers.replace(':', '-').replace('+', '_')}")
    )
    recipe = args[3] if len(args) > 3 else "default"
    model_yaml = args[4] if len(args) > 4 else "yolo26s-cls.yaml"
    data = args[5] if len(args) > 5 else resume_args.get("data", DATA_7SRC_DEFAULT)
    epochs = int(args[6]) if len(args) > 6 else resume_args.get("epochs")
    r = RECIPES[recipe]

    # Resume drift guard: refuse silent switches that corrupt mid-run state — distill_path /
    # adaptor_arch change graph topology + loss_items labels; data change invalidates the run.
    if resume_args:
        for key, now, default in (
            ("distill_path", distill_path, "adaptor"),
            ("adaptor_arch", adaptor_arch, "mlp"),
            ("data", data, DATA_7SRC_DEFAULT),
            ("sample_t", sample_t, 0.0),
            ("optimizer", optimizer, "AdamW"),
            ("normalize_teacher_input", normalize_teacher_input, False),
            ("loss_type", loss_type, "cos_l1"),
            ("student_scales", student_scales, None),
            ("high_res_final_epochs", high_res_final_epochs, None),
        ):
            prev = resume_args.get(key, default)
            if now != prev:
                raise ValueError(
                    f"Refusing resume: --{key} mismatch (ckpt={prev!r} vs cli={now!r}). "
                    f"Either drop the flag or start a fresh run."
                )
        # Device guard: check_resume's whitelist behavior is version-brittle; bake device into
        # the checkpoint explicitly to avoid silent CLI vs ckpt mismatches on resume.
        prev_device = str(resume_args.get("device", "0"))
        if str(gpu) != prev_device:
            raise ValueError(
                f"Refusing resume: device mismatch (ckpt={prev_device!r} vs cli={gpu!r}). "
                f"To resume on different GPUs, bake the new device into the checkpoint first:\n"
                f'  python -c "from callbacks.paths import patch_resume; '
                f"patch_resume('{resume}', device='{gpu}')\"\n"
                f"Then re-run with the same --resume path."
            )

    world_size = len(gpu.split(",")) if "," in gpu else 1
    global_batch = (
        int(batch_override) * world_size if batch_override else int(resume_args.get("batch", 64 * world_size))
    )
    # nbs = effective (post-accumulation) batch. Default rises to the global step-batch (floored at
    # NBS_CANONICAL); --nbs pins it higher so a memory-capped micro-batch still reaches a target effective
    # batch via gradient accumulation (trainer.py:281). lr0/warmup scale off nbs (the effective batch), so a
    # small micro-batch + --nbs matches the recipe LR of a same-effective-batch run with no manual --lr pin.
    nbs = int(nbs_override) if nbs_override else max(global_batch, NBS_CANONICAL)
    scale = max(1.0, nbs / NBS_CANONICAL)
    lr0 = float(lr_override or r["lr0"]) * scale
    warmup_epochs = r["warmup_epochs"] * scale

    # A .pt model arg forks a finished run's trained backbone: pretrained=True then skips the
    # reset_parameters() wipe in ImageEncoderTrainer.get_model that would re-randomize the loaded
    # weights back to a cold start. A .yaml build stays pretrained=False (fresh init).
    fork_pretrained = str(model_yaml).endswith(".pt")
    model = YOLO(model_yaml)
    # grad_clip, beta2, nfs_sync registered inside ImageEncoderTrainer (survives DDP respawn).
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            teachers=teachers,
            recipe=recipe,
            cos_weight=cos_weight,
            l1_weight=l1_weight,
            cls_l1=cls_l1,
            distill_path=distill_path,
            adaptor_arch=adaptor_arch,
            sample_t=sample_t,
            optimizer=optimizer,
            normalize_teacher_input=normalize_teacher_input,
            loss_type=loss_type,
            student_scales=student_scales,
            high_res_final_epochs=high_res_final_epochs,
            grad_clip=r["grad_clip"],
            beta2=r["beta2"],
            wandb_group="distill",
        ),
    )
    train_args = dict(
        trainer=ImageEncoderTrainer,
        teachers=teachers,
        data=data,
        knn_eval="/data/shared-datasets/imagenet",
        normalize_teacher_input=normalize_teacher_input,
        cos_weight=cos_weight,
        l1_weight=l1_weight,
        cls_l1=cls_l1,
        distill_path=distill_path,
        adaptor_arch=adaptor_arch,
        sample_t=sample_t,
        loss_type=loss_type,
        student_scales=student_scales,
        high_res_final_epochs=high_res_final_epochs,
        device=gpu,
        **paths.run_paths(name),
        epochs=epochs or r["epochs"],
        batch=global_batch,
        imgsz=224,
        patience=20,
        nbs=nbs,
        cos_lr=True,
        lr0=lr0,
        lrf=0.01,
        momentum=r["momentum"],
        weight_decay=r["weight_decay"],
        grad_clip=r["grad_clip"],
        beta2=r["beta2"],
        warmup_epochs=warmup_epochs,
        warmup_bias_lr=0,
        dropout=0,
        optimizer=optimizer,
        pretrained=fork_pretrained,
        amp=True,
        seed=0,
        deterministic=True,
        fliplr=0.5,
        # Single-scale distill is teacher-compute-bound, so 2 workers suffice. Multi-scale loads every
        # batch at the largest scale (e.g. 640), making CPU augmentation ~8x heavier per image, so raise
        # to 4 (the shared-NFS cap; 8+ triggers an EPERM remount) to keep the teacher forward fed.
        workers=4 if student_scales else 2,
        nfs_sync=True,
    )
    # Recipe-driven aug overrides — applied only when present so legacy recipes inherit
    # Ultralytics's DEFAULT_CFG (auto_augment=randaugment, erasing=0.4, hsv_h=0.015, hsv_s=hsv_v=0.4).
    # Reference recipes (DINOv3 / EUPE / UNIC / DUNE) explicitly disable RandAugment + RandomErasing
    # and rely on a hand-tuned photometric stack — see RECIPES["dinov3"] docstring above.
    for k in ("wd_end", "auto_augment", "erasing", "hsv_h", "hsv_s", "hsv_v", "grayscale", "gaussian_blur", "solarize"):
        if k in r:
            train_args[k] = r[k]
    if resume:
        train_args["resume"] = resume
    if fork_from:
        parent_id, fork_step = fork_from.split(":")
        wandb_config.fork_and_attach(parent_id, int(fork_step), name)
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
