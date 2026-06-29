#!/usr/bin/env python
"""CE-on-ImageNet supervised pretrain matching exp5b-ce-baseline (single-GPU only).

Recipe: byte-exact match to exp5b-ce-baseline (encoder-distillation.md baselines
table line: batch=256 nbs=256 MuSGD lr=0.1 muon_w=0.1 warmup_epochs=0 cos_lr).
Epochs compressed 200 -> 114 to match the phase-1 distill epoch budget. Output
weights are intended as arch-matched CE baselines for downstream finetuning.

Usage:
    python run_cls_imagenet.py <gpu> <model_yaml> <name>
    python run_cls_imagenet.py <gpu> --resume <last.pt>

Flags:
    --resume <path>: resume from checkpoint (auto-calls paths.patch_resume).
    --data <path>: override ImageNet root (e.g. local /data/datasets/imagenet
        when a host has rsynced a local copy; default falls through to NFS).
    --lr <float>: override recipe lr0 (default 0.1; not auto-scaled here since
        runner is single-GPU and there is no canonical batch to scale against).
    --batch <int>: override per-GPU batch (default 256). nbs stays at 256.
    --epochs <int>: override epoch budget (default 114).
    --tags <csv>: override wandb tags (comma-separated). Defaults derived from
        model_yaml: imagenet-pretrain, ce-baseline-114ep, exp5b-recipe,
        yolo26-{conv|fastvit}, scale-{s|l}.
    --notes <str>: override wandb run notes (markdown string). Defaults to a
        per-arch summary of recipe, reference top-1, single-gpu constraint,
        and data path.

Single-GPU only: muon_w + nfs_sync + wandb_config callbacks are registered via
model.add_callback() and silently no-op under DDP respawn (utils/dist.py:79
serializes the overrides dict, not the model callback list). For multi-GPU,
subclass ClassificationTrainer and register inside __init__ instead.
"""
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch

from callbacks import muon_w, nfs_sync, paths, wandb_config
from ultralytics import YOLO


def _pop_flag(argv: list[str], flag: str, is_bool: bool = False) -> tuple[list[str], str]:
    """Pop a --flag [value] pair from argv, return (remaining_argv, value).

    Matches the helper used in run_enc_distill_phase{1,2}.py for CLI consistency.
    """
    if flag not in argv:
        return argv, ""
    i = argv.index(flag)
    if is_bool:
        return argv[:i] + argv[i + 1:], "true"
    return argv[:i] + argv[i + 2:], argv[i + 1]


def _load_train_args(resume: str) -> dict:
    """Load saved training arguments from a checkpoint."""
    return torch.load(Path(resume), map_location="cpu", weights_only=False)["train_args"]


def main(argv: list[str]) -> None:
    """Launch a fresh CE-on-ImageNet 114ep run or resume from a checkpoint.

    Args:
        argv: [gpu, model_yaml, name] --resume <path>, --data <path>, --lr <float>, --batch <int>, --epochs <int>
    """
    args = argv[1:]
    args, resume = _pop_flag(args, "--resume")
    args, data_override = _pop_flag(args, "--data")
    args, lr_override = _pop_flag(args, "--lr")
    args, batch_override = _pop_flag(args, "--batch")
    args, epochs_override = _pop_flag(args, "--epochs")
    args, tags_override = _pop_flag(args, "--tags")
    args, notes_override = _pop_flag(args, "--notes")

    if resume:
        resume = paths.patch_resume(resume)
    resume_args = _load_train_args(resume) if resume else {}

    gpu = args[0] if args else "0"
    if "," in gpu:
        raise ValueError(
            f"Single-GPU only (gpu={gpu!r}). Callbacks added via model.add_callback() "
            f"silently no-op under DDP. For multi-GPU, subclass ClassificationTrainer "
            f"and register muon_w / wandb_config inside __init__."
        )

    model_yaml = args[1] if len(args) > 1 else resume_args.get("model", "yolo26s-cls.yaml")
    name = args[2] if len(args) > 2 else resume_args.get("name", "cls-imagenet-114ep")
    data = data_override or resume_args.get("data", "/data/shared-datasets/imagenet")
    lr0 = float(lr_override) if lr_override else 0.1
    batch = int(batch_override) if batch_override else 256
    epochs = int(epochs_override) if epochs_override else resume_args.get("epochs", 114)

    stem = Path(model_yaml).stem
    arch = next(
        (a for a in ("fastvit", "cls-attn") if a in stem),
        "vit" if "-vit-" in stem else "conv",
    )
    scale_match = re.match(r"yolo26([nslmx])", stem)
    scale = scale_match.group(1) if scale_match else "s"

    if tags_override:
        tags = [t.strip() for t in tags_override.split(",") if t.strip()]
    else:
        tags = ["imagenet-pretrain", "ce-baseline-114ep", "exp5b-recipe", f"yolo26-{arch}", f"scale-{scale}"]

    notes = notes_override or None

    model = YOLO(model_yaml)
    model.add_callback("on_train_start", muon_w.override(0.1))
    sync_start, sync_end = nfs_sync.setup(str(paths.NFS_MIRROR_ROOT), interval_sec=paths.SYNC_INTERVAL_SEC)
    model.add_callback("on_train_start", sync_start)
    model.add_callback("on_train_end", sync_end)
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            recipe="exp5b-114ep",
            muon_w=0.1,
            wandb_group="ce-pretrain-imagenet",
            tags=tags,
            notes=notes,
        ),
    )
    train_args = dict(
        data=data,
        epochs=epochs,
        patience=100,
        batch=batch,
        imgsz=224,
        workers=2,
        pretrained=False,
        optimizer="MuSGD",
        seed=0,
        deterministic=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        lr0=lr0,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=0,
        warmup_momentum=0.8,
        warmup_bias_lr=0,
        nbs=256,
        mosaic=1,
        auto_augment="randaugment",
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        fliplr=0.5,
        device=gpu,
        **paths.run_paths(name),
    )
    if resume:
        train_args["resume"] = resume
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
