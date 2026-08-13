#!/usr/bin/env python
"""Phase 1: Encoder distillation pretraining on the 7-source mix, 12-source with the domain pools."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # before torch: BLAS pools size at init, ignore torch.set_num_threads

import torch

from callbacks import paths, wandb_config
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train_image_encoder import KNN_EVERY_DEFAULT, ImageEncoderTrainer
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import YAML

# Reference global step-batch the recipes' lr0 and warmup_epochs are tuned for. When
# per_gpu_batch * world_size exceeds this, lr0 and warmup_epochs scale linearly and nbs rises
# to the global batch so wd_eff stays at the recipe value.
NBS_CANONICAL = 512

_PHASE1_RECIPE = Path(_REPO_ROOT) / "cfg" / "recipes" / "phase1.yaml"

DATA_7SRC_DEFAULT = "/data/shared-datasets/imagenet,/data/shared-datasets/coco,/data/shared-datasets/yoloe26_data/Objects365v1/images/train,/data/shared-datasets/yoloe26_data/mixed_grounding/gqa/paired/train,/data/shared-datasets/yoloe26_data/flickr/paired/train,/data/shared-datasets/DOTAv1-split/paired/train,/data/shared-datasets/SODA-A-split/images/train"


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


def _load_source_args(checkpoint: str) -> dict:
    """Load source arguments from adjacent YAML without deserializing checkpoint weights.

    Args:
        checkpoint (str): Source checkpoint path.

    Returns:
        (dict): Saved training arguments.
    """
    checkpoint = Path(checkpoint)
    args_yaml = checkpoint.parent.parent / "args.yaml"
    return (
        YAML.load(args_yaml)
        if checkpoint.parent.name == "weights" and args_yaml.exists()
        else _load_train_args(checkpoint)
    )


def main(argv: list[str]) -> None:
    """Launch a fresh phase 1 run or resume from a checkpoint.

    Args:
        argv: [gpu, teachers, name, recipe, model_yaml, data, epochs]
        --resume <path>: resume from checkpoint
        --cos_weight <float>: cosine loss weight (default 0.9)
        --l1_weight <float>: smooth L1 loss weight (default 0.1)
        --cls_l1: add smooth L1 to CLS token loss (default False)
        --loss_type <str>: patch loss "cos_l1" (default, 0.9cos+0.1L1) or "l2" (pure MSE on un-normalized features)
        --gram_weight <float>: add DINOv3 Gram loss on raw P5 patch similarities. Default 0 disables it.
        --lr <float>: override recipe lr0 (applied before batch scaling)
        --batch <int>: per-GPU (per-rank) batch. Global batch = per-GPU * world_size. When the
            global batch exceeds NBS_CANONICAL (512), lr0 and warmup_epochs scale linearly and
            nbs is raised to the global batch so wd_eff is invariant.
        --sample_t <float>: per-source temperature for ConcatDataset sampling. 0=uniform (default,
            existing behavior), 0.5=sqrt-balanced (EUPE / DINOv3 convention), 1=fully balanced.
            Active only when the dataset is a ConcatDataset (multi-source ``data=`` arg).
        --optimizer <name>: ultralytics optimizer name (default ``AdamW``). ``MuSGD`` swaps in
            Muon-based updates for distillation ablations. Recipe ``beta2`` is ignored when non-AdamW.
        --proj_hidden_dim <1280|1536>: override the profile's adaptor MLP hidden width.
        --normalize_teacher_input: presence-only flag (no value). When set, convert the pipeline's ImageNet-normalized
            input to each teacher's training-time distribution: no-op for EUPE/DINOv3 (which already match ImageNet
            stats), SigLIP-style ``2x - 1`` for SigLIP2/MoonViT/SAM3. Default off matches all existing phase1 anchors.
            On resume, inherits from the checkpoint when not re-passed.
        --high_res_final_epochs <imgsz:epochs>: e.g. "640:12" runs the student at <imgsz> for the last <epochs>
            epochs (DINOv3 high-resolution adaptation) so its frozen P5 attention meets the larger token count it
            will see at detection resolution. DINOv3 and EUPE teachers use the same size in that tail.
            ``--hires_tail`` is the legacy alias. Unset = student runs at ``imgsz`` throughout.
        --eupe_multires: start a separate 15-epoch EUPE Stage 3-style post-training run from the positional ``.pt``.
            Uses independent student/teacher scales from 256, 384, and 512, effective batch 512, lr0 2e-4, and wd 0.02.
        --parent_wandb_id <id>: original Phase 1 W&B run recorded as lineage on a new multi-resolution run.
        --knn_every <epochs>: run ImageNet kNN every N epochs. Default 10.
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
    args, proj_hidden_dim_str = _pop_flag(args, "--proj_hidden_dim")
    args, sample_t_str = _pop_flag(args, "--sample_t")
    args, optimizer = _pop_flag(args, "--optimizer")
    args, norm_in_str = _pop_flag(args, "--normalize_teacher_input", is_bool=True)
    args, loss_type = _pop_flag(args, "--loss_type")
    args, gram_weight_str = _pop_flag(args, "--gram_weight")
    args, high_res_final_epochs = _pop_flag(args, "--high_res_final_epochs")  # "<imgsz>:<epochs>" e.g. "384:12"
    args, _hires_legacy = _pop_flag(args, "--hires_tail")  # legacy alias for --high_res_final_epochs
    args, eupe_multires_str = _pop_flag(args, "--eupe_multires", is_bool=True)
    args, parent_wandb_id = _pop_flag(args, "--parent_wandb_id")
    args, knn_every_str = _pop_flag(args, "--knn_every")
    if unknown_flags := [arg for arg in args if arg.startswith("--")]:
        raise ValueError(f"Unknown arguments: {', '.join(unknown_flags)}")

    post_training_overrides = [
        flag
        for value, flag in (
            (cos_w, "--cos_weight"),
            (l1_w, "--l1_weight"),
            (cls_l1_str, "--cls_l1"),
            (distill_path, "--distill_path"),
            (adaptor_arch, "--adaptor_arch"),
            (proj_hidden_dim_str, "--proj_hidden_dim"),
            (sample_t_str, "--sample_t"),
            (optimizer, "--optimizer"),
            (norm_in_str, "--normalize_teacher_input"),
            (loss_type, "--loss_type"),
            (gram_weight_str, "--gram_weight"),
        )
        if value
    ]
    cos_weight = float(cos_w) if cos_w else 0.9
    l1_weight = float(l1_w) if l1_w else 0.1
    cls_l1 = bool(cls_l1_str)
    distill_path = distill_path or "adaptor"
    adaptor_arch = adaptor_arch or "mlp"
    sample_t = float(sample_t_str) if sample_t_str else 0.0
    optimizer = optimizer or "AdamW"
    normalize_teacher_input = bool(norm_in_str)
    loss_type = loss_type or "cos_l1"
    gram_weight = float(gram_weight_str) if gram_weight_str else 0.0
    high_res_final_epochs = high_res_final_epochs or _hires_legacy or None
    eupe_multires = bool(eupe_multires_str)
    recipe_cfg = YAML.load(_PHASE1_RECIPE)
    multires_recipe = recipe_cfg["post_training"]["eupe_multires"] if eupe_multires else {}

    if resume:
        resume = paths.patch_resume(resume)
    resume_args = _load_train_args(resume) if resume else {}

    gpu = args[0] if args else "0"
    teachers = args[1] if len(args) > 1 else resume_args.get("teachers", "eupe:vitb16")
    name = (
        args[2] if len(args) > 2 else resume_args.get("name", f"phase1-{teachers.replace(':', '-').replace('+', '_')}")
    )
    recipe = args[3] if len(args) > 3 else "default"
    model_yaml = args[4] if len(args) > 4 else "yolo26s-cls.yaml"
    data = args[5] if len(args) > 5 else resume_args.get("data", DATA_7SRC_DEFAULT)
    epochs = int(args[6]) if len(args) > 6 else resume_args.get("epochs")
    r = {**recipe_cfg["defaults"], **recipe_cfg["profiles"][recipe]}
    checkpoint_args = resume_args or (_load_source_args(model_yaml) if str(model_yaml).endswith(".pt") else {})
    proj_hidden_dim = int(
        proj_hidden_dim_str
        or (checkpoint_args.get("proj_hidden_dim", 1280) if checkpoint_args else r["proj_hidden_dim"])
    )
    if proj_hidden_dim not in (1280, 1536):
        raise ValueError(f"--proj_hidden_dim must be 1280 or 1536, got {proj_hidden_dim}")
    standardize_teacher_outputs = bool(
        checkpoint_args.get("standardize_teacher_outputs", False)
        if checkpoint_args
        else r["standardize_teacher_outputs"]
    )
    knn_every = int(knn_every_str) if knn_every_str else int(resume_args.get("knn_every", KNN_EVERY_DEFAULT))
    if knn_every < 1:
        raise ValueError(f"--knn_every must be positive, got {knn_every}")

    # Presence-only flags cannot express a saved True value. With no explicit override, resumes inherit the checkpoint.
    if resume_args and not norm_in_str:
        normalize_teacher_input = bool(resume_args.get("normalize_teacher_input", False))
    if resume_args and not gram_weight_str:
        gram_weight = float(resume_args.get("gram_weight", 0.0))

    post_training_args = {}
    if eupe_multires:
        if resume or fork_from or high_res_final_epochs:
            raise ValueError("--eupe_multires requires a fresh run without --resume, --fork_from, or high-res tail")
        if not str(model_yaml).endswith(".pt"):
            raise ValueError("--eupe_multires requires a finished Phase 1 .pt checkpoint as the positional model")
        if len(args) <= 2:
            raise ValueError("--eupe_multires requires a new run name as the third positional argument")
        if epochs is not None and epochs != multires_recipe["epochs"]:
            raise ValueError(f"--eupe_multires fixes post-training to {multires_recipe['epochs']} epochs, got {epochs}")
        if lr_override:
            raise ValueError(f"--eupe_multires fixes lr0 at {multires_recipe['lr0']} and does not accept --lr")
        if nbs_override and int(nbs_override) != multires_recipe["nbs"]:
            raise ValueError(f"--eupe_multires fixes effective batch at {multires_recipe['nbs']}")
        if post_training_overrides:
            raise ValueError(
                f"--eupe_multires inherits {', '.join(post_training_overrides)} from its checkpoint. Drop the override."
            )
        checkpoint = Path(model_yaml).resolve()
        post_training_args = checkpoint_args
        parent_teachers = post_training_args.get("teachers", teachers)
        parent_data = post_training_args.get("data", data)
        if len(args) > 1 and teachers != parent_teachers:
            raise ValueError(f"Post-training teacher mismatch: checkpoint={parent_teachers!r} vs cli={teachers!r}")
        if len(args) > 5 and data != parent_data:
            raise ValueError(f"Post-training data mismatch: checkpoint={parent_data!r} vs cli={data!r}")
        teachers, data = parent_teachers, parent_data
        if teachers not in {"dinov3:vitl16", "dinov3:convnextl"}:
            raise ValueError(f"--eupe_multires does not support {teachers!r}")
        if post_training_args.get("distill_path", "adaptor") != "adaptor":
            raise ValueError("--eupe_multires requires the P5 adaptor loss path, not feat_map/P4 supervision")
        if post_training_args.get("optimizer", "AdamW") != "AdamW":
            raise ValueError("--eupe_multires requires an AdamW Phase 1 checkpoint")

        cos_weight = float(post_training_args.get("cos_weight", 0.9))
        l1_weight = float(post_training_args.get("l1_weight", 0.1))
        cls_l1 = bool(post_training_args.get("cls_l1", False))
        distill_path = post_training_args.get("distill_path", "adaptor")
        adaptor_arch = post_training_args.get("adaptor_arch", "mlp")
        proj_hidden_dim = int(post_training_args.get("proj_hidden_dim", 1280) or 1280)
        sample_t = float(post_training_args.get("sample_t", 0.0))
        normalize_teacher_input = bool(post_training_args.get("normalize_teacher_input", False))
        standardize_teacher_outputs = bool(post_training_args.get("standardize_teacher_outputs", False))
        loss_type = post_training_args.get("loss_type", "cos_l1")
        gram_weight = float(post_training_args.get("gram_weight", 0.0))
        optimizer = "AdamW"
        parent_wandb_id = parent_wandb_id or wandb_config.resolve_run_id_by_name(checkpoint.parents[1].name)
    elif parent_wandb_id:
        raise ValueError("--parent_wandb_id is only valid with --eupe_multires")

    # Resume drift guard: refuse silent switches that corrupt mid-run state — distill_path /
    # adaptor_arch change graph topology + loss_items labels; data change invalidates the run.
    if resume_args:
        for key, now, default in (
            ("distill_path", distill_path, "adaptor"),
            ("adaptor_arch", adaptor_arch, "mlp"),
            ("proj_hidden_dim", proj_hidden_dim, 1280),
            ("data", data, DATA_7SRC_DEFAULT),
            ("sample_t", sample_t, 0.0),
            ("optimizer", optimizer, "AdamW"),
            ("normalize_teacher_input", normalize_teacher_input, False),
            ("loss_type", loss_type, "cos_l1"),
            ("gram_weight", gram_weight, 0.0),
            ("high_res_final_epochs", high_res_final_epochs, None),
            ("knn_every", knn_every, KNN_EVERY_DEFAULT),
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
    if eupe_multires:
        global_batch = int(batch_override) * world_size if batch_override else multires_recipe["batch"]
        if global_batch > multires_recipe["nbs"]:
            raise ValueError(
                f"--eupe_multires global micro-batch cannot exceed its effective batch of "
                f"{multires_recipe['nbs']}, got {global_batch}"
            )
        schedule = {**multires_recipe, "batch": global_batch}
        print(f"[recipe] {_PHASE1_RECIPE.name}:eupe_multires -> {schedule}")
    else:
        global_batch = (
            int(batch_override) * world_size if batch_override else int(resume_args.get("batch", 64 * world_size))
        )
        # nbs = effective batch after gradient accumulation. --nbs pins it so a memory-capped micro-batch still
        # trains at the target effective batch, and lr0/warmup scale off it.
        nbs = int(nbs_override) if nbs_override else max(global_batch, NBS_CANONICAL)
        scale = max(1.0, nbs / NBS_CANONICAL)
        schedule = {
            "epochs": epochs or r["epochs"],
            "batch": global_batch,
            "imgsz": 224,
            "nbs": nbs,
            "lr0": float(lr_override or r["lr0"]) * scale,
            "warmup_epochs": r["warmup_epochs"] * scale,
            "weight_decay": r["weight_decay"],
        }
    momentum_v = post_training_args.get("momentum", r["momentum"])
    grad_clip_v = post_training_args.get("grad_clip", r["grad_clip"])
    beta2_v = post_training_args.get("beta2", r["beta2"])

    # A .pt model arg forks a finished run's trained backbone: pretrained=True then skips the
    # reset_parameters() wipe in ImageEncoderTrainer.get_model that would re-randomize the loaded
    # weights back to a cold start. A .yaml build stays pretrained=False (fresh init).
    fork_pretrained = str(model_yaml).endswith(".pt")
    if not resume and not fork_pretrained and not guess_model_scale(model_yaml):
        raise SystemExit(
            f"[phase1] fresh yaml run needs an explicit scale letter in the model name: {model_yaml!r}. Relaunch as "
            f"e.g. yolo26x-...-cls.yaml (the x-file need not exist, Ultralytics unifies it). A scale-less name silently "
            f"binds to the scales-block's first key."
        )
    model = YOLO(model_yaml)
    post_training_log = (
        {
            "stage": "eupe_multires",
            "multires_sizes": multires_recipe["multires_sizes"],
            "pretrained_from": model_yaml,
            "parent_wandb_id": parent_wandb_id or None,
            "tags": ["eupe-multires"],
        }
        if eupe_multires
        else {}
    )
    # beta2 is registered inside ImageEncoderTrainer, and nfs_sync starts in its _setup_train, so both
    # survive DDP respawn. grad_clip and muon/sgd are plain train args now, read straight from self.args.
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
            proj_hidden_dim=proj_hidden_dim,
            sample_t=sample_t,
            optimizer=optimizer,
            normalize_teacher_input=normalize_teacher_input,
            standardize_teacher_outputs=standardize_teacher_outputs,
            loss_type=loss_type,
            gram_weight=gram_weight,
            high_res_final_epochs=high_res_final_epochs,
            knn_every=knn_every,
            grad_clip=grad_clip_v,
            beta2=beta2_v,
            wandb_group="distill",
            **post_training_log,
        ),
    )
    train_args = dict(
        trainer=ImageEncoderTrainer,
        teachers=teachers,
        data=data,
        knn_eval="/data/shared-datasets/imagenet",
        knn_every=knn_every,
        normalize_teacher_input=normalize_teacher_input,
        standardize_teacher_outputs=standardize_teacher_outputs,
        cos_weight=cos_weight,
        l1_weight=l1_weight,
        cls_l1=cls_l1,
        distill_path=distill_path,
        adaptor_arch=adaptor_arch,
        proj_hidden_dim=proj_hidden_dim,
        sample_t=sample_t,
        loss_type=loss_type,
        gram_weight=gram_weight,
        high_res_final_epochs=high_res_final_epochs,
        device=gpu,
        **paths.run_paths(name),
        **schedule,
        patience=20,
        cos_lr=True,
        lrf=0.01,
        momentum=momentum_v,
        grad_clip=grad_clip_v,
        beta2=beta2_v,
        warmup_bias_lr=0,
        dropout=0,
        optimizer=optimizer,
        pretrained=fork_pretrained,
        amp=True,
        seed=0,
        deterministic=True,
        fliplr=0.5,
        # Distillation is teacher-compute-bound, so 2 workers keep the teacher forward fed.
        workers=2,
        nfs_sync=True,
    )
    # Recipe-driven aug overrides — applied only when present so legacy recipes inherit
    # Ultralytics's DEFAULT_CFG (auto_augment=randaugment, erasing=0.4, hsv_h=0.015, hsv_s=hsv_v=0.4).
    # Reference recipes (DINOv3 / EUPE / UNIC / DUNE) explicitly disable RandAugment + RandomErasing
    # and rely on a hand-tuned photometric stack defined by the selected Phase 1 profile.
    aug_source = post_training_args if eupe_multires else r
    for k in (
        "wd_end",
        "auto_augment",
        "erasing",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "color_jitter",
        "grayscale",
        "gaussian_blur",
        "solarize",
    ):
        if k in aug_source and not (eupe_multires and k == "wd_end"):
            train_args[k] = aug_source[k]
    if resume:
        train_args["resume"] = resume
    if fork_from:
        parent_id, fork_step = fork_from.split(":")
        wandb_config.fork_and_attach(parent_id, int(fork_step), name)
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
