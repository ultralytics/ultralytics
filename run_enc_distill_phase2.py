#!/usr/bin/env python
"""Phase 2: Downstream evaluation with distilled backbone.

Usage:
    python run_enc_distill_phase2.py <gpu> <phase1_weights> <mode> [name] [phase1_wandb_id] [epochs] [patience]
    python run_enc_distill_phase2.py <gpu> --resume <last.pt>

    mode: "inet_finetune" (ImageNet MuSGD ft), "inet_linear_probe" (ImageNet AdamW linear probe),
          "inet_adamw_finetune" (ImageNet AdamW ft), "coco_det_finetune" (COCO detection,
          yolo26s.pt-aligned recipe), "coco_det_finetune_frozen" (COCO det, frozen backbone),
          "coco_pose_finetune" (COCO pose), "dota_obb_finetune" (DOTA-v1.0 OBB,
          yolo26s-obb.pt-aligned recipe), "multi_det_finetune" (sequential per-dataset
          det fine-tune + val over a list of YOLO-format datasets; logs per-dataset and
          macro-averaged mAP to a CSV; same yolo26s.pt recipe per dataset)

Flags:
    --resume <path>: resume from checkpoint (all single-dataset modes)
    --fork_from <parent_id>:<fork_step>: wandb-fork continuation (all single-dataset modes)
    --lr <val>: override lr0. For coco_det_finetune, dota_obb_finetune, and
                multi_det_finetune this is the RECIPE lr0 at canonical bs and gets
                scaled by --batch. For other modes it is the FINAL lr0.
    --batch <int>: override batch size. For coco_det_finetune (canonical bs=128, nbs=64),
                dota_obb_finetune (canonical bs=32, nbs=64), and multi_det_finetune
                (canonical bs=128, nbs=64), also scales lr0, nbs, and warmup_epochs
                linearly so wd_eff and lr/sample stay invariant. For other modes it is
                applied as-is.
    --nbs <int>: explicit nbs override (coco_det_finetune, dota_obb_finetune,
                multi_det_finetune; bypasses auto-scaling).
    --datasets <path>: multi_det_finetune only. Either a file with one YOLO data.yaml
                path per line (#-comments and blanks ignored), or a directory scanned
                one level deep for ``*/data.yaml``.
"""

import sys
from pathlib import Path
import os

_REPO_ROOT = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch

from callbacks import grad_clip, muon_w, nfs_sync, paths, wandb_config
from ultralytics import YOLO


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


_COCO_DET_MODES = ("coco_det_finetune", "coco_det_finetune_frozen")
_SCALED_MODES = _COCO_DET_MODES + ("dota_obb_finetune",)
_SINGLE_GPU_DET_MODES = _COCO_DET_MODES + ("dota_obb_finetune", "multi_det_finetune")

_AUG_ARGS = dict(
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1,
    auto_augment="randaugment",
    erasing=0.4,
    crop_fraction=1,
)


def _infer_model_yaml(phase1_weights: str, head_suffix: str = "") -> str:
    """Resolve the task-specific model yaml from a phase-1 cls weights path.

    Reads ``args.yaml`` next to the weights and rewrites the ``-cls`` suffix into ``head_suffix`` (e.g. ``""`` for det,
    ``"-pose"`` for pose, ``"-obb"`` for obb).

    Args:
        phase1_weights (str): Path to a phase-1 ``weights/best.pt`` or ``weights/last.pt``.
        head_suffix (str, optional): Suffix to substitute for ``-cls``.

    Returns:
        (str): Model yaml name, e.g. ``"yolo26s.yaml"`` or ``"yolo26nexta-cls.yaml".replace(...)``.
    """
    cls_yaml = "yolo26s-cls.yaml"
    args_yaml = Path(phase1_weights).parent.parent / "args.yaml"
    if args_yaml.exists():
        for line in args_yaml.read_text().splitlines():
            if line.startswith("model:"):
                cls_yaml = line.split(":", 1)[1].strip()
                break
    return cls_yaml.replace("-cls", head_suffix)


def _build_det_train_args(
    epochs: int | None,
    patience: int | None,
    batch_override: str,
    lr_override: str,
    nbs_override: str,
) -> dict:
    """Build the yolo26s.pt-aligned detection recipe.

    Scales lr0, nbs, and warmup_epochs linearly with batch so lr/sample, effective weight decay (wd * batch / nbs), and
    warmup span in samples stay invariant when the canonical batch=128 is overridden.

    Args:
        epochs (int, optional): Override default epochs (70).
        patience (int, optional): Override default patience (100).
        batch_override (str): CLI --batch override.
        lr_override (str): CLI --lr override (RECIPE lr at canonical bs).
        nbs_override (str): CLI --nbs override (bypasses auto-scaling).

    Returns:
        (dict): train_args fragment containing the recipe (no data, no device, no save_dir).
    """
    batch = int(batch_override) if batch_override else 128
    scale = batch / 128.0
    nbs = max(1, int(nbs_override) if nbs_override else int(round(64 * scale)))
    base_lr = float(lr_override) if lr_override else 0.00038
    lr0 = base_lr * scale
    warmup = 0.98745 * scale
    return dict(
        epochs=epochs or 70,
        batch=batch,
        imgsz=640,
        nbs=nbs,
        patience=patience or 100,
        lr0=lr0,
        lrf=0.88219,
        momentum=0.94751,
        weight_decay=0.00027,
        warmup_epochs=warmup,
        warmup_momentum=0.54064,
        warmup_bias_lr=0.05684,
        cos_lr=False,
        close_mosaic=10,
        end2end=True,
        box=9.83241,
        cls=0.64896,
        dfl=0.95824,
        pose=12.0,
        kobj=1.0,
        mosaic=0.99182,
        mixup=0.05,
        cutmix=0.00082,
        copy_paste=0.40413,
        copy_paste_mode="flip",
        scale=0.9,
        fliplr=0.30393,
        translate=0.27484,
        degrees=0.00012,
        shear=0.00136,
        hsv_h=0.01315,
        hsv_s=0.35348,
        hsv_v=0.19383,
        erasing=0.4,
        auto_augment="randaugment",
        optimizer="MuSGD",
    )


def _resolve_dataset_list(datasets_arg: str) -> list[Path]:
    """Resolve --datasets argument to a sorted list of YOLO data.yaml paths.

    Args:
        datasets_arg (str): Either a file containing one yaml path per line (#-comments and blanks ignored), or a
            directory scanned one level deep for ``*/data.yaml``.

    Returns:
        (list[Path]): absolute, sorted, deduplicated paths to existing data.yaml files.
    """
    p = Path(datasets_arg).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"--datasets path does not exist: {p}")
    if p.is_dir():
        yamls = sorted(p.glob("*/data.yaml"))
    else:
        yamls = []
        for line in p.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                yamls.append(Path(s).expanduser().resolve())
    yamls = sorted(set(yamls))
    missing = [y for y in yamls if not y.exists()]
    if missing:
        raise SystemExit(f"missing data.yaml files: {missing}")
    if not yamls:
        raise SystemExit(f"--datasets resolved zero data.yaml files from {p}")
    return yamls


def _run_multi_det(
    gpu: str,
    phase1_weights: str,
    parent_name: str,
    phase1_wandb_id: str,
    epochs: int | None,
    patience: int | None,
    batch_override: str,
    lr_override: str,
    nbs_override: str,
    datasets_arg: str,
) -> None:
    """Sequentially train + val on a list of YOLO-format detection datasets.

    Per dataset: fresh YOLO(model_yaml) with backbone from phase1_weights, train using the canonical yolo26s.pt det
    recipe (see _build_det_train_args), then val. Each dataset is its own W&B run named ``{parent_name}-{basename}``.
    Aggregate metrics are written to ``{parent save_dir}/multi_results.csv`` and printed as a macro average at the end.

    Single-GPU only (same DDP-callback-loss caveat as other det modes).

    Args:
        gpu (str): Single GPU id (e.g. "0").
        phase1_weights (str): Path to backbone checkpoint for `pretrained=`.
        parent_name (str): Run name prefix; sub-runs append "-{basename}".
        phase1_wandb_id (str): Optional W&B parent ID forwarded to wandb_config.
        epochs (int, optional): Per-dataset epochs (default 70).
        patience (int, optional): Per-dataset patience (default 100).
        batch_override (str): CLI --batch override (scales lr/nbs/warmup).
        lr_override (str): CLI --lr override.
        nbs_override (str): CLI --nbs override.
        datasets_arg (str): Path to dataset list (file or directory). See _resolve_dataset_list.
    """
    if "," in gpu:
        raise SystemExit(
            "ERROR: mode='multi_det_finetune' requires a single GPU. DetectionTrainer drops "
            f"add_callback registrations under DDP. Got gpu={gpu!r}; pass a single id like '0'."
        )
    dataset_yamls = _resolve_dataset_list(datasets_arg)
    model_yaml = _infer_model_yaml(phase1_weights)

    parent_paths = paths.run_paths(parent_name)
    parent_save_dir = Path(parent_paths["project"]) / parent_paths["name"]
    parent_save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = parent_save_dir / "multi_results.csv"
    if not csv_path.exists():
        csv_path.write_text("dataset,map50,map50_95,fitness\n")

    print(f"[multi_det_finetune] parent={parent_name} datasets={len(dataset_yamls)} model={model_yaml}")
    print(f"[multi_det_finetune] aggregate csv -> {csv_path}")

    results = []
    for i, ds_yaml in enumerate(dataset_yamls, start=1):
        basename = ds_yaml.parent.name
        sub_name = f"{parent_name}-{basename}"
        print(f"\n=== [{i}/{len(dataset_yamls)}] {basename} -> {sub_name} ===")

        model = YOLO(model_yaml)
        model.add_callback("on_train_start", grad_clip.override(1.0))
        model.add_callback("on_train_start", muon_w.override(0.4355))
        sync_start, sync_end = nfs_sync.setup(str(paths.NFS_MIRROR_ROOT), interval_sec=paths.SYNC_INTERVAL_SEC)
        model.add_callback("on_train_start", sync_start)
        model.add_callback("on_train_end", sync_end)
        model.add_callback(
            "on_pretrain_routine_start",
            wandb_config.log_config(
                model=model_yaml,
                pretrained_from=phase1_weights,
                phase1_wandb_id=phase1_wandb_id,
                mode="multi_det_finetune",
                cls_to_det_remap=True,
                wandb_group="downstream-multi-det",
                parent_run=parent_name,
                dataset=basename,
            ),
        )
        det_args = _build_det_train_args(epochs, patience, batch_override, lr_override, nbs_override)
        train_args = dict(
            pretrained=phase1_weights,
            device=int(gpu),
            **paths.run_paths(sub_name),
            warmup_bias_lr=0,
            dropout=0,
            amp=True,
            seed=0,
            deterministic=True,
            workers=2,
            data=str(ds_yaml),
            **det_args,
        )
        model.train(**train_args)
        metrics = model.val()
        row = {
            "dataset": basename,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "fitness": float(metrics.fitness),
        }
        results.append(row)
        with csv_path.open("a") as f:
            f.write(f"{row['dataset']},{row['map50']:.4f},{row['map50_95']:.4f},{row['fitness']:.4f}\n")
        print(f"[done] {basename} mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f} fitness={row['fitness']:.4f}")

    macro = {k: sum(r[k] for r in results) / len(results) for k in ("map50", "map50_95", "fitness")}
    with csv_path.open("a") as f:
        f.write(f"MACRO,{macro['map50']:.4f},{macro['map50_95']:.4f},{macro['fitness']:.4f}\n")
    print(
        f"\n[multi_det_finetune] MACRO over {len(results)} datasets: "
        f"mAP50={macro['map50']:.4f} mAP50-95={macro['map50_95']:.4f} fitness={macro['fitness']:.4f}"
    )


def main(argv: list[str]) -> None:
    """Launch a fresh phase 2 run or resume from a checkpoint."""
    argv = argv[1:]
    argv, resume = _pop_flag(argv, "--resume")
    argv, fork_from = _pop_flag(argv, "--fork_from")
    argv, lr_override = _pop_flag(argv, "--lr")
    argv, batch_override = _pop_flag(argv, "--batch")
    argv, nbs_override = _pop_flag(argv, "--nbs")
    argv, scratch = _pop_flag(argv, "--scratch", is_bool=True)
    argv, datasets_arg = _pop_flag(argv, "--datasets")
    if resume:
        resume = paths.patch_resume(resume)
    resume_args = _load_train_args(resume) if resume else {}
    gpu = argv[0] if argv else "0"
    phase1_weights = (
        argv[1]
        if len(argv) > 1
        else resume_args.get("pretrained", "runs/classify/yolo-next-encoder/phase1-d7-dinov3-convnextb/weights/best.pt")
    )
    mode = argv[2] if len(argv) > 2 else ("inet_linear_probe" if resume_args.get("freeze") else "inet_finetune")
    if mode in _SINGLE_GPU_DET_MODES and "," in gpu:
        raise SystemExit(
            f"ERROR: mode={mode!r} requires a single GPU. Phase-2 DetectionTrainer/OBBTrainer "
            f"re-spawns workers under DDP and silently drops our model.add_callback registrations "
            f"(grad_clip/muon_w/nfs_sync would no-op). Got gpu={gpu!r}. Pass a single GPU id like '0'."
        )
    name = argv[3] if len(argv) > 3 else resume_args.get("name", f"phase2-{mode}-d7")
    phase1_wandb_id = argv[4] if len(argv) > 4 else ""
    epochs = int(argv[5]) if len(argv) > 5 else resume_args.get("epochs")
    patience = int(argv[6]) if len(argv) > 6 else resume_args.get("patience")

    if mode == "multi_det_finetune":
        if not datasets_arg:
            raise SystemExit("ERROR: mode='multi_det_finetune' requires --datasets <file|dir>.")
        if resume or fork_from:
            raise SystemExit("ERROR: --resume and --fork_from are not supported for multi_det_finetune.")
        _run_multi_det(
            gpu=gpu,
            phase1_weights=phase1_weights,
            parent_name=name,
            phase1_wandb_id=phase1_wandb_id,
            epochs=epochs,
            patience=patience,
            batch_override=batch_override,
            lr_override=lr_override,
            nbs_override=nbs_override,
            datasets_arg=datasets_arg,
        )
        return

    # Resume auto-fallback: pre-fill cli overrides from saved args so resume preserves the run's
    # lr/batch/nbs. Both the per-mode scaling blocks and the post-dispatch apply block read these
    # vars, so populating once here covers both code paths.
    if resume_args:
        batch_override = batch_override or (str(resume_args["batch"]) if "batch" in resume_args else "")
        lr_override = lr_override or (str(resume_args["lr0"]) if "lr0" in resume_args else "")
        nbs_override = nbs_override or (str(resume_args["nbs"]) if "nbs" in resume_args else "")

    if mode in ("coco_det_finetune", "coco_det_finetune_frozen", "coco_pose_finetune", "dota_obb_finetune"):
        head_suffix = {"coco_pose_finetune": "-pose", "dota_obb_finetune": "-obb"}.get(mode, "")
        model_yaml = _infer_model_yaml(phase1_weights, head_suffix)
    else:
        model_yaml = "yolo26s-cls.yaml"
    wandb_group = {
        "coco_det_finetune": "downstream-coco",
        "coco_pose_finetune": "downstream-coco-pose",
        "dota_obb_finetune": "downstream-dota-obb",
    }.get(mode, "downstream-imagenet")

    model = YOLO(model_yaml)
    # NOTE: C2PSA remap tested and abandoned (17.77% vs 28.02% without remap).
    # Standard pretrained= flow transfers backbone layers 0-8 via intersect_dicts.
    if mode == "inet_finetune":
        model.add_callback("on_train_start", muon_w.override(0.1))
    model.add_callback("on_train_start", grad_clip.override(1.0))
    sync_start, sync_end = nfs_sync.setup(str(paths.NFS_MIRROR_ROOT), interval_sec=paths.SYNC_INTERVAL_SEC)
    model.add_callback("on_train_start", sync_start)
    model.add_callback("on_train_end", sync_end)
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            pretrained_from=phase1_weights,
            phase1_wandb_id=phase1_wandb_id,
            mode=mode,
            cls_to_det_remap=mode == "coco_det_finetune",
            wandb_group=wandb_group,
        ),
    )
    train_args = dict(
        pretrained=phase1_weights,
        device=gpu if mode in ("coco_det_finetune", "dota_obb_finetune") else int(gpu),
        **paths.run_paths(name),
        cos_lr=True,
        warmup_bias_lr=0,
        dropout=0,
        amp=True,
        seed=0,
        deterministic=True,
        workers=2,
    )
    if mode == "inet_linear_probe":
        train_args.update(
            data="/data/shared-datasets/imagenet",
            epochs=epochs or 50,
            batch=256,
            imgsz=224,
            nbs=256,
            freeze=10,
            patience=patience or 10,
            lr0=1e-3,
            lrf=0.01,
            weight_decay=1e-3,
            warmup_epochs=1,
            optimizer="AdamW",
        )
    elif mode == "inet_adamw_finetune":
        train_args.update(
            data="/data/shared-datasets/imagenet",
            epochs=epochs or 50,
            batch=256,
            imgsz=224,
            nbs=256,
            patience=patience or 30,
            lr0=1e-3,
            lrf=0.01,
            weight_decay=1e-3,
            warmup_epochs=5,
            momentum=0.9,
            optimizer="AdamW",
            **_AUG_ARGS,
        )
    elif mode in _COCO_DET_MODES:
        det_args = _build_det_train_args(epochs, patience, batch_override, lr_override, nbs_override)
        print(
            f"[coco_det_finetune] batch={det_args['batch']} nbs={det_args['nbs']} lr0={det_args['lr0']:.5f} "
            f"warmup_epochs={det_args['warmup_epochs']:.3f} (scale={det_args['batch'] / 128.0:.2f}x vs canonical bs=128)"
        )
        train_args.update(data="coco.yaml", **det_args)
        # NOTE: sgd_w/cls_w/o2m/detach_epoch from yolo26s.pt recipe are not exposed
        # as train_args in our ultralytics checkout (cfg validator rejects). muon_w
        # is set via callback since it isn't in DEFAULT_CFG_DICT either.
        model.add_callback("on_train_start", muon_w.override(0.4355))
        if mode == "coco_det_finetune_frozen":
            train_args["freeze"] = 9
    elif mode == "coco_pose_finetune":
        train_args.update(
            data="coco-pose.yaml",
            epochs=epochs or 70,
            batch=128,
            imgsz=640,
            nbs=64,
            patience=patience or 30,
            lr0=0.00125,
            lrf=0.5,
            momentum=0.937,
            weight_decay=0.0007,
            warmup_epochs=1,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            optimizer="MuSGD",
            close_mosaic=5,
            cache="disk",
            cos_lr=False,
            pose=24,
            kobj=4.0,
            mosaic=1.0,
            mixup=0,
            copy_paste=0.0,
            scale=0.9,
            fliplr=0.5,
            degrees=0.0,
            shear=0.0,
            translate=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.4,
            auto_augment="randaugment",
        )
    elif mode == "dota_obb_finetune":
        # yolo26s-obb.pt canonical: batch=32, nbs=64, lr0=0.00125, warmup_epochs=1, degrees=180.
        # Scale linearly when --batch overrides canonical so lr/sample, wd_eff = wd*batch/nbs,
        # and warmup span (in samples) stay invariant.
        obb_batch = int(batch_override) if batch_override else 32
        obb_scale = obb_batch / 32.0
        obb_nbs = max(1, int(nbs_override) if nbs_override else int(round(64 * obb_scale)))
        obb_base_lr = float(lr_override) if lr_override else 0.00125
        obb_lr0 = obb_base_lr * obb_scale
        obb_warmup = 1.0 * obb_scale
        print(
            f"[dota_obb_finetune] batch={obb_batch} nbs={obb_nbs} lr0={obb_lr0:.5f} "
            f"warmup_epochs={obb_warmup:.3f} (scale={obb_scale:.2f}x vs canonical bs=32)"
        )
        train_args.update(
            data="DOTAv1.yaml",
            epochs=epochs or 50,
            batch=obb_batch,
            imgsz=1024,
            nbs=obb_nbs,
            patience=patience or 100,
            lr0=obb_lr0,
            lrf=0.5,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=obb_warmup,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=False,
            close_mosaic=5,
            end2end=True,
            box=7.5,
            cls=0.5,
            dfl=6,
            pose=12.0,
            kobj=1.0,
            mosaic=1.0,
            mixup=0.1,
            cutmix=0.0,
            copy_paste=0.0,
            copy_paste_mode="flip",
            scale=0.9,
            fliplr=0.5,
            translate=0.1,
            degrees=180,
            shear=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.4,
            auto_augment="randaugment",
            optimizer="MuSGD",
        )
        model.add_callback("on_train_start", muon_w.override(0.5))
    else:  # inet_finetune (default)
        train_args.update(
            data="/data/shared-datasets/imagenet",
            epochs=epochs or 50,
            batch=256,
            imgsz=224,
            nbs=256,
            patience=patience or 30,
            lr0=0.1,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            warmup_epochs=0,
            optimizer="MuSGD",
            **_AUG_ARGS,
        )
    # Resume drift guard: refuse silent data mismatch. The mode-inference at line 97 uses
    # freeze as a proxy and can land on a different mode than the saved run (e.g. resumed
    # coco_det_finetune defaults to inet_finetune). Fail loud rather than truncate the dataset.
    if resume_args and "data" in resume_args and train_args["data"] != resume_args["data"]:
        raise ValueError(
            f"Refusing resume: mode-implied data mismatch (ckpt={resume_args['data']!r} vs "
            f"mode={mode!r} → {train_args['data']!r}). Pass the correct mode positionally."
        )

    # lr/batch/nbs are handled per-mode for scaled modes (coco_det_finetune, dota_obb_finetune);
    # for other modes they apply as final values.
    if mode not in _SCALED_MODES:
        if lr_override:
            train_args["lr0"] = float(lr_override)
        if batch_override:
            train_args["batch"] = int(batch_override)
        if nbs_override:
            train_args["nbs"] = int(nbs_override)
    if resume:
        train_args["resume"] = resume
    if fork_from:
        parent_id, fork_step = fork_from.split(":")
        wandb_config.fork_and_attach(parent_id, int(fork_step), name)
    if scratch:
        train_args["pretrained"] = False
        print("[scratch] pretrained=False, backbone will be randomly initialized")
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
