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
          macro-averaged mAP to a CSV; ul33 table recipe, batch 64 and lr0 0.0015 by default)

Flags:
    --resume <path>: resume from checkpoint (all single-dataset modes)
    --fork_from <parent_id>:<fork_step>: wandb-fork continuation (all single-dataset modes)
    --lr <val>: override lr0. For coco_det_finetune, dota_obb_finetune, and multi_det_finetune
                this is the RECIPE lr0 at canonical bs and gets scaled by --batch. For other
                modes it is the FINAL lr0. multi_det_finetune defaults to lr0 0.0015 (the ul33
                table recipe) at any batch when --lr is omitted.
    --batch <int>: override batch size. For coco_det_finetune (canonical bs=128, nbs=64),
                dota_obb_finetune (canonical bs=32, nbs=64), and multi_det_finetune
                (default bs=64, nbs=32), also scales nbs and warmup_epochs linearly so
                wd_eff and warmup/sample stay invariant. coco and dota also scale lr0
                linearly. multi_det's default lr0 stays 0.0015 at any batch (an explicit
                --lr is still scaled). For other modes --batch is applied as-is.
    --nbs <int>: explicit nbs override (coco_det_finetune, dota_obb_finetune,
                multi_det_finetune; bypasses auto-scaling).
    --datasets <path>: multi_det_finetune only. Either a file with one YOLO data.yaml
                path per line (#-comments and blanks ignored), or a directory scanned
                one level deep for ``*/data.yaml``.
    --imgsz <int>: multi_det_finetune/teacher_frozen_det only. Override the canonical det
                imgsz (640), e.g. 224 to run the frozen backbone at its phase-1 grid.
"""

import os
import re
import shutil
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch

from callbacks import grad_clip, muon_w, nfs_sync, paths, wandb_config
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.nn.teacher_model import TEACHER_REGISTRY, safe_key
from ultralytics.utils import YAML

# teacher_frozen_det: frozen foundation teacher (yolo26-teacherdet.yaml layer 0) + trainable ViTDet pyramid + Detect,
# the frozen-feature detection ceiling. Supported = ImageNet-stat ViT/ConvNeXt teachers audited (2026-06-24) to run at
# det imgsz 640 with row-major square token grids and stats matching their phase-1 distillation. Other teachers each
# need per-teacher work first: siglip2:g learned pos-embed (needs interpolate_pos_encoding=True or a 384 lock),
# moonvit:so400m needs imgsz % 224, sam3:l needs set_imgsz + a 1008 registry fix + its missing asset, tips:* is unbuilt.
# dinov3:vit7b is correctness-valid but excluded until checkpoints strip the frozen teacher: at 6.7B its best.pt is
# ~13GB and nfs_sync would mirror that x16 datasets. The 86M-300M teachers below stay at 170-600MB, mirrored fine.
_TEACHERDET_YAML = str(Path(_REPO_ROOT) / "ultralytics" / "cfg" / "models" / "26" / "yolo26-teacherdet.yaml")
_TEACHER_FROZEN_DET_SUPPORTED = frozenset(
    {"eupe:vitb16", "eupe:vits16", "eupe:convnextb", "dinov3:vitb16", "dinov3:vitl16", "dinov3:convnextb"}
)
# det imgsz per teacher (default 640 for patch-16 ViT + ConvNeXt). siglip2/moonvit values are pre-staged for when
# those teachers join _TEACHER_FROZEN_DET_SUPPORTED; until then the .get() default 640 is what runs.
_TEACHER_DET_IMGSZ = {"siglip2:g": 384, "moonvit:so400m": 448}


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


def _is_cls_yaml(model) -> bool:
    """Return True when model names a classification yaml."""
    return bool(model) and Path(str(model)).suffix in ("", ".yaml", ".yml") and "-cls" in Path(str(model)).stem


def _checkpoint_cls_yaml(weights: Path) -> str:
    """Read classification yaml metadata from a checkpoint."""
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = ckpt.get("model")
    for yaml_file in (ckpt.get("train_args", {}).get("model"), getattr(model, "yaml", {}).get("yaml_file", "")):
        if _is_cls_yaml(yaml_file):
            return str(yaml_file)
    raise ValueError(f"Could not infer a classification yaml from checkpoint metadata: {weights}")


def _export_hf_token() -> None:
    """Export HF_TOKEN from .env so HF-gated teachers (dinov3 L/convnextb/7b) load. No-op if already set or no .env."""
    env = Path(".env")
    if os.environ.get("HF_TOKEN") or not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
            return


_COCO_DET_MODES = ("coco_det_finetune", "coco_det_finetune_frozen")
_SCALED_MODES = _COCO_DET_MODES + ("dota_obb_finetune",)
_SINGLE_GPU_DET_MODES = _COCO_DET_MODES + ("dota_obb_finetune", "multi_det_finetune", "teacher_frozen_det")

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
    """Resolve the task-specific model yaml from a weights path.

    Reads ``args.yaml`` next to the weights when the path follows the standard ``<run>/weights/<file>.pt`` layout
    (phase-1 distill checkpoints); otherwise derives the cls yaml from the weights filename so bare ultralytics weights
    like ``yolo26l-cls.pt`` pick up the right scale. The trailing ``-cls.yaml`` is rewritten to ``{head_suffix}.yaml``
    (e.g. ``""`` for det, ``"-pose"`` for pose, ``"-obb"`` for obb), and any ``-sppf`` arch tag is dropped since det
    yamls already carry SPPF (``yolo26x-cls-sppf.yaml`` -> ``yolo26x.yaml``).

    Args:
        phase1_weights (str): Path to a weights ``.pt`` file.
        head_suffix (str, optional): Suffix to substitute for ``-cls``.

    Returns:
        (str): Model yaml name, e.g. ``"yolo26s.yaml"`` or ``"yolo26l-obb.yaml"``.
    """
    w = Path(phase1_weights)
    cls_yaml = w.stem + ".yaml"
    if w.parent.name == "weights":
        args_yaml = w.parent.parent / "args.yaml"
        cls_yaml = YAML.load(args_yaml).get("model") if args_yaml.exists() else cls_yaml
        if not _is_cls_yaml(cls_yaml):
            cls_yaml = _checkpoint_cls_yaml(w)
    # Strip the `-cls` task suffix. Lookahead matches both end-position (`yolo26s-cls.yaml` -> `yolo26s.yaml`)
    # and middle-position when a custom arch suffix follows (`yolo26s-cls-attn.yaml` -> `yolo26s-attn.yaml`).
    out = re.sub(r"-cls(?=[-.])", head_suffix, cls_yaml, count=1)
    # Drop the `-sppf` tag: det yamls already carry SPPF, no `-sppf` det counterparts exist.
    out = re.sub(r"-sppf(?=[-.])", "", out, count=1)
    if "-cls" in out:
        raise ValueError(
            f"_infer_model_yaml: stripped -cls failed for {cls_yaml!r} -> {out!r} "
            f"(source weights: {phase1_weights}). Would route to ClassificationTrainer and crash."
        )
    return out


def _build_det_train_args(
    epochs: int | None,
    patience: int | None,
    batch_override: str,
    lr_override: str,
    nbs_override: str,
    default_batch: int = 128,
    default_lr0: float | None = None,
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
    batch = int(batch_override) if batch_override else default_batch
    scale = batch / 128.0
    nbs = max(1, int(nbs_override) if nbs_override else int(round(64 * scale)))
    # An explicit --lr is the recipe lr0 at bs=128, scaled by batch (all modes). Absent --lr, multi_det (ul33) uses a
    # fixed default_lr0 at any batch, while coco/dota fall back to their scaled 0.00038 base.
    if lr_override:
        lr0 = float(lr_override) * scale
    elif default_lr0 is not None:
        lr0 = default_lr0
    else:
        lr0 = 0.00038 * scale
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


# Default flat recipe for multi_det. Epochs/patience apply literally to every dataset (no dataset-size scaling).
# Pass positional epochs/patience to override. Macros at different budgets are not directly comparable, so
# _run_multi_det warns on any override. The flat default is 100 for the multi_det per-dataset suite. The
# single-dataset coco/dota modes keep _build_det_train_args' own default of 70.
_MULTI_DET_BASE_EPOCHS = 100
_MULTI_DET_BASE_PATIENCE = 100
# ul33 table recipe: batch 64 with lr0 held fixed at any batch (the coco/dota helper default is bs=128, lr0 scaled).
_MULTI_DET_BASE_BATCH = 64
_MULTI_DET_BASE_LR0 = 0.0015


def _dataset_train_stats(data_yaml: Path, batch: int) -> tuple[int, int]:
    """Count train images and iterations per epoch for one dataset (logging only).

    Args:
        data_yaml (Path): Dataset config whose `train:` entry resolves to an image dir or a `.txt` list.
        batch (int): Effective batch size for the iters-per-epoch estimate.

    Returns:
        (int): Number of training images.
        (int): Iterations per epoch at the given batch.
    """
    d = YAML.load(data_yaml)
    root = Path(d.get("path", data_yaml.parent))
    train = d.get("train", "images/train")
    train_path = Path(train) if Path(train).is_absolute() else root / train
    if train_path.is_file() and train_path.suffix == ".txt":
        with train_path.open() as f:
            n_imgs = sum(1 for line in f if line.strip() and not line.startswith("#"))
    else:
        n_imgs = sum(1 for p in train_path.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
    return n_imgs, max(1, (n_imgs + batch - 1) // batch)


def load_multi_results(parent_name: str, expected: list[str] | None = None) -> tuple[dict, dict]:
    """Read a multi_det aggregate CSV as the layout-agnostic source of truth for one parent run.

    Prefers the shared NFS mirror, falls back to the host-local copy. Raises on a missing file, a missing MACRO row, a
    duplicate dataset row, or any dataset in ``expected`` without a row, so a partial chain or a wrong-layout lookup
    fails loudly instead of silently treating absent values as zero (the failure mode that produced the inet-l 0.0000
    artifact when results were re-derived from wandb groups across the flat-vs-nested layout split).

    Args:
        parent_name (str): multi_det parent run name; the CSV lives at ``<root>/<parent_name>/multi_results.csv``.
        expected (list, optional): Dataset basenames that must each be present.

    Returns:
        (dict): Per-dataset metrics keyed by basename, each with map50, map50_95, fitness.
        (dict): The MACRO-row metrics with map50, map50_95, fitness.
    """
    for root in (paths.NFS_MIRROR_ROOT, paths.LOCAL_ROOT):
        csv_path = paths.multi_results_csv(parent_name, root)
        try:
            text = csv_path.read_text()
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(f"multi_results.csv not found for {parent_name!r} under NFS or local root")

    per_dataset = {}
    macro = None
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        name, map50, map50_95, fitness = line.split(",")
        row = {"map50": float(map50), "map50_95": float(map50_95), "fitness": float(fitness)}
        if name == "MACRO":
            macro = row
        elif name in per_dataset:
            raise ValueError(f"{csv_path}: duplicate row for dataset {name!r}")
        else:
            per_dataset[name] = row
    if macro is None:
        raise ValueError(f"{csv_path}: no MACRO row, the run is incomplete")
    missing = sorted(set(expected or ()) - per_dataset.keys())
    if missing:
        raise ValueError(f"{csv_path}: missing rows for {missing}")
    return per_dataset, macro


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
    freeze_override: str = "",
    imgsz_override: str = "",
    teacher_spec: str | None = None,
    seed: int = 0,
) -> None:
    """Sequentially train + val on a list of YOLO-format detection datasets.

    Per dataset: fresh YOLO(model_yaml) with backbone from phase1_weights, train using the canonical yolo26s.pt det
    recipe (see _build_det_train_args), then val. Each dataset is its own W&B run named ``{parent_name}-{basename}``.
    Aggregate metrics are written to ``{parent save_dir}/multi_results.csv`` (mirrored to the NFS run dir) and printed
    as a macro average at the end.

    Single-GPU only (same DDP-callback-loss caveat as other det modes).

    Args:
        gpu (str): Single GPU id (e.g. "0").
        phase1_weights (str): Path to backbone checkpoint for `pretrained=`.
        parent_name (str): Run name prefix; sub-runs append "-{basename}".
        phase1_wandb_id (str): Optional W&B parent ID forwarded to wandb_config.
        epochs (int, optional): Per-dataset epochs (default 100).
        patience (int, optional): Per-dataset patience (default 100).
        batch_override (str): CLI --batch override (scales lr/nbs/warmup).
        lr_override (str): CLI --lr override.
        nbs_override (str): CLI --nbs override.
        datasets_arg (str): Path to dataset list (file or directory). See _resolve_dataset_list.
        freeze_override (str): Distilled-student freeze depth. When set (and no teacher_spec), freezes det layers 0..N-1
            via the trainer freeze arg (e.g. 10 for yolo26l = transferred backbone 0-8 + SPPF 9), so only C2PSA + neck +
            Detect head train. Mirrors the frozen-teacher ceiling probe for the distilled backbone.
        teacher_spec (str, optional): Frozen-teacher registry key (e.g. "eupe:vitb16"). When set, runs the
            teacher_frozen_det mode: build yolo26-teacherdet.yaml with this teacher, freeze=1, no phase1 weights or
            parent push. When None, the standard distilled-student multi_det_finetune mode.
        seed (int, optional): Training seed for detection-head init and augmentation RNG. Default 0 reproduces prior
            runs, vary it to sample per-dataset run-to-run variance.
    """
    if "," in gpu:
        raise SystemExit(
            "ERROR: mode='multi_det_finetune' requires a single GPU. DetectionTrainer drops "
            f"add_callback registrations under DDP. Got gpu={gpu!r}; pass a single id like '0'."
        )
    dataset_yamls = _resolve_dataset_list(datasets_arg)
    parent_save_dir = paths.LOCAL_ROOT / parent_name
    parent_save_dir.mkdir(parents=True, exist_ok=True)
    if teacher_spec:
        # Inject the chosen teacher into a resolved copy of the teacherdet yaml (safe_key form; a colon would crash the
        # parse_model ast.literal_eval arg handler). Written to the parent dir as run provenance.
        cfg = YAML.load(_TEACHERDET_YAML)
        cfg["backbone"][0][3] = [safe_key(teacher_spec)]
        model_yaml = str(parent_save_dir / "teacherdet.yaml")
        YAML.save(model_yaml, cfg)
    else:
        model_yaml = _infer_model_yaml(phase1_weights)
        # Fail fast on a wrong parent id (e.g. a dir basename) before training the full dataset suite, since
        # push_summary_to_parent would otherwise drop the downstream link silently at the final step.
        wandb_config.assert_parent_resolvable(phase1_wandb_id)

    csv_path = paths.multi_results_csv(parent_name, paths.LOCAL_ROOT)
    nfs_csv = paths.multi_results_csv(parent_name)
    if not csv_path.exists():
        csv_path.write_text("dataset,map50,map50_95,fitness\n")

    def _mirror_csv() -> None:
        # nfs_sync mirrors only per-dataset save_dirs, never this parent-level CSV, so it stays host-local otherwise.
        # Warn-only on filesystem/NFS errors so a transient mirror failure never kills the run (nfs_sync is non-blocking for the same reason).
        try:
            nfs_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, nfs_csv)
        except OSError as e:
            print(f"[multi_det_finetune] NFS mirror of multi_results.csv failed (continuing): {e}")

    base_epochs = epochs or _MULTI_DET_BASE_EPOCHS
    base_patience = patience or _MULTI_DET_BASE_PATIENCE
    if base_epochs != _MULTI_DET_BASE_EPOCHS or base_patience != _MULTI_DET_BASE_PATIENCE:
        print(
            f"[multi_det_finetune] NOTE non-default recipe epochs={base_epochs} patience={base_patience}, "
            f"default flat is {_MULTI_DET_BASE_EPOCHS}/{_MULTI_DET_BASE_PATIENCE}. Macros at different epoch budgets "
            f"are not directly comparable. Omit the positional epochs/patience to use the default."
        )
    batch_actual = int(batch_override) if batch_override else _MULTI_DET_BASE_BATCH
    print(f"[multi_det_finetune] parent={parent_name} datasets={len(dataset_yamls)} model={model_yaml}")
    print(f"[multi_det_finetune] aggregate csv -> {csv_path}")
    print(f"[multi_det_finetune] flat recipe epochs={base_epochs} patience={base_patience} batch={batch_actual}")

    results = []
    for i, ds_yaml in enumerate(dataset_yamls, start=1):
        basename = ds_yaml.parent.name
        n_imgs, iters_per_ep = _dataset_train_stats(ds_yaml, batch_actual)
        print(f"\n=== [{i}/{len(dataset_yamls)}] {basename} ===")
        print(
            f"[multi_det_finetune] {basename}: n_train={n_imgs} iters/ep={iters_per_ep} "
            f"epochs={base_epochs} patience={base_patience}"
        )

        model = YOLO(model_yaml)
        model.add_callback("on_train_start", grad_clip.override(1.0))
        model.add_callback("on_train_start", muon_w.override(0.4355))
        # Nest NFS mirror under parent so different parents' same-basename sub-runs (e.g. two parents both training
        # `aerial-cows`) don't collide on the flat `NFS_MIRROR_ROOT / Path(save_dir).name` mapping in nfs_sync.setup.
        sync_start, sync_end = nfs_sync.setup(
            str(paths.NFS_MIRROR_ROOT / parent_name), interval_sec=paths.SYNC_INTERVAL_SEC
        )
        model.add_callback("on_train_start", sync_start)
        model.add_callback("on_train_end", sync_end)
        model.add_callback(
            "on_pretrain_routine_start",
            wandb_config.log_config(
                model=model_yaml,
                pretrained_from=teacher_spec or phase1_weights,
                phase1_wandb_id=phase1_wandb_id,
                mode="teacher_frozen_det" if teacher_spec else "multi_det_finetune",
                teacher=teacher_spec,
                wandb_group=parent_name,
                parent_run=parent_name,
                dataset=basename,
                n_train_images=n_imgs,
                iters_per_epoch=iters_per_ep,
            ),
        )
        det_args = _build_det_train_args(
            base_epochs, base_patience, batch_override, lr_override, nbs_override,
            default_batch=_MULTI_DET_BASE_BATCH, default_lr0=_MULTI_DET_BASE_LR0,
        )
        if teacher_spec:
            # Freeze layer 0 (the teacher) via the trainer freeze arg: BaseTrainer re-enables requires_grad for any
            # non-frozen-listed param (trainer.py:319), so freezing only in __init__ is undone. imgsz is per-teacher.
            det_args["freeze"] = 1
            det_args["imgsz"] = _TEACHER_DET_IMGSZ.get(teacher_spec, 640)
        elif freeze_override:
            # Frozen distilled-student backbone; same trainer re-enable caveat as the teacher branch above.
            det_args["freeze"] = int(freeze_override)
        if imgsz_override:
            # Ablation lever: run the whole detector (and thus the frozen backbone) at a non-640 imgsz, e.g. 224 to
            # match the backbone's phase-1 distillation grid. Overrides the canonical 640 and any per-teacher imgsz.
            det_args["imgsz"] = int(imgsz_override)
        train_args = dict(
            pretrained=False if teacher_spec else phase1_weights,
            device=int(gpu),
            project=paths.WANDB_PROJECT,
            name=basename,
            save_dir=str(parent_save_dir / basename),
            exist_ok=False,
            dropout=0,
            amp=True,
            seed=seed,
            deterministic=True,
            workers=4,
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
        _mirror_csv()
        print(f"[done] {basename} mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f} fitness={row['fitness']:.4f}")

    macro = {k: sum(r[k] for r in results) / len(results) for k in ("map50", "map50_95", "fitness")}
    with csv_path.open("a") as f:
        f.write(f"MACRO,{macro['map50']:.4f},{macro['map50_95']:.4f},{macro['fitness']:.4f}\n")
    _mirror_csv()
    print(
        f"\n[multi_det_finetune] MACRO over {len(results)} datasets: "
        f"mAP50={macro['map50']:.4f} mAP50-95={macro['map50_95']:.4f} fitness={macro['fitness']:.4f}"
    )
    if not teacher_spec:  # frozen-teacher runs have no phase1 distillation parent to push the downstream link to
        # Auto-resolve the phase-1 run from the backbone dir when no id was passed, so the sweep view self-links.
        parent_id = phase1_wandb_id or wandb_config.resolve_run_id_by_name(Path(phase1_weights).parents[1].name)
        print(f"[multi_det_finetune] downstream link -> phase1 wandb id: {parent_id or '(unresolved, skipped)'}")
        wandb_config.push_summary_to_parent(
            parent_id,
            {
                "downstream_multi_macro_map50_95": float(macro["map50_95"]),
                "downstream_multi_n_datasets": len(results),
            },
        )


def main(argv: list[str]) -> None:
    """Launch a fresh phase 2 run or resume from a checkpoint."""
    argv = argv[1:]
    argv, resume = _pop_flag(argv, "--resume")
    argv, fork_from = _pop_flag(argv, "--fork_from")
    argv, lr_override = _pop_flag(argv, "--lr")
    argv, batch_override = _pop_flag(argv, "--batch")
    argv, nbs_override = _pop_flag(argv, "--nbs")
    argv, freeze_override = _pop_flag(argv, "--freeze")
    argv, scratch = _pop_flag(argv, "--scratch", is_bool=True)
    argv, datasets_arg = _pop_flag(argv, "--datasets")
    argv, imgsz_override = _pop_flag(argv, "--imgsz")
    argv, seed_override = _pop_flag(argv, "--seed")
    seed = int(seed_override) if seed_override else 0
    argv, teacher_spec = _pop_flag(argv, "--teacher")
    if teacher_spec:
        # Layout: <gpu> teacher_frozen_det <name> --teacher <spec> --datasets <file>. The frozen-teacher backbone
        # builds itself from --teacher, so there is no phase1_weights slot; the mode keyword is optional padding.
        if teacher_spec not in TEACHER_REGISTRY:
            raise SystemExit(f"--teacher {teacher_spec!r} not in TEACHER_REGISTRY: {sorted(TEACHER_REGISTRY)}")
        if teacher_spec not in _TEACHER_FROZEN_DET_SUPPORTED:
            raise SystemExit(
                f"teacher_frozen_det does not yet support {teacher_spec!r}. Supported (audited to run at det imgsz 640 "
                f"with correct stats + row-major grids): {sorted(_TEACHER_FROZEN_DET_SUPPORTED)}. siglip2/moonvit/sam3/"
                f"tips each need per-teacher work first (see the _TEACHER_FROZEN_DET_SUPPORTED note)."
            )
        if not datasets_arg:
            raise SystemExit("ERROR: teacher_frozen_det requires --datasets <file|dir>.")
        if resume or fork_from:
            raise SystemExit("ERROR: --resume and --fork_from are not supported for teacher_frozen_det.")
        if freeze_override:
            raise SystemExit("ERROR: --freeze is not supported with --teacher (teacher_frozen_det already freezes layer 0).")
        gpu = argv[0] if argv else "0"
        if "," in gpu:
            raise SystemExit(f"ERROR: teacher_frozen_det requires a single GPU (DDP drops add_callback). Got gpu={gpu!r}.")
        positionals = [a for a in argv[1:] if a != "teacher_frozen_det"]
        name = positionals[0] if positionals else f"phase2-teacherfrozen-{safe_key(teacher_spec)}"
        _export_hf_token()
        _run_multi_det(
            gpu=gpu,
            phase1_weights="",
            parent_name=name,
            phase1_wandb_id="",
            epochs=None,
            patience=None,
            batch_override=batch_override,
            lr_override=lr_override,
            nbs_override=nbs_override,
            datasets_arg=datasets_arg,
            imgsz_override=imgsz_override,
            seed=seed,
            teacher_spec=teacher_spec,
        )
        return
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
            freeze_override=freeze_override,
            imgsz_override=imgsz_override,
            seed=seed,
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
    # Standard pretrained= flow transfers the backbone via intersect_dicts (layers 0-8, or 0-10 for -sppf cls yamls).
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
        seed=seed,
        deterministic=True,
        workers=4,
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
