"""Run kNN evaluation on a distilled encoder run directory or on a released reference encoder.

Usage:
    python run_knn_eval.py <gpu_id> <run_dir|teacher_spec> [--imgsz N] [--crop_ratio R] [--csv PATH] [--wandb]

Everything is scored at k=20 / T=0.07. A target is either a ``TEACHER_REGISTRY`` key (``tips:v2b14`` or
``tips_v2b14``) or a run directory, which supplies weights/best.pt, falling back to last.pt with a warning.

Preprocessing is derived, not passed, because it is a property of how the weights were fed. A distilled run
(``teachers`` in args.yaml) and every released encoder take the 'imagenet' entry in ``KNN_PROTOCOLS``, a CE
classification run takes 'unit'. See that table for what reading one family through the other's entry costs.

Flags:
    --imgsz       eval resolution, default 224. Batch scales down with it to hold activation memory ~constant.
    --crop_ratio  center-crop ratio, default 1.0. 0.875 is the 256/224 protocol DINOv2/TIPS/EUPE publish under, and
                  as a research override it blocks --wandb.
    --csv         append one row per call, so a sweep stays readable while it runs.
    --wandb       overwrite the finished run's knn/top1 summary, run directories only. History keeps the
                  pre-overwrite value.

Examples:
    python run_knn_eval.py 3 /data/shared-datasets/fatih-runs/classify/yolo-next-encoder/phase1-d1-eupe-vitb16
    python run_knn_eval.py 6 tips:v2b14 --imgsz 518 --csv /home/fatih/runs/knn-reference.csv
"""

import sys
import time
from pathlib import Path

import torch

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.knn_eval import build_knn_loaders, extract_features, knn_accuracy, yolo_cls_features

IMAGENET = "/data/shared-datasets/imagenet"


def _update_wandb(run_dir, knn_top1):
    """Overwrite a finished WandB run's knn/top1 summary with a re-measured value.

    It overwrites knn/top1 rather than adding a key because that is the key every reader hits (four scripts under
    .claude/skills/wandb-check), so a new one would leave all of them on the stale number. Nothing is lost: history is
    append-only and the summary is its last point, so the pre-overwrite value stays readable through scan_history.
    """
    link = Path(run_dir) / "wandb" / "latest-run"
    if not link.is_symlink():
        print("  WandB: no wandb/latest-run symlink found")
        return
    run_id = link.resolve().name.split("-", 2)[2]
    try:
        import wandb

        run = wandb.Api().run(f"fca/yolo-next-encoder/{run_id}")
        run.summary["knn/top1"] = knn_top1
        run.summary.update()
        print(f"  WandB updated: {run.name} -> knn/top1={knn_top1:.2f}%")
    except Exception as e:
        print(f"  WandB update failed: {e}")


def _load_teacher(spec, imgsz, device):
    """Build a frozen released encoder from a TEACHER_REGISTRY spec.

    Args:
        spec (str): Registry key such as 'tips:v2b14' or 'dune:vits14'.
        imgsz (int): Eval resolution, which must be a whole number of patches.
        device (torch.device): Device to load onto.

    Returns:
        (tuple): (model, feature_fn, name, protocol).
    """
    from ultralytics.nn.teacher_model import TEACHER_REGISTRY, build_teacher_model

    mult = TEACHER_REGISTRY[spec]["imgsz_multiple"]
    if imgsz % mult:
        print(f"Error: {spec} has patch stride {mult}, so imgsz {imgsz} would be cropped to {imgsz // mult * mult}.")
        sys.exit(1)
    # Rebases input onto each teacher's own distribution: a real shift for TIPS (raw [0, 1]), a no-op for the
    # rest. It reads its input as ImageNet-normalized, hence the "imagenet" protocol below.
    teacher = build_teacher_model(spec, device=device, normalize_input=True)
    return teacher, lambda m, imgs: m.encode(imgs).cls, spec, "imagenet"


def _load_run_dir(run_dir, device):
    """Build a student from a run directory's checkpoint and args.yaml, and resolve its eval protocol.

    Args:
        run_dir (Path): Training run directory.
        device (torch.device): Device to load onto.

    Returns:
        (tuple): (model, feature_fn, name, protocol).
    """
    weight_path = run_dir / "weights" / "best.pt"
    if not weight_path.exists():
        weight_path = run_dir / "weights" / "last.pt"
        print(f"WARNING: no best.pt in {run_dir.name}, falling back to last.pt (different checkpoint provenance)")
    if not weight_path.exists():
        print(f"Error: no weights found in {run_dir / 'weights'}")
        sys.exit(1)

    # Distillation feeds ImageNet stats, CE classification raw [0, 1]. ClassificationTrainer records this on the
    # checkpoint (classify/train.py:168-170) but ImageEncoderTrainer overrides get_dataloader without calling up, so
    # no distilled checkpoint carries it and args.yaml is the only record. Stamping it there would retire this line.
    protocol = "imagenet" if YAML.load(run_dir / "args.yaml").get("teachers") else "unit"
    print(f"  weights: {weight_path}")

    # The checkpoint carries its own architecture and load_checkpoint prefers ema, so args.yaml's model: key goes unread.
    return load_checkpoint(weight_path, device=device)[0], yolo_cls_features, run_dir.name, protocol


def main():
    """Run kNN evaluation on a run directory or a released reference encoder."""
    from ultralytics.nn.teacher_model import (
        TEACHER_REGISTRY,
        resolve_teacher_key,
    )

    argv = sys.argv[1:]

    def pop(flag):
        """Remove ``flag`` and its value from argv, returning the value or None."""
        if flag not in argv:
            return None
        i = argv.index(flag)
        value = argv[i + 1]
        del argv[i : i + 2]
        return value

    imgsz = int(pop("--imgsz") or 224)
    crop_ratio = float(pop("--crop_ratio") or 1.0)
    csv_path = pop("--csv")
    use_wandb = "--wandb" in argv
    positional = [a for a in argv if not a.startswith("--")]

    if len(positional) < 2:
        print("Usage: run_knn_eval.py <gpu> <target> [--imgsz N] [--crop_ratio R] [--csv PATH] [--wandb]")
        sys.exit(1)

    gpu_id, target = int(positional[0]), resolve_teacher_key(positional[1])
    is_teacher = target in TEACHER_REGISTRY
    if use_wandb and is_teacher:
        print("Error: --wandb writes to a training run's summary, which a released teacher does not have.")
        sys.exit(1)
    if use_wandb and crop_ratio != 1.0:
        print(f"Error: crop_ratio {crop_ratio} is a research override, so its result cannot claim the knn/top1 key.")
        sys.exit(1)

    device = torch.device(f"cuda:{gpu_id}")
    print(f"Evaluating: {target}")
    print(f"  imgsz: {imgsz}")
    print(f"  wandb: {'on' if use_wandb else 'off'}")

    # Resolve the target before the ImageNet scan so a bad spec or resolution fails in milliseconds, not minutes.
    model, feature_fn, name, protocol = (
        _load_teacher(target, imgsz, device) if is_teacher else _load_run_dir(Path(target), device)
    )
    train_loader, val_loader, num_classes = build_knn_loaders(Path(IMAGENET), imgsz, protocol, crop_ratio)

    t0 = time.time()
    train_feats, train_labels = extract_features(model, train_loader, device, feature_fn)
    val_feats, val_labels = extract_features(model, val_loader, device, feature_fn)
    top1 = knn_accuracy(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        k=20,
        temp=0.07,
        num_classes=num_classes,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"\nkNN top-1: {top1:.2f}% ({elapsed:.0f}s)")

    if csv_path:
        header = not Path(csv_path).exists()
        with open(csv_path, "a", encoding="utf-8") as f:
            if header:
                f.write("model,imgsz,crop_ratio,norm,top1,seconds\n")
            f.write(f"{name},{imgsz},{crop_ratio},{protocol},{top1:.2f},{elapsed:.0f}\n")
        print(f"  appended to {csv_path}")

    if use_wandb:
        _update_wandb(Path(target), top1)


if __name__ == "__main__":
    main()
