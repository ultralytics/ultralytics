"""Run kNN evaluation on a distilled encoder run directory.

Usage:
    python run_knn_eval.py <gpu_id> <run_dir> [--imgsz N] [--wandb]

Finds weights/best.pt and model config from args.yaml automatically.
--imgsz sets the eval resolution (default 224); the loader batch scales down with imgsz to hold
activation memory roughly constant. With --wandb, updates the finished WandB run's summary with knn/top1.

Examples:
    python run_knn_eval.py 3 /data/shared-datasets/fatih-runs/classify/yolo-next-encoder/phase1-d1-eupe-vitb16
    python run_knn_eval.py 3 /home/fatih/runs/classify/yolo-next-encoder/phase1-yolo26l-eupe-vitb16 --wandb
"""

import sys
import time
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.data import ClassificationDataset
from ultralytics.data.build import build_dataloader
from ultralytics.utils.knn_eval import extract_features, knn_accuracy

IMAGENET = "/data/shared-datasets/imagenet"


def _update_wandb(run_dir, knn_top1):
    """Update a finished WandB run's summary with kNN top-1 accuracy."""
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


def main():
    """Run kNN evaluation on a run directory."""
    args_list = sys.argv[1:]
    use_wandb = "--wandb" in args_list
    imgsz = 224
    if "--imgsz" in args_list:
        i = args_list.index("--imgsz")
        imgsz = int(args_list[i + 1])
        del args_list[i : i + 2]
    argv = [a for a in args_list if not a.startswith("--")]

    if len(argv) < 2:
        print("Usage: python run_knn_eval.py <gpu_id> <run_dir> [--imgsz N] [--wandb]")
        sys.exit(1)

    gpu_id = int(argv[0])
    run_dir = Path(argv[1])

    # Validate run directory
    weight_path = run_dir / "weights" / "best.pt"
    if not weight_path.exists():
        weight_path = run_dir / "weights" / "last.pt"
    if not weight_path.exists():
        print(f"Error: no weights found in {run_dir / 'weights'}")
        sys.exit(1)

    # Get model config from args.yaml
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        print(f"Error: no args.yaml in {run_dir}")
        sys.exit(1)
    model_cfg = None
    for line in args_yaml.read_text().splitlines():
        if line.startswith("model:"):
            model_cfg = line.split(":", 1)[1].strip()
            break
    if not model_cfg:
        print(f"Error: no 'model:' key in {args_yaml}")
        sys.exit(1)

    device = torch.device(f"cuda:{gpu_id}")
    print(f"Evaluating: {run_dir.name}")
    print(f"  weights: {weight_path}")
    print(f"  model_cfg: {model_cfg}")
    print(f"  imgsz: {imgsz}")
    print(f"  wandb: {'on' if use_wandb else 'off'}")

    # Build dataloaders
    from types import SimpleNamespace

    root = Path(IMAGENET)
    ds_args = SimpleNamespace(
        imgsz=imgsz,
        cache=False,
        fraction=1.0,
        auto_augment="",
        erasing=0.0,
        crop_fraction=1.0,
        scale=0.92,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
    )
    train_ds = ClassificationDataset(str(root / "train"), args=ds_args, augment=False, prefix="knn-train")
    val_ds = ClassificationDataset(str(root / "val"), args=ds_args, augment=False, prefix="knn-val")
    bs = max(8, round(256 * (224 / imgsz) ** 2))  # hold activation memory ~constant across imgsz
    train_loader = build_dataloader(train_ds, bs, 8, shuffle=False, rank=-1)
    val_loader = build_dataloader(val_ds, bs, 8, shuffle=False, rank=-1)
    num_classes = len(train_ds.base.classes)

    # Load model from yaml + distillation checkpoint
    model = YOLO(model_cfg)
    ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    src = ckpt.get("ema") or ckpt.get("model")
    state = src.float().state_dict()
    loaded = model.model.load_state_dict(state, strict=False)
    print(f"  Loaded: {len(state) - len(loaded.unexpected_keys)}/{len(state)} keys")
    model.model.to(device).float()

    # Evaluate
    t0 = time.time()
    train_feats, train_labels = extract_features(model.model, train_loader, device)
    val_feats, val_labels = extract_features(model.model, val_loader, device)
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
    print(f"\nkNN top-1: {top1:.2f}% ({time.time() - t0:.0f}s)")

    if use_wandb:
        _update_wandb(run_dir, top1)


if __name__ == "__main__":
    main()
