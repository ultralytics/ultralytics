"""Run kNN evaluation on a distilled encoder run directory or on a released reference encoder.

Usage:
    python run_knn_eval.py <gpu_id> <run_dir|teacher_spec> [--imgsz N] [--crop_ratio R] [--csv PATH] [--wandb]

A target matching ``TEACHER_REGISTRY`` in either form (``tips:v2b14`` or ``tips_v2b14``) is read as a teacher spec and
scored under the same k=20 / T=0.07 protocol as our students. Anything else is a run directory, where weights/best.pt
and the model config come from args.yaml automatically, warning and falling back to last.pt when best.pt is absent.

--imgsz sets the eval resolution (default 224), and the loader batch scales down with imgsz to hold activation memory
roughly constant. --crop_ratio selects the published eval protocol (bicubic shortest-side resize to imgsz/ratio, then
a center crop), which is what DINOv2/TIPS/EUPE kNN numbers are measured under. Pass 0.875 to match them, or 1.0 to
isolate interpolation from crop. Absent, the Ultralytics bilinear ratio-1.0 transform is used, as in every existing
table row. --csv appends one result row per call so a sweep stays readable while it runs. With --wandb, updates the
finished WandB run's summary with knn/top1 (run directories only).

Examples:
    python run_knn_eval.py 3 /data/shared-datasets/fatih-runs/classify/yolo-next-encoder/phase1-d1-eupe-vitb16
    python run_knn_eval.py 6 tips:v2b14 --imgsz 518 --csv /home/fatih/runs/knn-reference.csv
"""

import sys
import time
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.data import ClassificationDataset
from ultralytics.data.build import build_dataloader
from ultralytics.utils import YAML
from ultralytics.utils.knn_eval import extract_features, knn_accuracy, yolo_cls_features

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


def _load_teacher(spec, imgsz, device):
    """Build a frozen released encoder from a TEACHER_REGISTRY spec.

    Args:
        spec (str): Registry key such as 'tips:v2b14' or 'dune:vits14'.
        imgsz (int): Eval resolution, which must be a whole number of patches.
        device (torch.device): Device to load onto.

    Returns:
        (tuple): (model, feature_fn, name).
    """
    from ultralytics.nn.teacher_model import TEACHER_REGISTRY, build_teacher_model

    mult = TEACHER_REGISTRY[spec]["imgsz_multiple"]
    if imgsz % mult:
        print(f"Error: {spec} has patch stride {mult}, so imgsz {imgsz} would be cropped to {imgsz // mult * mult}.")
        sys.exit(1)
    # normalize_input=True converts the loader's ImageNet-normalized tensor to each teacher's own training
    # distribution: a real conversion for TIPS (raw [0, 1]), a no-op for the ImageNet-stat teachers.
    teacher = build_teacher_model(spec, device=device, normalize_input=True)
    return teacher, lambda m, imgs: m.encode(imgs).cls, spec


def _load_run_dir(run_dir, device):
    """Build a distilled student from a run directory's checkpoint and args.yaml.

    Args:
        run_dir (Path): Training run directory.
        device (torch.device): Device to load onto.

    Returns:
        (tuple): (model, feature_fn, name).
    """
    weight_path = run_dir / "weights" / "best.pt"
    if not weight_path.exists():
        weight_path = run_dir / "weights" / "last.pt"
        print(f"WARNING: no best.pt in {run_dir.name}, falling back to last.pt (different checkpoint provenance)")
    if not weight_path.exists():
        print(f"Error: no weights found in {run_dir / 'weights'}")
        sys.exit(1)

    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        print(f"Error: no args.yaml in {run_dir}")
        sys.exit(1)
    model_cfg = YAML.load(args_yaml).get("model")
    if not model_cfg:
        print(f"Error: no 'model:' key in {args_yaml}")
        sys.exit(1)
    print(f"  weights: {weight_path}")
    print(f"  model_cfg: {model_cfg}")

    model = YOLO(model_cfg)
    ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    src = ckpt.get("ema") or ckpt.get("model")
    state = src.float().state_dict()
    loaded = model.model.load_state_dict(state, strict=False)
    print(f"  Loaded: {len(state) - len(loaded.unexpected_keys)}/{len(state)} keys")
    model.model.to(device).float()
    return model.model, yolo_cls_features, run_dir.name


def main():
    """Run kNN evaluation on a run directory or a released reference encoder."""
    from ultralytics.nn.teacher_model import TEACHER_REGISTRY, resolve_teacher_key

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
    crop_ratio = float(pop("--crop_ratio") or 0)
    csv_path = pop("--csv")
    use_wandb = "--wandb" in argv
    positional = [a for a in argv if not a.startswith("--")]

    if len(positional) < 2:
        print("Usage: run_knn_eval.py <gpu> <run_dir|teacher_spec> [--imgsz N] [--crop_ratio R] [--csv PATH] [--wandb]")
        sys.exit(1)

    gpu_id, target = int(positional[0]), resolve_teacher_key(positional[1])
    is_teacher = target in TEACHER_REGISTRY
    if use_wandb and is_teacher:
        print("Error: --wandb writes to a training run's summary, which a released teacher does not have.")
        sys.exit(1)
    if use_wandb and crop_ratio:
        print("Error: the knn/top1 summary key has no protocol axis, so a --crop_ratio result would overwrite it.")
        sys.exit(1)

    device = torch.device(f"cuda:{gpu_id}")
    print(f"Evaluating: {target}")
    print(f"  imgsz: {imgsz}")
    print(f"  wandb: {'on' if use_wandb else 'off'}")

    # Resolve the target before the ImageNet scan so a bad spec or resolution fails in milliseconds, not minutes.
    model, feature_fn, name = (
        _load_teacher(target, imgsz, device) if is_teacher else _load_run_dir(Path(target), device)
    )

    from types import SimpleNamespace

    root = Path(IMAGENET)
    ds_args = SimpleNamespace(
        imgsz=imgsz,
        cache=False,
        fraction=1.0,
        auto_augment="",
        erasing=0.0,
        scale=0.92,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
    )
    train_ds = ClassificationDataset(str(root / "train"), args=ds_args, augment=False, prefix="knn-train")
    val_ds = ClassificationDataset(str(root / "val"), args=ds_args, augment=False, prefix="knn-val")
    if crop_ratio:
        # ``classify_transforms`` welds resize to crop and rejects ``crop_fraction`` (augment.py:2812 raises), so the
        # published protocol is built here: eupe/eval/knn.py:199-203 asserts a 256/224 ratio, transforms.py:132 BICUBIC.
        import torchvision.transforms as T

        resize = round(imgsz / crop_ratio)

        from ultralytics.data.augment import DEFAULT_MEAN, DEFAULT_STD

        train_ds.torch_transforms = val_ds.torch_transforms = T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(imgsz),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(DEFAULT_MEAN), std=torch.tensor(DEFAULT_STD)),
            ]
        )
        print(f"  published protocol: resize {resize} bicubic -> center crop {imgsz} (ratio {crop_ratio})")
    bs = max(8, round(256 * (224 / imgsz) ** 2))  # hold activation memory ~constant across imgsz
    train_loader = build_dataloader(train_ds, bs, 8, shuffle=False, rank=-1)
    val_loader = build_dataloader(val_ds, bs, 8, shuffle=False, rank=-1)

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
        num_classes=len(train_ds.base.classes),
        device=device,
    )
    elapsed = time.time() - t0
    print(f"\nkNN top-1: {top1:.2f}% ({elapsed:.0f}s)")

    if csv_path:
        header = not Path(csv_path).exists()
        with open(csv_path, "a", encoding="utf-8") as f:
            if header:
                f.write("model,imgsz,crop_ratio,top1,seconds\n")
            f.write(f"{name},{imgsz},{crop_ratio or 1.0},{top1:.2f},{elapsed:.0f}\n")
        print(f"  appended to {csv_path}")

    if use_wandb:
        _update_wandb(Path(target), top1)


if __name__ == "__main__":
    main()
