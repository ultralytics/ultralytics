"""Train s3d on the drive-disjoint three-way split with real model selection and no contaminated warm start.

Replaces the `val=False`, warm-started recipe in s3d_tools/rebuild3.py, which had three defects that all
inflate the reported number:

1. `val=False` meant `self.fitness` was never set, so trainer.py's `if self.best_fitness == self.fitness:`
   compared None to None and rewrote best.pt every epoch — best.pt and last.pt differed in 0 of 806
   tensors. There was no model selection and no early stopping, and the run reported epoch 1000 even
   though a recorded 400-epoch run scored 9.11 AP3D@0.7 Mod against 6.07 at 1000.
2. It warm-started from released `yolo26*-s3d.pt`, trained on a split with 99.6% drive leakage into val.
3. It sampled frames uniformly, letting the top five of 71 drives supply 37.8% of every epoch. Drive
   balancing now lives in the dataset YAML's `drives:` / `drive_balance:` keys, not here.

Selection, and why it is post hoc rather than `patience`
--------------------------------------------------------
The dev split is 540 frames over 14 drives with 88.3% of them in the top five drives, and only ~230
Pedestrian / ~76 Cyclist instances. Two consequences:

* s3d's built-in `fitness` is the mean AP3D@0.5 Moderate over all three classes, so two thirds of the
  selection signal comes from ~306 VRU instances and is mostly noise. This driver therefore does not
  select on `fitness`; it selects on `AP3D_Car_Mod_50` (1913 Car instances, and @0.5 rather than @0.7
  because the tighter threshold is the spikier of the two).
* Per-epoch dev AP on 540 frames swings well beyond this project's ~0.8-1.2 AP A/B noise floor, so a
  `patience` counter on it would stop semi-randomly, and `cos_lr` means a truncated run is not the run
  you would have got from a shorter schedule anyway. So the full horizon is always trained and the
  checkpoint is chosen afterwards from an EMA-smoothed dev curve. `--patience` remains as a plateau
  safety net and deliberately does not participate in selection.

Post-hoc selection also survives DDP, which a callback would not: `ultralytics/utils/dist.py` rebuilds
the trainer in each subprocess from `args` alone, so callbacks registered here would never run on a
multi-GPU box.

Deliberately fast, not cheap: `--device` defaults to every visible GPU. `--batch` is the GLOBAL batch and
defaults to 8 per GPU, which is 64 on an 8-GPU box — exactly the nominal batch (`nbs`) the recipe's
`lr0=0.01` was tuned at, so no LR rescaling is implied. To trade fidelity for throughput, raise it and
scale `lr0` linearly (e.g. `--batch 256 --lr0 0.04`). `--save-period 1` keeps every epoch's checkpoint so
selection can land on any epoch; for the larger scales use `--save-period 5` and accept that selection
snaps to the nearest saved epoch.

Usage:
    python research/s3d_train_holdout.py                      # defaults to the VAL-IS-DEV yaml
    python research/s3d_train_holdout.py --scale s --batch 256 --lr0 0.04
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils import SETTINGS, YAML
from ultralytics.utils.checks import check_yaml

# The shipped config resolves by name and auto-downloads, so this runs unchanged on a fresh box. Its
# `val:`/`val_split:` point at the DEV split, never at test — see cfg/datasets/kitti-stereo-chen.yaml.
DEV_YAML = "kitti-stereo-chen.yaml"
# Not shipped inside the dataset archive, so the drive map is a local path by default and the sampler is
# opt-in; pass --drives "" to train without it.
DRIVES = "/home/rick/s3d_tools/chen3way_split.json"
SELECT_KEY = "AP3D_Car_Mod_50"


def parse_args() -> argparse.Namespace:
    """Parse driver arguments, defaulting device and batch to the whole machine."""
    ngpu = torch.cuda.device_count()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEV_YAML, help="dataset YAML whose val/val_split point at DEV, never test")
    p.add_argument("--drives", default=DRIVES, help="drive map for drive-balanced sampling; empty string disables")
    p.add_argument("--drive-balance", type=float, default=1.0, help="per-drive cap in multiples of the mean")
    p.add_argument("--scale", default="n", choices=("n", "s", "m", "l"), help="model scale")
    p.add_argument("--epochs", type=int, default=400, help="cos_lr horizon; the recorded optimum was near 400")
    p.add_argument("--batch", type=int, default=8 * max(ngpu, 1), help="GLOBAL batch across all GPUs")
    p.add_argument("--device", default=",".join(str(i) for i in range(ngpu)) or "cpu", help="e.g. 0,1,2,3")
    p.add_argument("--workers", type=int, default=8, help="dataloader workers per rank")
    p.add_argument("--val-period", type=int, default=1, help="dev is only 540 frames, so evaluate every epoch")
    p.add_argument("--save-period", type=int, default=1, help="epochs between checkpoints selection can choose from")
    p.add_argument("--select-span", type=int, default=10, help="EMA span in epochs used to smooth the dev curve")
    p.add_argument("--patience", type=int, default=150, help="plateau safety net only; selection ignores it")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--project", default="runs/s3d_holdout")
    p.add_argument("--name", default="train")
    return p.parse_args()


def resolve_dataset_yaml(path: str, drives: str, balance: float, project: str, name: str) -> str:
    """Validate the split wiring and, if needed, emit a copy carrying the drive-balancing keys.

    The s3d loader takes split DIRECTORY names from `train_split` / `val_split` (see
    Stereo3DDetTrainer.get_dataset), not from the `train:` / `val:` image paths, and both default to
    "train" / "val". A YAML that only redirects `val:` at dev makes the loader look for `images/val/`,
    which does not exist in the three-way tree. The reverse mistake is worse: a `val_split` of "test"
    would burn the sealed split on every epoch. Both are checked here rather than trusted.

    The derived YAML is written to a stable path, not a temp file: DDP subprocesses re-resolve
    `args.data` from disk, so it has to outlive this process.

    Args:
        path (str): Source dataset YAML.
        drives (str): Drive-map JSON to inject, or "" to leave sampling uniform.
        balance (float): `drive_balance` to inject alongside it.
        project (str): Run project directory, used to host the derived YAML.
        name (str): Run name, used to name the derived YAML.

    Returns:
        (str): YAML path to train on — the original when no injection was needed.
    """
    cfg = YAML.load(check_yaml(path))
    val_split = cfg.get("val_split", "val")
    if "test" in str(val_split) or "test" in str(cfg.get("val", "")):
        raise SystemExit(f"{path} points validation at the sealed test split (val_split={val_split!r}) — refusing")
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = Path(SETTINGS["datasets_dir"]) / root
    for key, split in (("train_split", cfg.get("train_split", "train")), ("val_split", val_split)):
        if not (root / "images" / split / "left").is_dir():
            raise SystemExit(f"{path}: {key}={split!r} but {root / 'images' / split / 'left'} does not exist")

    if not drives:
        print("NOTE: --drives empty — training with uniform, drive-proportional sampling")
        return path
    if not Path(drives).is_file():
        raise SystemExit(f"drive map {drives} does not exist")
    if cfg.get("drives") == drives and cfg.get("drive_balance") == balance:
        return path
    cfg["drives"], cfg["drive_balance"] = drives, balance
    derived = Path(project) / f"{name}_data.yaml"
    derived.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(derived, cfg)
    print(f"NOTE: wrote {derived} with drives={drives} drive_balance={balance}")
    return str(derived)


def select_checkpoint(save_dir: Path, span: int) -> dict:
    """Pick the checkpoint at the peak of the EMA-smoothed dev curve for SELECT_KEY.

    Args:
        save_dir (Path): Run directory holding results.csv and weights/.
        span (int): EMA span in epochs; the smoothing weight is 2 / (span + 1).

    Returns:
        (dict): Selection record — chosen epoch, its raw and smoothed score, the raw argmax for
            comparison, and the checkpoint that was copied to weights/selected.pt.
    """
    rows = list(csv.DictReader((save_dir / "results.csv").open()))
    curve = [(int(float(r["epoch"])), float(r[SELECT_KEY])) for r in rows if r.get(SELECT_KEY)]
    if not curve:
        raise SystemExit(f"no {SELECT_KEY} column in {save_dir / 'results.csv'} — was val=True?")

    alpha, ema, smoothed = 2.0 / (span + 1), None, []
    for epoch, value in curve:
        ema = value if ema is None else alpha * value + (1 - alpha) * ema
        smoothed.append((epoch, value, ema))
    best = max(smoothed, key=lambda t: t[2])
    raw_best = max(curve, key=lambda t: t[1])

    # save_model() names checkpoints by the 0-based epoch, while results.csv logs the 1-based one.
    available = sorted(int(p.stem[5:]) for p in (save_dir / "weights").glob("epoch*.pt"))
    wanted = best[0] - 1
    ckpt = None
    if available:
        nearest = min(available, key=lambda k: abs(k - wanted))
        ckpt = save_dir / "weights" / f"epoch{nearest}.pt"
        if nearest != wanted:
            print(f"NOTE: epoch{wanted}.pt absent, selection snapped to epoch{nearest}.pt (save_period too coarse)")
    else:
        ckpt = save_dir / "weights" / "last.pt"
        print("NOTE: no per-epoch checkpoints kept, falling back to last.pt")
    selected = save_dir / "weights" / "selected.pt"
    shutil.copy2(ckpt, selected)
    return {
        "criterion": f"EMA(span={span}) of {SELECT_KEY} on dev",
        "selected_epoch": best[0],
        "selected_raw": best[1],
        "selected_smoothed": best[2],
        "raw_argmax_epoch": raw_best[0],
        "raw_argmax_value": raw_best[1],
        "source_checkpoint": str(ckpt),
        "selected_checkpoint": str(selected),
    }


def main() -> None:
    """Train the full horizon, select post hoc on the smoothed dev curve, then score the selection."""
    a = parse_args()
    data = resolve_dataset_yaml(a.data, a.drives, a.drive_balance, a.project, a.name)

    # No .load(): the released yolo26*-s3d.pt saw the held-out drives, so warm-starting from them leaks.
    model = YOLO(f"yolo26{a.scale}-s3d.yaml")
    model.train(
        data=data,
        epochs=a.epochs,
        imgsz=[384, 1248],
        batch=a.batch,
        device=a.device,
        optimizer="SGD",
        lr0=a.lr0,
        cos_lr=True,
        val=True,  # the whole point: fitness exists, so best.pt and results.csv mean something
        val_period=a.val_period,
        save_period=a.save_period,
        patience=a.patience,
        plots=False,
        workers=a.workers,
        project=a.project,
        name=a.name,
        exist_ok=True,
    )

    out = Path(model.trainer.save_dir)
    selection = select_checkpoint(out, a.select_span)
    res = YOLO(selection["selected_checkpoint"]).val(
        data=data, imgsz=[384, 1248], batch=a.batch, device=a.device, plots=False
    )
    metrics = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in getattr(res, "results_dict", {}).items()}
    payload = {"scale": a.scale, "data": data, "epochs": a.epochs, "selection": selection, "dev_metrics": metrics}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    print("WROTE", out / "metrics.json")
    print(
        f"  selected epoch {selection['selected_epoch']} "
        f"(smoothed {selection['selected_smoothed']:.2f}, raw {selection['selected_raw']:.2f}); "
        f"raw argmax was epoch {selection['raw_argmax_epoch']} at {selection['raw_argmax_value']:.2f}"
    )
    for k in ("AP3D_Car_Easy_70", "AP3D_Car_Mod_70", "AP3D_Car_Hard_70", "AP3D_Car_Mod_50"):
        print(f"  {k} = {metrics.get(k)}")
    print("  NOTE: dev numbers only. The sealed test split is scored once, separately.")


if __name__ == "__main__":
    main()
