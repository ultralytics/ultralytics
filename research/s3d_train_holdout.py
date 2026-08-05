"""Train s3d on a drive-disjoint three-way split with real model selection and no contaminated warm start.

Replaces the `val=False`, warm-started recipe in s3d_tools/rebuild3.py, which had three defects that all
inflate the reported number:

1. `val=False` meant `self.fitness` was never set, so trainer.py's `if self.best_fitness == self.fitness:`
   compared None to None and rewrote best.pt every epoch — best.pt and last.pt differed in 0 of 806
   tensors. There was no model selection and no early stopping, and the run reported epoch 1000 even
   though a recorded 400-epoch run scored 9.11 AP3D@0.7 Mod against 6.07 at 1000.
2. It warm-started from released `yolo26*-s3d.pt`, trained on a split with 99.6% drive leakage into val.
3. It sampled frames uniformly, letting the top five of 71 drives supply 37.8% of every epoch.

This driver trains on the three-way `train` split, validates on `dev` (`test` stays sealed for the single
final number), and leaves drive balancing to the `drives:` key of the dataset YAML.

Deliberately fast, not cheap: `--device` defaults to every visible GPU and DDP is used whenever more than
one is present. `--batch` is the GLOBAL batch and defaults to 8 per GPU, which is 64 on an 8-GPU box —
exactly the nominal batch (`nbs`) the recipe's `lr0=0.01` was tuned at, so no LR rescaling is implied. To
trade fidelity for throughput, raise it and scale `lr0` linearly (e.g. `--batch 256 --lr0 0.04`).

Usage:
    python research/s3d_train_holdout.py --data kitti-stereo-chen3way-3cls.yaml --scale n
    python research/s3d_train_holdout.py --data ... --batch 256 --lr0 0.04 --device 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse driver arguments, defaulting device and batch to the whole machine."""
    ngpu = torch.cuda.device_count()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset YAML whose `val:` points at the DEV split")
    p.add_argument("--scale", default="n", choices=("n", "s", "m", "l"), help="model scale")
    p.add_argument("--epochs", type=int, default=400, help="cos_lr horizon; the recorded optimum was near 400")
    p.add_argument("--batch", type=int, default=8 * max(ngpu, 1), help="GLOBAL batch across all GPUs")
    p.add_argument("--device", default=",".join(str(i) for i in range(ngpu)) or "cpu", help="e.g. 0,1,2,3")
    p.add_argument("--workers", type=int, default=8, help="dataloader workers per rank")
    p.add_argument("--val-period", type=int, default=5, help="epochs between dev evaluations")
    p.add_argument("--patience", type=int, default=60, help="epochs without a dev improvement before stopping")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--project", default="runs/s3d_holdout")
    p.add_argument("--name", default="train")
    return p.parse_args()


def main() -> None:
    """Train, then report the dev metrics of the selected checkpoint."""
    a = parse_args()
    # No .load(): the released yolo26*-s3d.pt saw the val drives, so warm-starting from them leaks.
    model = YOLO(f"yolo26{a.scale}-s3d.yaml")
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=[384, 1248],
        batch=a.batch,
        device=a.device,
        optimizer="SGD",
        lr0=a.lr0,
        cos_lr=True,
        val=True,  # the whole point: fitness exists, so best.pt and patience mean something
        val_period=a.val_period,
        patience=a.patience,
        plots=False,
        workers=a.workers,
        project=a.project,
        name=a.name,
        exist_ok=True,
    )

    trainer = model.trainer
    out = Path(trainer.save_dir)
    res = model.val(data=a.data, imgsz=[384, 1248], batch=a.batch, device=a.device, plots=False)
    metrics = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in getattr(res, "results_dict", {}).items()}
    payload = {
        "scale": a.scale,
        "data": a.data,
        "epochs_requested": a.epochs,
        "epochs_run": trainer.epoch + 1,
        "best_fitness": None if trainer.best_fitness is None else float(trainer.best_fitness),
        "selected_checkpoint": str(trainer.best),
        "dev_metrics": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    print("WROTE", out / "metrics.json")
    print(f"  stopped at epoch {payload['epochs_run']} of {a.epochs}, best dev fitness {payload['best_fitness']}")
    for k in ("AP3D_Car_Easy_70", "AP3D_Car_Mod_70", "AP3D_Car_Hard_70", "AP3D_Car_Mod_50"):
        print(f"  {k} = {metrics.get(k)}")


if __name__ == "__main__":
    main()
