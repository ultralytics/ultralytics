#!/usr/bin/env python
"""Run optimizer-specific W&B Bayesian sweeps for the distilled UltraViT-S COCO detector.

Create the sweep once, then attach any GPU on any Ultra server with the returned sweep ID:

    python run_coco_tune_sweep.py create AdamW
    python run_coco_tune_sweep.py create MuSGD
    python run_coco_tune_sweep.py agent <sweep-id> --gpu 6
    python run_coco_tune_sweep.py agent <sweep-id> --gpu 3
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before W&B: BLAS pools size at init, ignore torch.set_num_threads

ENTITY = "fca"
PROJECT = "yolo-next-encoder"
SWEEP_NAME = "tune-coco-ultravit-s-290726"
MODEL_YAML = Path(__file__).parent / "ultralytics/cfg/models/26/yolo26s-ultravit-290726.yaml"
PRETRAINED = Path(
    "/data/shared-datasets/fatih-runs/classify/yolo-next-encoder/"
    "ph1-12src-ultravit-s-290726-dinov3-vitl16/weights/best.pt"
)
LR0_RANGES = {"AdamW": (2.5e-4, 1e-3), "MuSGD": (1.9e-4, 7.6e-4)}
SCORE_HORIZON_FLOOR = 8  # keep projecting this many epochs at completion so end slope still counts
SCORE_PROJECTION_CAP = 0.01  # slope may tip close calls but never dominate measured mAP

SWEEP_CONFIG = {
    "name": SWEEP_NAME,
    "method": "bayes",
    "run_cap": 48,
    "metric": {"name": "tune/score", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 75, "eta": 2},
    "parameters": {
        "backbone_lr_ratio": {"distribution": "log_uniform_values", "min": 0.05, "max": 1.0},
        "lr0_scale": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "lrf": {"distribution": "uniform", "min": 0.1, "max": 1.0},
        "weight_decay": {"distribution": "log_uniform_values", "min": 5e-5, "max": 5e-4},
        "momentum": {"distribution": "uniform", "min": 0.85, "max": 0.97},
        "warmup_epochs": {"values": [0.0, 1.0, 2.0, 3.0]},
        "warmup_momentum": {"distribution": "uniform", "min": 0.5, "max": 0.9},
        "warmup_bias_lr": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        "close_mosaic": {"values": [0, 5, 10, 15]},
    },
}


def train() -> None:
    """Train one configuration assigned by an optimizer-specific W&B sweep."""
    import wandb

    with wandb.init(entity=ENTITY, project=PROJECT, group=SWEEP_NAME, job_type="hparam-tune") as run:
        from callbacks import nfs_sync, paths
        from run_enc_distill_phase2 import _load_recipe
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        if not SETTINGS["wandb"]:
            raise RuntimeError("W&B logging is disabled. Run 'yolo settings wandb=True' once on this server.")
        if not PRETRAINED.exists():
            raise FileNotFoundError(f"Distilled checkpoint not found: {PRETRAINED}")

        run.name = f"{SWEEP_NAME}-{run.id}"
        recipe = _load_recipe("coco-preserve", str(MODEL_YAML), batch=128, nbs=64)
        sweep_config = dict(run.config)
        lr0_min, lr0_max = LR0_RANGES[sweep_config["optimizer"]]
        recipe["lr0"] = lr0_min * (lr0_max / lr0_min) ** sweep_config.pop("lr0_scale")
        recipe.update(sweep_config)
        run.config.update(
            {
                **recipe,
                "model": str(MODEL_YAML),
                "pretrained": str(PRETRAINED),
                "data": "coco.yaml",
                "muon": 0.5,
                "sgd": 0.5,
            },
            allow_val_change=True,
        )

        history = []

        def log_score(trainer) -> None:
            """Log projected mAP with trend removed from the noise penalty."""
            history.append(float(trainer.metrics["metrics/mAP50-95(B)"]))
            window = history[-20:]
            x_mean, y_mean = (len(window) - 1) / 2, statistics.mean(window)
            denominator = sum((i - x_mean) ** 2 for i in range(len(window)))
            slope = (
                sum((i - x_mean) * (value - y_mean) for i, value in enumerate(window)) / denominator
                if denominator
                else 0.0
            )
            noise = statistics.pstdev(value - y_mean - slope * (i - x_mean) for i, value in enumerate(window))
            horizon = min(len(window) - 1, max(trainer.epochs - trainer.epoch - 1, SCORE_HORIZON_FLOOR))
            projected_map = history[-1] + max(min(horizon * slope, SCORE_PROJECTION_CAP), -SCORE_PROJECTION_CAP)
            wandb.log(
                {
                    "tune/score": projected_map - noise,
                    "tune/projected_map": projected_map,
                    "tune/map_slope": slope,
                    "tune/residual_std": noise,
                },
                step=trainer.epoch + 1,
                commit=False,
            )

        model = YOLO(str(MODEL_YAML))
        model.add_callback("on_model_save", log_score)
        run_paths = paths.run_paths(run.name)
        model.add_callback("on_train_end", nfs_sync.start(run_paths["save_dir"], interval_sec=60))
        model.train(
            device=0,
            data="coco.yaml",
            pretrained=str(PRETRAINED),
            cls_remap=True,
            muon=0.5,
            sgd=0.5,
            workers=4,
            plots=False,
            save=True,
            val=True,
            **run_paths,
            **recipe,
        )


def create_sweep(optimizer: str) -> None:
    """Create one optimizer-specific sweep and print commands for attaching GPU agents."""
    import wandb

    sweep_config = {
        **SWEEP_CONFIG,
        "name": f"{SWEEP_NAME}-{optimizer.lower()}",
        "parameters": {**SWEEP_CONFIG["parameters"], "optimizer": {"value": optimizer}},
    }
    sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
    print(f"Sweep ID: {sweep_id}")
    print(f"GPU 6: python {Path(__file__).name} agent {sweep_id} --gpu 6")
    print(f"GPU 3: python {Path(__file__).name} agent {sweep_id} --gpu 3")


def run_agent(sweep_id: str, gpu: str, count: int | None = None) -> None:
    """Attach one visible GPU to an existing shared sweep."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("WANDB_LOG_MODEL", "false")

    import wandb

    from ultralytics import settings

    settings.update({"wandb": True})

    wandb.agent(sweep_id, function=train, entity=ENTITY, project=PROJECT, count=count)


def main() -> None:
    """Parse sweep creation and agent commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create one optimizer-specific sweep")
    create.add_argument("optimizer", choices=LR0_RANGES)
    agent = subparsers.add_parser("agent", help="attach one GPU to an existing sweep")
    agent.add_argument("sweep_id", help="ID printed by the create command")
    agent.add_argument("--gpu", required=True, help="physical GPU index exposed to this agent")
    agent.add_argument(
        "--count", type=int, help=f"maximum trials for this agent; global cap is {SWEEP_CONFIG['run_cap']}"
    )
    args = parser.parse_args()

    create_sweep(args.optimizer) if args.command == "create" else run_agent(args.sweep_id, args.gpu, args.count)


if __name__ == "__main__":
    main()
