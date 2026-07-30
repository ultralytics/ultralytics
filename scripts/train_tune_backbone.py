# © 2014-2025 Ultralytics Inc. 🚀 All rights reserved. CONFIDENTIAL: Unauthorized use or distribution prohibited.

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main(device: str | None = None, compile_mode: str = "", workers: int = 32):
    """Tune model training hyperparameters."""
    model = YOLO("yolo26s-p4p5-wide-deep16-sni-ultravit.yaml")
    model.load("ultravit_s.pt")
    model.tune(
        device=device,
        data="coco.yaml",
        epochs=70,
        imgsz=640,
        batch=128,
        compile=compile_mode or False,
        plots=False,
        val=True,
        save=True,
        workers=workers,
        project="tune-yolo26s-backbone",
        iterations=6000,
        optimizer="MuSGD",
        space={  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-2),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "blr_ratio": (0.05, 1.0),
            "cls_lr_mult": (1.0, 5.0),
            "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_bias_lr": (0.0, 0.2),  # warmup bias lr
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "epochs": (50, 300),
            "close_mosaic": (0, 20),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="Device to use, e.g. 'cpu', 'cuda', or 'mps'")
    parser.add_argument("--compile", default="", help="Compile mode, e.g. 'default', '' for disabled")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    args = parser.parse_args()
    main(args.device, args.compile, args.workers)
