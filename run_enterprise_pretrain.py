# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Launch one arm of the Enterprise vs Objects365v1 detection pretraining ablation.

The paired recipes share optimization and augmentation settings. Data, schedule, and classification handling differ.
The Enterprise arm uses source-pure batches and masked federated classification. The exhaustively annotated O365
control uses the stock loss.

Usage:
    python run_enterprise_pretrain.py 0,1 yolo26s-p4p5-wide-deep16-sni.yaml enterprise
    python run_enterprise_pretrain.py 2,3 yolo26s-p4p5-wide-slim19-ultravit.yaml o365 --no-amp
"""

from __future__ import annotations

import argparse
from pathlib import Path

from callbacks import nfs_sync
from callbacks.paths import run_paths
from run_enc_distill_phase2 import _TRAIN_DEFAULTS, _load_recipe
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train_federated import FederatedDetectionTrainer

ARMS = {
    "enterprise": (
        "enterprise",
        "/data/shared-datasets/domain-det/_merged/data.yaml",
        FederatedDetectionTrainer,
    ),
    "o365": ("enterprise-o365-control", "Objects365v1.yaml", None),
}
INIT = Path("/data/shared-datasets/fatih-runs/enterprise-init")
FEDERATED_CLI_KEYS = (
    "federated_cls_loss",
    "focal_alpha",
    "focal_gamma",
    "asl_gamma_pos",
    "asl_gamma_neg",
    "asl_clip",
    "federated_cls_normalize",
)


def main() -> None:
    """Resolve the arm's recipe, data and trainer, then train."""
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("device", help="physical GPU ids, e.g. 0,1")
    a.add_argument("model", help="detector yaml under cfg/models/26/, scale must be in the filename")
    a.add_argument("arm", choices=sorted(ARMS))
    a.add_argument("name", nargs="?", help="run name, defaults to ph2-<arm>-<model stem>")
    a.add_argument("--pretrained", default=None, help="backbone init .pt, defaults to the arm's staged init")
    a.add_argument("--no-amp", action="store_true", help="ultravit arms, which are unstable under fp16")
    a.add_argument("--epochs", type=int, default=None)
    a.add_argument("--batch", type=int, default=None)
    a.add_argument("--backbone-lr-ratio", type=float, default=None)
    a.add_argument("--quota-alpha", type=float, default=None)
    a.add_argument("--federated-cls-loss", choices=("bce", "focal", "asl"), default=None)
    a.add_argument("--focal-alpha", type=float, default=None)
    a.add_argument("--focal-gamma", type=float, default=None)
    a.add_argument("--asl-gamma-pos", type=float, default=None)
    a.add_argument("--asl-gamma-neg", type=float, default=None)
    a.add_argument("--asl-clip", type=float, default=None)
    a.add_argument("--federated-cls-normalize", choices=("none", "active_classes"), default=None)
    a.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = a.parse_args()
    if args.arm == "o365" and any(getattr(args, key) is not None for key in FEDERATED_CLI_KEYS):
        a.error("federated classification options apply only to the enterprise arm")

    recipe_name, data, trainer = ARMS[args.arm]
    recipe = _load_recipe(
        recipe_name,
        args.model,
        epochs=args.epochs,
        batch=args.batch,
        backbone_lr_ratio=args.backbone_lr_ratio,
        quota_alpha=args.quota_alpha,
        **{key: getattr(args, key) for key in FEDERATED_CLI_KEYS},
    )
    recipe["amp"] = not args.no_amp
    stem = Path(args.model).stem
    pretrained = args.pretrained or str(INIT / ("ultravit_s.pt" if "ultravit" in stem else "c3k2_s.pt"))
    name = args.name or f"ph2-{args.arm}-{stem.replace('yolo26s-p4p5-wide-', '')}"
    print(f"[arm] {args.arm}  model={args.model}  init={pretrained}  data={data}")
    train_args = {**_TRAIN_DEFAULTS, **recipe, **run_paths(name, exist_ok=bool(args.resume))}
    sync_stop = nfs_sync.start(train_args["save_dir"])  # orc reads progress from the mirrored results.csv
    try:
        YOLO(args.resume or args.model).train(
            data=data,
            device=args.device,
            pretrained=pretrained,
            trainer=trainer,
            resume=bool(args.resume),
            **train_args,
        )
    finally:
        sync_stop()


if __name__ == "__main__":
    main()
