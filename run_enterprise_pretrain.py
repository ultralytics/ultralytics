# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Launch one arm of the enterprise-corpus vs Objects365v1 detection pretraining ablation.

Both arms share every hyperparameter except the pretrain data and the epoch-denominated keys that follow from it, so
the recipe lives in `cfg/recipes/enterprise-{corpus,o365-control}.yaml` rather than here. The corpus arm trains through
`FederatedDetectionTrainer` (dataset-pure batches, owning-slice cls loss), the o365 arm through the stock trainer.

Usage:
    python run_enterprise_pretrain.py 0,1 yolo26s-p4p5-wide-deep16-sni.yaml corpus
    python run_enterprise_pretrain.py 2,3 yolo26s-p4p5-wide-slim19-ultravit.yaml o365 --no-amp
"""

from __future__ import annotations

import argparse
from pathlib import Path

from callbacks.paths import run_paths
from run_enc_distill_phase2 import _load_recipe
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train_federated import FederatedDetectionTrainer

ARMS = {
    "corpus": ("enterprise-corpus", "/data/shared-datasets/domain-det/_merged/data.yaml", FederatedDetectionTrainer),
    "o365": ("enterprise-o365-control", "/data/shared-datasets/yoloe26_data/Objects365v1/Objects365v1.yaml", None),
}
INIT = Path("/data/shared-datasets/fatih-runs/enterprise-init")


def main() -> None:
    """Resolve the arm's recipe, data and trainer, then train."""
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("device", help="physical GPU ids, e.g. 0,1")
    a.add_argument("model", help="detector yaml under cfg/models/26/, scale must be in the filename")
    a.add_argument("arm", choices=sorted(ARMS))
    a.add_argument("name", nargs="?", help="run name, defaults to ph-<arm>-<model stem>")
    a.add_argument("--pretrained", default=None, help="backbone init .pt, defaults to the arm's staged init")
    a.add_argument("--no-amp", action="store_true", help="ultravit arms, which are unstable under fp16")
    a.add_argument("--epochs", type=int, default=None)
    a.add_argument("--batch", type=int, default=None)
    a.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = a.parse_args()

    recipe_name, data, trainer = ARMS[args.arm]
    recipe = _load_recipe(recipe_name, args.model, epochs=args.epochs, batch=args.batch)
    recipe["amp"] = not args.no_amp
    stem = Path(args.model).stem
    pretrained = args.pretrained or str(INIT / ("ultravit_s.pt" if "ultravit" in stem else "c3k2_s.pt"))
    name = args.name or f"ph-{args.arm}-{stem.replace('yolo26s-p4p5-wide-', '')}"
    print(f"[arm] {args.arm}  model={args.model}  init={pretrained}  data={data}")
    YOLO(args.resume or args.model).train(
        data=data,
        device=args.device,
        pretrained=pretrained,
        trainer=trainer,
        resume=bool(args.resume),
        **recipe,
        **run_paths(name, exist_ok=bool(args.resume)),
    )


if __name__ == "__main__":
    main()
