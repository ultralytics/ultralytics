#!/usr/bin/env python3
"""Train RT-DETR on merge_data_v8.2_binary, then OOD eval on all 15 MVTec categories.

Matches the YOLOA v8.2 experiment setup — same training data, same OOD categories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import RTDETR

# --- ultra6 server paths ---
V82_DATA_YAML = (
    "/data/shared-datasets/louis_data/AnomalyDataset/"
    "merge_data_v8.2_binary/data.yaml"
)
MVTEC_ROOT = Path("/data/shared-datasets/louis_data/MVTec-YOLO/MVTec-YOLO")
LAPTOP_PREFIX = "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO"

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

OUTPUT_ROOT = Path("/data/shared-datasets/louis_data/MVTec-YOLO/rtdetr_yamls")


def prepare_ood_yaml(category: str) -> Path:
    """Write a server-pathed data YAML for one MVTec category (OOD eval).

    The source train/val lists may contain laptop paths — rewrite them to server paths.
    """
    src_train = MVTEC_ROOT / category / "train.txt"
    src_val = MVTEC_ROOT / category / "val.txt"

    if not src_train.is_file() or not src_val.is_file():
        raise FileNotFoundError(f"Missing {src_train} or {src_val}")

    cat_out = OUTPUT_ROOT / category
    cat_out.mkdir(parents=True, exist_ok=True)

    for src, dst_name in [(src_train, "train.txt"), (src_val, "val.txt")]:
        lines = []
        for ln in src.read_text().splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith(LAPTOP_PREFIX):
                s = str(MVTEC_ROOT) + s[len(LAPTOP_PREFIX):]
            lines.append(s)
        (cat_out / dst_name).write_text("\n".join(lines) + "\n")

    data = {
        "path": str(MVTEC_ROOT),
        "train": str(cat_out / "train.txt"),
        "val": str(cat_out / "val.txt"),
        "nc": 1,
        "names": ["anomaly"],
    }
    yaml_path = cat_out / f"{category}_binary.yaml"
    yaml_path.write_text(yaml.dump(data))
    return yaml_path


def train(args: argparse.Namespace) -> Path:
    """Train RT-DETR on merge_data_v8.2_binary."""
    print(f"Training data: {V82_DATA_YAML}")
    model = RTDETR("rtdetr-l.pt")
    results = model.train(
        data=V82_DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="runs/rtdetr_mvtec",
        name=args.name,
        exist_ok=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        amp=args.amp,
        seed=args.seed,
        save=True,
        save_period=1,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training done. Best: {best_pt}")
    return best_pt


def ood_eval(weights_path: Path, args: argparse.Namespace) -> dict:
    """Run OOD evaluation on all 15 MVTec categories."""
    metrics = {}
    for cat in MVTEC_CATEGORIES:
        data_yaml = prepare_ood_yaml(cat)
        model = RTDETR(str(weights_path))
        result = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project="runs/rtdetr_mvtec",
            name=f"ood_{cat}",
            exist_ok=True,
            split="val",
        )
        metrics[cat] = {
            "mAP50": round(float(result.box.map50), 4),
            "mAP50-95": round(float(result.box.map), 4),
        }
        print(f"  {cat}: mAP50={metrics[cat]['mAP50']:.4f}  mAP50-95={metrics[cat]['mAP50-95']:.4f}")

    map50_vals = [m["mAP50"] for m in metrics.values()]
    map_vals = [m["mAP50-95"] for m in metrics.values()]
    print(f"\nOOD Summary — {len(MVTEC_CATEGORIES)} categories:")
    print(f"  mAP50:      mean={sum(map50_vals)/len(map50_vals):.4f}  "
          f"range=[{min(map50_vals):.4f}, {max(map50_vals):.4f}]")
    print(f"  mAP50-95:   mean={sum(map_vals)/len(map_vals):.4f}  "
          f"range=[{min(map_vals):.4f}, {max(map_vals):.4f}]")

    out_path = Path("runs/rtdetr_mvtec") / "ood_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train RT-DETR on merge_data_v8.2_binary + MVTec OOD eval"
    )
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", type=str, default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--cos_lr", action="store_true", default=True)
    parser.add_argument("--close_mosaic", type=int, default=10)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="v1", help="Experiment name")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--weights", type=str, default=None, help="Weights path for --skip_train")
    args, _ = parser.parse_known_args()

    # DDP guard: only rank 0 runs OOD eval
    import os as _os
    _local_rank = int(_os.environ.get("LOCAL_RANK", -1))
    if args.skip_train:
        if not args.weights:
            raise ValueError("--weights required with --skip_train")
        ood_eval(Path(args.weights), args)
    else:
        best_pt = train(args)
        if _local_rank <= 0:
            print("\n" + "=" * 60)
            print("OOD evaluation on all 15 MVTec categories...")
            print("=" * 60 + "\n")
            ood_eval(best_pt, args)


if __name__ == "__main__":
    main()
