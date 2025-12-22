import argparse
import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Ultralytics validation script")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to .pt weights file (e.g. runs/detect/train/weights/best.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to eval YAML config with val settings",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name for validation (overrides cfg if set)",
    )
    parser.add_argument(
        "--val",
        nargs="*",
        default=[],
        help="Additional Ultralytics val args, e.g. half=True plots=True",
    )
    return parser.parse_args()


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    return overrides


def load_cfg(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def main():
    args = parse_args()
    cfg_overrides = load_cfg(args.config)

    model_path = args.weights or cfg_overrides.pop("model", None)
    if not model_path:
        raise SystemExit("Provide --weights or set `model` in --config.")
    model = YOLO(model_path)

    val_kwargs = dict(cfg_overrides)

    if args.name is not None:
        val_kwargs["name"] = args.name

    val_kwargs.update(parse_overrides(args.val))

    metrics = model.val(**val_kwargs)

    if metrics is not None and hasattr(metrics, "box"):
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
