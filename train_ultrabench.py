# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train and score one Platform-backed Ultra Benchmark multirun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.trainer import MultiTrainer
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.patches import torch_load
from val_ultrabench import DATASETS, FRACTION, evaluate_run


def _validate_predictions(path: Path) -> None:
    """Validate a COCO-format prediction artifact."""
    predictions = json.loads(path.read_text())
    required = {"image_id", "file_name", "category_id", "bbox", "score"}
    if not isinstance(predictions, list) or any(
        not isinstance(prediction, dict)
        or not required <= prediction.keys()
        or not isinstance(prediction["bbox"], list)
        or len(prediction["bbox"]) != 4
        for prediction in predictions
    ):
        raise ValueError(f"Invalid COCO predictions: {path}")


def _load_completed_args(child: Path, uri: str, provenance: dict) -> dict | None:
    """Return validated completion provenance, or None for an incomplete child."""
    predictions_path, args_path = child / "predictions.json", child / "args.yaml"
    if not predictions_path.is_file() or not args_path.is_file():
        return None
    saved = YAML.load(args_path) or {}
    if any(key not in saved for key in (*provenance, "platform_uri", "data", "max_det", "train_metrics")):
        return None
    _validate_predictions(predictions_path)
    mismatched = {key: (saved.get(key), value) for key, value in provenance.items() if saved.get(key) != value}
    if saved["platform_uri"] != uri:
        mismatched["platform_uri"] = (saved["platform_uri"], uri)
    if not saved["data"] or not Path(saved["data"]).is_file():
        mismatched["data"] = (saved["data"], "existing local YAML")
    if not isinstance(saved["train_metrics"], dict) or not saved["train_metrics"]:
        mismatched["train_metrics"] = (saved["train_metrics"], "non-empty metrics dictionary")
    if type(saved["max_det"]) is not int or saved["max_det"] < 1:
        mismatched["max_det"] = (saved["max_det"], "positive integer")
    if mismatched:
        raise ValueError(f"Refusing to skip {child.name} with changed provenance: {mismatched}")
    return saved


def main() -> None:
    """Run one Ultra Benchmark multirun."""
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("model")
    parser.add_argument("name")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    datasets = [line for raw in DATASETS.read_text().splitlines() if (line := raw.strip()) and not line.startswith("#")]
    names = [Path(uri).name for uri in datasets]
    if (
        len(datasets) != 37
        or len(set(datasets)) != 37
        or len(set(names)) != 37
        or not all(uri.startswith("ul://") for uri in datasets)
    ):
        raise ValueError("Ultra Benchmark requires exactly 37 unique Platform dataset URIs")

    run_dir = (args.project / args.name / "multitrain").resolve()
    loaded = YOLO(args.model)
    source_model = Path(loaded.ckpt_path or loaded.cfg)
    source_file = Path(
        loaded.ckpt_path
        or check_yaml(re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", loaded.cfg), hard=False)
        or check_yaml(loaded.cfg)
    ).resolve()
    source_model = source_model.resolve() if source_model.exists() else source_model
    source_hash = hashlib.sha256()
    with source_file.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            source_hash.update(chunk)
    common = {
        "device": args.device,
        "project": str(run_dir.parent),
        "exist_ok": True,
        "epochs": 100,
        "imgsz": args.imgsz,
        "workers": 4,
        "deterministic": True,
        "channels_last": True,
        "fraction": FRACTION,
        "save_json": True,
        "plots": False,
    }
    source_sha256 = source_hash.hexdigest()
    provenance = {
        "source_name": source_model.name,
        "source_sha256": source_sha256,
        **{key: common[key] for key in ("epochs", "imgsz", "fraction", "save_json", "deterministic")},
        "cls_remap": True,
        "seed": 0,
    }
    os.environ.update(ULTRALYTICS_PLATFORM="false", WANDB_LOG_MODEL="false", WANDB_RUN_GROUP=args.name)
    results = {}
    pending = []
    for uri, name in zip(datasets, names):
        child = run_dir / name
        if (completed := _load_completed_args(child, uri, provenance)) is not None:
            results[name] = completed["train_metrics"]
            continue

        if child.exists():
            shutil.rmtree(child)
        pending.append((uri, name))

    if pending:
        trained = loaded.train(data=[uri for uri, _ in pending], **common)
        results.update({name: trained.get(name) for _, name in pending})

    del loaded
    for uri, name in pending:
        if not results[name]:
            continue
        child = run_dir / name
        best, last = (child / "weights" / filename for filename in ("best.pt", "last.pt"))
        checkpoint = best if best.is_file() else last
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing final checkpoint for {name}")
        predictions_path = child / "predictions.json"
        if not predictions_path.is_file():
            predictions_path.write_text("[]")
        _validate_predictions(predictions_path)
        train_args = torch_load(checkpoint)["train_args"]
        resolved_data = Path(train_args["data"]).resolve()
        if not resolved_data.is_file():
            raise FileNotFoundError(f"Resolved dataset YAML is missing for {name}: {resolved_data}")
        saved = YAML.load(child / "args.yaml")
        saved.update(
            data=str(resolved_data),
            platform_uri=uri,
            source_model=str(source_model),
            source_name=source_model.name,
            source_sha256=source_sha256,
            max_det=train_args["max_det"],
            train_metrics={key: float(value) for key, value in results[name].items()},
        )
        YAML.save(child / "args.yaml", saved)

    failed = [name for _, name in pending if not results[name]]
    if failed:
        raise RuntimeError(f"Training failed for {failed}")

    summary = MultiTrainer(None, {"task": "detect"}, None)
    summary.metrics = results
    with tempfile.TemporaryDirectory(dir=run_dir) as temporary:
        summary.save_dir = Path(temporary)
        summary.save_results().replace(run_dir / "multitrain_results.json")
    evaluate_run(run_dir, args.imgsz)
    for name in names:
        weights = run_dir / name / "weights"
        if weights.is_dir():
            shutil.rmtree(weights)


if __name__ == "__main__":
    main()
