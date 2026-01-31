"""
Batch evaluation utilities for comparing multiple YOLO models on a single dataset.

Scope:
- N models in -> summary (and optional confidence sweep) out.
- Built on top of the public YOLO().val(...) API.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.utils import LOGGER


@dataclass
class ModelSpec:
    """Resolved model description for batch evaluation."""

    name: str
    weights_path: Path


@dataclass
class BatchEvalResult:
    """Result of evaluating a single model within a batch."""

    model_name: str
    metrics: dict[str, float]


def _iter_model_paths(pattern: str) -> Iterable[Path]:
    """Yield paths for a user-specified model identifier.

    Supports:
    - Explicit file or directory paths.
    - Simple glob patterns like 'runs/detect/train*'.
    """
    p = Path(pattern)
    # If it exists directly, just yield it.
    if p.exists():
        yield p
        return

    # Fallback to globbing relative to CWD.
    has_wildcard = any(ch in pattern for ch in "*?[]")
    if has_wildcard:
        matched = sorted(Path(".").glob(pattern))
        if matched:
            yield from matched
            return

    # If nothing matched, surface a clear error.
    raise FileNotFoundError(f"No files or directories found matching model pattern: {pattern}")


def resolve_models(models: Sequence[str]) -> list[ModelSpec]:
    """Resolve user-provided model identifiers into concrete weight paths.

    Accepts:
        - Run directories (e.g. 'runs/detect/train38') -> uses 'weights/best.pt'.
        - Direct .pt files (e.g. 'path/to/model.pt').
        - Simple glob patterns that expand to either of the above.
    """
    resolved: list[ModelSpec] = []
    for item in models:
        for path in _iter_model_paths(item):
            if path.is_dir():
                best = path / "weights" / "best.pt"
                if not best.exists():
                    raise FileNotFoundError(f"No best.pt found under run dir: {path}")
                resolved.append(ModelSpec(name=path.name, weights_path=best))
            else:
                if not path.exists():
                    raise FileNotFoundError(f"Model weights not found: {path}")
                resolved.append(ModelSpec(name=path.stem, weights_path=path))
    return resolved


def evaluate_model(
    spec: ModelSpec,
    data: str,
    split: str = "val",
    **val_kwargs: Any,
) -> BatchEvalResult:
    """Run Ultralytics val for a single model and extract key metrics.

    This is a thin wrapper around YOLO(weights).val(...).
    """
    LOGGER.info(f"Running batcheval for model '{spec.name}' from {spec.weights_path} on split '{split}'.")
    model = YOLO(str(spec.weights_path))
    metrics = model.val(data=data, split=split, **val_kwargs)

    # Prefer the standardized results_dict interface when available.
    metrics_dict: dict[str, float]
    if hasattr(metrics, "results_dict"):
        metrics_dict = dict(metrics.results_dict)
    elif isinstance(metrics, Mapping):
        metrics_dict = dict(metrics)
    else:
        try:
            metrics_dict = dict(metrics)  # type: ignore[arg-type]
        except TypeError:
            LOGGER.warning("Unexpected metrics object from YOLO.val(); summary will be empty.")
            metrics_dict = {}

    return BatchEvalResult(model_name=spec.name, metrics=metrics_dict)


def _default_save_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / "batcheval" / ts


def _collect_summary_rows(results: Sequence[BatchEvalResult]) -> list[dict[str, Any]]:
    """Normalize batch results into a list of flat dicts suitable for CSV export."""
    rows: list[dict[str, Any]] = []
    for res in results:
        row: dict[str, Any] = {"model_name": res.model_name}
        for k, v in res.metrics.items():
            row[k] = v
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a list of mapping rows to CSV, inferring a unified header."""
    if not rows:
        return

    # Collect union of keys across all rows to keep output stable.
    keys: list[str] = ["model_name"]
    extra_keys: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k == "model_name":
                continue
            extra_keys.add(k)
    keys.extend(sorted(extra_keys))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _run_conf_sweep(
    specs: Sequence[ModelSpec],
    data: str,
    split: str,
    conf_min: float,
    conf_max: float,
    conf_step: float,
    save_root: Path,
    **val_kwargs: Any,
) -> None:
    """Run an optional confidence sweep by re-running validation with different thresholds.

    This keeps the implementation simple and uses the public `conf` argument to YOLO().val(...).
    """
    if conf_step <= 0:
        raise ValueError("conf_step must be > 0 for confidence sweeps.")

    sweep_rows: list[dict[str, Any]] = []
    # Drop any incoming 'conf' override so the sweep's own thresholds take precedence
    # and we don't pass duplicate keyword arguments to YOLO().val(...).
    safe_val_kwargs = {k: v for k, v in val_kwargs.items() if k != "conf"}

    for spec in specs:
        model = YOLO(str(spec.weights_path))
        conf = conf_min
        while conf <= conf_max + 1e-9:
            LOGGER.info(
                f"Running confidence sweep for model '{spec.name}' at conf={conf:.3f} on split '{split}'.",
            )
            metrics = model.val(data=data, split=split, conf=conf, **safe_val_kwargs)
            if hasattr(metrics, "results_dict"):
                metrics_dict = dict(metrics.results_dict)
            elif isinstance(metrics, Mapping):
                metrics_dict = dict(metrics)
            else:
                try:
                    metrics_dict = dict(metrics)  # type: ignore[arg-type]
                except TypeError:
                    metrics_dict = {}

            row: dict[str, Any] = {"model_name": spec.name, "conf": conf}
            row.update(metrics_dict)
            sweep_rows.append(row)
            conf += conf_step

    if sweep_rows:
        _write_csv(save_root / "sweep.csv", sweep_rows)


def batcheval(
    models: Sequence[str],
    data: str,
    split: str = "val",
    sweep_conf: bool = False,
    conf_min: float = 0.05,
    conf_max: float = 0.95,
    conf_step: float = 0.05,
    save_dir: str | Path | None = None,
    **val_kwargs: Any,
) -> list[BatchEvalResult]:
    """Evaluate multiple models on the same dataset split and write a unified report.

    This function returns a list of `BatchEvalResult` objects and writes:
    - `summary.csv` (always): one row per model with flattened scalar metrics.
    - `sweep.csv` (optional): metrics across confidence thresholds when `sweep_conf=True`.
    """
    if not models:
        raise ValueError("No models provided to batcheval().")

    save_root = Path(save_dir) if save_dir is not None else _default_save_dir()
    save_root.mkdir(parents=True, exist_ok=True)

    specs = resolve_models(models)
    if not specs:
        raise ValueError("No models resolved from provided identifiers.")

    LOGGER.info(f"Starting batcheval for {len(specs)} model(s). Results will be written to: {save_root}")
    results: list[BatchEvalResult] = []

    for spec in specs:
        results.append(evaluate_model(spec, data=data, split=split, **val_kwargs))

    # Write main summary table.
    summary_rows = _collect_summary_rows(results)
    _write_csv(save_root / "summary.csv", summary_rows)
    LOGGER.info(f"Batcheval summary written to {save_root / 'summary.csv'}.")

    # Optional confidence sweep.
    if sweep_conf:
        _run_conf_sweep(
            specs=specs,
            data=data,
            split=split,
            conf_min=conf_min,
            conf_max=conf_max,
            conf_step=conf_step,
            save_root=save_root,
            **val_kwargs,
        )

    return results


def run_cli(args: dict[str, Any]) -> list[BatchEvalResult]:
    """CLI entrypoint wrapper.

    Expected args keys:
        models (str | Sequence[str]): comma-separated list or sequence of models.
        data (str): dataset YAML path.
        split (str): dataset split, e.g. 'val' or 'test'.
        sweep_conf (bool): whether to perform confidence sweeps.
        conf_min, conf_max, conf_step (floats): sweep configuration.
        save_dir (str | Path, optional): explicit output directory.
        Other keys are forwarded to YOLO().val(...) as keyword arguments.
    """
    raw_models = args.get("models")
    if not raw_models:
        raise SyntaxError("Missing required 'models' argument for batcheval CLI.")

    if isinstance(raw_models, str):
        models = [m.strip() for m in raw_models.split(",") if m.strip()]
    else:
        models = list(raw_models)

    data = args.get("data")
    if not data:
        raise SyntaxError("Missing required 'data' argument for batcheval CLI.")

    split = args.get("split", "val")
    sweep_conf = bool(args.get("sweep_conf", False))
    conf_min = float(args.get("conf_min", 0.05))
    conf_max = float(args.get("conf_max", 0.95))
    conf_step = float(args.get("conf_step", 0.05))
    save_dir = args.get("save_dir")

    # Everything else is passed through to YOLO().val(...)
    passthrough_keys = {
        "models",
        "data",
        "split",
        "sweep_conf",
        "conf_min",
        "conf_max",
        "conf_step",
        "save_dir",
    }
    val_kwargs = {k: v for k, v in args.items() if k not in passthrough_keys}

    return batcheval(
        models=models,
        data=data,
        split=split,
        sweep_conf=sweep_conf,
        conf_min=conf_min,
        conf_max=conf_max,
        conf_step=conf_step,
        save_dir=save_dir,
        **val_kwargs,
    )
