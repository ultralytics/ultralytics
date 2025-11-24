# Ultralytics — `yolo batcheval` Starter Pack

A focused, drop-in starter you can paste into the Ultralytics repo (or a fork). Includes design notes, skeleton code, tests, docs snippet, and a draft PR description.

Target: **single PR** that adds a `yolo batcheval` command and supporting Python API.

---

## 📁 Scope

This canvas is intended to live as a single file (e.g., `batcheval_canvas.md`) in a `design/` or `notes/` directory in your fork of Ultralytics.

It contains:

- Design doc (`DESIGN_BATCHEVAL`)
- Implementation sketch (`ultralytics/analytics/batcheval.py` skeleton)
- Tests sketch (`tests/test_batcheval.py` skeleton)
- Docs snippet (`docs/batcheval.md` sketch)
- Draft PR description (`PR_DRAFT_BATCHEVAL`)

You can split these into separate files later if you want, but this single canvas is enough to seed Cursor context.

---

## DESIGN_BATCHEVAL

### Goal

Add a **built-in batch evaluation command** to Ultralytics that:

- Evaluates **multiple YOLO models** on the **same dataset/split** in one go.
- Produces a **unified metrics summary** (CSV) and optionally a **confidence-threshold sweep** table.
- Stays **small and generic** – no DB logging, no dashboards, no Syght/mmWave-specific logic.

This is a “lite”, upstream-friendly version of the LB2YOLO batch test pipeline.

---

### User Story

> As a practitioner training several YOLO models (e.g., different sizes or hyperparams), I want a single command to evaluate all of them on the same dataset split so that I can quickly pick the best model without wiring up external tools or writing my own comparison scripts.

---

### Public API (Proposed)

#### Python

```python
from ultralytics.analytics import batcheval

results = batcheval(
    models=["runs/detect/train38", "runs/detect/train42/weights/best.pt"],
    data="data.yaml",
    split="val",
    sweep_conf=False,
)
```

Parameters:

- `models`: list of run directories and/or `.pt` files.
- `data`: dataset YAML (same as `yolo val`).
- `split`: `"val"` (default), `"test"`, etc.
- `sweep_conf`: whether to run confidence threshold sweeps.
- `conf_min`, `conf_max`, `conf_step`: sweep settings (if enabled).
- `save_dir`: optional; default `runs/batcheval/<timestamp>/`.

Return:

- `List[BatchEvalResult]` (small dataclass with `model_name` and `metrics` dict).

#### CLI

```bash
# Basic: evaluate multiple models on the same dataset split
yolo batcheval models="runs/detect/train38,models/exp2.pt" data=data.yaml split=val

# Using a glob for runs
yolo batcheval models="runs/detect/train*" data=data.yaml split=test

# With optional confidence sweep
yolo batcheval \
  models="runs/detect/train38,models/exp2.pt" \
  data=data.yaml \
  split=test \
  sweep_conf=True \
  conf_min=0.05 \
  conf_max=0.95 \
  conf_step=0.05
```

---

### Design Notes

#### Model Resolution

We accept:

- Run directories (e.g., `runs/detect/train38`) → use `weights/best.pt`.
- Direct `.pt` files (e.g., `models/exp2.pt`).

We normalize into:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelSpec:
    name: str  # for reporting
    weights_path: Path  # full path to .pt
```

#### Evaluation Strategy

For v1 we keep it simple:

- For each `ModelSpec`:
  - `model = YOLO(weights_path)`
  - `metrics = model.val(data=data, split=split, **val_kwargs)`
- We then normalize `metrics` into a plain dict with keys like:
  - `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `metrics/precision(B)`, `metrics/recall(B)`, etc.
- We extract a subset for the summary table:
  - `mAP50`, `mAP50_95`, `precision`, `recall`, maybe `f1` if available.

#### Confidence Sweeps (Optional)

If `sweep_conf=True`, we have two options:

1. **Simple but slower**\
   Re-run validation with different confidence thresholds by passing `conf` or equivalent arg to `model.val()`. This is easiest and keeps code minimal but runs N× slower.

2. **Smarter (if supported)**\
   If Ultralytics exposes a way to get raw predictions and recompute metrics offline, we can:
   - Cache predictions for one run.
   - Sweep thresholds offline using a helper similar to LB2YOLO.

For MVP, (1) is acceptable; we can start with a coarse grid and document the tradeoff.

We write a `sweep.csv` with columns:

- `model_name, class_id, conf, precision, recall, f1`

#### Output Layout

Default root:

```text
runs/batcheval/<timestamp>/
  summary.csv
  sweep.csv      # only if sweep_conf=True
```

`summary.csv` has one row per model:

```text
model_name,mAP50,mAP50_95,precision,recall,f1,...
```

---

### Non-Goals

- No DB logging.
- No external dashboards.
- No RTSP / streaming integration.
- No refactors of core training or validation code.

This should be a **thin wrapper** around public APIs (`YOLO().val(...)`) and simple CSV exports.

````

---

## `ultralytics/analytics/batcheval.py` (skeleton)

```python
"""
Batch evaluation utilities for comparing multiple YOLO models on a single dataset.

Scope:
- N models in -> summary (and optional confidence sweep) out.
- Built on top of existing YOLO().val(...) API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


@dataclass
class ModelSpec:
    name: str
    weights_path: Path


@dataclass
class BatchEvalResult:
    model_name: str
    metrics: dict  # TODO: align with Ultralytics metrics type if needed


def resolve_models(models: Sequence[str]) -> list[ModelSpec]:
    """Resolve user-provided model identifiers into concrete weight paths.

    Accepts:
        - Run directories (e.g. runs/detect/train38) -> uses weights/best.pt
        - Direct .pt files (e.g. path/to/model.pt)
    """
    resolved: list[ModelSpec] = []
    for item in models:
        p = Path(item)
        if p.is_dir():
            best = p / "weights" / "best.pt"
            if not best.exists():
                raise FileNotFoundError(f"No best.pt found under run dir: {p}")
            resolved.append(ModelSpec(name=p.name, weights_path=best))
        else:
            if not p.exists():
                raise FileNotFoundError(f"Model weights not found: {p}")
            resolved.append(ModelSpec(name=p.stem, weights_path=p))
    return resolved


def evaluate_model(
    spec: ModelSpec,
    data: str,
    split: str = "val",
    **val_kwargs,
) -> BatchEvalResult:
    """Run Ultralytics val for a single model and extract key metrics.

    This is a thin wrapper around YOLO(weights).val(...).
    """
    model = YOLO(str(spec.weights_path))
    metrics = model.val(data=data, split=split, **val_kwargs)
    # TODO: inspect metrics object and convert to a stable dict
    metrics_dict = dict(metrics)
    return BatchEvalResult(model_name=spec.name, metrics=metrics_dict)


def _default_save_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / "batcheval" / ts


def batcheval(
    models: Sequence[str],
    data: str,
    split: str = "val",
    sweep_conf: bool = False,
    conf_min: float = 0.05,
    conf_max: float = 0.95,
    conf_step: float = 0.05,
    save_dir: str | Path | None = None,
    **val_kwargs,
) -> list[BatchEvalResult]:
    """Evaluate multiple models on the same dataset split and write a unified report.

    Returns a list of BatchEvalResult objects; also writes summary.csv (and optional sweep.csv) to save_dir.
    """
    if save_dir is None:
        save_root = _default_save_dir()
    else:
        save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    specs = resolve_models(models)
    results: list[BatchEvalResult] = []

    for spec in specs:
        results.append(evaluate_model(spec, data=data, split=split, **val_kwargs))

    # TODO: normalize metrics across models and write summary.csv to save_root
    # TODO: if sweep_conf=True, run confidence sweeps and write sweep.csv

    return results


def run_cli(args: dict):
    """CLI entrypoint wrapper.

    Expected args keys:
        models (str): comma-separated list
        data (str)
        split (str)
        sweep_conf (bool)
        conf_min, conf_max, conf_step (floats)
        save_dir (str, optional)
    """
    models = [m.strip() for m in args["models"].split(",")]
    data = args["data"]
    split = args.get("split", "val")
    sweep_conf = bool(args.get("sweep_conf", False))
    conf_min = float(args.get("conf_min", 0.05))
    conf_max = float(args.get("conf_max", 0.95))
    conf_step = float(args.get("conf_step", 0.05))
    save_dir = args.get("save_dir")

    batcheval(
        models=models,
        data=data,
        split=split,
        sweep_conf=sweep_conf,
        conf_min=conf_min,
        conf_max=conf_max,
        conf_step=conf_step,
        save_dir=save_dir,
    )
````

---

## tests/test_batcheval.py (skeleton)

```python
from ultralytics.analytics.batcheval import ModelSpec, resolve_models


def test_resolve_models_with_pt(tmp_path):
    m = tmp_path / "m.pt"
    m.write_bytes(b"dummy")  # fake weights
    specs = resolve_models([str(m)])
    assert len(specs) == 1
    assert isinstance(specs[0], ModelSpec)
    assert specs[0].weights_path == m
    assert specs[0].name == "m"


def test_resolve_models_with_run_dir(tmp_path):
    run = tmp_path / "train99"
    wdir = run / "weights"
    wdir.mkdir(parents=True)
    best = wdir / "best.pt"
    best.write_bytes(b"dummy")

    specs = resolve_models([str(run)])
    assert len(specs) == 1
    spec = specs[0]
    assert spec.weights_path == best
    assert spec.name == "train99"
```

---

## docs/batcheval.md (sketch)

````md
# yolo batcheval

`yolo batcheval` evaluates multiple YOLO models on the same dataset split and writes a unified metrics summary.

## Examples

```bash
# Compare two specific models
yolo batcheval models="runs/detect/train38,models/exp2.pt" data=data.yaml split=val

# Compare all runs under runs/detect/
yolo batcheval models="runs/detect/train*" data=data.yaml split=test

# With confidence threshold sweep
yolo batcheval \
  models="runs/detect/train38,models/exp2.pt" \
  data=data.yaml \
  split=test \
  sweep_conf=True \
  conf_min=0.05 \
  conf_max=0.95 \
  conf_step=0.05
```
````

Outputs are written to `runs/batcheval/<timestamp>/`:

- `summary.csv` — one row per model with key metrics.
- `sweep.csv` — (optional) metrics vs confidence thresholds.

````

---

## PR_DRAFT_BATCHEVAL

```md
# Draft PR: yolo batcheval

**Title:**
feat: add `yolo batcheval` command for multi-model evaluation

---

## Summary

This PR introduces a new `yolo batcheval` command and corresponding
`ultralytics.analytics.batcheval` module that provide a simple, built-in way to:

- Evaluate **multiple YOLO models** on the **same dataset split**, and
- Produce a unified metrics summary as CSV (and optionally a confidence-threshold sweep).

The goal is to support lightweight, local model comparison without requiring
external experiment tracking tools.

---

## Motivation

- Today, users typically:
  - Run `yolo val` once per model and compare metrics manually, or
  - Rely on external tools (W&B, HUB, MLflow, etc.) for comparisons.
- Many workflows only need a **small local helper** to compare N models on a dataset.
- This PR adds that helper as a thin wrapper around existing public APIs, with minimal risk.

---

## What This PR Does

- Adds `ultralytics/analytics/batcheval.py` with:
  - `batcheval(models, data, split, ...)` Python API.
  - `resolve_models(models)` utility for run dirs + `.pt` files.
  - `run_cli(args)` for CLI integration.
- Extends the `yolo` CLI:
  - Adds a `batcheval` command that accepts:
    - `models`: comma-separated list of runs and/or `.pt` files.
    - `data`: dataset YAML (same as `yolo val`).
    - `split`: validation split (default `"val"`).
    - Optional confidence sweep arguments (`sweep_conf`, `conf_min`, `conf_max`, `conf_step`).
- Writes outputs to `runs/batcheval/<timestamp>/`:
  - `summary.csv` with one row per model and key metrics.
  - `sweep.csv` (if `sweep_conf=True`) with metrics vs confidence thresholds.

---

## What This PR Does NOT Do

- Does not modify or refactor the core training/validation engines.
- Does not add database logging, dashboards, or any external integrations.
- Does not attempt to replace tools like W&B or HUB; it is intentionally a small, local utility.

---

## Testing

- New tests in `tests/test_batcheval.py`:
  - `test_resolve_models_with_pt` — verifies `.pt` inputs resolve correctly.
  - `test_resolve_models_with_run_dir` — verifies run directories resolve to `weights/best.pt`.
- Additional tests can be added (in follow-ups) once we finalize how to normalize the `metrics` object; e.g.:
  - A small integration test that checks `summary.csv` is created and contains one row per model.

---

## Notes

- This feature is designed to be opt-in and low-risk:
  - It only calls public APIs (`YOLO().val(...)`).
  - It writes new files under `runs/` without touching existing outputs.
- Follow-up work (if desired) could:
  - Add richer per-class metrics to `summary.csv`.
  - Integrate with any existing or future `dashboard` functionality that can consume these CSVs.
````
