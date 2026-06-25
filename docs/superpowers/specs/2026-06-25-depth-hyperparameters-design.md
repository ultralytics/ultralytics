# Depth hyperparameters: migrate DEPTH_* env vars to cfg

**Date:** 2026-06-25
**Branch:** depth_anything
**Status:** Approved (pending spec review)

## Problem

Seven `DEPTH_*` environment variables currently control depth training/calibration
behavior. They are read once at object-construction time via `os.environ.get`, so they
are effectively launch-time config that lives outside the normal Ultralytics cfg system:
they cannot be set in a model/data/args YAML, do not appear in `default.yaml`, are not
type-validated, and are invisible to anyone reading the config surface. This makes runs
hard to reproduce and inconsistent with every other loss gain / training knob in the repo.

## Goal

Make all seven proper Ultralytics hyperparameters in `cfg/default.yaml`, read from
`model.args` / `self.args` like every other gain (`box`, `cls`, `dfl`, …). Remove the
env-var reads entirely (clean break — no fallback). Behavior is unchanged when no value
is passed, because the new defaults equal the current env defaults.

## Naming convention

Follows the existing `default.yaml` loss-gain convention: short, lowercase, **unprefixed**
by task (e.g. `pose`/`kobj`/`angle` are not `pose_`/`obb_`); the task is noted in the
comment via `(depth tasks)`. Sub-parameters take a descriptive bare name.

| Old env var | New cfg key | Default | Type | Read by |
|---|---|---|---|---|
| `DEPTH_SILOG_WEIGHT`   | `silog`        | `1.0`  | float | loss |
| `DEPTH_GRAD_WEIGHT`    | `silog_grad`   | `0.5`  | float | loss |
| `DEPTH_SILOG_LAMBDA`   | `silog_lambda` | `0.5`  | float | loss |
| `DEPTH_L1_WEIGHT`      | `silog_l1`     | `0.0`  | float | loss |
| `DEPTH_DIST_POWER`     | `dist_pw`      | `0.0`  | float | loss |
| `DEPTH_CAL_DIST_POWER` | `cal_dist_pw`  | `0.0`  | float | val / calibrate |
| `DEPTH_AUTO_CALIBRATE` | `auto_calibrate` | `True` | bool | train |

## Changes

### 1. `cfg/default.yaml`
Add a "Depth" block at the end of the Hyperparameters section:
```yaml
silog: 1.0          # (float) SILog depth loss gain (depth tasks)
silog_grad: 0.5     # (float) gradient/edge depth loss gain (depth tasks)
silog_lambda: 0.5   # (float) SILog variance focus: 1.0=scale-invariant, 0.0=log-RMSE (depth tasks)
silog_l1: 0.0       # (float) scale-anchored L1 depth loss gain (depth tasks)
dist_pw: 0.0        # (float) far-pixel distance weighting power in depth loss (depth tasks)
cal_dist_pw: 0.0    # (float) distance weighting power in depth scale calibration (depth tasks)
auto_calibrate: True # (bool) auto-calibrate depth output scale after training (depth tasks)
```

### 2. `cfg/__init__.py` (type validation)
- Add to `CFG_FLOAT_KEYS`: `silog`, `silog_grad`, `silog_lambda`, `silog_l1`, `dist_pw`, `cal_dist_pw`.
- Add to `CFG_BOOL_KEYS`: `auto_calibrate`.

### 3. `utils/loss.py` — `v8DepthLoss.__init__`
Replace the five `os.environ.get(...)` reads and the local `import os` with reads from
`model.args` (mirroring `v8DetectionLoss`: `h = model.args`):
- `self.silog_weight = h.silog`
- `self.grad_weight = h.silog_grad`
- `self.silog_lambda = h.silog_lambda`
- `self.l1_weight = h.silog_l1`
- `self.dist_power = h.dist_pw`

(Internal attribute names may stay as-is; only the source changes.)

### 4. `models/yolo/depth/val.py` — `get_stats`
`dist_power=float(os.environ.get("DEPTH_CAL_DIST_POWER", 0.0))` → `dist_power=self.args.cal_dist_pw`.
Drop the local `import os` if otherwise unused.

### 5. `models/yolo/depth/calibrate.py` — `calibrate_checkpoint`
Add parameter `dist_power: float = 0.0`; use it instead of the env read when calling
`lstsq_affine`. Drop the local `import os` if otherwise unused.

### 6. `models/yolo/depth/train.py` — `final_eval`
- `os.environ.get("DEPTH_AUTO_CALIBRATE", "1") == "0"` → `not self.args.auto_calibrate` (bool).
- Pass `dist_power=self.args.cal_dist_pw` into `calibrate_checkpoint(...)`.
- Update the docstring that references `DEPTH_AUTO_CALIBRATE=0`.

### 7. Tests — `tests/depth/test_loss.py`
Replace the `monkeypatch.setenv("DEPTH_*", …)` helper with one that attaches a small
`model.args` namespace carrying the five loss keys (`silog`, `silog_grad`, `silog_lambda`,
`silog_l1`, `dist_pw`). No other depth test reads these vars.

## Out of scope (YAGNI)
- No env-var fallback / deprecation path (clean break, per decision).
- No new wiring into model `.yaml` files — these are train-time args like `box`/`cls`.
- No changes to the calibration algorithm or loss math; values and defaults are preserved.

## Verification
- `pytest tests/depth/` → all pass (test_loss rewritten, others unchanged).
- Smoke: build a depth model and run one train step; confirm `model.args.silog_lambda`
  etc. are present and flow into the loss.
- Grep: no remaining `DEPTH_` env references in `ultralytics/`.
