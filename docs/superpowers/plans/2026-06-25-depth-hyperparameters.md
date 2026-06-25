# Depth Hyperparameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the seven `DEPTH_*` environment variables with proper Ultralytics cfg hyperparameters read from `model.args` / `self.args`.

**Architecture:** Add the keys to `cfg/default.yaml`, register their types in `cfg/__init__.py`, and change the five `os.environ.get` read sites (loss, val, calibrate, train) to read from args — mirroring how `v8DetectionLoss` reads `box`/`cls`/`dfl`. Clean break: no env fallback.

**Tech Stack:** Python, PyTorch, Ultralytics cfg system, pytest.

## Global Constraints

- **Execution host:** all edits and commands target the **depth_dev** host — repo at `/root/autodl-tmp/ultralytics_depth`, branch `depth_anything`, python = `~/miniconda3/bin/python`. Run pytest over ssh, e.g. `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -m pytest …'`.
- **Defaults must equal current env defaults** (zero behavior change when unset): `silog=1.0`, `silog_grad=0.5`, `silog_lambda=0.5`, `silog_l1=0.0`, `dist_pw=0.0`, `cal_dist_pw=0.0`, `auto_calibrate=True`.
- **Naming:** descriptive bare keys, no task prefix; comment each `(depth tasks)`.
- **No env fallback** — remove every `DEPTH_*` `os.environ` read.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Add depth cfg keys + register types

**Files:**
- Modify: `ultralytics/cfg/default.yaml` (end of Hyperparameters section, after `kobj`/`rle`/`angle` block, before `nbs`)
- Modify: `ultralytics/cfg/__init__.py` (`CFG_FLOAT_KEYS`, `CFG_BOOL_KEYS`)
- Test: `tests/depth/test_cfg.py`

**Interfaces:**
- Produces: cfg args `silog`, `silog_grad`, `silog_lambda`, `silog_l1`, `dist_pw`, `cal_dist_pw` (float) and `auto_calibrate` (bool), available on any `get_cfg()` namespace and on `model.args` / `self.args`.

- [ ] **Step 1: Write the failing test**

Add to `tests/depth/test_cfg.py`:
```python
def test_depth_hyperparameters_in_default_cfg():
    """Depth loss/calibration knobs are real cfg args with the documented defaults."""
    from ultralytics.cfg import get_cfg

    args = get_cfg()
    assert args.silog == 1.0
    assert args.silog_grad == 0.5
    assert args.silog_lambda == 0.5
    assert args.silog_l1 == 0.0
    assert args.dist_pw == 0.0
    assert args.cal_dist_pw == 0.0
    assert args.auto_calibrate is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -m pytest tests/depth/test_cfg.py::test_depth_hyperparameters_in_default_cfg -v'`
Expected: FAIL — `AttributeError: 'IterableSimpleNamespace' object has no attribute 'silog'`.

- [ ] **Step 3: Add the keys to `default.yaml`**

In `ultralytics/cfg/default.yaml`, immediately after the `angle: 1.0 …` line (end of the loss-gain group, before `nbs:`), insert:
```yaml
silog: 1.0          # (float) SILog depth loss gain (depth tasks)
silog_grad: 0.5     # (float) gradient/edge depth loss gain (depth tasks)
silog_lambda: 0.5   # (float) SILog variance focus: 1.0=scale-invariant, 0.0=log-RMSE (depth tasks)
silog_l1: 0.0       # (float) scale-anchored L1 depth loss gain (depth tasks)
dist_pw: 0.0        # (float) far-pixel distance weighting power in depth loss (depth tasks)
cal_dist_pw: 0.0    # (float) distance weighting power in depth scale calibration (depth tasks)
auto_calibrate: True # (bool) auto-calibrate depth output scale after training (depth tasks)
```

- [ ] **Step 4: Register the types in `cfg/__init__.py`**

In `CFG_FLOAT_KEYS` frozenset, add the six float keys (place after `"dfl",`):
```python
        "silog",
        "silog_grad",
        "silog_lambda",
        "silog_l1",
        "dist_pw",
        "cal_dist_pw",
```
In `CFG_BOOL_KEYS` frozenset, add (after `"end2end",`):
```python
        "auto_calibrate",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -m pytest tests/depth/test_cfg.py -v'`
Expected: PASS (all tests in file).

- [ ] **Step 6: Commit**

```bash
ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && git add ultralytics/cfg/default.yaml ultralytics/cfg/__init__.py tests/depth/test_cfg.py && git commit -m "feat(depth): add depth loss/calibration hyperparameters to cfg

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"'
```

---

### Task 2: Migrate v8DepthLoss to read model.args

**Files:**
- Modify: `ultralytics/utils/loss.py` (`v8DepthLoss.__init__`, ~lines 1159–1185)
- Test: `tests/depth/test_loss.py` (rewrite helper)

**Interfaces:**
- Consumes: `model.args.silog`, `.silog_grad`, `.silog_lambda`, `.silog_l1`, `.dist_pw` (Task 1).
- Produces: `v8DepthLoss(model)` requires `model.args` to carry those five keys (no env reads).

- [ ] **Step 1: Rewrite the test helper to set `model.args` instead of env**

Replace the top of `tests/depth/test_loss.py` (imports + `_loss_for_scaled_pred`) with:
```python
from types import SimpleNamespace

import torch

from ultralytics.utils.loss import v8DepthLoss


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))


def _loss_for_scaled_pred(lam, scale, l1=0.0):
    model = _Tiny()
    # grad weight 0 isolates the silog/scale terms
    model.args = SimpleNamespace(
        silog=1.0, silog_grad=0.0, silog_lambda=lam, silog_l1=l1, dist_pw=0.0
    )
    crit = v8DepthLoss(model)
    gt = torch.rand(2, 1, 16, 16) * 5 + 1.0
    pred = (gt * scale).clone().requires_grad_(True)  # perfect structure, wrong global scale
    total, _ = crit({"depth": pred}, {"depth": gt})
    return float(total)
```
Then update the two test bodies to drop the `monkeypatch` argument and the `l1="0"`/`l1="1.0"` strings (use floats):
```python
def test_lower_lambda_penalizes_scale_error_more():
    """A globally scale-shifted prediction is ~free under scale-invariant silog (lambda=1) but
    must be heavily penalized as lambda drops (loss becomes scale-dependent)."""
    loss_invariant = _loss_for_scaled_pred(lam=1.0, scale=2.0)
    loss_anchored = _loss_for_scaled_pred(lam=0.15, scale=2.0)
    assert loss_invariant < 0.05
    assert loss_anchored > 5 * max(loss_invariant, 1e-6)


def test_l1_weight_adds_scale_penalty():
    """The scale-anchored L1 term penalizes a scale shift even when silog is scale-invariant."""
    no_l1 = _loss_for_scaled_pred(lam=1.0, scale=2.0, l1=0.0)
    with_l1 = _loss_for_scaled_pred(lam=1.0, scale=2.0, l1=1.0)
    assert with_l1 > no_l1 + 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -m pytest tests/depth/test_loss.py -v'`
Expected: FAIL — the loss still calls `os.environ.get`, ignoring `model.args.silog_lambda`, so `test_lower_lambda_penalizes_scale_error_more` fails (both losses computed at the default lambda 0.5).

- [ ] **Step 3: Change `v8DepthLoss.__init__` to read `model.args`**

In `ultralytics/utils/loss.py`, in `v8DepthLoss.__init__`, replace the `import os` line and the five `os.environ.get(...)` assignments with:
```python
        device = next(model.parameters()).device
        self.device = device
        h = model.args  # hyperparameters
        self.silog_weight = h.silog
        self.grad_weight = h.silog_grad
        # SILog variance-focus: 1.0 = fully scale-invariant, 0.0 = plain log-RMSE (scale-dependent).
        self.silog_lambda = h.silog_lambda
        # Optional scale-anchored L1 term on the log-depth residual (penalizes absolute offset).
        self.l1_weight = h.silog_l1
        # Depth-distance weighting: weight each valid pixel by gt**dist_power (normalized to mean 1).
        self.dist_power = h.dist_pw
```
Leave the `self.max_depth` detection loop below unchanged. Confirm no other use of `os` remains in this method (the `import os` was local to it).

- [ ] **Step 4: Run test to verify it passes**

Run: `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -m pytest tests/depth/test_loss.py -v'`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && git add ultralytics/utils/loss.py tests/depth/test_loss.py && git commit -m "refactor(depth): read depth loss gains from model.args, not env

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"'
```

---

### Task 3: Migrate calibration + training env vars to args

**Files:**
- Modify: `ultralytics/models/yolo/depth/calibrate.py` (`fit_calibration` ~line 55, `calibrate_checkpoint` ~line 129)
- Modify: `ultralytics/models/yolo/depth/val.py` (`get_stats`, ~lines 95–108)
- Modify: `ultralytics/models/yolo/depth/train.py` (`final_eval`, ~lines 100–120)
- Test: `tests/depth/test_no_env.py` (new — regression guard)

**Interfaces:**
- Consumes: `self.args.cal_dist_pw`, `self.args.auto_calibrate` (Task 1).
- Produces: `fit_calibration(..., dist_power=0.0)` and `calibrate_checkpoint(..., dist_power=0.0)` new keyword params.

- [ ] **Step 1: Write the failing guard test**

Create `tests/depth/test_no_env.py`:
```python
import pathlib
import re


def test_no_depth_env_reads_in_source():
    """All DEPTH_* knobs are cfg args now; no os.environ reads of them may remain."""
    root = pathlib.Path(__file__).resolve().parents[2] / "ultralytics"
    offenders = []
    pattern = re.compile(r"(environ|getenv).*DEPTH_|DEPTH_.*(environ|getenv)")
    for path in root.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, "Found DEPTH_* env reads:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -m pytest tests/depth/test_no_env.py -v'`
Expected: FAIL — lists the remaining reads in `calibrate.py`, `val.py`, `train.py`.

- [ ] **Step 3: Thread `dist_power` through `calibrate.py`**

In `fit_calibration`, change the signature to:
```python
def fit_calibration(model, dataloader, device, max_images: int = 200, set_buffers: bool = True, dist_power: float = 0.0):
```
and replace the `import os` + `lstsq_affine(...)` env call with:
```python
    a, b = lstsq_affine(np.concatenate(logp_all), np.concatenate(logg_all), dist_power=dist_power)
```
In `calibrate_checkpoint`, change the signature to:
```python
def calibrate_checkpoint(ckpt_path, dataloader, device, dist_power: float = 0.0) -> None:
```
and pass it through:
```python
    res = fit_calibration(work, dataloader, device, set_buffers=True, dist_power=dist_power)
```

- [ ] **Step 4: Update `val.py` `get_stats`**

Replace the `import os` + env read in the `lstsq_affine` call with `self.args.cal_dist_pw`:
```python
            self.calib = lstsq_affine(
                np.concatenate(self._cal_logp), np.concatenate(self._cal_logg),
                dist_power=self.args.cal_dist_pw,
            )
```

- [ ] **Step 5: Update `train.py` `final_eval`**

Replace the env guard and the `calibrate_checkpoint` call:
```python
        super().final_eval()
        if RANK not in {-1, 0} or not self.args.auto_calibrate:
            return
        try:
            from .calibrate import calibrate_checkpoint

            LOGGER.info("Auto-calibrating depth output scale on the validation set...")
            for ckpt in (self.best, self.last):
                if ckpt.exists():
                    calibrate_checkpoint(ckpt, self.test_loader, self.device, dist_power=self.args.cal_dist_pw)
```
Update the docstring line `Disable with the environment variable ``DEPTH_AUTO_CALIBRATE=0``.` →
`Disable with ``auto_calibrate=False``.`
Then check whether `os` is still used elsewhere in `train.py`: run `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && grep -n "os\\." ultralytics/models/yolo/depth/train.py'`. If no matches remain, remove the now-unused `import os`.

- [ ] **Step 6: Run guard + existing calibration/validator tests**

Run: `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -m pytest tests/depth/test_no_env.py tests/depth/test_calibration.py tests/depth/test_validator.py -v'`
Expected: PASS — guard finds no offenders; calibration/validator tests still pass (new params default to 0.0).

- [ ] **Step 7: Commit**

```bash
ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && git add ultralytics/models/yolo/depth/calibrate.py ultralytics/models/yolo/depth/val.py ultralytics/models/yolo/depth/train.py tests/depth/test_no_env.py && git commit -m "refactor(depth): read calibration/auto-calibrate settings from args, not env

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"'
```

---

### Task 4: Full-suite verification + end-to-end smoke

**Files:** none (verification only).

- [ ] **Step 1: Run the full depth test suite**

Run: `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -m pytest tests/depth/ -q'`
Expected: all pass (≥46 — the 44 prior + the new cfg test + the guard).

- [ ] **Step 2: End-to-end smoke — args flow into the loss**

Run:
```bash
ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && ~/miniconda3/bin/python -c "
from ultralytics import YOLO
from ultralytics.utils.loss import v8DepthLoss
m = YOLO(\"ultralytics/cfg/models/26/yolo26-depth-log.yaml\")
m.model.args = __import__(\"ultralytics.cfg\", fromlist=[\"get_cfg\"]).get_cfg(overrides={\"silog_lambda\": 0.3, \"cal_dist_pw\": 2.0})
loss = v8DepthLoss(m.model)
assert loss.silog_lambda == 0.3, loss.silog_lambda
print(\"OK silog_lambda flows:\", loss.silog_lambda)
"'
```
Expected: prints `OK silog_lambda flows: 0.3`.

- [ ] **Step 3: Confirm no `DEPTH_` env references remain anywhere**

Run: `ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && grep -rIn "DEPTH_[A-Z]" ultralytics/ tests/ || echo CLEAN'`
Expected: `CLEAN` (or only unrelated matches like `DEPTH_*` in dataset-path comments — review any output).

- [ ] **Step 4: Final commit if any verification touch-ups were needed**

If no changes were required, nothing to commit. Otherwise:
```bash
ssh depth_dev 'cd ~/autodl-tmp/ultralytics_depth && git add -A && git commit -m "test(depth): verify hyperparameter migration end-to-end

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"'
```

---

## Self-Review

**Spec coverage:**
- default.yaml keys → Task 1 ✓
- cfg type registration → Task 1 ✓
- loss reads args → Task 2 ✓
- val.py cal_dist_pw → Task 3 (Step 4) ✓
- calibrate.py dist_power param → Task 3 (Step 3) ✓
- train.py auto_calibrate + cal_dist_pw passthrough + docstring → Task 3 (Step 5) ✓
- test_loss.py rewrite → Task 2 ✓
- clean break / no env fallback → enforced by Task 3 guard test ✓
- verification (suite, smoke, grep) → Task 4 ✓

**Placeholder scan:** none — all steps contain concrete code/commands.

**Type consistency:** `dist_power` kwarg name matches `lstsq_affine`'s existing `dist_power`; `fit_calibration`/`calibrate_checkpoint` use the same name; args keys (`silog`, `silog_grad`, `silog_lambda`, `silog_l1`, `dist_pw`, `cal_dist_pw`, `auto_calibrate`) are identical across default.yaml, cfg type sets, loss/val/train reads, and tests.
