# Calibrated Val Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After depth auto-calibration in `final_eval`, write `val_batch{0,1,2}_calibrated.jpg` (4-column `RGB | GT | raw | calibrated` panels) to the run dir.

**Architecture:** Extract the panel-rendering from `DepthValidator.plot_predictions` into a reusable module-level `plot_depth_panels()` (val.py). `calibrate_checkpoint()` gains a `plot_dir` param: after fitting, it runs the loaded float copy with identity calibration for a raw forward, computes the calibrated column as `exp(a·log(raw)+b)`, and writes the panels. `DepthTrainer.final_eval` passes `plot_dir=save_dir` for the checkpoint that represents the run (best.pt, falling back to last.pt).

**Tech Stack:** PyTorch, OpenCV, NumPy, pytest. Spec: `docs/superpowers/specs/2026-07-03-depth-calibrated-val-plots-design.md`.

## Global Constraints

- No env-var configuration — all behavior via explicit function params / `args` (repo policy, see `tests/depth/test_no_env.py`).
- A plot failure must never break calibration or training: plot calls wrapped in try/except → `LOGGER.warning`.
- Plotting only on `RANK in {-1, 0}` and only when `self.args.plots` is set (final_eval is already RANK-gated).
- Existing `val_batch{ni}.jpg` output layout must not change (3 columns, no header strip).
- Header strip is 24 px tall, white background, black `cv2.putText` labels (font `FONT_HERSHEY_SIMPLEX`, scale 0.5). Use ASCII `x` (not `×`) in the scale label — cv2 can't render non-ASCII.
- Repo comment style: docstrings explain *why*, not just what. Match surrounding code.
- Run tests with the repo venv's pytest: `python -m pytest` from repo root `/home/rick/ultralytics_depth`.

---

### Task 1: `plot_depth_panels()` renderer + `plot_predictions` refactor

**Files:**
- Modify: `ultralytics/models/yolo/depth/val.py` (replace `plot_predictions` at lines 161–193; add module-level `plot_depth_panels` after the class)
- Test: `tests/depth/test_plots.py` (create)

**Interfaces:**
- Consumes: `DepthValidator._colorize_depth(depth, vmin, vmax)` (existing staticmethod, unchanged).
- Produces: `plot_depth_panels(imgs, gt, preds, fname, titles=None, max_images=4)` — module-level in `ultralytics/models/yolo/depth/val.py`. `imgs`: `(B,3,H,W)` float in [0,1]; `gt`: `(B,1,H,W)` or `(B,H,W)`; `preds`: **list** of `(B,1,H,W)` or `(B,H,W)` tensors, one extra column each; `fname`: `Path`; `titles`: optional list of `2 + len(preds)` strings → 24 px header strip. Task 2 imports this.

- [ ] **Step 1: Write the failing tests**

Create `tests/depth/test_plots.py`:

```python
"""Unit tests for depth panel plotting (val_batch grids and calibrated variants)."""

import cv2
import torch

from ultralytics.models.yolo.depth.val import DepthValidator, plot_depth_panels


def test_plot_depth_panels_writes_grid(tmp_path):
    """Grid = one row per image, columns RGB | GT | one per preds entry, panel size = img size."""
    imgs = torch.rand(2, 3, 32, 32)
    gt = torch.rand(2, 32, 32) * 5 + 0.5
    preds = [torch.rand(2, 1, 32, 32) * 5 + 0.5, torch.rand(2, 32, 32) * 5 + 0.5]  # both shapes accepted
    fname = tmp_path / "panels.jpg"
    plot_depth_panels(imgs, gt, preds, fname)
    img = cv2.imread(str(fname))
    assert img is not None
    assert img.shape == (2 * 32, 4 * 32, 3)  # 2 rows, RGB|GT|pred|pred


def test_plot_depth_panels_titles_add_header_strip(tmp_path):
    """Passing titles prepends a 24 px labeled header strip."""
    imgs = torch.rand(1, 3, 32, 32)
    gt = torch.rand(1, 32, 32) * 5 + 0.5
    fname = tmp_path / "panels.jpg"
    plot_depth_panels(imgs, gt, [torch.rand(1, 1, 32, 32) * 5], fname, titles=["RGB", "GT", "pred"])
    img = cv2.imread(str(fname))
    assert img.shape == (24 + 32, 3 * 32, 3)


def test_plot_depth_panels_respects_max_images(tmp_path):
    """Rows are capped at max_images."""
    imgs = torch.rand(6, 3, 32, 32)
    gt = torch.rand(6, 32, 32) * 5 + 0.5
    fname = tmp_path / "panels.jpg"
    plot_depth_panels(imgs, gt, [torch.rand(6, 1, 32, 32) * 5], fname, max_images=4)
    img = cv2.imread(str(fname))
    assert img.shape == (4 * 32, 3 * 32, 3)


def test_plot_predictions_layout_unchanged(tmp_path):
    """The validator wrapper still writes 3-column val_batch{ni}.jpg with no header strip."""
    v = DepthValidator.__new__(DepthValidator)  # skip __init__ (needs full args); wrapper only uses save_dir
    v.save_dir = tmp_path
    batch = {"img": torch.rand(2, 3, 32, 32), "depth": torch.rand(2, 32, 32) * 5 + 0.5}
    preds = {"depth": torch.rand(2, 1, 32, 32) * 5 + 0.5}
    v.plot_predictions(batch, preds, ni=0)
    img = cv2.imread(str(tmp_path / "val_batch0.jpg"))
    assert img.shape == (2 * 32, 3 * 32, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/depth/test_plots.py -v`
Expected: 4 FAIL/ERROR with `ImportError: cannot import name 'plot_depth_panels'`.

- [ ] **Step 3: Implement**

In `ultralytics/models/yolo/depth/val.py`, replace the whole `plot_predictions` method (currently lines 161–193) with:

```python
    def plot_predictions(self, batch, preds, ni, max_images: int = 4):
        """Save a RGB | GT depth | predicted depth panel for the batch to val_batch{ni}.jpg.

        Depth has no boxes/classes, so the detection-style plotters are replaced with a
        side-by-side depth visualization (see plot_depth_panels). Called by BaseValidator
        for the first few batches when args.plots is set.
        """
        if "depth" not in batch:
            return
        try:
            plot_depth_panels(
                batch["img"], batch["depth"], [self._extract_pred(preds)],
                self.save_dir / f"val_batch{ni}.jpg", max_images=max_images,
            )
        except Exception as e:
            LOGGER.warning(f"DepthValidator: failed to plot val_batch{ni}: {e}")
```

Then append at the end of the file (module level, after the class — it references `DepthValidator._colorize_depth`, resolved at call time):

```python
def plot_depth_panels(imgs, gt, preds, fname, titles=None, max_images: int = 4):
    """Write a depth panel grid: one row per image, columns RGB | GT | one per entry of ``preds``.

    All depth columns share the GT valid-pixel range per row, so a scale error between GT and any
    prediction shows up directly as a color mismatch. Panels are resized to the RGB image size,
    so predictions at head stride need no prior interpolation.

    Args:
        imgs: (B,3,H,W) float image tensor in [0,1].
        gt: (B,1,H,W) or (B,H,W) ground-truth depth in meters (pixels <= 0 are invalid, drawn black).
        preds: List of (B,1,H,W) or (B,H,W) predicted depth tensors; each adds one column.
        fname: Output image path.
        titles: Optional list of ``2 + len(preds)`` column labels, drawn in a 24 px header strip.
            None (the val_batch{ni}.jpg default) keeps the historical strip-free layout.
        max_images: Maximum number of rows.
    """
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    preds = [p.unsqueeze(1) if p.ndim == 3 else p for p in preds]
    h, w = imgs.shape[-2:]
    rows = []
    for i in range(min(imgs.shape[0], max_images)):
        rgb = (imgs[i].detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        panels = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
        g = gt[i, 0]
        gv = g[g > 0]
        vmin = float(gv.min()) if gv.numel() else 0.0
        vmax = float(gv.max()) if gv.numel() else 1.0
        for d in [g] + [p[i, 0] for p in preds]:
            panels.append(cv2.resize(DepthValidator._colorize_depth(d, vmin, vmax), (w, h), interpolation=cv2.INTER_NEAREST))
        rows.append(np.hstack(panels))
    grid = np.vstack(rows)
    if titles:
        strip = np.full((24, grid.shape[1], 3), 255, dtype=np.uint8)
        for j, t in enumerate(titles):
            cv2.putText(strip, str(t), (j * w + 4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        grid = np.vstack([strip, grid])
    cv2.imwrite(str(fname), grid)
```

`cv2` and `np` are already imported at the top of val.py. Do not touch `_colorize_depth` — `DepthTrainer.plot_training_samples` uses it via the class.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/depth/test_plots.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Run the neighboring depth tests to catch regressions**

Run: `python -m pytest tests/depth/ tests/test_depth_calibration.py -q`
Expected: all PASS (the refactor changes no observable behavior of `plot_predictions`).

- [ ] **Step 6: Commit**

```bash
git add ultralytics/models/yolo/depth/val.py tests/depth/test_plots.py
git commit -m "refactor(depth): extract reusable plot_depth_panels from val plotting"
```

---

### Task 2: Calibrated plot pass in `calibrate.py`

**Files:**
- Modify: `ultralytics/models/yolo/depth/calibrate.py` (add `_plot_calibrated_batches`; extend `calibrate_checkpoint` signature/body at lines 297–329; add `from pathlib import Path` to the top-level imports)
- Test: `tests/depth/test_plots.py` (append)

**Interfaces:**
- Consumes: `plot_depth_panels(imgs, gt, preds, fname, titles=None, max_images=4)` from Task 1; existing `_depth_head`, `_extract`, `fit_calibration_selective`.
- Produces: `calibrate_checkpoint(ckpt_path, dataloader, device, dist_power=0.0, plot_dir=None)` — when `plot_dir` (str | Path) is set and calibration succeeded, writes `val_batch{0..2}_calibrated.jpg` there. Task 3 passes `plot_dir`.

- [ ] **Step 1: Write the failing test**

Append to `tests/depth/test_plots.py`:

```python
def test_calibrate_checkpoint_writes_calibrated_plots(tmp_path):
    """calibrate_checkpoint(plot_dir=...) writes 4-column val_batch{ni}_calibrated.jpg panels."""
    from ultralytics.models.yolo.depth.calibrate import calibrate_checkpoint
    from ultralytics.nn.tasks import DepthModel

    torch.manual_seed(0)
    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    batches = [
        {"img": (torch.rand(2, 3, 64, 64) * 255).to(torch.uint8), "depth": torch.rand(2, 64, 64) * 5 + 0.5}
        for _ in range(4)
    ]
    path = tmp_path / "ckpt.pt"
    torch.save({"model": model}, path)
    calibrate_checkpoint(path, batches, device="cpu", plot_dir=tmp_path)
    for ni in range(3):  # max_batches=3 even though 4 batches are available
        img = cv2.imread(str(tmp_path / f"val_batch{ni}_calibrated.jpg"))
        assert img is not None, f"val_batch{ni}_calibrated.jpg missing"
        assert img.shape == (24 + 2 * 64, 4 * 64, 3)  # header strip + 2 rows, RGB|GT|raw|calibrated
    assert not (tmp_path / "val_batch3_calibrated.jpg").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/depth/test_plots.py::test_calibrate_checkpoint_writes_calibrated_plots -v`
Expected: FAIL with `TypeError: calibrate_checkpoint() got an unexpected keyword argument 'plot_dir'`.

- [ ] **Step 3: Implement**

In `ultralytics/models/yolo/depth/calibrate.py`, add to the top-level imports (after `from __future__ import annotations`):

```python
from pathlib import Path
```

Insert this function directly above `calibrate_checkpoint`:

```python
def _plot_calibrated_batches(model, dataloader, device, a, b, name, plot_dir, max_batches: int = 3, max_images: int = 4):
    """Write ``val_batch{ni}_calibrated.jpg`` panels (RGB | GT | raw | calibrated) to ``plot_dir``.

    Runs the model with calibration buffers at identity to get the raw prediction; the calibrated
    column is its deterministic affine ``exp(a·log(raw) + b)`` — no second forward. The first
    ``max_batches`` batches are the same ones BaseValidator plots as ``val_batch{ni}.jpg`` (val
    loaders are not shuffled), so the files are directly comparable. With the "only if it helps"
    policy the selected ``name`` may be ``identity``; the panels are still written (raw ==
    calibrated), which documents that calibration was a no-op. Buffers are restored afterwards.
    """
    from .val import plot_depth_panels

    head = _depth_head(model)
    a0, b0 = float(head.cal_a), float(head.cal_b)
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)
    model = model.to(device).eval()
    titles = ["RGB", "GT", "raw", f"calibrated ({name} x{np.exp(b):.2f})"]
    plot_dir = Path(plot_dir)
    with torch.no_grad():
        for ni, batch in enumerate(dataloader):
            if ni >= max_batches:
                break
            img = batch["img"].to(device).float() / 255
            gt = batch["depth"].to(device).float()
            raw = _extract(model(img)).float()
            if raw.ndim == 3:
                raw = raw.unsqueeze(1)
            cal = torch.exp(a * torch.log(raw.clamp(min=1e-6)) + b)
            plot_depth_panels(
                img, gt, [raw, cal], plot_dir / f"val_batch{ni}_calibrated.jpg",
                titles=titles, max_images=max_images,
            )
    head.cal_a.fill_(a0)
    head.cal_b.fill_(b0)
```

Then update `calibrate_checkpoint`. New signature and docstring addition:

```python
def calibrate_checkpoint(ckpt_path, dataloader, device, dist_power: float = 0.0, plot_dir=None) -> None:
```

Add to its docstring Args section:

```
        plot_dir: If set, also write ``val_batch{ni}_calibrated.jpg`` comparison panels
            (RGB | GT | raw | calibrated) for the first val batches into this directory.
```

And append at the end of the function body, after the final `LOGGER.info(...)`:

```python
    if plot_dir is not None:
        try:
            _plot_calibrated_batches(work, dataloader, device, a, b, res["name"], plot_dir)
        except Exception as e:
            LOGGER.warning(f"Calibrated val plots skipped ({type(e).__name__}: {e})")
```

Note: `work` is the float copy `fit_calibration_selective` already ran on — reusing it avoids reloading the checkpoint. The try/except keeps a plot failure from breaking calibration (the checkpoint is already saved at this point).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/depth/test_plots.py tests/depth/test_calibration.py tests/test_depth_calibration.py -v`
Expected: all PASS (including the pre-existing `test_calibrate_checkpoint_applies_selective_policy`, which calls without `plot_dir`).

- [ ] **Step 5: Commit**

```bash
git add ultralytics/models/yolo/depth/calibrate.py tests/depth/test_plots.py
git commit -m "feat(depth): write calibrated val_batch comparison plots from calibrate_checkpoint"
```

---

### Task 3: Hook in `DepthTrainer.final_eval`

**Files:**
- Modify: `ultralytics/models/yolo/depth/train.py` (`final_eval`, lines 175–193)
- Test: `tests/depth/test_plots.py` (append)

**Interfaces:**
- Consumes: `calibrate_checkpoint(ckpt_path, dataloader, device, dist_power=0.0, plot_dir=None)` from Task 2.
- Produces: nothing new — trainer behavior only.

- [ ] **Step 1: Write the failing test**

Append to `tests/depth/test_plots.py`:

```python
def test_final_eval_plots_only_representative_checkpoint(tmp_path, monkeypatch):
    """final_eval passes plot_dir for best.pt only; last.pt is calibrated without plotting."""
    from types import SimpleNamespace

    import ultralytics.models.yolo.depth.calibrate as calibrate
    from ultralytics.models import yolo
    from ultralytics.models.yolo.depth.train import DepthTrainer

    calls = []
    monkeypatch.setattr(yolo.detect.DetectionTrainer, "final_eval", lambda self: None)  # skip real eval
    monkeypatch.setattr(
        calibrate, "calibrate_checkpoint",
        lambda ckpt, dl, dev, dist_power=0.0, plot_dir=None: calls.append((ckpt.name, plot_dir)),
    )
    t = DepthTrainer.__new__(DepthTrainer)  # skip __init__ (needs data/model); final_eval uses only these attrs
    t.args = SimpleNamespace(auto_calibrate=True, plots=True, cal_dist_pw=0.0)
    t.best, t.last = tmp_path / "best.pt", tmp_path / "last.pt"
    t.best.touch()
    t.last.touch()
    t.save_dir, t.test_loader, t.device = tmp_path, [], "cpu"
    t.final_eval()
    assert calls == [("best.pt", tmp_path), ("last.pt", None)]

    calls.clear()
    t.args.plots = False  # plots disabled -> calibrate both, plot neither
    t.final_eval()
    assert calls == [("best.pt", None), ("last.pt", None)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/depth/test_plots.py::test_final_eval_plots_only_representative_checkpoint -v`
Expected: FAIL — `calls == [("best.pt", None), ("last.pt", None)]` in the first assertion (current code never passes `plot_dir`).

- [ ] **Step 3: Implement**

In `ultralytics/models/yolo/depth/train.py`, replace the body of the `try:` block in `final_eval` (keep the surrounding method, docstring, RANK/auto_calibrate guard, and except clause):

```python
        try:
            from .calibrate import calibrate_checkpoint

            LOGGER.info("Auto-calibrating depth output scale on the validation set...")
            # Calibrated comparison plots come from the checkpoint that represents the run:
            # best.pt, or last.pt when best was never saved. Each checkpoint is fitted separately.
            plot_ckpt = self.best if self.best.exists() else self.last
            for ckpt in (self.best, self.last):
                if ckpt.exists():
                    plot_dir = self.save_dir if self.args.plots and ckpt == plot_ckpt else None
                    calibrate_checkpoint(
                        ckpt, self.test_loader, self.device, dist_power=self.args.cal_dist_pw, plot_dir=plot_dir
                    )
        except Exception as e:
            LOGGER.warning(f"Auto-calibration skipped ({type(e).__name__}: {e}); checkpoints left uncalibrated.")
```

Also update the method docstring's last line to mention the plots:

```python
        """Run the standard final evaluation, then auto-calibrate the saved checkpoints.

        After training, fits the scale-only log-affine (``cal_a``/``cal_b``) on the validation
        set and writes it into best.pt/last.pt, so the model outputs metric-scaled depth out of
        the box. Disable with ``auto_calibrate=False``. When ``plots`` is set, also writes
        ``val_batch{ni}_calibrated.jpg`` (RGB | GT | raw | calibrated) comparison panels.
        """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/depth/test_plots.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full depth test suite**

Run: `python -m pytest tests/depth/ tests/test_depth_calibration.py tests/test_depth_loss.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add ultralytics/models/yolo/depth/train.py tests/depth/test_plots.py
git commit -m "feat(depth): calibrated val_batch plots after auto-calibration in final_eval"
```

---

### Task 4: End-to-end verification on depthx1

**Files:** none created locally (remote smoke test). SSH host `depthx1` (`connect.westd.seetacloud.com:38011`, user `root`), repo at `/root/ult`, python at `/root/miniconda3/bin/python`.

**Interfaces:**
- Consumes: the three modified source files from Tasks 1–3.
- Produces: verified `val_batch{ni}_calibrated.jpg` from a real 1-epoch training run; one image copied back for visual inspection.

- [ ] **Step 1: Confirm SSH access and inspect the server repo**

```bash
ssh depthx1 "cd /root/ult && git log --oneline -3 && git status --short | head"
```

Expected: a depth-branch commit log. Note how far behind `f6218ba5d` it is.

- [ ] **Step 2: Sync the code**

Preferred (if the box has GitHub access): push the branch and pull it there.

```bash
git push -u origin depth-calibrated-val-plots
ssh depthx1 "cd /root/ult && git fetch origin depth-calibrated-val-plots && git checkout depth-calibrated-val-plots && git reset --hard origin/depth-calibrated-val-plots"
```

Fallback (seetacloud boxes often have no outbound internet — see the `autodl-proxy` skill): copy just the changed files. Only valid if Step 1 showed the server repo at or near `f6218ba5d` (these three files must be based on the same code):

```bash
scp ultralytics/models/yolo/depth/val.py depthx1:/root/ult/ultralytics/models/yolo/depth/val.py
scp ultralytics/models/yolo/depth/calibrate.py depthx1:/root/ult/ultralytics/models/yolo/depth/calibrate.py
scp ultralytics/models/yolo/depth/train.py depthx1:/root/ult/ultralytics/models/yolo/depth/train.py
```

If the server repo is on older code where scp would mix versions, use a git bundle instead:

```bash
git bundle create /tmp/claude-1000/-home-rick-ultralytics-depth/d48bdbf9-e4a4-4f78-8b70-ea2511228d49/scratchpad/cal-plots.bundle origin/depth_anything..depth-calibrated-val-plots depth-calibrated-val-plots
scp /tmp/claude-1000/-home-rick-ultralytics-depth/d48bdbf9-e4a4-4f78-8b70-ea2511228d49/scratchpad/cal-plots.bundle depthx1:/root/
ssh depthx1 "cd /root/ult && git fetch /root/cal-plots.bundle depth-calibrated-val-plots && git checkout FETCH_HEAD"
```

(The bundle needs `origin/depth_anything`'s objects present on the server; if that also fails, bundle from the merge-base the server has.)

- [ ] **Step 3: Find a depth dataset on the box**

```bash
ssh depthx1 "ls /root/ult/runs 2>/dev/null; find /root -maxdepth 3 -name 'args.yaml' -newer /root/miniconda3 2>/dev/null | head -3; grep -l 'depth' /root/ult/ultralytics/cfg/datasets/*.yaml 2>/dev/null"
```

Expected: a previous depth run's `args.yaml` reveals the `data:` yaml used on this box (`ssh depthx1 "grep '^data' <args.yaml>"`). Use that as `<DEPTH_DATA_YAML>` below.

- [ ] **Step 4: Run a 1-epoch training with plots + auto-calibration**

```bash
ssh depthx1 "cd /root/ult && /root/miniconda3/bin/python -c \"
from ultralytics import YOLO
m = YOLO('yolo26n-depth.yaml')
m.train(data='<DEPTH_DATA_YAML>', epochs=1, imgsz=640, batch=8, plots=True, project='runs/calplots', name='e2e', exist_ok=True)
\""
```

Expected in the log: `Auto-calibrating depth output scale on the validation set...` then `Depth calibration selected '<name>' ...`, no `Calibrated val plots skipped` warning.

- [ ] **Step 5: Verify the plots exist and pull one back**

```bash
ssh depthx1 "ls -la /root/ult/runs/calplots/e2e/val_batch*"
```

Expected: `val_batch0.jpg val_batch1.jpg val_batch2.jpg` AND `val_batch0_calibrated.jpg val_batch1_calibrated.jpg val_batch2_calibrated.jpg` (fewer if the val set has < 3 batches — then the same count for both).

```bash
scp depthx1:/root/ult/runs/calplots/e2e/val_batch0_calibrated.jpg /tmp/claude-1000/-home-rick-ultralytics-depth/d48bdbf9-e4a4-4f78-8b70-ea2511228d49/scratchpad/
```

Read the image and confirm visually: 4 labeled columns, header strip readable, calibrated column's colors closer to GT than raw (or identical if identity was selected — check the header label).

- [ ] **Step 6: No commit (remote verification only)**

If anything failed, fix locally with a test reproducing it, re-run Tasks 1–3 test suites, commit the fix, and repeat Steps 2–5.
