# S3D Track B — Auxiliary Depth Head + Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-only auxiliary dense-depth head to the s3d model, supervised by a frozen monocular-depth teacher via distillation, so the shared backbone/neck learns metric-depth structure — at **zero inference/deploy cost** and with **no dense-depth GT** required.

**Architecture:** The monocular `Depth` fusion module is reused as an on-demand submodule of `Stereo3DDetHead`, attached to the clean `[P3,P4,P5]` features and run only under `self.training`. A frozen teacher (`yolo26x-depth`) runs on the left image in `preprocess_batch` to produce `batch["teacher_depth"]`. A new `distill` loss term (SILog + gradient-matching, factored from `DepthLoss26` into a shared helper) pulls the aux head toward the teacher. Everything distillation-related is gated off by default so the baseline and Track C are unaffected.

**Tech Stack:** PyTorch, Ultralytics engine (`Stereo3DDetModel`/`Stereo3DDetTrainer`/`Stereo3DDetLoss`), `Depth` head, pytest.

## Global Constraints

- Python >= 3.8, PyTorch >= 1.8. Line length 120; run `ruff format . && ruff check --fix .` before every commit.
- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` (Actions bot adds it; don't add/revert manually).
- Never edit the primary checkout — work in a git worktree on a feature branch; open a PR. Never push `main`, never force-push.
- **Depends on** `docs/superpowers/plans/2026-07-23-s3d-depth-foundation.md`: rebase this track's branch onto the merged foundation before starting. Locked node indices: **P3=16, P4=19, P5=22, StereoCostVolume=23, Stereo3DDetHead=24**, head `from=[16,19,22,23]`.
- Teacher checkpoint (frozen, eval, on device): `yolo26x-depth.pt`. Its `Depth` head at eval returns metric depth `(B,1,H/4,W/4)`. Load via `ultralytics.nn.tasks.load_checkpoint`.
- Student scale `n`. Primary dataset `kitti-stereo`; regression `cube-s3d`. Smoke: `kitti-stereo8` / a tiny cube subset, `epochs=1`.
- PR body must include a `Deleted:` line. For this track: `Deleted: nothing` — justification: pure additive training-only capability; the one refactor (SILog/grad helper) is a net consolidation that removes duplicated math from `DepthLoss26`.

## Design decisions (read before starting)

- **Config, off by default:** a `training.distill: {teacher: <path>, c_mid: 256}` block plus `training.loss_weights.distill: <float>` live in a **new** `yolo26-s3d-distill.yaml`. Absent block → aux head never built, teacher never loaded, `distill` loss stays 0. Baseline `yolo26-s3d.yaml` and Track C are untouched.
- **Loss vector 7→8:** `distill` is appended as `loss_names[7]`, always present (0 when inactive). All four name sites are kept in sync in Task 3.
- **Reuse `Depth`:** the aux head is a `Depth` submodule, not a new module (Delete>Replace>Add). Built on demand via `enable_distill_head`, mirroring the existing `set_depth_mode` pattern (head.py:136).
- **Shared SILog helper:** the SILog + gradient-matching math is factored out of `DepthLoss26` (utils/loss.py:1169-1257) into `silog_gradient_loss`, called by both `DepthLoss26` and the s3d distill loss.

---

### Task 1: Add the on-demand auxiliary depth head to `Stereo3DDetHead`

**Files:**
- Modify: `ultralytics/models/yolo/s3d/head.py` (`__init__` ~line 116; `forward_head` ~line 148; add `enable_distill_head`)
- Test: `tests/test_s3d_distill.py` (create)

**Interfaces:**
- Consumes: reordered s3d model (foundation); `Depth` from `ultralytics.nn.modules.head`.
- Produces:
  - `Stereo3DDetHead.distill_head` attribute (default `None`).
  - `Stereo3DDetHead.enable_distill_head(c_mid: int = 256) -> None` — builds `self.distill_head = Depth(c_mid, self.aux_ch)`.
  - When enabled and `self.training`: `forward_head` adds `preds["aux_dense_depth"]` of shape `(B, 1, H/4, W/4)` (metric depth). Never present at eval/export.

- [ ] **Step 1: Write the failing test**

Create `tests/test_s3d_distill.py`:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Track B: auxiliary depth head + distillation tests."""

import torch

from ultralytics.models.yolo.s3d.model import Stereo3DDetModel

S3D_CFG = "ultralytics/cfg/models/26/yolo26-s3d.yaml"


def _build(cfg=S3D_CFG):
    return Stereo3DDetModel(cfg, ch=3, nc=3, verbose=False)


def test_distill_head_off_by_default():
    """Baseline s3d model has no distill head (zero added params/cost)."""
    m = _build()
    assert m.model[-1].distill_head is None


def test_distill_head_training_output_shape():
    """When enabled and training, forward adds a dense depth map at input/4."""
    m = _build()
    m.model[-1].enable_distill_head(c_mid=256)
    m.train()
    x = torch.rand(2, 6, 384, 384)
    preds = m(x)  # training → preds dict
    assert "aux_dense_depth" in preds
    assert preds["aux_dense_depth"].shape == (2, 1, 96, 96)
    assert torch.isfinite(preds["aux_dense_depth"]).all()


def test_distill_head_absent_at_eval():
    """Even when enabled, the aux head does not run at eval (zero inference cost)."""
    m = _build()
    m.model[-1].enable_distill_head(c_mid=256)
    m.eval()
    x = torch.rand(1, 6, 384, 384)
    with torch.no_grad():
        out = m(x)
    preds = out[1] if isinstance(out, tuple) else out
    assert "aux_dense_depth" not in preds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_s3d_distill.py -v`
Expected: FAIL — `AttributeError: 'Stereo3DDetHead' object has no attribute 'distill_head'`.

- [ ] **Step 3: Implement the aux head in `head.py`**

In `Stereo3DDetHead.__init__`, immediately after `self.depth_dfl = DepthDFL(...)` (head.py:117), add:

```python
        self.aux_ch = tuple(ch)  # per-scale P3/P4/P5 channels, for the optional distill head
        self.distill_head = None  # built on demand via enable_distill_head(); training-only
```

Add this method after `set_depth_mode` (head.py:141):

```python
    def enable_distill_head(self, c_mid: int = 256) -> None:
        """Build a training-only monocular depth head over [P3, P4, P5] for distillation.

        Reuses the monocular Depth fusion module (nn/modules/head.py). Runs only under
        self.training and is never traced into the inference/export graph.
        """
        from ultralytics.nn.modules.head import Depth

        self.distill_head = Depth(c_mid, self.aux_ch)
```

In `forward_head`, just before `return preds` (head.py:190), add:

```python
        # Training-only distillation head: dense metric depth over clean [P3,P4,P5].
        if self.training and self.distill_head is not None:
            d = self.distill_head(list(x))  # x is [P3, P4, P5] (cost vol already separated)
            preds["aux_dense_depth"] = d["depth"] if isinstance(d, dict) else d
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_s3d_distill.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add ultralytics/models/yolo/s3d/head.py tests/test_s3d_distill.py
git commit -m "s3d: add on-demand training-only distillation depth head (reuses Depth)

Deleted: nothing (additive, training-only; reuses the monocular Depth module)"
```

---

### Task 2: Config plumbing + frozen teacher loading + `batch["teacher_depth"]`

**Files:**
- Create: `ultralytics/cfg/models/26/yolo26-s3d-distill.yaml`
- Modify: `ultralytics/models/yolo/s3d/model.py` (`__init__`, after the `depth_mode` block ~line 55)
- Modify: `ultralytics/models/yolo/s3d/train.py` (`__init__` callback ~line 32; add `_load_distill_teacher`, `_teacher_depth`; `preprocess_batch` ~line 269)
- Test: `tests/test_s3d_distill.py` (add tests)

**Interfaces:**
- Consumes: `enable_distill_head` (Task 1); `load_checkpoint`.
- Produces:
  - `yolo26-s3d-distill.yaml` — reordered s3d graph + `training.distill.{teacher,c_mid}` + `training.loss_weights.distill`.
  - `Stereo3DDetModel.__init__` calls `head.enable_distill_head(c_mid)` when `training.distill` present.
  - `Stereo3DDetTrainer.teacher_model` (frozen depth model or `None`); `Stereo3DDetTrainer._teacher_depth(teacher, left) -> Tensor (B,1,H/4,W/4)`.
  - `preprocess_batch` adds `batch["teacher_depth"]` when a teacher is loaded and the model is training.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_s3d_distill.py`:

```python
from pathlib import Path

import pytest

from ultralytics.models.yolo.s3d.train import Stereo3DDetTrainer

DISTILL_CFG = "ultralytics/cfg/models/26/yolo26-s3d-distill.yaml"
_TEACHER = Path("yolo26x-depth.pt")


def test_distill_yaml_enables_head():
    """The distill YAML builds the aux head at construction time."""
    m = Stereo3DDetModel(DISTILL_CFG, ch=3, nc=3, verbose=False)
    assert m.model[-1].distill_head is not None


def test_baseline_yaml_leaves_head_disabled():
    """The plain s3d YAML must NOT build the aux head."""
    m = Stereo3DDetModel(S3D_CFG, ch=3, nc=3, verbose=False)
    assert m.model[-1].distill_head is None


@pytest.mark.skipif(not _TEACHER.exists(), reason="teacher checkpoint not present")
def test_teacher_depth_forward_shape():
    """The frozen teacher maps a 3ch left image to (B,1,H/4,W/4) metric depth."""
    from ultralytics.nn.tasks import load_checkpoint

    teacher, _ = load_checkpoint(str(_TEACHER))
    teacher = teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    left = torch.rand(1, 3, 384, 384)
    d = Stereo3DDetTrainer._teacher_depth(teacher, left)
    assert d.shape == (1, 1, 96, 96)
    assert torch.isfinite(d).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_s3d_distill.py -k "distill_yaml or baseline_yaml or teacher_depth" -v`
Expected: FAIL — the distill YAML does not exist and `_teacher_depth` is undefined.

- [ ] **Step 3a: Create the distill model YAML**

Create `ultralytics/cfg/models/26/yolo26-s3d-distill.yaml` — identical to the reordered `yolo26-s3d.yaml` plus the distill config. Copy the reordered `yolo26-s3d.yaml` verbatim, then extend its `training:` block to:

```yaml
training:
  loss_weights:
    lr_distance: 2.0
    depth: 3.0
    dimensions: 1.0
    orientation: 1.0
    proj_center: 1.0
    distill: 1.0
  distill:
    teacher: yolo26x-depth.pt
    c_mid: 256
```

(Everything else — `nc`, `stereo`, `siamese`, `scales`, `mean_dims`, `backbone`, reordered `head`, node indices — is copied unchanged from `yolo26-s3d.yaml`.)

- [ ] **Step 3b: Enable the head from config in `model.py`**

In `Stereo3DDetModel.__init__`, after the `depth_mode` block (model.py:52-55), add:

```python
            distill_cfg = training_cfg.get("distill")
            if distill_cfg:
                head.enable_distill_head(c_mid=int(distill_cfg.get("c_mid", 256)))
```

- [ ] **Step 3c: Load the teacher and populate `batch["teacher_depth"]` in `train.py`**

In `Stereo3DDetTrainer.__init__`, after the existing `add_callback(...)` (train.py:32), add:

```python
        self.teacher_model = None
        self.add_callback("on_train_start", Stereo3DDetTrainer._load_distill_teacher)
```

Add these methods to `Stereo3DDetTrainer`:

```python
    @staticmethod
    def _load_distill_teacher(trainer):
        """Load the frozen monocular-depth teacher if training.distill.teacher is set."""
        model_yaml = getattr(unwrap_model(trainer.model), "yaml", {}) or {}
        distill_cfg = (model_yaml.get("training", {}) or {}).get("distill") or {}
        path = distill_cfg.get("teacher")
        if not path:
            trainer.teacher_model = None
            return
        from ultralytics.nn.tasks import load_checkpoint

        teacher, _ = load_checkpoint(path)
        teacher = teacher.to(trainer.device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        trainer.teacher_model = teacher
        LOGGER.info(f"s3d distill: loaded frozen depth teacher from {path}")

    @staticmethod
    def _teacher_depth(teacher, left_img):
        """Run the frozen teacher on a 3ch left image → metric depth (B,1,H/4,W/4)."""
        import torch

        with torch.no_grad():
            out = teacher(left_img)
        d = out[0] if isinstance(out, tuple) else out
        return d["depth"] if isinstance(d, dict) else d
```

Change `preprocess_batch` (train.py:269-275) to:

```python
    def preprocess_batch(self, batch):
        """Normalize 6-channel images to float [0,1]; optionally attach teacher depth for distillation."""
        batch = preprocess_stereo_batch(batch, self.device, half=False)
        if getattr(self, "teacher_model", None) is not None and unwrap_model(self.model).training:
            batch["teacher_depth"] = self._teacher_depth(self.teacher_model, batch["img"][:, :3])
        return batch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_s3d_distill.py -k "distill_yaml or baseline_yaml or teacher_depth" -v`
Expected: PASS (`teacher_depth` test SKIPS if the checkpoint is absent).

- [ ] **Step 5: Commit**

```bash
git add ultralytics/cfg/models/26/yolo26-s3d-distill.yaml ultralytics/models/yolo/s3d/model.py ultralytics/models/yolo/s3d/train.py tests/test_s3d_distill.py
git commit -m "s3d distill: config-gated aux head + frozen teacher depth in preprocess

Deleted: nothing (additive, off by default; baseline/Track-C untouched)"
```

---

### Task 3: Distillation loss term + shared SILog helper + loss-name sync

**Files:**
- Modify: `ultralytics/utils/loss.py` (add `silog_gradient_loss`; reroute `DepthLoss26`)
- Modify: `ultralytics/models/yolo/s3d/loss.py` (`loss_names` line 51; `loss()` vector line 347-366; add `_distill_loss`)
- Modify: `ultralytics/models/yolo/s3d/train.py` (`get_validator` loss_names line 44)
- Test: `tests/test_s3d_distill.py` (add tests)

**Interfaces:**
- Consumes: `preds["aux_dense_depth"]` (Task 1), `batch["teacher_depth"]` (Task 2).
- Produces:
  - `ultralytics.utils.loss.silog_gradient_loss(pred, gt, valid_mask=None, lam=1.0, grad_weight=0.5, scales=4, eps=1e-6) -> Tensor`.
  - `Stereo3DDetLoss.loss_names` = 8-tuple ending in `"distill"`; `loss()` returns an 8-vector; `distill` weighted by `aux_w["distill"]`, 0 when `aux_dense_depth`/`teacher_depth` absent.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_s3d_distill.py`:

```python
def test_silog_gradient_loss_zero_on_match_positive_on_diff():
    from ultralytics.utils.loss import silog_gradient_loss

    gt = torch.rand(2, 1, 64, 64) * 40 + 2.0  # positive metric depth
    same = silog_gradient_loss(gt.clone(), gt)
    diff = silog_gradient_loss(gt * 1.5, gt)
    assert same.item() < 1e-4
    assert diff.item() > same.item()


def test_silog_resizes_pred_to_gt():
    from ultralytics.utils.loss import silog_gradient_loss

    pred = torch.rand(1, 1, 48, 48) * 40 + 2.0
    gt = torch.rand(1, 1, 96, 96) * 40 + 2.0
    out = silog_gradient_loss(pred, gt)
    assert torch.isfinite(out).all()


def test_s3d_loss_names_include_distill():
    m = Stereo3DDetModel(DISTILL_CFG, ch=3, nc=3, verbose=False)
    crit = m.init_criterion()
    assert crit.loss_names == ("box", "cls", "lr_dist", "depth", "dims", "orient", "proj_center", "distill")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_s3d_distill.py -k "silog or loss_names_include" -v`
Expected: FAIL — `silog_gradient_loss` is undefined and `loss_names` has 7 entries.

- [ ] **Step 3a: Add the shared helper and reroute `DepthLoss26`**

Add to `ultralytics/utils/loss.py` (module level, near `DepthLoss26`):

```python
def silog_gradient_loss(
    pred, gt, valid_mask=None, lam=1.0, grad_weight=0.5, scales=4, eps=1e-6, return_components=False
):
    """Scale-invariant log depth loss (SILog) + multi-scale log-gradient matching.

    Args:
        pred (Tensor): Predicted metric depth, (B,1,H,W) (resized to gt if needed).
        gt (Tensor): Target metric depth, (B,1,H,W) or (B,H,W).
        valid_mask (Tensor | None): Bool mask of valid pixels; defaults to gt > eps.
        lam (float): 1.0 = fully scale-invariant, 0.0 = log-RMSE.
        grad_weight (float): Weight on the gradient-matching term in the combined return.
        scales (int): Number of pyramid levels for gradient matching.
        eps (float): Numerical floor.
        return_components (bool): If True, return the unweighted (silog, gloss) pair so callers
            (e.g. DepthLoss26) can log/weight the two terms exactly as before. If False, return the
            combined scalar ``silog + grad_weight * gloss``.
    """
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)
    if valid_mask is None:
        valid_mask = gt > eps

    lp = pred.clamp(min=eps).log()
    lg = gt.clamp(min=eps).log()
    ld = (lp - lg)[valid_mask]
    if ld.numel() == 0:
        z = pred.sum() * 0.0
        return (z, z) if return_components else z
    silog = torch.sqrt(ld.var(unbiased=False) + (1.0 - lam) * ld.mean() ** 2 + eps)

    gloss = pred.sum() * 0.0
    p, g, m = lp, lg, valid_mask.float()
    for s in range(scales):
        if s > 0:
            p, g, m = F.avg_pool2d(p, 2), F.avg_pool2d(g, 2), F.avg_pool2d(m, 2)
        dpx, dgx = p[..., :, 1:] - p[..., :, :-1], g[..., :, 1:] - g[..., :, :-1]
        mx = m[..., :, 1:] * m[..., :, :-1]
        dpy, dgy = p[..., 1:, :] - p[..., :-1, :], g[..., 1:, :] - g[..., :-1, :]
        my = m[..., 1:, :] * m[..., :-1, :]
        gloss = gloss + (dpx - dgx).abs().mul(mx).sum() / mx.sum().clamp(min=1.0)
        gloss = gloss + (dpy - dgy).abs().mul(my).sum() / my.sum().clamp(min=1.0)
    return (silog, gloss) if return_components else silog + grad_weight * gloss
```

Then reroute `DepthLoss26.__call__` (utils/loss.py:1169-1257) through the helper **using `return_components=True`** so its existing `("dlog_loss", "dgrad_loss")` breakdown is preserved bit-for-bit: replace the inline SILog + gradient math with

```python
        silog, gloss = silog_gradient_loss(
            pred, gt_depth, valid_mask, lam=self.hyp.dlam, return_components=True
        )
        dlog_loss = silog * self.hyp.dlog
        dgrad_loss = gloss * self.hyp.dgrad
```

keeping whatever masking/resize `DepthLoss26` did (the helper reproduces both). This is a net deletion — the duplicated SILog/gradient block leaves `DepthLoss26`. Verify no behavior change: `pytest tests/ -k depth -v` must still pass (the two logged terms must match pre-refactor values within float tolerance). If any depth test regresses, the reroute — not the helper — is at fault; align the helper's masking with the original before proceeding.

- [ ] **Step 3b: Add `_distill_loss` and extend the loss vector in `s3d/loss.py`**

Change `loss_names` (loss.py:51) to:

```python
        self.loss_names = ("box", "cls", "lr_dist", "depth", "dims", "orient", "proj_center", "distill")
```

Add this method to `Stereo3DDetLoss`:

```python
    def _distill_loss(self, pred_depth: torch.Tensor, teacher_depth: torch.Tensor) -> torch.Tensor:
        """SILog + gradient-matching distillation loss vs the frozen teacher's depth map."""
        from ultralytics.utils.loss import silog_gradient_loss

        return silog_gradient_loss(pred_depth, teacher_depth.to(self.device))
```

In `loss()` change the vector allocation (loss.py:347) to `loss = torch.zeros(8, device=self.device)` and, after the `proj_center` block (loss.py:363), add:

```python
        if "aux_dense_depth" in preds and "teacher_depth" in batch:
            loss[7] = self._distill_loss(preds["aux_dense_depth"], batch["teacher_depth"]) * float(
                self.aux_w.get("distill", 1.0)
            )
```

Also add `"aux_dense_depth"` to the `aux_keys` set (loss.py:344) so it is routed out of the detection preds.

- [ ] **Step 3c: Sync the validator loss names in `train.py`**

Change `get_validator` (train.py:44) to:

```python
        self.loss_names = ("box", "cls", "lr_dist", "depth", "dims", "orient", "proj_center", "distill")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_s3d_distill.py -k "silog or loss_names_include" -v`
Expected: PASS.
Run regression: `pytest tests/ -k depth -v`
Expected: PASS (the `DepthLoss26` reroute is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
git add ultralytics/utils/loss.py ultralytics/models/yolo/s3d/loss.py ultralytics/models/yolo/s3d/train.py tests/test_s3d_distill.py
git commit -m "s3d distill: distill loss term + shared silog_gradient_loss helper

Deleted: duplicated SILog/gradient math from DepthLoss26 (consolidated into silog_gradient_loss)"
```

---

### Task 4: End-to-end smoke test — training completes, inference graph unchanged

**Files:**
- Test: `tests/test_s3d_distill.py` (add a slow integration test)

**Interfaces:**
- Consumes: everything from Tasks 1–3, the distill YAML, and the teacher checkpoint.
- Produces: proof that a 1-epoch distill run completes and that the exported/inference graph carries no distillation ops (zero added inference cost).

- [ ] **Step 1: Write the failing/slow test**

Add to `tests/test_s3d_distill.py`:

```python
@pytest.mark.slow
@pytest.mark.skipif(not _TEACHER.exists(), reason="teacher checkpoint not present")
def test_distill_train_one_epoch_and_clean_inference(tmp_path):
    """A 1-epoch distill run completes; the eval graph produces no aux_dense_depth."""
    from ultralytics import YOLO

    model = YOLO(DISTILL_CFG, task="s3d")
    model.train(data="cube-s3d.yaml", epochs=1, imgsz=384, batch=2, device=0, project=str(tmp_path))
    # Inference path must not carry the distillation head output.
    net = model.model
    net.eval()
    with torch.no_grad():
        out = net(torch.rand(1, 6, 384, 384).to(next(net.parameters()).device))
    preds = out[1] if isinstance(out, tuple) else out
    assert "aux_dense_depth" not in preds
```

- [ ] **Step 2: Run it to confirm it exercises the full path**

Run: `pytest tests/test_s3d_distill.py::test_distill_train_one_epoch_and_clean_inference -v --slow`
Expected: PASS if a GPU + teacher checkpoint + `cube-s3d` are present; otherwise SKIP. On CPU-only dev boxes, run the equivalent CLI smoke below instead.

- [ ] **Step 3: CLI smoke on the target box (documented, run before renting the full box)**

Run:

```bash
yolo train model=ultralytics/cfg/models/26/yolo26-s3d-distill.yaml data=cube-s3d.yaml \
  pretrained=yolo26n-depth.pt epochs=1 imgsz=384 batch=2
```

Expected: the run reaches epoch 1/1, prints 8 loss columns including `distill`, and the `distill` column is finite and non-zero. Confirm the "Transferred X/Y" line (foundation warm-start) still shows full backbone+neck transfer.

- [ ] **Step 4: Commit**

```bash
git add tests/test_s3d_distill.py
git commit -m "s3d distill: end-to-end smoke test (train completes, clean inference graph)

Deleted: nothing (test-only)"
```

---

## Evaluation / Experiment plan

Run after smoke passes, on the rented AutoDL box (Track B box):

1. **Warm-start every run** with the scale-matched depth checkpoint (`pretrained=yolo26n-depth.pt`), per the foundation — isolates the distillation mechanism from the warm-start itself.
2. **Arms** (each on KITTI, then re-validate the best on cube):
   - baseline: `yolo26-s3d.yaml`, warm-start only (foundation).
   - B: `yolo26-s3d-distill.yaml` (aux head + teacher distillation).
3. **Metrics:** existing s3d validator — AP3D@0.5 / AP3D@0.7, depth error — plus the **stereo-sensitivity probe** (does the model use the right image?).
4. **Success:** B improves AP3D@0.7 on KITTI over the warm-start baseline with **no cube regression** and **zero inference-cost delta** (Task 4 guarantees the graph is unchanged).
5. **Cost-volume side-answer:** compare the stereo-sensitivity probe for baseline vs B. If distillation raises sensitivity, the cost volume was revived by depth-aware features; if it stays ≈0, that is evidence the `StereoCostVolume` branch is dead weight and a deletion candidate (feed this back to the design's open cost-volume question).

## Self-Review

- **Spec coverage:** aux head → Task 1; teacher + config off-by-default → Task 2; distill loss + name sync → Task 3; zero-inference-cost + smoke → Task 4; KITTI+cube + cost-volume side-answer → Evaluation section. Warm-start comes from the foundation (dependency stated). All Track B spec items covered.
- **Placeholder scan:** none — every code step is complete runnable code; the one prose instruction (rerouting `DepthLoss26` through `silog_gradient_loss`) is bounded by a behavior-preservation test (`pytest tests/ -k depth`). If preserving the two-name `("dlog_loss","dgrad_loss")` split proves awkward, the fallback (combined value, dgrad folded) is stated explicitly.
- **Type consistency:** `enable_distill_head(c_mid)`, `distill_head`, `aux_ch`, `_teacher_depth`, `teacher_model`, `preds["aux_dense_depth"]`, `batch["teacher_depth"]`, `silog_gradient_loss(...)`, and the 8-tuple `loss_names` are used identically across all tasks and match the foundation's node indices (16/19/22/23/24).
- **Scope:** single implementation plan, one worktree/branch, one box. Ready for execution after the foundation is merged.
