# S3D Track C — Dense Depth-Prior Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed a frozen monocular-depth model's dense metric depth map into the s3d 3D head, **replacing** the inert `StereoCostVolume`, so the depth/lr branches read from a strong dense depth prior at train *and* inference.

**Architecture:** A new `DepthPriorEncoder` module occupies the head's 4th-input slot (node 23) where `StereoCostVolume` used to sit, outputting the same `[B, 64, H/8, W/8]` shape so the head is unchanged. The `Stereo3DDetModel` owns a frozen, `eval()`, `requires_grad=False` depth model (hidden from the module registry so it is never trained, saved, or toggled to train mode). A dedicated monocular-left forward runs the frozen depth model on the left RGB once, then injects the encoded dense-depth features at node 23 — reusing the existing cost-volume injection *seam* (`if m.i == <cv slot>`) but dropping the now-unnecessary right-image pass.

**Tech Stack:** PyTorch, Ultralytics `parse_model`/`BaseModel`, `attempt_load_one_weight`, pytest.

## Global Constraints

- Python >= 3.8, PyTorch >= 1.8 (repo floor).
- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` (Actions bot adds; don't add manually — but keep it if present).
- Line length 120; run `ruff format . && ruff check --fix .` before each commit.
- **Delete > Replace > Add.** This track's PR body **must** contain: `Deleted: StereoCostVolume from the s3d graph (replaced by DepthPriorEncoder in the depth-prior variant)`.
- Never edit the primary checkout — work in a git worktree on a feature branch; open a PR. Never push `main`, never force-push.
- **Depends on** `2026-07-23-s3d-depth-foundation.md` (reordered YAML). Branch from / rebase onto the foundation branch before starting.
- Depth-prior checkpoint (frozen, eval, scale-independent): `/home/rick/autoresearch_depth/weights/yolo26x-depth-640-best.pt`.
- Warm-start (scale-matched, still applies): `yolo26n-depth.pt`.
- Student scale `n`. Eval datasets: `kitti-stereo` (primary) + `cube-s3d` (regression). Smoke datasets: `kitti-stereo8` / small cube, `epochs=1`.

## Locked interfaces (from foundation)

- Reordered `yolo26-s3d.yaml`: P3=**16**, P4=**19**, P5=**22**, cost-vol slot=**23**, head=**24**, head `from`=`[16,19,22,23]`.
- Head 4th input (`cv_ch`) is `64`; `Stereo3DDetHead.forward_head` (`head.py:158-161,174-175`) concatenates it onto P3 for the `lr_distance`/`depth` branches only. **The head is not modified in this track.**
- Existing injection seam: `Stereo3DDetModel._predict_once` (`model.py:91-143`) special-cases the cost-vol layer at `model.py:121-122`; slot discovery is at `model.py:38-44`; export path `forward_export` at `model.py:69-89`.

---

### Task 1: `DepthPriorEncoder` module + parser wiring

**Files:**
- Modify: `ultralytics/nn/modules/block.py` (add `DepthPriorEncoder` after `StereoCostVolume` at `block.py:2259`)
- Modify: `ultralytics/nn/modules/__init__.py` (export `DepthPriorEncoder`)
- Modify: `ultralytics/nn/tasks.py` (import + parser branch near `tasks.py:2089-2092`)
- Test: `tests/test_s3d_depth_prior.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `DepthPriorEncoder(c1=1, c2=64)`; `forward(depth_map[B,1,H,W]) -> [B, c2, H/2, W/2]`. Parser registers output channels `c2` (default 64), input fixed at 1 (a depth map, not a feature map).

- [ ] **Step 1: Write the failing test**

Create `tests/test_s3d_depth_prior.py`:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for Track C: dense depth-prior fusion into the s3d head."""

import torch

from ultralytics.nn.modules.block import DepthPriorEncoder


def test_depth_prior_encoder_shape():
    """Encoder maps a dense depth map [B,1,H/4,W/4] to P3-scale features [B,64,H/8,W/8]."""
    enc = DepthPriorEncoder(c1=1, c2=64)
    depth_map = torch.rand(2, 1, 96, 96)  # input/4 for a 384px image
    out = enc(depth_map)
    assert out.shape == (2, 64, 48, 48), f"expected [2,64,48,48], got {tuple(out.shape)}"
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_s3d_depth_prior.py::test_depth_prior_encoder_shape -v`
Expected: FAIL — `ImportError: cannot import name 'DepthPriorEncoder'`.

- [ ] **Step 3: Implement the module**

Append to `ultralytics/nn/modules/block.py` (after `StereoCostVolume`, at line 2259):

```python
class DepthPriorEncoder(nn.Module):
    """Encode a dense metric depth map into P3-scale features for the s3d head.

    Replaces StereoCostVolume as the head's 4th input in the monocular depth-prior s3d variant:
    a frozen depth model produces a dense depth map at input/4, which this module downsamples to
    input/8 (P3 stride) and refines to c2 channels — matching the cost-volume output contract so
    the head needs no change.

    Args:
        c1: Input channels (1 = single-channel depth map).
        c2: Output channels (must equal the head's cost-volume channel count, default 64).
        refine_layers: Number of conv layers after the stride-2 stem (default 2).
    """

    def __init__(self, c1: int = 1, c2: int = 64, refine_layers: int = 2):
        super().__init__()
        self.stem = Conv(c1, c2, 3, s=2)  # input/4 -> input/8
        self.refine = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(refine_layers)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode depth map [B, 1, H, W] -> [B, c2, H/2, W/2]."""
        return self.refine(self.stem(x))
```

- [ ] **Step 4: Export + parser wiring**

In `ultralytics/nn/modules/__init__.py`, add `DepthPriorEncoder` to the `from .block import (...)` list and to `__all__` (alongside `StereoCostVolume`).

In `ultralytics/nn/tasks.py`, add `DepthPriorEncoder` to the `from ultralytics.nn.modules import (...)` block (next to `StereoCostVolume`), then add a parser branch immediately after the `StereoCostVolume` branch (`tasks.py:2089-2092`):

```python
        elif m is DepthPriorEncoder:
            c1 = 1  # depth map is single-channel; head input is injected, not routed from ch[f]
            c2 = args[0]  # output channels, NOT width-scaled (must equal head cv_ch)
            args = [c1, c2]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_s3d_depth_prior.py::test_depth_prior_encoder_shape -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ultralytics/nn/modules/block.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py tests/test_s3d_depth_prior.py
git commit -m "Add DepthPriorEncoder: dense depth map -> P3-scale head input

Deleted: nothing yet (module + parser wiring; replaces StereoCostVolume in the depth-prior YAML added later)"
```

---

### Task 2: Depth-prior YAML variant + build test

**Files:**
- Create: `ultralytics/cfg/models/26/yolo26-s3d-depthprior.yaml`
- Modify: `tests/test_s3d_depth_prior.py`

**Interfaces:**
- Consumes: `DepthPriorEncoder` (Task 1); reordered head indices (foundation).
- Produces: a model YAML where node 23 is `DepthPriorEncoder` (not `StereoCostVolume`) and `training.depth_prior.model` names the frozen checkpoint. Head `from` stays `[16,19,22,23]`; `cv_ch` stays 64.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_s3d_depth_prior.py`:

```python
from ultralytics.nn.modules.block import DepthPriorEncoder as _DPE
from ultralytics.models.yolo.s3d.head import Stereo3DDetHead


def _build_depthprior(monkeypatch=None):
    # Build without loading the frozen depth model (tested separately in Task 3).
    from ultralytics.models.yolo.s3d.model import Stereo3DDetModel

    return Stereo3DDetModel("ultralytics/cfg/models/26/yolo26-s3d-depthprior.yaml", ch=3, nc=3, verbose=False)


def test_depthprior_yaml_node23_is_encoder():
    m = _build_depthprior().model
    assert isinstance(m[23], _DPE), f"node 23 should be DepthPriorEncoder, got {type(m[23])}"
    head = m[24]
    assert isinstance(head, Stereo3DDetHead)
    assert list(head.f) == [16, 19, 22, 23]
    assert head.cv_ch == 64, f"head cv_ch should stay 64, got {head.cv_ch}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_s3d_depth_prior.py::test_depthprior_yaml_node23_is_encoder -v`
Expected: FAIL — YAML file does not exist yet (`FileNotFoundError`).

- [ ] **Step 3: Create the YAML variant**

Create `ultralytics/cfg/models/26/yolo26-s3d-depthprior.yaml` (identical to the reordered `yolo26-s3d.yaml` except node 23 and the new `depth_prior` block):

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# YOLO26 Stereo 3D — Track C: monocular dense depth-prior variant.
# Node 23 is DepthPriorEncoder (replaces StereoCostVolume); a frozen depth model
# supplies the dense depth map, injected at node 23 by Stereo3DDetModel.

nc: 3
stereo: true
siamese: true # keeps the 3ch backbone (warm-start compatible); right pass is dropped at inference
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

training:
  loss_weights:
    lr_distance: 2.0
    depth: 3.0
    dimensions: 1.0
    orientation: 1.0
    proj_center: 1.0
  depth_prior:
    model: /home/rick/autoresearch_depth/weights/yolo26x-depth-640-best.pt

mean_dims:
  Car: [3.88, 1.63, 1.53]
  Pedestrian: [0.88, 0.60, 1.73]
  Cyclist: [1.72, 0.60, 1.77]

backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11
  - [[-1, 6], 1, Concat, [1]] # 12
  - [-1, 2, C3k2, [512, True]] # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 4], 1, Concat, [1]] # 15
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8, clean)

  - [16, 1, Conv, [256, 3, 2]] # 17
  - [[-1, 13], 1, Concat, [1]] # 18
  - [-1, 2, C3k2, [512, True]] # 19 (P4/16)
  - [-1, 1, Conv, [512, 3, 2]] # 20
  - [[-1, 10], 1, Concat, [1]] # 21
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 22 (P5/32)

  # Dense depth-prior encoder (from=0 is a placeholder; the model injects the depth map here)
  - [0, 1, DepthPriorEncoder, [64]] # 23: [B, 64, H/8, W/8]

  - [[16, 19, 22, 23], 1, Stereo3DDetHead, [nc]] # 24
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_s3d_depth_prior.py::test_depthprior_yaml_node23_is_encoder -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ultralytics/cfg/models/26/yolo26-s3d-depthprior.yaml tests/test_s3d_depth_prior.py
git commit -m "Add yolo26-s3d-depthprior YAML: node 23 DepthPriorEncoder + depth_prior block

Deleted: StereoCostVolume from this variant's graph (replaced by DepthPriorEncoder)"
```

---

### Task 3: Own a frozen depth model + inject the prior at the cost-vol seam

**Files:**
- Modify: `ultralytics/models/yolo/s3d/model.py` (`__init__` slot discovery `model.py:37-44`; add `_depth_prior_model`, `_forward_depth_prior`; dispatch in `_predict_once` `model.py:91-104`)
- Modify: `tests/test_s3d_depth_prior.py`

**Interfaces:**
- Consumes: `yolo26-s3d-depthprior.yaml` (Task 2); `attempt_load_one_weight`.
- Produces:
  - `Stereo3DDetModel.depth_prior_model` (property) → the frozen `DepthModel` or `None`.
  - `_prior_layer: int` — the `DepthPriorEncoder` node index (23).
  - `_forward_depth_prior(x)` — monocular-left forward that injects the encoded depth prior at `_prior_layer` and returns the head preds. Handles 6ch (train/val) and 3ch (stride computation).
  - Invariant: the frozen depth model's parameters are **not** in `model.parameters()` (never optimized) and stay in `eval()` mode.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_s3d_depth_prior.py`:

```python
from pathlib import Path

import pytest

_PRIOR_CKPT = "/home/rick/autoresearch_depth/weights/yolo26x-depth-640-best.pt"
_have_prior = Path(_PRIOR_CKPT).exists()
prior_only = pytest.mark.skipif(not _have_prior, reason="depth-prior checkpoint not present")


@prior_only
def test_depth_prior_frozen_and_excluded_from_params():
    from ultralytics.models.yolo.s3d.model import Stereo3DDetModel

    m = Stereo3DDetModel("ultralytics/cfg/models/26/yolo26-s3d-depthprior.yaml", ch=3, nc=3, verbose=False)
    dpm = m.depth_prior_model
    assert dpm is not None, "depth_prior_model should be loaded from the YAML depth_prior block"
    assert all(not p.requires_grad for p in dpm.parameters()), "prior model params must be frozen"
    prior_ids = {id(p) for p in dpm.parameters()}
    train_ids = {id(p) for p in m.parameters()}
    assert not (prior_ids & train_ids), "frozen depth model must NOT be part of the trainable module tree"


@prior_only
def test_depth_prior_forward_finite():
    from ultralytics.models.yolo.s3d.model import Stereo3DDetModel

    m = Stereo3DDetModel("ultralytics/cfg/models/26/yolo26-s3d-depthprior.yaml", ch=3, nc=3, verbose=False)
    m.eval()
    x = torch.rand(1, 6, 384, 384)
    with torch.no_grad():
        out = m(x)
    preds = out[1] if isinstance(out, tuple) else out
    assert "depth" in preds and torch.isfinite(preds["depth"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_s3d_depth_prior.py::test_depth_prior_frozen_and_excluded_from_params -v`
Expected: FAIL — `AttributeError: 'Stereo3DDetModel' object has no attribute 'depth_prior_model'` (or it returns None).

- [ ] **Step 3: Slot discovery + frozen prior loading in `__init__`**

In `ultralytics/models/yolo/s3d/model.py`, update the imports at the top:

```python
from ultralytics.nn.modules.block import DepthPriorEncoder, StereoCostVolume
```

Replace the slot-discovery block (`model.py:37-44`) with one that recognizes either occupant of the head's 4th slot:

```python
        # Now enable siamese mode and find tap/cv/prior layer indices
        self._prior_layer = None
        if siamese:
            for m in self.model:
                if isinstance(m, StereoCostVolume):
                    self._tap_layer = m.f  # backbone layer whose output feeds cost volume
                    self._cv_layer = m.i  # StereoCostVolume layer index
                    self._siamese = True
                    break
                if isinstance(m, DepthPriorEncoder):
                    self._prior_layer = m.i  # depth-prior injection slot (replaces cost volume)
                    self._siamese = True  # keep 3ch backbone; right pass is skipped in prior forward
                    break
```

At the end of `__init__` (after the `depth_mode` block, `model.py:56`), load the frozen prior. Store it in a **list** so it is not registered as a submodule (kept out of `state_dict`, `parameters()`, and `.train()`):

```python
        # Track C: own a frozen monocular depth model as a dense-depth prior source.
        self._depth_prior_model = []
        dp_cfg = training_cfg.get("depth_prior")
        if dp_cfg and dp_cfg.get("model"):
            from ultralytics.nn.tasks import attempt_load_one_weight

            dpm, _ = attempt_load_one_weight(dp_cfg["model"])
            dpm.eval()
            for p in dpm.parameters():
                p.requires_grad_(False)
            self._depth_prior_model = [dpm]  # list wrapper hides it from the nn.Module registry
```

Add a property (place near `init_criterion`):

```python
    @property
    def depth_prior_model(self):
        """Frozen monocular depth model used as the dense-depth prior (or None)."""
        return self._depth_prior_model[0] if self._depth_prior_model else None
```

- [ ] **Step 4: Implement the monocular-left prior forward**

Add these methods to `Stereo3DDetModel` (near `_predict_once`), and dispatch to it at the top of `_predict_once`:

```python
    def _compute_depth_prior(self, left_rgb):
        """Run the frozen depth model on left RGB → dense metric depth map [B, 1, h, w].

        Falls back to a zeros depth map at input/4 when the frozen model is not yet loaded
        (e.g. the stride-computation forward inside super().__init__, before the prior is set up).
        """
        if not getattr(self, "_depth_prior_model", []):
            h, w = left_rgb.shape[-2:]
            return left_rgb.new_zeros(left_rgb.shape[0], 1, h // 4, w // 4)
        dpm = self._depth_prior_model[0].to(left_rgb.device)
        with torch.no_grad():
            out = dpm(left_rgb)
        depth_map = out["depth"] if isinstance(out, dict) else out
        if isinstance(depth_map, (tuple, list)):  # eval heads may return (pred, ...)
            depth_map = depth_map[0]
        return depth_map

    def _forward_depth_prior(self, x, profile=False, visualize=False, embed=None):
        """Monocular-left forward: inject the encoded depth prior at the DepthPriorEncoder slot.

        Uses only the left image (no right pass — the cost volume is gone), so inference is a
        single s3d backbone pass plus one frozen depth-model pass.
        """
        left = x[:, :3] if x.shape[1] == 6 else x  # 3ch path = stride computation
        depth_map = self._compute_depth_prior(left)

        y = []
        xf = left
        for m in self.model:
            if m.f != -1:
                xf = y[m.f] if isinstance(m.f, int) else [xf if j == -1 else y[j] for j in m.f]
            if m.i == self._prior_layer:
                feat = m(depth_map)  # DepthPriorEncoder → [B, 64, H/8, W/8]
                p3 = y[16]  # align to P3 spatial size (head concatenates onto P3)
                if feat.shape[-2:] != p3.shape[-2:]:
                    feat = F.interpolate(feat, size=p3.shape[-2:], mode="bilinear", align_corners=False)
                xf = feat
            else:
                xf = m(xf)
            y.append(xf if m.i in self.save else None)
        return xf
```

Add the dispatch as the first lines of `_predict_once` (before the existing `if not self._siamese ...` check at `model.py:103`). It **lazily discovers** the prior slot so it also fires during the stride-computation forward inside `super().__init__` (when the post-super discovery block has not run yet and `_prior_layer` is unset). `self.model` already exists at that point, so the scan succeeds; `_compute_depth_prior` returns a zeros map until the frozen model is loaded:

```python
        # Lazy prior-slot discovery so this fires during super().__init__ stride computation too.
        if getattr(self, "_prior_layer", None) is None:
            for _m in self.model:
                if isinstance(_m, DepthPriorEncoder):
                    self._prior_layer = _m.i
                    break
        if getattr(self, "_prior_layer", None) is not None:
            return self._forward_depth_prior(x, profile, visualize, embed)
```

Add `import torch.nn.functional as F` at the top of `model.py` if not present (it currently imports only `torch`).

**Why the lazy scan matters (stride computation):** the plain `StereoCostVolume` tolerates a feature-map input, so the standard stride forward works for the baseline. `DepthPriorEncoder(c1=1)` does **not** — routing node 0's feature map into it (the YAML `from` is a placeholder) would shape-crash during `super().__init__`. Dispatching to `_forward_depth_prior` (with the zeros fallback) during stride computation avoids running the encoder on a feature map. The post-super discovery block in Step 3 still sets `_prior_layer` explicitly for the normal case; the lazy scan only covers the pre-super window.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_s3d_depth_prior.py -v -k "frozen or forward_finite"`
Expected: PASS (both prior tests). If the checkpoint is absent they SKIP — run on the box where it exists before launch.

- [ ] **Step 6: Commit**

```bash
git add ultralytics/models/yolo/s3d/model.py tests/test_s3d_depth_prior.py
git commit -m "s3d: inject frozen dense depth prior at the cost-vol slot (Track C)

Deleted: StereoCostVolume runtime path in the depth-prior variant (right-image pass dropped; monocular-left forward)"
```

---

### Task 4: End-to-end smoke test + inference-cost measurement

**Files:**
- Modify: `tests/test_s3d_depth_prior.py`

**Interfaces:**
- Consumes: full Track C model (Tasks 1-3) + trainer wiring (unchanged — same `Stereo3DDetTrainer`/`Stereo3DDetModel` path, so `train.py:233-239` builds it and the depth range is set at `train.py:242`).
- Produces: a 1-epoch train+val run that completes with finite loss; a recorded inference-cost delta vs. the plain-cost-volume baseline.

- [ ] **Step 1: Write the smoke test**

Add to `tests/test_s3d_depth_prior.py`:

```python
@prior_only
@pytest.mark.slow
def test_depth_prior_train_smoke(tmp_path):
    """One epoch on a tiny stereo dataset trains and validates without error."""
    from ultralytics import YOLO

    model = YOLO("ultralytics/cfg/models/26/yolo26-s3d-depthprior.yaml")
    results = model.train(
        data="kitti-stereo8.yaml",
        epochs=1,
        imgsz=384,
        batch=2,
        workers=0,
        project=str(tmp_path),
        name="c_smoke",
        pretrained="yolo26n-depth.pt",  # warm-start (foundation)
    )
    assert results is not None
```

- [ ] **Step 2: Run it (box with GPU + checkpoints)**

Run: `pytest tests/test_s3d_depth_prior.py::test_depth_prior_train_smoke -v --slow`
Expected: PASS — training completes 1 epoch, validation runs, no NaN/shape errors. On a machine without the prior checkpoint or dataset it SKIPS.

- [ ] **Step 3: Measure the inference-cost delta (manual, recorded in PR)**

Run baseline then Track C prediction timing on identical inputs:

```bash
# Baseline (plain cost-volume s3d)
yolo predict model=/home/rick/weights/yolo26n-s3d.pt source=<sample-left-dir> imgsz=384 verbose=True 2>&1 | grep -i "ms"
# Track C (needs a trained depth-prior .pt from the run below, or the smoke best.pt)
yolo predict model=<c_smoke>/weights/best.pt source=<sample-left-dir> imgsz=384 verbose=True 2>&1 | grep -i "ms"
```

Record both per-image inference times in the PR body (the frozen depth model adds one forward pass; the dropped right backbone pass claws some of it back). This delta is a required deliverable, not a gate.

- [ ] **Step 4: Commit**

```bash
git add tests/test_s3d_depth_prior.py
git commit -m "Add Track C train+val smoke test and inference-cost measurement recipe

Deleted: nothing (test-only)"
```

---

### Task 5 (optional ablation): depth prior + cost volume together

Run this only if the prior-only variant beats the warm-start baseline — it isolates how much of the gain is the prior vs. the (revived) cost volume.

**Files:**
- Create: `ultralytics/cfg/models/26/yolo26-s3d-depthprior-plus-cv.yaml`
- Modify: `ultralytics/models/yolo/s3d/model.py` (combined forward)
- Modify: `tests/test_s3d_depth_prior.py`

**Interfaces:**
- Consumes: `DepthPriorEncoder`, `StereoCostVolume`, siamese right pass, depth prior.
- Produces: a variant whose head 4th input is `Concat([StereoCostVolume, DepthPriorEncoder]) → 128ch`; `cv_ch` auto-becomes 128 (the head pops the last channel count, so no head change). The model runs BOTH the siamese right pass (for the cost volume) AND the depth-prior pass.

- [ ] **Step 1: Write the build test**

```python
@prior_only
def test_depthprior_plus_cv_builds():
    from ultralytics.models.yolo.s3d.model import Stereo3DDetModel

    m = Stereo3DDetModel("ultralytics/cfg/models/26/yolo26-s3d-depthprior-plus-cv.yaml", ch=3, nc=3, verbose=False)
    head = m.model[-1]
    assert head.cv_ch == 128, f"combined 4th input should be 128ch, got {head.cv_ch}"
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(1, 6, 384, 384))
    preds = out[1] if isinstance(out, tuple) else out
    assert torch.isfinite(preds["depth"]).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_s3d_depth_prior.py::test_depthprior_plus_cv_builds -v`
Expected: FAIL — YAML absent.

- [ ] **Step 3: Create the combined YAML**

Create `ultralytics/cfg/models/26/yolo26-s3d-depthprior-plus-cv.yaml` — identical to `yolo26-s3d-depthprior.yaml` up to node 22, then:

```yaml
  - [1, 1, StereoCostVolume, [64, 48, 24]] # 23: [B, 64, H/8, W/8]
  - [0, 1, DepthPriorEncoder, [64]] # 24: [B, 64, H/8, W/8] (injected)
  - [[23, 24], 1, Concat, [1]] # 25: [B, 128, H/8, W/8]
  - [[16, 19, 22, 25], 1, Stereo3DDetHead, [nc]] # 26
```

Keep the `training.depth_prior` block.

- [ ] **Step 4: Implement the combined forward**

In `model.py`, generalize discovery so both `_cv_layer` and `_prior_layer` may be set (remove the `break` that stops at the first match; record whichever modules exist). Add a `_forward_prior_and_cv` that runs the siamese batch trick (reusing the existing `_predict_once` right-tap logic) AND injects the depth prior at `_prior_layer`. Dispatch to it when both `_cv_layer` and `_prior_layer` are set. Concretely, extend the existing siamese loop in `_predict_once` so that, in addition to `if m.i == self._cv_layer: x = m((y[self._tap_layer][:B], right_tap))`, it also handles `elif m.i == self._prior_layer: x = <encode+align depth prior>` using the same alignment code as `_forward_depth_prior` Step 4. Compute `depth_map` once at the top of the loop from `left`.

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_s3d_depth_prior.py::test_depthprior_plus_cv_builds -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ultralytics/cfg/models/26/yolo26-s3d-depthprior-plus-cv.yaml ultralytics/models/yolo/s3d/model.py tests/test_s3d_depth_prior.py
git commit -m "Add depth-prior + cost-volume ablation variant (Track C)

Deleted: nothing (ablation keeps both sources to isolate contribution)"
```

---

## Experiment / evaluation plan

- **Baselines:** (1) plain `yolo26-s3d` warm-started from `yolo26n-depth.pt` (foundation); (2) current s3d with no transfer.
- **Track C run:** `yolo train model=yolo26-s3d-depthprior.yaml data=kitti-stereo.yaml pretrained=yolo26n-depth.pt imgsz=<train> epochs=<full>`; then the same on `cube-s3d.yaml` (regression guard).
- **Metrics:** existing s3d validator — AP3D@0.5 / AP3D@0.7, depth error, and the stereo-sensitivity probe (expected ~0 here since stereo is gone; the signal must now come from the prior).
- **Success:** AP3D@0.7 on KITTI materially beats both baselines; cube shows no regression. Report the inference-cost delta from Task 4.
- **Export note (measured, not solved):** the frozen depth model is a *second* network, so single-graph ONNX/Hailo export of `forward_export` does not cover it. Document the two-network export cost (two engines, or a fused export as future work) in the results — it is an explicit cost of Track C, weighed against its accuracy ceiling. Do not attempt to make `forward_export` trace the prior in this plan.
- **Parallelization:** this track runs on its own AutoDL box, launched after Tasks 1-4 pass locally (smoke). Track B runs on a second box concurrently.

## Self-Review

- **Spec coverage:** design §"Track C" → Tasks 1-4 (encoder, frozen prior, injection, smoke+cost); "replace the inert cost volume" → Task 2 YAML (node 23) + Task 3 forward (right pass dropped); "concat ablation" → Task 5; "export complexity measured not solved" → Experiment plan. Warm-start (foundation) reused via `pretrained=` in the smoke/run commands.
- **Placeholder scan:** no TBD/TODO; every code step has complete runnable code except Task 5 Step 4, which is deliberately described (optional ablation) rather than fully coded to avoid over-investing before the prior-only result is known — flagged as optional.
- **Type consistency:** `DepthPriorEncoder(c1=1, c2=64)`, `_prior_layer`, `_depth_prior_model` (list), `depth_prior_model` (property), `_compute_depth_prior`, `_forward_depth_prior`, head `cv_ch=64` (128 in ablation) used consistently across tasks and tests. `_build_depthprior`/`prior_only` fixtures shared across tests.
- **Fixes applied inline:** added the P3-size interpolation guard in `_forward_depth_prior` so the head's `torch.cat` never mismatches; used the list-wrapper idiom (verified: `nn.Module.__setattr__` does not register plain lists) so the frozen model is excluded from `parameters()`/`state_dict()`/`.train()`; noted the `import torch.nn.functional as F` addition to `model.py`.
- **Construction-time fix (review):** `DepthPriorEncoder(c1=1)` cannot accept the feature map that the standard stride-computation forward would route into node 23, so building the model would crash inside `super().__init__`. Fixed by lazily discovering the prior slot in `_predict_once` (self.model exists during the stride forward) and dispatching to `_forward_depth_prior`, which synthesizes a zeros depth map until the frozen model is loaded. Verified conceptually against the ordering: stride forward → lazy scan finds the encoder → zeros map → no feature-into-encoder crash; post-super block then sets `_prior_layer` for real forwards.
