# S3D Depth-Transfer Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the `yolo26-s3d` YAML so a monocular `yolo26-depth` checkpoint warm-starts the *entire* s3d backbone + FPN + PAN (today only backbone + top-down FPN through P3 transfers), and verify the transfer.

**Architecture:** `BaseModel.load` uses `intersect_dicts` (match by exact key name + tensor shape; non-matches silently skipped). The `StereoCostVolume` node inserted mid-head shifts the bottom-up PAN indices out of alignment with `yolo26-depth.yaml`, so the PAN weights fail to name-match and are dropped. Moving `StereoCostVolume` to a leaf position *after* the PAN restores index alignment for nodes 0–22 while preserving identical runtime wiring (the cost volume is a leaf that only feeds the head's depth branches).

**Tech Stack:** PyTorch, Ultralytics `parse_model`/`BaseModel.load`, pytest.

## Global Constraints

- Python >= 3.8, PyTorch >= 1.8 (repo floor).
- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` (Actions bot adds; don't add manually).
- Line length 120; `ruff format . && ruff check --fix .` before commit.
- Never edit the primary checkout — work in a git worktree on a feature branch; open a PR. Never push to `main`, never force-push.
- Warm-start checkpoint (scale-matched): `yolo26n-depth.pt` (local at `/home/rick/ultralytics_depth/.claude/worktrees/cfg-simplify/weights/yolo26n-depth.pt`, or auto-downloaded release asset `yolo26n-depth.pt`).
- Teacher/prior checkpoint (both tracks, scale-independent): `/home/rick/autoresearch_depth/weights/yolo26x-depth-640-best.pt`.
- Student scale: `n`.

## Locked interfaces (consumed by Track B and Track C plans)

- **Reordered `yolo26-s3d.yaml` head node indices:** P3 = **16**, P4 = **19**, P5 = **22**, `StereoCostVolume` = **23**, `Stereo3DDetHead` = **24**. Head `from` list = `[16, 19, 22, 23]`.
- These P3/P4/P5 indices (16/19/22) are **identical to `yolo26-depth.yaml`**, so any module attached by node index in one aligns with the other.
- Warm-start entrypoint: `Stereo3DDetModel(...).load("<depth>.pt")` (already called in `Stereo3DDetTrainer.get_model` at `train.py:238-239` when `weights` given, i.e. `yolo train ... pretrained=<depth>.pt`).

---

### Task 1: Reorder the s3d YAML to align PAN indices with the depth model

**Files:**
- Modify: `ultralytics/cfg/models/26/yolo26-s3d.yaml:49-71` (the `head:` block)
- Test: `tests/test_s3d_depth_transfer.py` (create)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: reordered YAML with P3=16, P4=19, P5=22, cost-vol=23, head=24, head `from`=`[16,19,22,23]`. Runtime output identical to before the reorder.

- [ ] **Step 1: Write the failing test** — build the s3d model from the (not-yet-reordered) YAML and assert node types/indices at the new positions, plus a forward-shape equivalence check.

Create `tests/test_s3d_depth_transfer.py`:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for warm-starting s3d from a monocular depth checkpoint (foundation)."""

import torch

from ultralytics.nn.modules.block import StereoCostVolume
from ultralytics.models.yolo.s3d.head import Stereo3DDetHead
from ultralytics.models.yolo.s3d.model import Stereo3DDetModel


def _build_s3d(scale="n"):
    return Stereo3DDetModel(f"ultralytics/cfg/models/26/yolo26-s3d.yaml", ch=3, nc=3, verbose=False)


def test_s3d_yaml_pan_indices_aligned():
    """After reorder: P3=16, P4=19, P5=22, StereoCostVolume=23, head=24 with from=[16,19,22,23]."""
    m = _build_s3d().model
    assert isinstance(m[23], StereoCostVolume), f"node 23 should be StereoCostVolume, got {type(m[23])}"
    head = m[24]
    assert isinstance(head, Stereo3DDetHead), f"node 24 should be Stereo3DDetHead, got {type(head)}"
    assert list(head.f) == [16, 19, 22, 23], f"head from-list should be [16,19,22,23], got {list(head.f)}"


def test_s3d_forward_runs_after_reorder():
    """Model still produces finite 3D preds on a 6ch stereo input after the reorder."""
    m = _build_s3d()
    m.eval()
    x = torch.rand(1, 6, 384, 384)
    with torch.no_grad():
        out = m(x)
    # eval returns (inference_tensor, preds_dict)
    preds = out[1] if isinstance(out, tuple) else out
    assert "depth" in preds and torch.isfinite(preds["depth"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_s3d_depth_transfer.py::test_s3d_yaml_pan_indices_aligned -v`
Expected: FAIL — node 23 is currently `Stereo3DDetHead` (cost volume is at 17), so the `isinstance` assert fails.

- [ ] **Step 3: Reorder the YAML head block**

Replace the `head:` block of `ultralytics/cfg/models/26/yolo26-s3d.yaml` (lines 49-71) with:

```yaml
# YOLO26 head — PAN indices 0-22 aligned with yolo26-depth.yaml for full warm-start.
# Cost volume is a leaf (feeds only depth branches) moved AFTER the PAN so it does
# not shift the PAN indices out of name-match alignment with the depth checkpoint.
head:
  # Top-down FPN
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11
  - [[-1, 6], 1, Concat, [1]] # 12
  - [-1, 2, C3k2, [512, True]] # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 4], 1, Concat, [1]] # 15
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8, clean — no cost vol)

  # Bottom-up PAN from clean P3 (indices 17-22 match yolo26-depth.yaml)
  - [16, 1, Conv, [256, 3, 2]] # 17
  - [[-1, 13], 1, Concat, [1]] # 18
  - [-1, 2, C3k2, [512, True]] # 19 (P4/16)
  - [-1, 1, Conv, [512, 3, 2]] # 20
  - [[-1, 10], 1, Concat, [1]] # 21
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 22 (P5/32)

  # Stereo cost volume (leaf; separate, NOT merged into P3) — moved after PAN
  - [1, 1, StereoCostVolume, [64, 48, 24]] # 23: [B, 64, H/8, W/8]

  # Head: [P3, P4, P5, cost_vol] — cost vol only feeds depth branches
  - [[16, 19, 22, 23], 1, Stereo3DDetHead, [nc]] # 24
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_s3d_depth_transfer.py -v`
Expected: PASS (both `test_s3d_yaml_pan_indices_aligned` and `test_s3d_forward_runs_after_reorder`).

- [ ] **Step 5: Commit**

```bash
git add ultralytics/cfg/models/26/yolo26-s3d.yaml tests/test_s3d_depth_transfer.py
git commit -m "Reorder s3d YAML: move StereoCostVolume to leaf so depth PAN weights align

Deleted: nothing (pure reorder; runtime wiring unchanged, warm-start alignment restored)"
```

---

### Task 2: Verify full-neck warm-start from the depth checkpoint

**Files:**
- Modify: `tests/test_s3d_depth_transfer.py` (add a test)

**Interfaces:**
- Consumes: reordered YAML from Task 1; `Stereo3DDetModel.load` (inherited `BaseModel.load` → `intersect_dicts`).
- Produces: a regression test asserting backbone+neck (nodes 0–22) parameters match the depth checkpoint after `load`, guarding the alignment against future YAML drift.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_s3d_depth_transfer.py`:

```python
from pathlib import Path

import pytest

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.torch_utils import intersect_dicts

# Prefer local checkpoint; fall back to the release asset name for auto-download.
_DEPTH_CKPT_LOCAL = Path("/home/rick/ultralytics_depth/.claude/worktrees/cfg-simplify/weights/yolo26n-depth.pt")
_DEPTH_CKPT = str(_DEPTH_CKPT_LOCAL) if _DEPTH_CKPT_LOCAL.exists() else "yolo26n-depth.pt"


def test_warmstart_transfers_full_backbone_and_neck():
    """model.load(depth.pt) must transfer every param for nodes 0-22 (backbone + FPN + PAN)."""
    m = _build_s3d()
    depth_model, _ = attempt_load_one_weight(_DEPTH_CKPT)
    depth_sd = depth_model.float().state_dict()

    # Keys the s3d model shares with the depth checkpoint (nodes 0-22 only; exclude head/cost-vol).
    s3d_sd = m.model.float().state_dict()
    shared = intersect_dicts(depth_sd, s3d_sd)
    node_022 = {k for k in shared if k.split(".")[1].isdigit() and int(k.split(".")[1]) <= 22}
    assert len(node_022) > 0, "no shared backbone/neck params found — index alignment is broken"

    m.load(_DEPTH_CKPT)
    loaded_sd = m.model.state_dict()
    # Every shared node-0..22 tensor in the s3d model must now equal the depth checkpoint's.
    mismatched = [k for k in node_022 if not torch.equal(loaded_sd[k].float(), depth_sd[k].float())]
    assert not mismatched, f"{len(mismatched)} backbone/neck tensors did not warm-start: {mismatched[:5]}"
```

- [ ] **Step 2: Run test to verify it fails (or is the guard)**

Run: `pytest tests/test_s3d_depth_transfer.py::test_warmstart_transfers_full_backbone_and_neck -v`
Expected: PASS after Task 1 (this test *documents and guards* the fix). If run against the pre-reorder YAML it would FAIL because PAN nodes 17-22 differ. If the depth checkpoint is unavailable and cannot auto-download, the test errors on load — resolve by placing the checkpoint locally (see Global Constraints).

- [ ] **Step 3: (No implementation needed — Task 1 already provides the behavior.)**

This task adds a guard test only. If it fails, the bug is in Task 1's reorder; fix there.

- [ ] **Step 4: Confirm the transfer count in a real run**

Run: `yolo train model=ultralytics/cfg/models/26/yolo26-s3d.yaml data=cube-s3d.yaml pretrained=<depth>.pt epochs=1 imgsz=384 2>&1 | grep -i transferred`
Expected: the "Transferred X/Y items" line shows X near the full backbone+neck param-tensor count (not ~half). Record the number in the PR body.

- [ ] **Step 5: Commit**

```bash
git add tests/test_s3d_depth_transfer.py
git commit -m "Guard: assert depth warm-start transfers full s3d backbone+neck (nodes 0-22)

Deleted: nothing (test-only guard for the reorder in the prior commit)"
```

---

## Self-Review

- **Spec coverage:** F1 (YAML reorder) → Task 1. F2 (warm-start path) → already wired in `train.py:238-239`; Task 2 verifies it end to end. Both foundation requirements covered.
- **Placeholder scan:** none — all steps contain runnable code/commands.
- **Type consistency:** `_build_s3d()`, `_DEPTH_CKPT`, node indices 16/19/22/23/24 used identically across both tasks.
- **Note for executors:** this plan is a prerequisite for `2026-07-23-s3d-depth-track-b-distillation.md` and `2026-07-23-s3d-depth-track-c-dense-prior.md`. Merge this to the base branch (or rebase both track branches onto it) before starting either track.
