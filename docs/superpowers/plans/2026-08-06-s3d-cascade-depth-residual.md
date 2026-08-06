# S3D Cascade Depth Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the depth head's _range_ from its _precision_, so depth resolution stops being a function of `DEPTH_BINS`. Stage 1 keeps the existing coarse log-depth distribution; stage 2 predicts a continuous residual inside a window whose width is set by stage 1's own uncertainty (the CasMVSNet / UCSNet pattern, adapted to a per-anchor detection head rather than a plane-sweep volume).

**Architecture:** `DepthDFL` gains an optional residual path. The coarse expectation `mu` and its spread `sigma` are already computed from the bin distribution; the residual head emits one extra channel `delta_raw` per anchor and the decoded log-depth becomes `mu + tanh(delta_raw) * k * sigma`. Bounding by `sigma` means the correction is large where the coarse distribution is unsure and near-zero where it is confident, so the residual cannot fight a well-localised coarse prediction. Everything lives in the existing `depth` aux branch — no new branch, no new tensor plumbed through decode.

**Tech Stack:** Python/PyTorch only. No new dependencies. Reuses `DepthDFL`, `DFLoss`, `_deep_branch`, `get_aux_specs(depth_mode=...)`, `Stereo3DDetMetrics`.

## Global Constraints

- Branch off `001-stereo-centernet-gaps` in a NEW git worktree (`superpowers:using-git-worktrees`); never edit the primary checkout; PR targets `001-stereo-centernet-gaps`. **Cut the worktree from `origin/001-stereo-centernet-gaps`, not the local branch** — the local one goes stale within hours (this cost a rebase already: `git log` showed 1 commit while `git diff` showed 11 files / 772 deletions).
- All experiments run on **`kitti-chen-small`** (`/home/rick/datasets/kitti-chen-small.yaml`, 799 train / 189 val, drive-disjoint). It is a **screening** subset: absolute numbers are not comparable to the full Chen split and must never be quoted as benchmark results.
- Every new Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license`. Google-style docstrings, `from __future__ import annotations`, ruff line length 120. Run `ruff format . && ruff check .` before every commit. `E741` at `preprocess.py:637` is pre-existing — do not "fix" it in this PR.
- Tests in `tests/test_s3d.py`, CPU-only, no network. Full gate before PR: `pytest tests/test_s3d.py -q`.
- **Default must stay off.** `residual_depth: false` unless a YAML opts in, so every existing checkpoint and every other s3d experiment is bit-unchanged.
- PR body needs a `Deleted:` line. This is additive; the rule-2 justification is that decoupling range from precision cannot be expressed as a deletion. Name what was reused: `DepthDFL`, `DFLoss`, the existing depth aux branch.

## Key domain facts implementers must know

- **Bins are already log-spaced and the decode is already continuous.** `DepthDFL` is `softmax → Σ p_i·b_i` over `linspace(log 2, log 80, 16)`, and `DFLoss` splits each target across the two adjacent bins (`wl = tr - target`). So neither decode nor supervision quantises to a bin centre. Anyone who assumes "bin size = depth resolution" will design the wrong thing.
- **The arithmetic that motivates this work.** 16 bins over `ln(80/2)` = 3.689 gives a bin ratio of `e^(3.689/15)` = 1.279, i.e. **27.9% depth step per bin — 8.4 m wide at 30 m**. Prior measurement puts the heads' sub-bin resolution at ~7.9% of a bin ⇒ **~1.96% relative, 0.59 m at 30 m**. Measured system depth error is ~1.2 m. **The representational floor is therefore ~2× below the current error — this work is bounded, and Task 1 exists to measure the bound before any of it is built.**
- **Depth is a fusion of two cues, and the bins are only one.** `preprocess.py:299` fuses `z_from_disp` (from `lr_distance`, a bin-free continuous regression — the _primary_ cue) with `z_from_direct` (the bins) by inverse variance. Improving the bin branch can be invisible end-to-end. Use `get_aux_specs(depth_mode="depth_only")` to isolate the branch under test.
- The depth-cue fusion weight comes from `_dfl_variance`, which reads `outputs["depth_bin_values"]` — the head's real grid (fixed in `ca521ed98`; it used to rebuild the grid from `DEPTH_MIN`/`DEPTH_MAX` and mis-weight every non-KITTI rig by up to 2.13×). The residual must reuse that same `sigma`, not recompute it.
- `_depth_bin_loss` (`loss.py:298`) converts GT log-depth to a fractional bin index via `(gathered - depth_log_min) / depth_log_range * (n_bins - 1)`; `depth_log_min`/`depth_log_range` are set from the dataset range. A residual changes the decoded value but **not** this target — keep the coarse DFL loss exactly as is, or the coarse stage decalibrates and `sigma` becomes meaningless.
- Depth branches are built at `head.py:129`: `_deep_branch(x + self.cv_ch, out_c, depth_hidden)` per FPN scale, `out_c` from `AUX_SPECS`. Adding a residual channel means `AUX_SPECS["depth"] = n_bins + 1` when enabled — check `PER_ANCHOR_AUX_KEYS` (`preprocess.py:33`) and the export concat (`head.py:213`) still line up.
- Runs are seedable as of `a5e766bc7`, so the noise floor can actually be measured. Use it.

## File structure

- Modify: `ultralytics/models/yolo/s3d/head.py` — `DepthDFL` residual path + one extra output channel when enabled (~35 lines).
- Modify: `ultralytics/models/yolo/s3d/loss.py` — optional L1 term on the final decoded log-depth (~15 lines).
- Modify: `ultralytics/cfg/models/26/yolo26-s3d.yaml` (+ `-kitti`) — the opt-in flag only.
- Create: `ultralytics/data/scripts/depth_precision_probe.py` — the Task 1 oracle probe (~120 lines).
- Modify: `tests/test_s3d.py` — gates listed per task.

---

### Task 1 (KILL GATE): measure the ceiling before building anything

- [ ] Write `depth_precision_probe.py`: run val on `kitti-chen-small`, and for every TAL-matched detection substitute the predicted depth with (a) exact GT depth, (b) GT depth snapped to the **current 16-bin** grid, (c) GT snapped to a 64-bin grid. Report AP3D Car @0.7/@0.5 Mod for each.
- [ ] **Decision rule.** If (b) is within the AP noise floor of (a), bin resolution is _not_ the binding constraint and this plan stops here — the finding goes in `research/findings.md` and the effort moves to the disparity cue. Proceed only if (a) − (b) exceeds the noise floor measured in Task 2.
- [ ] Record all three numbers in the PR body regardless of outcome.

### Task 2: establish the noise floor on this subset

- [ ] Train the unmodified baseline 3× on `kitti-chen-small` with different seeds; report per-seed AP3D Car @0.7 Mod and median |Δlog z| on matched detections.
- [ ] **Expect AP on 189 val images to be much noisier than the ~0.8–1.2 AP floor measured on the full split.** If the AP spread swamps the Task 1 gap, switch the primary metric to **median |Δlog z| on matched detections**, which is far more stable, and demote AP3D to a secondary check.
- [ ] Write both floors into the plan file before running any arm.

### Task 3: the residual path (only if Task 1 passes)

- [ ] `DepthDFL.__init__(..., residual: bool = False, k: float = 1.0)`. When enabled, `forward` splits the last channel off as `delta_raw`, computes `mu` and `sigma` from the bin distribution, and returns `mu + torch.tanh(delta_raw) * k * sigma`.
- [ ] `sigma` must be the **std of the same distribution** `_dfl_variance` reports, computed on `self.bin_values` — one definition, used by both.
- [ ] Gate: `test_residual_is_bounded_by_coarse_sigma` — with a razor-sharp coarse distribution (σ→0) the decoded depth must equal the non-residual decode to within 1e-6 whatever `delta_raw` is; with a flat distribution it must be able to move by ~k·σ.
- [ ] Gate: `test_residual_default_off_is_bitwise_identical` — same input, `residual=False` vs current `main` decode, `torch.equal`.

### Task 4: supervision

- [ ] Keep `_depth_bin_loss` unchanged on the coarse logits (it keeps `sigma` calibrated).
- [ ] Add an L1 term on the final decoded log-depth vs GT log-depth, weighted by a new `dres` hyperparameter (default 1.0), applied to foreground anchors only.
- [ ] Gate: `test_residual_loss_is_zero_when_decode_is_exact`.

### Task 5: the A/B on kitti-chen-small

- [ ] Arms, ≥3 seeds each, identical everything else: **(0)** baseline 16 bins · **(1)** 64 bins, no residual · **(2)** 16 bins + residual · **(3)** 64 bins + residual.
- [ ] Arm (1) is the control that matters: **if more bins alone matches the cascade, ship the bins and delete the residual.** 16→64 costs ~9k params and no new code path.
- [ ] Run every arm twice: once normally, once with `depth_mode="depth_only"` to isolate the branch from the disparity cue it is fused with.
- [ ] Report median |Δlog z| (primary), AP3D Car @0.7/@0.5 Mod (secondary, with the Task 2 floor beside it), and the fraction of anchors where `|tanh(delta_raw)| > 0.9` — **a saturating residual means `k` is too small and the window is clipping.**

### Task 6: promote or stop

- [ ] Only if an arm beats baseline by more than the Task 2 floor on the primary metric, confirm on the **full Chen split** before claiming anything.
- [ ] Write the outcome — including a null — into `research/findings.md`. Both previous attempts at this class of fix (item 5 surface residual, quality head) were nulls; a third null is a useful result, not a failure.

## Verification

```bash
cd <worktree>
/home/rick/ultralytics/.venv/bin/python -m pytest tests/test_s3d.py -q     # 48 passed, 1 skipped at baseline
ruff format . && ruff check .
# screening A/B (GPU box):
yolo s3d train model=yolo26n-s3d.yaml data=kitti-chen-small.yaml epochs=200 imgsz=384,1248 seed=0
```

- The regression guard is `test_residual_default_off_is_bitwise_identical`: if a future change makes the residual path affect the default configuration, it fails.
- Ground truth for "how much depth precision is actually available from the images" is `disp_gate.py` in `/home/rick/s3d_tools` — a model-free NCC block matcher that reaches **0.98 px / 1.74 m median**. Any learned depth claim should be read against it.
