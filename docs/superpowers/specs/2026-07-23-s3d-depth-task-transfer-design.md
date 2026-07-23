# S3D ← Depth-task transfer: warm-start, distillation, and dense-depth prior

**Date:** 2026-07-23
**Status:** Design — awaiting review
**Branch base:** `001-stereo-centernet-gaps`

## Motivation

The stereo-3D (`s3d`) task is **effectively monocular** — measured stereo sensitivity ≈ 0.00
on both cube_s3d and KITTI; the `StereoCostVolume` branch is barely used. The dominant AP3D
bottleneck is **depth accuracy**: a depth-bias-correction oracle showed a ~5.3× AP3D@0.7 upper
bound. The 3D head already infers depth monocularly, and depth is what caps the score. Therefore
a strong monocular-depth prior — injected as pretrained weights, a distillation teacher, or a
dense-depth input — attacks the exact weak spot.

The monocular **depth task** already lives in this repo (`yolo26-depth.yaml`, `Depth` head,
`DepthLoss26`) and shares an **identical backbone and top-down neck** with s3d. We exploit that
overlap three ways, run as **two parallel experiment tracks** plus a shared foundation.

Goal framing: a **general, reusable method** (not cube-specific). Validate on depth-diverse
**KITTI**, then confirm no regression on the **cube_s3d** client deliverable.

## Assets (no depth training needed)

- **Warm-start (scale-matched):** `yolo26n-depth.pt` (local; also an auto-downloadable release
  asset — `yolo26{n..x}-depth.pt`). Must match the student scale for name+shape matching.
- **Teacher / prior (stronger, scale-independent):** `yolo26x-depth-640-best.pt`
  (`/home/rick/autoresearch_depth/weights/`) — strongest available depth model.
- **Student:** `yolo26n-s3d` (scale `n`; matches prior s3d/cube work).
- **Baseline:** current `yolo26-s3d` with no depth transfer.

## Shared foundation (applies to BOTH tracks)

### F1. Reorder the s3d YAML so warm-start transfers the *whole* neck

`BaseModel.load` uses `intersect_dicts` (name+shape matched; non-matches silently skipped).
Today the `StereoCostVolume` inserted at node 17 shifts the bottom-up PAN indices, so a depth
checkpoint only transfers backbone + top-down FPN through P3 (nodes 0–16); the P4/P5 PAN weights
silently fail to match.

**Fix:** move `StereoCostVolume` to just before the head so PAN indices 0–22 align exactly with
`yolo26-depth.yaml`:

```
16: C3k2 256          # P3   (unchanged)
17: Conv 256 s2       # was 18
18: Concat [-1, 13]   # was 19
19: C3k2 512          # P4   (was 20)  ← matches depth node 19
20: Conv 512 s2       # was 21
21: Concat [-1, 10]   # was 22
22: C3k2 1024         # P5   (was 23)  ← matches depth node 22
23: StereoCostVolume [1, ...]          # moved here (leaf → head input)
24: Stereo3DDetHead [[16, 19, 22, 23], nc]
```

After this, `model.load(depth.pt)` transfers backbone + full FPN+PAN (nodes 0–22); only node 23
(cost volume) and the head are new. **Verification gate:** the "Transferred X/Y items" log line
must show the full backbone+neck transferring (not ~half).

### F2. Warm-start path (A)

Build the s3d model, then `model.load(yolo26n-depth.pt)` (equivalently `pretrained=`). No new code
beyond F1. Nearly free, fully reversible, applies to both tracks. Ablation: warm-start on/off.

## Track B — auxiliary depth head + distillation (zero inference cost)

**Hypothesis:** forcing the shared backbone/neck to encode dense metric depth (via a training-only
head distilled from a frozen depth teacher) sharpens the per-anchor depth-DFL branch and lifts
AP3D@0.7, at **no inference/deploy cost**.

### B1. Training-only auxiliary depth head
- Attach the monocular `Depth` head to the left-branch neck features `[16, 19, 22]` (P3/P4/P5),
  mirroring the Pose `cv4` / `one2many` pattern used by `Stereo3DDetHead`.
- Gate it with `if self.training`: it produces `preds["aux_depth"]` (dense metric depth at
  input/4) during training only; not built into the inference/export graph.

### B2. Frozen distillation teacher
- Load `yolo26x-depth-640-best.pt` once, frozen, `eval()`, on the trainer device (new hook,
  mirroring the existing `on_train_epoch_start` `epoch_frac` precedent).
- In `preprocess_batch`, run the teacher on the **left** image (no grad) → `batch["teacher_depth"]`
  (metric depth, resized to aux-head resolution).
- No dense-depth GT required → **dataset-agnostic**. On VKITTI2/KITTI, real GT depth (where present)
  MAY additionally supervise the aux head; the general path uses the teacher only.

### B3. Distillation loss
- New s3d loss term `distill`: `DepthLoss26`-style SILog + gradient-matching between `aux_depth`
  and `teacher_depth`, weighted by a new `training.loss_weights.distill`.
- Added to `_compute_aux_losses` / `loss()` and `loss_names`. Aux head + teacher never run at
  inference → zero cost. **Deleted:** none for B; but B produces the evidence for the cost-volume
  deletion decision (does distillation revive it, or confirm it is dead weight?).

## Track C — dense depth prior fused into the head (highest ceiling)

**Hypothesis:** handing the 3D head a strong dense metric-depth field directly maximizes depth
accuracy — accepting a second forward pass at inference.

### C1. Depth prior source
- Frozen `yolo26x-depth-640-best.pt` produces a **dense metric depth map** for the left image at
  **train and inference**.

### C2. Fusion — replace the inert cost volume
- Encode the dense depth map (one stride-2 conv to reach P3 stride/8; small conv encoder) and feed
  it into the depth/lr branches at P3 **in place of `StereoCostVolume`** (node 23 slot).
- This is the delete-first form: **Deleted:** `StereoCostVolume` from the s3d graph for this track
  (the branch is inert; C tests whether an external depth prior is the better occupant of that slot).
- Ablation vs. keeping the cost volume alongside the prior (concat) to isolate contribution.

### C3. Inference/deploy
- Two forward passes (frozen depth model + s3d) or a single fused model. Export complexity is
  explicitly in scope as a cost to measure, not to solve now.

## Evaluation

- **Datasets:** KITTI (`kitti-stereo`, primary — depth-diverse, AP3D@0.7 is depth-bound) and
  cube_s3d (regression check on the client deliverable).
- **Metrics:** existing s3d validator — AP3D@0.5 / AP3D@0.7, depth error, plus the
  **stereo-sensitivity probe** (does the model use the right image?).
- **Baselines:** (1) current s3d; (2) warm-start-only (F2) to isolate the transfer contribution
  from each track's mechanism.
- **Success:** a track "wins" if AP3D@0.7 on KITTI improves materially over both baselines with no
  cube regression. Track B additionally must add zero inference cost; Track C reports its inference
  cost delta.

## Parallelization / execution

- **Two git worktrees** off `001-stereo-centernet-gaps`, one per track (Core Principles: never edit
  the primary checkout; branch + PR per track).
- Each track passes a **local smoke test** (1–2 epochs, tiny subset — `kitti-stereo8`, `depth8`)
  proving the model builds, warm-start transfers the full neck, teacher/prior runs, and loss is
  finite — **before** any box is rented.
- **Then** rent **two AutoDL boxes** (one per track) and launch full KITTI+cube runs in parallel.
  Boxes are rented at launch, not now, to avoid idle billing.

## Open risks

- **Teacher domain gap:** the depth teacher was trained on its own datasets; on KITTI/cube its
  metric scale may drift. Mitigation: the aux/prior path uses scale-robust SILog; optionally fit a
  log-affine scale to the teacher output (the `Depth` head already carries `cal_a`/`cal_b`).
- **Track C export:** two-network inference complicates Hailo/ONNX export; measured, not solved here.
- **cube near-constant depth:** cube may show little movement for either track — expected; KITTI is
  the real testbed and cube is the regression guard.

## Deliverables

- F1 YAML reorder (shared) + warm-start verified.
- Track B: aux head + teacher hook + `distill` loss; smoke test; KITTI+cube run.
- Track C: dense-depth fusion replacing cost volume; smoke test; KITTI+cube run.
- Per-track PR with a `Deleted:` line (B: none + cost-volume evidence; C: `StereoCostVolume`).
- Results comparison vs. both baselines; recommendation on the cost-volume's fate.
