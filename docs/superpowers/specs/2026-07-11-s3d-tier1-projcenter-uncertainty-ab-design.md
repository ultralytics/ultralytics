# s3d Tier-1 A/B: projected-3D-center offset + depth uncertainty

**Date:** 2026-07-11
**Branch base:** `001-stereo-centernet-gaps` @ ec3af8bdc (post imgsz depth fix)
**Goal:** raise KITTI Car _Moderate_ **AP3D@0.7** (baseline 4.2; @0.5 is 34.3) by fixing the two structural localization gaps the code trace and the recent-literature survey both flagged as top levers.

## Motivation

The @0.7 collapse is a localization-precision problem, not detection. Two root gaps in the decode:

1. **3D center = back-projection of the 2D box center** (`preprocess.py`). The 2D box center is not the projection of the 3D centroid; the residual bias amplifies through the pinhole ray into metric X/Y error and, via `ray_angle`, into heading. This is the dominant strict-IoU localization error in the monocular/stereo literature (MonoDLE, SMOKE, MonoFlex).
2. **Depth fusion is an unweighted geometric mean with no uncertainty** (`preprocess.py`). One of the two depth cues is effectively wasted, and the detection score carries no localization-quality signal, so the ranked AP@0.7 curve cannot reward well-localized boxes (MonoFlex, GUPNet).

## Two levers

Each lever is gated by an independent flag in the model YAML `training:` block so one codebase runs every arm. A flag gates head-branch construction + its loss term + its decode use, so arms are cleanly attributable.

### Lever 1 — Projected-3D-center offset (`use_proj_center`)

- **Head** (`head.py`): new aux branch `proj_offset`, 2 channels (Δu, Δv), per scale, following the existing `_branch` aux pattern. Added to `AUX_SPECS` only when the flag is set.
- **Target** (`dataset.py`): project the 3D box **centroid** to the image with the sample's P2, express in letterbox-normalized coords, subtract the 2D box center → `(Δu, Δv)` residual, stored in `aux_targets["proj_offset"]`.
    - _KITTI subtlety to verify in implementation:_ `location_3d` is the box **bottom-center**; the centroid is `location + (0, -h/2, 0)` in the camera frame (y-down). Project the centroid, so the target matches the point the decoder reconstructs as `center_3d` for IoU. A unit round-trip test (encode centroid → project → decode offset → back-project → recover centroid X/Y) guards this, mirroring the existing dimension/orientation round-trip tests.
- **Loss** (`loss.py`): smooth-L1 on positives via the existing `_aux_loss` path; new `loss_weights.proj_center` entry. Pseudo-label weighting applies like the other aux terms.
- **Decode** (`preprocess.py`): when the flag is on, `u,v = 2D_box_center + predicted_offset` (un-normalized to letterbox px) before back-projection. Corrects `x_3d, y_3d` and, through `ray_angle = atan2(x_3d, z_3d)`, the heading.

### Lever 2 — Depth uncertainty (`use_depth_uncertainty`), split into two decode-time knobs

Uncertainty is _trained once_ (NLL); how it is _used_ is a decode-time toggle, so fusion and score-weighting are isolated on the same checkpoint.

- **Head** (`head.py`): add one log-variance channel `σ_lr` to the `lr_distance` branch. The direct-depth cue reuses the **DFL distribution spread** `σ_direct² = Σ pᵢ(bᵢ − μ)²` (free, no new channel).
- **Loss** (`loss.py`): train `lr_distance` with Laplacian NLL (attenuated L1) using `σ_lr` instead of plain smooth-L1. Direct-depth training is unchanged (still DFL). NLL is used only when the flag is set.
- **Decode knobs** (`preprocess.py`), each a separate config toggle:
    - `ivw_fusion`: fuse the two log-depths by inverse variance `w = 1/σ²` (weighted mean in log-space) — _replaces_ the current geometric mean (which is the equal-weight special case).
    - `score_weight`: multiply the detection score by `exp(−k·σ_total)` (k a small constant), demoting depth-uncertain boxes in the ranked AP.

## Experiment: 4 training runs → 8 eval arms

**Training runs** (weights differ only here):

| Run | Flags                   | Train-time change                            |
| --- | ----------------------- | -------------------------------------------- |
| T0  | —                       | current losses                               |
| Tc  | `use_proj_center`       | + proj_offset head + smooth-L1               |
| Tσ  | `use_depth_uncertainty` | + σ_lr channel, Laplacian NLL on lr_distance |
| Tcσ | both                    | center + σ                                   |

**Eval arms** — decode toggles `{ivw_fusion, score_weight}`, all reported vs the 34.3/4.2 baseline. A2–A4 share **Tσ**; A5–A7 share **Tcσ** (fusion-vs-score isolated with zero retraining):

| Arm              | Checkpoint | Fusion   | Score wt | Center     |
| ---------------- | ---------- | -------- | -------- | ---------- |
| A0 baseline      | T0         | geo-mean | —        | box-center |
| A1 +center       | Tc         | geo-mean | —        | ✓          |
| A2 +fusion       | Tσ         | IVW      | —        | —          |
| A3 +score        | Tσ         | geo-mean | ✓        | —          |
| A4 +fusion+score | Tσ         | IVW      | ✓        | —          |
| A5 full stack    | Tcσ        | IVW      | ✓        | ✓          |
| A6 center+fusion | Tcσ        | IVW      | —        | ✓          |
| A7 center+score  | Tcσ        | geo-mean | ✓        | ✓          |

**Protocol:** each training run matches the recorded baseline exactly — `yolo train task=s3d model=yolo26-s3d.yaml data=<kitti-stereo> epochs=200 batch=32 val=False`, then a `val` pass per eval arm (decode flags set at val time). Runs on the weste seetacloud box (SSH + autodl-proxy for internet; ship the branch via `git archive`, editable-install, set `yolo settings datasets_dir/runs_dir`, caches on the data disk). Arms train in parallel if the box has ≥4 GPUs, else queued.

**Primary metric:** Car Moderate AP3D@0.7. Report @0.5 / AP_BEV / AOS and Easy/Hard alongside. A lever "wins" if it improves @0.7 without regressing @0.5.

## Scope guards (YAGNI)

- No new depth-bin scheme (LID/SID is a separate Tier-1 item), no 3D-IoU loss, no keypoints, no cost-volume changes.
- No change to 2D detection or TAL assignment — the projected center is a decode/head addition, not a re-anchoring of the heatmap.
- Config flags are experiment scaffolding: after a winner is chosen, the losing branches and flags are deleted (the winning path becomes unconditional), per the repo's Delete-first principle.

## Testing

- Unit round-trip test for `proj_offset` encode/decode (centroid projection ↔ offset ↔ back-projection), CPU-only, alongside the existing s3d round-trip tests.
- Unit test that `ivw_fusion` with equal σ reduces to the geometric mean (continuity with A0), and that `score_weight` monotonically demotes higher-σ boxes.
- Existing s3d test suite (10 unit tests + `test_val`) must stay green.
