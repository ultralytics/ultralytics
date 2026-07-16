# Research Log — s3d capability & improvement

Chronological decision timeline. Newest at bottom.

## Bootstrap

- Deep-read the s3d subsystem (~10k LOC): siamese backbone, StereoCostVolume, aux heads (lr_distance/dims/orient/depth), KITTI R40 eval. Reported docs numbers ~48% AP3D@0.5 Mod but un-reproducible.
- Formed plan: Phase 0 metric integrity → Phase 1 stereo depth → Phase 2 orientation → Phase 3 portability. Goal = maximize accuracy.

## Phase 0 — metric integrity (CONFIRMED H0 groundwork)

- Found `compute_3d_iou` used axis-aligned approximation of rotated boxes → inflated IoU. Replaced with exact rotated BEV×height IoU. Analytic tests (45°→0.707, 90°rect→1/3, disjoint→0). Commit `da9eece4`.
- Added AP_BEV + AOS metrics + per-class reporting; left fitness unchanged. Commit `a6cc8303`. Benchmark script `f60fd565`.

## Phase 1 — cost-volume disparity (workflow-designed, then REFUTED)

- Ran a design→TDD→verify workflow; implemented soft-argmax disparity decoded from the cost volume as primary depth (commit `c63f929f`), with P4/P5 lr_distance fallback fix.
- A/B (from-scratch 200ep) showed it REGRESSED ~2.7× (Car AP3D@0.5 34.3→12.9). But both arms were ~0 at first...

## The real bug hunt (root cause of AP3D≈0)

- First A/B returned 0/0 (both arms). Probed the model directly: 2D detection + depth were ACCURATE (pred z within ~1 m). So eval, not model.
- Localized: predicted box **length ~10.8 m** (GT ~4 m). Root cause = dimension priors keyed by class NAME, looked up by INT id → generic defaults → garbage dimension _targets_; model learned them; decoder mis-expanded. Fixed by rekeying name→int (commit `e8d5501d`); round-trip exact.
- Re-ran A/B with the fix: corrected baseline = **34.3** Car AP3D@0.5 (real capability), Phase 1 = 12.9 → **Phase 1 rejected, reverted** (commit `1bc1d631`).

## Phase 2 — MultiBin orientation (implemented; A/B in progress)

- Implemented 2-bin MultiBin orientation (conf + residual), encode+decode in one module (`orientation.py`) with a round-trip test guard (lesson from the dimension bug). Commit `7bc2dc19`. All unit tests pass.

## A/B at the documented recipe (pretrained 1000ep)

- A (sin/cos baseline) launched on westd; B (MultiBin) launched on ultra1.
- ultra1 setup: 8× Blackwell, internet, venv ~/s3d/venv. B died once at epoch 44 (SIGKILL, no traceback — likely a concurrent `pkill python` from a co-located depth-DDP job); relaunched fresh in tmux (dropped the buggy resume path: resume incompatible with nc auto-expand).
- **B completed**: Car AP3D@0.5 **52.0**, @0.7 10.7, AP_BEV@0.5 63.0, **AOS@0.5 51.9** (≈AP → near-perfect heading on TPs). Pretrained+1000ep is a big win over scratch200 (34.3→52.0).
- **A completed**: Car AP3D@0.5 51.4, @0.7 10.4, AP_BEV@0.5 62.5, AOS@0.5 51.3, mean@0.5 16.3.
- **H2 verdict — SUPPORTED (small), MultiBin KEPT.** B > A on all 6 metrics (Car AP3D@0.5 +0.6, mean +1.5, AOS +0.6). Consistency (6/6) makes it a real if small gain; no downside → keep (HEAD already has it). Caveat: single-seed Car +0.6 ≈ noise; the +1.5 mean and 6/6 consistency are the real support — multi-seed would confirm.
- **Insight:** orientation was NOT the bottleneck — sin/cos already gets AOS@0.5 (51.3) ≈ AP3D@0.5 (51.4). The tight-IoU ceiling is vertical/height: at @0.7, AP_BEV (25.9) ≫ AP3D (10.4). Next lever = box-bottom/height precision, not orientation.
- Headline capability (replaces docs' stale 48%): **~52% AP3D@0.5 (Car, Moderate, true rotated IoU)**, pretrained backbone + 1000ep.

## Model-size scaling sweep (parallel, ultra7 GPUs 2-7)

- Using 6 idle Blackwell GPUs on ultra7 to run the corrected pipeline (MultiBin + dimfix + pretrained backbone + 1000ep SGD/cos, true rotated IoU) across all 5 model sizes, plus a 2nd seed of n for a noise-floor estimate.
- GPU2=n, GPU3=s, GPU4=m, GPU5=l, GPU6=x, GPU7=n(seed1). batch=32 all (x fits at 88.8/97GB). Dataset symlinked + label cache pre-built to avoid 6-way races; pretrained weights pre-fetched.
- Goal: regenerate the docs Models table with TRUSTWORTHY numbers (current docs table predates the IoU + dimension fixes), get the capability-vs-size curve, and quantify run-to-run noise (contextualizes the small MultiBin gain).
- Status: all 6 training, no OOM. Results pending (n first ~6h, x last ~15-20h).
