# Findings — Stereo 3D Detection (s3d) capability & improvement

_Project: investigate the YOLO26 stereo-3D ("s3d") task on branch `001-stereo-centernet-gaps`, establish true capability, and improve it. Metric: KITTI R40 AP3D / AP_BEV / AOS, Car @ Moderate, with **true rotated** 3D IoU._

Last updated: 2026-06-01. Status: H0 confirmed, H1 refuted, **H2 supported (small, consistent)** — A/B complete.

> **A/B verdict (H2):** MultiBin beats sin/cos on **all 6 metrics** (Car AP3D@0.5 +0.6, mean +1.5, AOS@0.5 +0.6) — a small but consistent (6/6) gain → **kept**. But orientation was _not_ the bottleneck: sin/cos already achieves AOS@0.5 (51.3) ≈ AP3D@0.5 (51.4), i.e. heading is near-perfect on TPs either way. The real ceiling is **tight-IoU vertical localization** (at @0.7, AP_BEV 25.9 ≫ AP3D 10.4 → height/Y, not BEV, is what fails).

---

## Current Understanding

The s3d task **appeared to have no capability (AP3D ≈ 0)**, but this was caused by **two evaluation/target bugs, not the model**. The underlying model produces accurate 2D detections and accurate metric depth (probed: predicted z within ~1 m of GT at 0.9+ conf). Once the bugs were fixed, the task shows real, competitive capability: **52.0% AP3D@0.5 (Car, Moderate)** with the documented training recipe.

Two levers were then tested by controlled A/B (same recipe, one change):

- **Cost-volume stereo disparity (Phase 1): REFUTED** — regressed AP3D ~2.7×.
- **MultiBin orientation (Phase 2): PENDING** — B (multibin) done at 52.0 AP3D@0.5 / 51.9 AOS@0.5; baseline A (sin/cos) finishing to complete the comparison.

## Hypotheses & Status

| ID     | Hypothesis                                                                                              | Prediction                                                   | Status                                                              |
| ------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- |
| **H0** | The reported AP3D≈0 is an eval/target bug, not model incapacity                                         | Fixing it yields nonzero AP3D with no retraining of the idea | **CONFIRMED**                                                       |
| **R0** | Pretrained backbone + 1000ep ≫ from-scratch 200ep                                                       | Large AP3D lift                                              | **CONFIRMED** (34.3→52.0 Car@0.5)                                   |
| **H1** | Decoding disparity from the cost volume (soft-argmax) as primary depth beats the lr_distance regression | AP3D up                                                      | **REFUTED** (34.3→12.9, ~2.7× worse)                                |
| **H2** | MultiBin orientation beats sin/cos at resolving heading → higher AOS/AP3D                               | AOS up vs sin/cos baseline                                   | **SUPPORTED, small** (B>A on 6/6 metrics; +0.6 AP3D@0.5, +1.5 mean) |

## Key Results

### The two bugs (root causes of AP3D≈0)

1. **3D IoU was an axis-aligned-bbox approximation** of rotated boxes (`utils/metrics.py:compute_3d_iou`) — inflated IoU for yawed boxes (e.g. two identical boxes 45° apart returned IoU 1.0; true 0.707). Replaced with exact rotated BEV-polygon × height-overlap IoU. Commit `da9eece4`.
2. **Dimension priors were keyed by class NAME in YAML but looked up by INT class_id** (`compute_dimension_offset`). Every lookup missed → generic default mean/std → the encoded 3D dimension _targets_ were garbage (a 4.2 m car encoded ΔL≈16). The model faithfully learned the garbage (dims-loss stayed low), and the decoder expanded it to **~10.8 m-long cars** → 3D IoU <0.5 → AP3D=0. Depth/orientation were unaffected (they don't use these priors), which is why only dimensions broke, length worst. Fixed by rekeying name→int. Commit `e8d5501d`. Decoded length round-trips exactly after the fix (4.2 → 4.200).

Also added the missing KITTI metrics **AP_BEV** and **AOS** (`a6cc8303`) and a reproducible benchmark script (`f60fd565`).

### Capability (corrected metrics, Car @ Moderate, true rotated IoU)

| Config                                    | AP3D@0.5 | AP3D@0.7 | AP_BEV@0.5 | AOS@0.5  |
| ----------------------------------------- | -------- | -------- | ---------- | -------- |
| pre-fix (any)                             | ~0       | ~0       | ~0         | ~0       |
| corrected baseline, **scratch 200ep**     | 34.3     | 4.2      | 46.1       | 34.0     |
| Phase 1 (cost-disp), scratch 200ep        | 12.9     | 1.3      | 16.8       | 12.8     |
| A: sin/cos, pretrained 1000ep             | 51.4     | 10.4     | 62.5       | 51.3     |
| **B: MultiBin, pretrained 1000ep (kept)** | **52.0** | **10.7** | **63.0**   | **51.9** |

## Patterns & Insights

- **The model was never the problem.** 2D detection and metric depth were accurate the whole time; two downstream bugs (eval IoU + dimension target encoding) hid all capability. Lesson: when a metric reads exactly 0, suspect the metric/target pipeline before the model.
- **Encode/decode symmetry is the recurring failure mode.** The dimension bug was an encode/decode mismatch. Phase 2 (MultiBin) was therefore built with encode+decode in one module guarded by a round-trip test.
- **The "biggest lever" hypothesis (stereo cost-volume depth) backfired.** The minimal cost-volume soft-argmax disparity, as primary depth with the monocular head demoted, is _worse_ than the simple lr_distance regression at this budget — likely the soft-argmax is noisier and cost volumes are iteration-hungry. Gating each lever behind a real training run prevented stacking a harmful change.
- **MultiBin helps a little, but orientation wasn't the bottleneck.** B > A on all 6 metrics (Car AP3D@0.5 +0.6, mean +1.5, AOS +0.6) — a real but small, possibly partly within-seed-noise gain; kept because it's consistently non-negative. Crucially, _sin/cos already_ achieves AOS@0.5 (51.3) ≈ AP3D@0.5 (51.4) → heading is near-perfect on TPs regardless of encoding, so there was little headroom for MultiBin to capture.
- **The real ceiling is tight-IoU vertical localization.** At IoU 0.7, AP_BEV (25.9) ≫ AP3D (10.4): the bird's-eye footprint is fine but adding the height/Y dimension collapses the score. So the @0.7 gap is a **height/Y precision** problem (box bottom + height), not orientation or BEV — the highest-value next lever.

## Lessons & Constraints (do-not-repeat)

- Do not trust dimension/orientation priors keyed by name; the dataset must rekey to int class ids (`_rekey_dims_to_int`). `train.py` auto-expand rekey (~L140-145) is also buggy (merges only Aux fallbacks, not ids 0-7) — dataset.py compensates; clean up later.
- Do **not** re-enable Phase 1 cost-volume disparity blindly — it regressed. Revisit only with its followups (multi-scale cost volume, L-R photometric consistency, learnable soft-argmax temperature).
- From-scratch 200ep is too weak for a trustworthy headline (AP3D@0.5 ≈ 34 vs 52 pretrained); use pretrained backbone + ~1000ep + SGD/cos-LR for real numbers.
- Resume + nc auto-expand are incompatible (rebuilds at nc=8 vs 66-class checkpoint) — train fresh, don't rely on `train(resume=True)`.
- Infra: see [[westd-test-env]], [[ultra1-test-env]]. ultra1 is shared — a concurrent `pkill python` from another job will SIGKILL training (run in tmux; even so, pkill-by-name isn't survivable).

## Open Questions / Next

- **Height/Y precision** is the top lever for AP3D@0.7 (BEV@0.7 25.9 vs 3D@0.7 10.4 → vertical extent is the failure). Try better Y/height supervision (e.g. supervise box-bottom directly, or an uncertainty-weighted height loss).
- Per-class Pedestrian/Cyclist far below Car (mean@0.5 16–18 vs Car 51–52) — small/rare classes.
- Confirm the small MultiBin gain with multiple seeds (single-seed +0.6 Car is near noise; +1.5 mean and 6/6 consistency are the support).
- Phase 3 (portability: strip hardcoded KITTI constants) untested.
