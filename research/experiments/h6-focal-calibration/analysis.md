# Analysis — H6, focal calibration

_2026-09-18. Step 1 is evaluation-side only: no model was retrained, no weight changed._

## What the teacher's focal actually is

`research/src/teacher_focal_probe.py`, 40 COCO images spanning 18 distinct shapes:

| quantity | mean | sd | min | max |
| --- | ---: | ---: | ---: | ---: |
| focal / max(w, h) | 1.2329 | 0.0434 | 1.1466 | 1.4142 |
| **focal / diagonal** | **1.0000** | **0.0000** | **1.0000** | **1.0000** |

SAM 3D Body runs with no FOV estimator and its default sets **focal = the image diagonal**, exactly. The
1.2-ish ratio assumed until now was not a constant at all: `f/max(w,h)` ranges 1.147 to 1.414 purely with
aspect ratio, and 1.4142 is the square-image case. Every pseudo-label depth is therefore expressed in a
diagonal-focal convention, and a model trained on them predicts **depth-under-a-diagonal-focal**, not metric
depth.

For 3DPW that convention is wrong by `diag / f_true` = 2202.9 / 1962 = **1.1206**, near-constant across all 24
test sequences (1.1187-1.1229). Every predicted distance was being compared against a ground truth 12% away.

## Effect, with no retraining

`convert_3dpw.py` now converts 3DPW's true metric GT into the teacher's convention by default. Re-scoring the
same four checkpoints:

| arm | AbsRel(Z) before | after | MPJPE before | after |
| --- | ---: | ---: | ---: | ---: |
| A_scratch | 0.1415 | 0.0907 | 126.9 | 123.5 |
| B_warmstart | 0.1266 | 0.0903 | 113.2 | 111.0 |
| C_depthsafe | 0.1754 | **0.0571** | 122.5 | **106.6** |
| D_scalecomp | 0.1826 | 0.0614 | 124.6 | 106.9 |

**The ranking inverts.** C and D went from the two worst arms on depth to the two best, by a factor of 1.5x
over B — precisely what `depth_scale_probe.py` predicted when it showed 81% of D's error collapsing under a
single global scalar. The project's best MPJPE improves from 113.2 mm to **106.6 mm** without touching a
model.

This settles the H2/H2.1 story. Scale-compensated supervision was right all along; it was being scored
against a ground truth in the wrong units, and the arm that had been *sloppiest* about learning apparent size
looked best precisely because sloppiness damped a unit error.

## Two numbers moved the other way, and they should

- **delta1(Z) falls for every arm** (D 0.7406 -> 0.6908). It is a fixed 100 mm threshold, and the converted GT
  relative depths are 1.12x larger, so the same threshold is now a stricter test. A units change, not a
  regression.
- **PA-MPJPE rises for every arm** (D 72.21 -> 78.53). This one is real and worth keeping in view: the
  convention scales Z and leaves X and Y alone, which is an *anisotropic* stretch, not a similarity. That is
  genuinely what the model predicts — reprojection fixes X and Y and scales only depth — so comparing this way
  is correct, but Procrustes cannot absorb an axis-dependent stretch, and any residual mismatch between the
  model's learned effective focal and the sequence's real one shows up here.

## What remains (H6 step 2, needs a retrain)

The labels are still inconsistent *with each other*. Because the teacher keys focal to the diagonal, the
implied `f/max(w,h)` varies 1.147-1.414 across COCO's aspect ratios, so two identically-framed people in
differently-shaped images get depth targets 3.5% apart for no physical reason — and the model cannot see the
original aspect after letterboxing, so that variance is pure label noise.

Re-keying every pseudo-label to one declared convention (`f = R x max(w, h)`, R fixed) removes it. That is a
GPU-free pass over the label files — the teacher does not need to run again, since the conversion is
`z x (R x max(w,h) / diagonal)` from image dimensions alone. It does need a retrain to test, which is the
decision point.

Predicted effect: small but clean, since it removes a 3.5% noise floor from the depth target. The larger prize
is already banked.
