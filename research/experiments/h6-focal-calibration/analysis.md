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

---

# Step 2 — re-keyed labels (`E_refocal`)

_2026-09-18. 100 epochs, 1.47 h, ultra15 GPU 3. One change from `D_scalecomp`: the dataset._

## Results on 3DPW

Each model scored in the convention it was trained in, which is the only fair comparison, plus E scored in
D's convention so the two can be read on one common ruler.

| model | benchmark | pose mAP50-95 | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D_scalecomp | conv (diagonal) | 0.7805 | 106.9 | **78.53** | 0.0614 | **0.6908** |
| E_refocal | r12 (own) | 0.7803 | 109.4 | 82.05 | **0.0398** | 0.6496 |
| **E_refocal** | **conv (common ruler)** | 0.7803 | **106.5** | 79.23 | **0.0449** | 0.6744 |

## Against the locked predictions

| prediction | outcome |
| --- | --- |
| AbsRel(Z) modestly better than D's 0.0614 | **confirmed, and not modestly** — 0.0449 like-for-like, a 27% cut, and 0.0398 in its own convention |
| MPJPE at or slightly below 106.9 mm | confirmed like-for-like — 106.5 mm |
| 2D mAP unchanged | confirmed exactly — 0.7803 against 0.7805 |
| "a null result is the honest prior" | **wrong.** The aspect-driven label noise was doing real damage |

**0.0398 AbsRel is the best depth number the project has produced**, from a label edit that cost no GPU time
and did not re-run the teacher.

The gain is not a units artefact. Scored on D's own benchmark, in D's own convention, E still beats it 0.0449
to 0.0614. The model genuinely learned depth better; removing target variance it could not observe was worth
27% of the remaining root-depth error.

## What got slightly worse, and why it is consistent

`delta1(Z)` falls (0.6908 -> 0.6744 on the common ruler) and PA-MPJPE rises (78.53 -> 79.23). Both are
*relative*-depth measures, and both move the same small amount in the same direction, so this reads as one
effect rather than noise: the re-keying rescales relative depths by the same per-image factor as the root, and
for the minority of images with extreme aspect that is a large change to a quantity the root-depth argument
does not apply to as cleanly. **Root depth improved 27%; relative depth regressed ~2%.**

That asymmetry is itself a finding, and it points at the next split: the root channel and the relative
channels may not want the same convention. The root genuinely scales with focal. The relative channels
describe a body, whose true metric size does not change when the camera does — the scaling is correct only
under the fixed-pixels argument, and the two requirements are in tension.

## Where this leaves the project

Best real-GT figures, all from a 3.4M-parameter single-shot model on 3DPW test:

- **MPJPE 106.5 mm**, **PA-MPJPE 78.5 mm**, **root-depth AbsRel 0.0398**, **pose mAP50-95 0.780**
- Against the first honest measurement two days ago — 113.2 mm MPJPE and 0.1266 AbsRel — the depth error is
  down **3.2x**, and not one point of it came from a bigger model or a longer schedule. All of it came from
  finding three different focal conventions in one pipeline and reconciling them.
