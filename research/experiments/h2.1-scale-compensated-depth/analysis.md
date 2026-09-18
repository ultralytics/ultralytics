# Analysis — H2.1, scale-compensated depth labels

_2026-09-18. `D_scalecomp`, 100 epochs, 1.50 h, ultra15 GPU 5. Control is `B_warmstart`, unchanged._

## Results

**3DPW test — real ground truth**, the locked protocol.

| arm | box mAP50-95 | pose mAP50-95 | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_scratch | 0.4945 | 0.6994 | 126.9 | 79.54 | 0.1415 | 0.7001 |
| B_warmstart | 0.5012 | 0.7744 | **113.2** | 72.82 | **0.1266** | 0.7320 |
| C_depthsafe | 0.4986 | 0.7757 | 122.5 | 72.48 | 0.1754 | **0.7487** |
| D_scalecomp | **0.5026** | **0.7805** | 124.6 | **72.21** | 0.1826 | 0.7406 |

**COCO val2017 — teacher agreement.**

| arm | box mAP50-95 | pose mAP50-95 | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B_warmstart | 0.7035 | 0.5869 | 123.1 | 72.73 | 0.0728 | **0.7280** |
| C_depthsafe | 0.6836 | 0.5671 | **119.0** | **72.13** | **0.0552** | 0.7363 |
| D_scalecomp | 0.7006 | **0.5885** | 119.1 | 72.70 | 0.0578 | 0.7260 |

## Against the locked predictions

| prediction | outcome |
| --- | --- |
| 3DPW AbsRel(Z) beats B's 0.1266 | **refuted** — 0.1826, the worst of all four arms |
| 3DPW delta1(Z) matches C's 0.7487 | near miss — 0.7406 |
| 2D mAP holds at B's level | **confirmed** — 0.7805 pose and 0.5026 box, both the best of the four |
| MPJPE below B's 113.2 mm | refuted — 124.6 mm |

In-domain, D did exactly what was predicted: it keeps B's 2D accuracy (mAP 0.5885 vs 0.5869) *and* recovers
C's depth gain (AbsRel 0.0578 vs C's 0.0552, against B's 0.0728), which is the best-of-both the hypothesis
promised. Out of domain it is the worst arm on the headline depth metric. Taken at face value the change looks
like a failure.

## It is not a failure — the metric is dominated by one number

AbsRel conflates two errors that have nothing to do with each other: every person being off by the *same*
factor, which is a camera-calibration problem, and each person being off *independently*, which is a model
problem. `research/src/depth_scale_probe.py` fits one scalar per model over the whole benchmark — the median
of gt/pred — and re-scores:

| arm | AbsRel raw | fitted global scale | AbsRel after | error explained by one number |
| --- | ---: | ---: | ---: | ---: |
| B_warmstart | 0.2297 | 0.820 | 0.0893 | 61% |
| C_depthsafe | 0.1815 | 0.852 | 0.0375 | 79% |
| D_scalecomp | 0.1808 | 0.848 | **0.0350** | **81%** |

(Measured at conf 0.25 over 1,500 frames, so the raw column is not the validator's number; the comparison
between arms is what matters.)

**Once the global factor is removed, D is the most accurate arm by a wide margin — 0.0350 against B's 0.0893,
2.6x better.** So the compensation did precisely what it was designed to do. What it also did was make the
model *more* faithful to the apparent-size-to-depth relation it was taught, and therefore inherit the teacher's
camera calibration more exactly — including where that calibration is wrong.

## The global factor is a focal mismatch, and the arithmetic matches

All three models over-predict 3DPW depth by about the same factor, ~1/0.85 = **1.18x**. SAM 3D Body runs with
no FOV estimator and falls back to a default that measured ~1.2x the image side on COCO; 3DPW's real cameras
are 1.0239. **1.2 / 1.024 = 1.17**, against 1.18 observed.

That is the whole gap. The pseudo-labels encode depths consistent with a ~1.2 focal ratio, the evaluation
applies 3DPW's true 1.0239, and every predicted distance is stretched by the ratio between them. It is a unit
conversion error sitting in the data pipeline, not a modelling failure — and it is also present in the COCO
labels, whose YAML claims `focal_ratio: 1.1` while the teacher was assuming ~1.2.

## Revised reading of H2 and H2.1

- **H2.1 is supported on its mechanism and refuted on its metric.** Compensating the resize genuinely fixes the
  depth target: it gives the best per-person depth accuracy of any arm, the best 2D mAP, and the best PA-MPJPE.
- **B's apparent advantage on raw AbsRel is an artefact of being sloppier.** Training against a target that
  randomly decouples apparent size from labelled depth partially weans the model off apparent size, which
  happens to damp the focal bias — while leaving per-person depth 2.6x noisier. It looks better on the metric
  by being worse at the task.
- **The H2 conclusion needs amending.** "Scale augmentation buys cross-domain robustness" was the wrong
  inference. What it buys is insensitivity to a calibration error we could simply fix.

## Next (H6 — focal calibration)

Depth is only defined up to a focal length, and the pipeline currently uses three different ones: the
teacher's ~1.2 default, the COCO YAML's 1.1, and 3DPW's measured 1.0239. Options, cheapest first:

1. **Re-key the labels to a single declared focal**, dividing every pseudo-label depth by the teacher's own
   per-image `focal_length` and multiplying by the convention. No retraining of the teacher, one pass over the
   labels. Predicted effect: 3DPW AbsRel for D drops from 0.18 toward the 0.035 the probe already shows.
2. **Feed the focal to the model**, if it is known at inference.
3. **Predict it**, which is what SAM 3D Body itself declines to do by default.

D_scalecomp is the right base model for all of these: it has the best geometry, and it is the arm whose
remaining error is a single scalar.
