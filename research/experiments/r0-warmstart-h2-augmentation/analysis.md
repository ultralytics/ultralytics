# Analysis — R0 and H2

_2026-09-17. Three arms, 100 epochs each, ~1.45 h per arm on one RTX 6000D (ultra15 GPUs 0/1/3)._

## Results

**3DPW test — real ground truth.** 4,907 frames, 7,097 persons, 12 joints, per-sequence intrinsics.

| arm | box mAP50-95 | pose mAP50-95 | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_scratch | 0.4945 | 0.6994 | 126.9 | 79.54 | 0.1415 | 0.7001 |
| B_warmstart | 0.5012 | 0.7744 | **113.2** | 72.82 | **0.1266** | 0.7320 |
| C_depthsafe | 0.4986 | 0.7757 | 122.5 | **72.48** | 0.1754 | **0.7487** |

**COCO val2017 — teacher agreement**, the fast inner-loop metric (not accuracy).

| arm | box mAP50-95 | pose mAP50-95 | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_scratch | 0.6702 | 0.4971 | 138.4 | 80.75 | 0.0760 | 0.6823 |
| B_warmstart | 0.7035 | 0.5869 | 123.1 | 72.73 | 0.0728 | 0.7280 |
| C_depthsafe | 0.6836 | 0.5671 | **119.0** | **72.13** | **0.0552** | **0.7363** |

## R0 — CONFIRMED

Warm-starting from `yolo26n-pose` beats scratch on **every metric on both datasets**: 3DPW MPJPE 126.9 -> 113.2
(-13.7 mm, -11%), PA-MPJPE 79.5 -> 72.8, pose mAP +0.075, AbsRel(Z) -0.015. This is the s3d finding again in a
different task: the cheapest large win available is a pretrained trunk, and 720 of 792 tensors transfer even
though the keypoint head cannot.

## H2 — SUPPORTED IN-DOMAIN, REFUTED OUT-OF-DOMAIN

This is the result worth having, and it is not the one predicted.

Disabling mosaic and scale did exactly what the mechanism said it would **on the distribution it was trained
on**: AbsRel(Z) 0.0728 -> 0.0552, a **24% relative cut in root-depth error** against the teacher. On real 3DPW
ground truth the same change moves AbsRel(Z) 0.1266 -> **0.1754, 39% worse**.

Both are real, and together they say something the single-dataset view could not:

- **The depth-target corruption is real.** C beats B on every *relative* depth measure, on both datasets:
  delta1(Z) 0.732 -> 0.749 on 3DPW, PA-MPJPE 72.82 -> 72.48. Scale augmentation genuinely does teach the
  network a wrong relationship between apparent size and depth.
- **Scale augmentation is also what teaches scale robustness.** Absolute root depth is precisely the quantity
  that depends on generalizing apparent-size statistics to an unseen camera and unseen subject distances.
  Removing the augmentation lets the model fit COCO's size distribution tightly, and 3DPW's is different.

So `mosaic=0, scale=0` is the wrong fix. It trades cross-domain robustness for in-domain target fidelity, and
the honest reading of H2 is: **the target is corrupt, but deleting the augmentation is not how to repair it.**

## What this implies next (H2.1)

Keep the augmentation and **compensate the label**: when a sample is scaled by `s`, an object's apparent size
changes exactly as if its depth were `z/s`, so the root-depth target should become `z_root / s`. That keeps the
regularizer and the scale-robustness it buys while removing the false supervision. The same argument applies to
the per-image letterbox ratio, which is the confound recorded in the protocol before any of these runs.

Predicted outcome: recovers C's relative-depth gain **and** B's out-of-domain AbsRel, i.e. beats both on 3DPW.

## Bug found by running the real benchmark

The first 3DPW pass reported **PA-MPJPE ~620 mm against MPJPE ~120 mm**, which is impossible: Procrustes
alignment can only reduce error. `procrustes_align` was fitting the similarity transform over all 17 joints,
including the five face joints 3DPW cannot annotate. Those are written as zeros and decode to a point ~2 m in
front of the camera and off-axis, so the fit was being dragged by five phantom joints per person. Fixed by
making the alignment visibility-weighted.

It never showed up on the COCO pseudo-labels, where all 17 joints are populated. That is the argument for the
real benchmark in one line: it is not only a harder test of the model, it is a test of the evaluator.
