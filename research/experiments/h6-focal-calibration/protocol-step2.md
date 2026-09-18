# Protocol — H6 step 2, re-keyed labels

_Locked 2026-09-18, after the relabel and before the run finishes. One arm, one change from `D_scalecomp`._

## Hypothesis

Step 1 showed the teacher keys focal to the image diagonal, exactly. That is self-consistent within an image
but not across them: `diagonal / max(w, h)` runs 1.147-1.414 with aspect ratio alone, and after letterboxing
the network cannot see the original aspect. So identical-looking people carry depth targets up to 39% apart
for a reason the network has no way to observe.

**H6.2: removing that unobservable variance from the target improves depth accuracy, by a small but clean
margin, because it lowers a noise floor rather than changing what is learned.**

## Arm

`E_refocal` — identical to `D_scalecomp` except the dataset: `coco-pose3d-refocal.yaml`, whose depths were
rescaled by `1.2 * max(w, h) / diagonal` per image. Measured over the relabel: factor mean 0.9757, range
**0.8485 to 1.1823** — that range is the noise being removed. No value hit an encoding bound. 56,599 images,
149,813 persons, both splits.

Scored against `3dpw-pose3d-r12.yaml`, whose GT is converted into the same declared convention
(factor diag/f_true -> 1.2*max/f_true = 1.1720, near-constant at 1.1700-1.1744).

## Predictions

- **3DPW AbsRel(Z) modestly better than D's 0.0614** — the step-1 correction already banked the systematic
  part; what is left here is variance. A large gain would be surprising and would mean the aspect noise was
  doing more damage than a 39% spread on a minority of images should.
- **MPJPE at or slightly below D's 106.9 mm.**
- **2D mAP unchanged** — nothing about the images or the 2D targets moved.
- **No change on teacher agreement against the re-keyed val set**, since both target and metric shifted
  together; the COCO numbers are not comparable across the two label sets and should not be read as progress.

A null result here is a perfectly good outcome and would say the aspect variance was never the binding
constraint — which, given the step-1 result, is the honest prior.
