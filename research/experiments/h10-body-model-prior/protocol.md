# Protocol — H10, does a parametric body model (as in SAM 3D Body) buy accuracy?

_Locked 2026-10-22, before any analysis ran. Task ft-mnvw._

## Question

SAM 3D Body regresses MHR rig parameters and reads joints off the posed mesh; our student regresses 18 joints ×
(x, y, vis, z) directly. Would putting a human body model into the student — as output representation (H4) or
as a train-time prior — improve its 3D accuracy?

A body model can only fix what it constrains: **bone lengths, left/right symmetry, proportions** (and, with the
full rig, joint limits). It does not see pixels, so it cannot fix 2D keypoint error, and it constrains root depth
only through metric body size. So the question reduces to: **how much of F_clip's error is structural
implausibility that a body model would remove?**

## Subject

`F_clip/weights/best.pt` (best 2D arm; 3DPW MPJPE 109.8 / PA 82.96 / AbsRel 0.04556). Every IoU-matched pair the
validator scores is dumped by `research/src/dump_matched_pairs.py`, which reproduces the validator to the printed
digit (checked: 109.85 / 82.96 / 0.04556 on 3DPW, 110.01 / 63.89 / 0.0619 on COCO val). 3DPW: 7,003 pairs, 12 body
joints with depth (face joints are vis 0). COCO val (teacher labels): 6,286 pairs, 17 joints.

## Measurements

1. **Decomposition.** MPJPE; MPJPE after a per-person scalar scale fit; PA-MPJPE. Swap in GT depth (keep
   predicted 2D) and GT 2D (keep predicted depth) — which residual dominates.
2. **Plausibility of the predictions.** Per-bone length error vs GT, L/R limb asymmetry, both against the same
   statistics of the GT itself.
3. **Skeleton refit under bone-length constraints** (Adam over 3D joints: stay close to the prediction in 2D
   and depth, soft-enforce bone lengths, pinned root):
   - **R-oracle**: the GT person's own bone lengths — upper bound for a body model that also knew the shape.
   - **R-mean**: population-mean bone lengths (from the training-set teacher labels... on 3DPW from the GT mean,
     which flatters it) — what a shape prior buys without per-person shape.
   - **R-sym**: the prediction's own lengths, L/R averaged — the weakest prior.
4. **Anthropometric root depth**: depth from metric torso+leg length over its pixel length, with GT and mean
   lengths, against F_clip's learned root depth.
5. **Teacher ceiling**: SAM 3D Body itself on 3DPW test from GT boxes, depth re-keyed from its diagonal focal to
   `1.2·max(w,h)` (the H6 trap), scored with the same lifting code.

## Locked predictions

| # | prediction |
| --- | --- |
| P1 | Depth-driven residual > 2D-driven residual on 3DPW. Scale fit removes < 10 mm. |
| P2 | Predicted bone lengths within 10% of GT on median; L/R asymmetry within 2x of GT's own. |
| P3 | R-oracle gains 3–8 mm MPJPE on 3DPW; R-mean gains 0–3 mm; R-sym ≈ 0. |
| P4 | Anthropometric root depth with mean lengths is worse than F_clip's 0.0456 AbsRel. |
| P5 | Teacher on 3DPW: MPJPE 80–95 mm. |

## Decision rule

- **R-mean ≥ 3 mm** on 3DPW → a body-model prior is worth a training experiment (bone/symmetry loss fine-tune).
- **R-oracle ≥ 10 mm but R-mean < 3 mm** → value exists only with per-person shape estimation, i.e. real
  MHR/SMPL regression (H4); escalate as a cost question.
- **Both small** → no body model for accuracy; the error is in 2D keypoints and depth regression.
