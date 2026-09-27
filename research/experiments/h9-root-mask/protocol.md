# Protocol — H9, masking the root when it leaves the frame

_Locked 2026-09-27, before the run. One arm, one change from `F_clip`: one column of the labels._

## Hypothesis

H8 found the cost of the recovered images: 3DPW root-depth AbsRel rose **0.0398 -> 0.0445 -> 0.0456** across
`E_refocal`, `G_stepmatch` and `F_clip` — monotone in the data and unmoved by doubling the schedule, so it is
the data and not the optimization. The mechanism was visible in what the recovered images are: people truncated
at the frame edge, whose mid-hip root is outside the picture. **10.4% of COCO persons** have such a root, and
every one of them was supervising an absolute depth whose evidence is not in the image, at a 2D position
clipped onto the border.

**H9: marking those roots unlabelled recovers the root-depth accuracy without giving back the 2D that H8
bought.**

## The change

One channel. `pseudo_label_sam3d.py` now gives the root the same in-frame test every other joint already gets,
and `research/src/mask_root_vis.py` applies it to the labels already on disk — an out-of-frame root was clipped
to exactly 0 or 1, so the clip marker recovers the flag without running the teacher again.

**Nothing in the training code changes.** `Pose3DLoss.calculate_keypoints_loss` already masks the root-depth
term, the root's 2D location and its objectness target off `keypoints[..., 2] != 0`; the root simply never had
a 0 to mask on. Verified against `coco-pose3d-refocal`: exactly one column differs (75, the root visibility),
15,638 rows in train and 676 in val, every depth byte-identical.

## Arm

`H_rootmask` — `F_clip` config exactly (warm start from `yolo26n-pose`, 100 epochs, batch 64, imgsz 640,
seed 0), on `coco-pose3d-rootmask.yaml`.

## Predictions

- **3DPW AbsRel(Z) below 0.0430**, against F's 0.0456, moving back toward E's 0.0398. At or above 0.0450 is a
  null and says the frame-edge roots were not what raised it.
- **delta1(Z) within 0.01 of F's 0.6568.** Relative-depth supervision is untouched.
- **MPJPE within 2 mm of F's 109.8.** It is root-relative and should not care.
- **COCO GT 2D within 0.005 of F's 0.4732 pose and unchanged at 0.6891 box.** The root is 1 of 18 joints in the
  pose loss and is masked for 10% of persons, so a small 2D cost is possible; a large one would mean the root
  was carrying more 2D signal than it should.

## What this experiment cannot see

**3DPW test has no truncated people**: 3 of 7,097 persons have a root at the border, 0.0%. So the benchmark
measures only the benefit of the mask — cleaner supervision for framed people — and is structurally blind to
its cost, which is that a truncated person now has no root-depth supervision at all and whatever the model
emits for them is unconstrained.

That asymmetry also reframes H8: training on frame-edge roots hurt a benchmark composed **entirely** of framed
roots. The regression it measured may always have been a distribution mismatch rather than label noise.

To see the cost, both arms are also scored on `coco-pose3d-refocal.yaml` val, whose labels keep every root
labelled including the truncated 10.6%. If AbsRel improves on 3DPW and degrades there, the mask is a trade
rather than a fix, and the honest recommendation becomes a down-weight rather than a mask.
