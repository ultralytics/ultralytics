# Findings — YOLOPose-3D

_Can SAM 3D Body's capability be compressed into a single YOLO forward pass? Repo `~/ultralytics_pose3d`,
branch `pose3d`. Last updated 2026-09-17. Status: **106.5 mm MPJPE and 0.0398 root-depth AbsRel on 3DPW
from a 3.4M-parameter single-shot model — depth error down 3.2x in two days, none of it from a bigger model.**_

## Research Question

SAM 3D Body recovers a metric 3D human mesh from one image, one person at a time, from a crop — about 3 s per
person on our hardware. YOLO finds every person in one pass and exports to edge runtimes. **How much of the
teacher's 3D capability survives compression into one YOLO head, and which part breaks first — root depth, or
root-relative pose?**

## Current Understanding

Nothing measured yet. What is established is the shape of the gap and the shape of the answer:

- The gap is **detector-shaped, not speed-shaped**. `Fast SAM 3D Body` already made the teacher 10.9× faster
  training-free, and it is still a crop model whose cost scales with the number of people. What nobody has built
  is a single-shot, anchor-free, edge-exportable head — the thing YOLO already is.
- The answer has **two independent halves**: root-relative pose (a shape problem, scale-invariant, what
  PA-MPJPE measures) and absolute root depth (a scale problem, destroyed by resize augmentation, what AbsRel
  measures). The task is designed to report them separately because they are likely to fail separately.

## Key Results

**H0 — the plumbing works (2026-09-16).** A `pose3d` task exists end to end in the repo: `Pose3D` head,
`Pose3DLoss`, `Pose3DModel`, trainer/validator/predictor, model and dataset YAMLs, task registrations. On the
synthetic `coco8-pose3d` smoke set (4+4 images, CPU, ultra1): 2 epochs train with all seven loss terms
(`box, pose, kobj, zrel, zroot, cls, l1`), validation prints `MPJPE / PA-MPJPE / AbsRel(Z) / d1(Z)`, the
checkpoint reloads as `task=pose3d` and dispatches to `Pose3DPredictor`, and ONNX export succeeds with output
`(1, 77, 2100)` = 4 box + 1 class + 18×4 keypoints.

No accuracy claim attaches to any of this: the smoke set's depth channel is synthetic (relative depth all zero,
root depth from a box-height heuristic), so its `d1(Z)=0.888` is an artefact of an untrained sigmoid sitting at
0.5, which decodes to exactly the 0 m the synthetic labels contain.

**The teacher is good enough to distil from (2026-09-16).** On COCO val2017, SAM 3D Body's *2D reprojection*
agrees with human annotation at **OKS mean 0.805 / median 0.893** (90.6% of 6,352 persons above 0.5, 79.6% above
0.75). That is a necessary condition, not a sufficient one — a reconstruction that reprojects correctly can still
be wrong in depth — but the failure mode where the teacher simply does not find the person is ruled out.

The depth distributions are well inside the chosen encoding: root depth median 5.00 m (p95 13.54, max 23.33
against a 50 m cap), relative depth p1 −0.53 / p99 +0.40 m (absmax 0.95 against a ±2 m range), and **not one
value in 120k landed on an encoding bound**. The relative range is about twice as wide as the data needs, which
costs nothing in float32 but would matter if the channel were ever quantized.

## Patterns and Insights

**A pretrained trunk is the cheapest large win, again.** R0 repeats the s3d result in a new task: warm-starting
from `yolo26n-pose` beats scratch on every metric on both datasets (3DPW MPJPE 126.9 -> 113.2 mm), even though
only 720 of 792 tensors transfer and the keypoint head cannot.

**The two halves of the task really do fail separately, and they disagree about augmentation.** Disabling
mosaic and scale improves every *relative* depth measure on both datasets (3DPW delta1(Z) 0.732 -> 0.749) and
cuts root-depth error 24% against the teacher — then makes root depth 39% *worse* on real 3DPW. Scale
augmentation corrupts the metric depth target and simultaneously teaches the scale robustness that absolute
depth depends on. The design's split of root depth from relative depth is what made this visible at all; a
single MPJPE number would have shown C as simply worse and hidden the mechanism.

**Metric depth is meaningless without the focal that defines it, and ours was never declared.** The teacher
keys focal to the image diagonal, exactly; the COCO YAML claimed 1.1x the long side; 3DPW's real cameras are
1.0239. Three conventions in one pipeline. Correcting only the evaluation cut 3DPW AbsRel ~3x for the
best arms and inverted the ranking of the whole H2 family — the arm that looked best on depth was the one
that had learned apparent size least well, so it inherited the unit error least. **Any absolute-depth number
should now be treated as untrustworthy until the convention it is expressed in is written down.**

**Teacher agreement and accuracy can point in opposite directions.** On the pseudo-labels C looks like a clean
win; on real ground truth it is a regression on the headline depth metric. The inner-loop metric is a cheap
proxy and nothing more — no direction should be committed to on it alone.

## Lessons and Constraints

Recorded from reading the code and from the s3d project's history, before they cost anything:

- **Keypoint channel 2 must stay visibility.** The augmentation pipeline reads `keypoints[..., 2]` as visibility
  in several places; putting depth there silently corrupts flips and copy-paste. Depth goes at index 3.
- **`RandomPerspective.apply_keypoints` drops channels past 2.** It rebuilds the array as
  `concatenate([xy, visible])`. It is the only augmentation function that needed patching.
- **`verify_image_label` asserts `lb.min() >= -0.01` over the whole row**, so signed metric depth cannot be
  written to disk. Both depth channels are stored encoded to [0, 1] (`ultralytics/utils/pose3d.py`).
- **`calculate_keypoints_loss` keys its visibility mask off `shape[-1] == 3`.** With a 4-wide keypoint that
  condition is False and the mask silently becomes all-True, supervising invisible joints against zeros. It is
  recomputed explicitly in `Pose3DLoss`. This one fails silently — no crash, just worse training.
- **`PoseTrainer.__init__` hard-sets `overrides["task"] = "pose"`**, so a subclass must restore its own task
  after `super().__init__` or checkpoints record the wrong task and reload with the wrong predictor.
- **The plotting stack also keys off `ndim == 3`.** `Keypoints.has_visible`, `Annotator.kpts`'s confidence
  filter and its skeleton branch all tested for exactly 3 channels, so with `plots=True` (the default) a 4-wide
  keypoint silently lost visibility filtering and drew no skeleton. Nothing crashed — which is why it had to be
  looked for rather than waited for.
- **MPJPE cannot be the fitness metric.** The framework maximizes fitness and MPJPE is an error; `delta1(Z)` is
  the higher-is-better form, mirroring the depth task.
- **From s3d (`~/ultralytics_3d_foundation/research/findings.md`): direct regression beat geometric decoding.**
  Cost-volume soft-argmax depth was 2.7× *worse* than regressing distance directly. Prefer regression here too,
  and treat any "decode the geometry properly" idea as a hypothesis with a bad prior.
- **From s3d: a pretrained backbone plus a long schedule was worth more than any architectural lever tested**
  (34.3 → 52.0 AP3D). Warm-start first, architect second.

## Open Questions

1. Does the student's error concentrate in root depth (scale) or in relative pose (shape)?
2. Does scale-changing augmentation have to be disabled for metric depth, or can the target be compensated?
3. Mosaic scaling is visibly the H2 hazard: the dataloader check shows the depth channel passing through the
   affine untouched, so a person scaled to half size keeps the same metric depth label. Compensate, or disable?
4. How much does the teacher's own bias cap the student — i.e. how far apart are teacher-agreement MPJPE and
   real-GT MPJPE?
5. Does the flat-in-person-count claim actually hold end to end, including NMS and postprocess?

## Optimization Trajectory

_(empty — no runs yet)_
