# Findings — YOLOPose-3D

_Can SAM 3D Body's capability be compressed into a single YOLO forward pass? Repo `~/ultralytics_pose3d`,
branch `pose3d`. Last updated 2026-09-16. Status: **task wired and smoke-verified; no scientific result yet.**_

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

## Patterns and Insights

_(empty — no experiments yet)_

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
3. How much does the teacher's own bias cap the student — i.e. how far apart are teacher-agreement MPJPE and
   real-GT MPJPE?
4. Does the flat-in-person-count claim actually hold end to end, including NMS and postprocess?

## Optimization Trajectory

_(empty — no runs yet)_
