# YOLOPose-3D — task design

_Written before any code, 2026-09-16. Repo `~/ultralytics_pose3d`, branch `pose3d`, forked from
upstream `main` @ `fa39b69cb` (8.4.153), which already carries both `pose` and `depth`._

## The question

SAM 3D Body (3DB) recovers a full-body metric 3D mesh from a single image, one person at a time,
from a crop, at roughly 3 s/person on our hardware. YOLO detects every person in one forward pass at
video rate and exports to ONNX/TensorRT/CoreML. **How much of 3DB's 3D capability survives
compression into a single YOLO head, and which part breaks first?**

This is not "accelerate 3DB" — `Fast SAM 3D Body` (arXiv:2603.15603) already did that, training-free,
10.9×, and it is still a crop model whose cost scales with the number of people. The gap nobody has
taken is the *detector-native* one: one shot, all people, one head, edge-exportable.

## Output contract

Per detected person, in camera space:

- 17 COCO keypoints in 2D pixels (what `pose` already gives),
- per-keypoint **root-relative depth** `Δz` in metres,
- the person's **root** (mid-hip): its 2D projection and its **absolute metric depth** `z_root`.

`(x, y) + Δz + z_root` plus a focal length lifts every joint to metric camera coordinates. Mesh
vertices and MHR parameters are explicitly **out of scope for v1** — see "Rejected for v1".

## Wiring decision: `kpt_shape = [18, 4]`

The whole task rides on the existing keypoint plumbing by widening the keypoint tensor and adding
one joint, rather than by adding a parallel head:

| | index | x | y | 2 | 3 |
|---|---|---|---|---|---|
| joints 0–16 | COCO-17 | px | px | visibility | `Δz`, root-relative metres |
| joint 17 | **root** (mid-hip) | px | px | visibility (always 1 if person labelled) | **`z_root`, absolute metres** |

Two deliberate choices here:

**Channel 2 stays visibility.** The obvious layout is `(x, y, z, vis)`, and it is wrong: the
augmentation pipeline reads `keypoints[..., 2]` as visibility in several places
(`RandomPerspective.apply_keypoints`, CopyPaste, the flip path). Putting `z` at index 3 means the
entire dataloader — `Instances.scale/normalize/denormalize/add_padding`, `RandomFlip`, `Format` —
works untouched, because all of it only ever indexes 0 and 1. The single exception is
`apply_keypoints`, which rebuilds the array as `concatenate([xy, visible])` and so *drops* any
channel past 2; that one function needs a patch to carry the tail through.

**Root depth is a joint, not a second head.** Storing `z_root` in the z-channel of an 18th joint
keeps the on-disk label format exactly what `verify_image_label` already validates
(`5 + nkpt*ndim` columns), needs no extra conv, no extra loss term wiring, and no new collate key —
and it supervises the root's 2D projection for free, which is what you need to lift the pose into
camera space anyway.

Decoupling `z_root` from `Δz` (rather than regressing absolute per-joint depth) is a bet, and it is
the bet the s3d project's history supports: s3d's `lr_distance` **direct regression** of object
distance beat cost-volume geometric decoding by 2.7× AP3D (hypothesis H1, refuted, in
`~/ultralytics_3d_foundation/research/findings.md`). Absolute per-joint depth would force one
regressor to cover both a ~1–20 m range and a ±0.5 m range. Whether that actually matters is H3.

## What changes, file by file

| File | Change |
|---|---|
| `ultralytics/nn/modules/head.py` | `Pose3D(Pose26)` — `kpts_decode` sigmoids ch 2, passes ch 3 through as metres |
| `ultralytics/utils/loss.py` | `Pose3DLoss` — existing OKS loss on xy, BCE on vis, masked L1 on `Δz`, SILog on `z_root` |
| `ultralytics/nn/tasks.py` | `Pose3DModel` |
| `ultralytics/models/yolo/pose3d/` | `train.py`, `val.py`, `predict.py`, `__init__.py` |
| `ultralytics/data/dataset.py:188` | allow `ndim == 4` |
| `ultralytics/data/augment.py:1291` | `apply_keypoints` must carry channels ≥3 through the affine |
| `ultralytics/cfg/__init__.py` | register task in `TASKS`, `TASK2DATA`, `TASK2MODEL`, `TASK2METRIC` |
| `ultralytics/cfg/models/26/yolo26-pose3d.yaml` | `yolo26-pose.yaml` with `kpt_shape: [18, 4]` and the `Pose3D` head |
| `ultralytics/cfg/datasets/coco8-pose3d.yaml` | 8-image smoke dataset, 18 kpt names, 18-long `flip_idx` |

## Metrics

Primary, and locked before any run:

- **MPJPE** (mm), root-relative, over the 17 COCO joints, on matched detections.
- **PA-MPJPE** (mm), Procrustes-aligned — separates pose shape error from global-orientation error.
- **root AbsRel** on `z_root`, plus `δ<1.25`, borrowed from the `depth` task's metric set.
- Detection is not free: report **PCK-matched recall** so a model cannot win MPJPE by only keeping
  the easy people.
- **Latency**: ms/image at 640 on one RTX PRO 6000, and ms/**person**, because the whole claim is
  that ours is flat in person count and the teacher's is linear.

Inner-loop (fast, minutes): **teacher agreement** — MPJPE against 3DB pseudo-labels on a held-out
split. Outer-loop (slow, honest): a real-GT benchmark. Neither 3DPW nor AGORA/EMDB/BEDLAM/H36M is on
disk; acquiring one is a bootstrap task and the numbers above are not trustworthy until it lands.

## Teacher / pseudo-label pipeline

`~/snowboard/snowpose/model.py` already wraps `SAM3DBodyEstimator` and is reused verbatim as the
label engine. MHR70 → COCO-17 is a **complete, exact** mapping (from `~/snowboard/snowpose/mhr.py`):

```
nose 0 · eyes 1,2 · ears 3,4 · shoulders 5,6 · elbows 7,8
wrists 62(L) 41(R) · hips 9,10 · knees 11,12 · ankles 13,14
root = midpoint(hip 9, hip 10)
```

Every COCO joint exists in MHR70 — no interpolation, no proxy joints. (MHR's feet, 15–20, are a
free extension for a later hypothesis.)

Source images: COCO train2017 person crops (boxes already labelled, so the detector supervision
stays real GT and only the 3D is pseudo). Teacher runs on an ultra host — 3DB checkpoints are
HF-gated and are **not** currently on ultra1–8 or ultra9/10/13; provisioning one is a bootstrap task.

## Rejected for v1

- **MHR/SMPL parameter regression + mesh.** It is what 3DB actually outputs, and it is the right
  long-term target, but it drags the MHR rig — a gated, separately-licensed asset — into the
  inference path of an AGPL repo, and it needs a differentiable rig forward pass in the loss. Keep
  as H4, behind the keypoint result.
- **Multi-view or temporal input.** Single image, single frame; anything else changes the task
  contract.

## Known hazards, recorded now so they are not rediscovered

1. **Scale augmentation breaks metric depth.** Mosaic/`scale`/random-resize change apparent object
   size, which is the main monocular depth cue; the `z_root` target is then a lie. The `depth` task
   hit this. Either disable scale-changing augs for the depth term or rescale `z_root` by the same
   factor. Untested — this is H2.
2. **Teacher bias becomes our ceiling.** arXiv:2601.06035 audits 3DB's anthropometric fidelity; our
   student cannot be better than its labels, so a real-GT benchmark is not optional.
3. **`Δz` in metres vs body-normalized.** Children and adults differ; a body-scale-normalized `Δz`
   may train better and is trivially convertible. Untested.
