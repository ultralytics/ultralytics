# Protocol — H11, does a YOLO-depth checkpoint make a better pose3d warm start?

_2026-10-23, task ft-syjg. Stage 0 (inventory + compatibility + frozen-feature probe) is done and reported below;
Stage 1 is the training A/B, not yet launched._

## Question

pose3d warm-starts from `yolo26n-pose` (2D pose, COCO). Its weakest channel is depth: per-joint relative depth is
55% of 3DPW MPJPE (H10). A YOLO monocular-depth model was trained on 2.19M depth images. **Does starting from its
weights improve the depth half of pose3d — and does it cost the 2D half?**

## Stage 0a — which YOLO-depth weights exist, and what transfers into `yolo26n-pose3d`

All in `ultra15:/data/shared-datasets/depth_anything/weights_all/` (README there). Transfer = tensors whose name and
shape match `yolo26n-pose3d` (`research/src/` dump + diff, `inspect_ckpt2.py` / `diff_sd.py` in `~/h10`).

| checkpoint | what it is | NYU δ1 | backbone 0-10 | neck 11-22 | head |
| --- | --- | ---: | --- | --- | --- |
| `yolo26n-sni-bbp3e5-depth-CONTROL-scratch.pt` | **shipped YOLO27n-depth**, 5.4M | 0.831 | 224/240 tensors, **98%** of params (layer 4 widened) | **0%** (different neck) | 0 |
| `yolo26n-sni-bbp3e5-depth.pt` | same arch, Objects365 warm start | 0.831 | 98% | 0% | 0 |
| `baseline-yolo26n-depth.pt` | Aug-31 baseline, stock yolo26 arch, 6.4M | 0.785 | **100%** | **100%** | 0 |
| `yolo26n.pt` (detect, control) | stock detector | — | 100% | 100% | partial |
| `yolo26n-pose.pt` (current init) | 2D pose | — | 100% | 100% | 252/324 tensors |

The sni checkpoints only unpickle inside a `depth-p3p5-sni` checkout (`ultra15:~/ultralytics_sni`, needs the
`Scale` block); their state dicts are dumped to plain tensors (`~/h10/*.sd.pt`) so pose3d can load them by name.
The `yolo26s` / `yolo27n_depth_mgv2gt` runs elsewhere are the wrong scale or architecture for `n`.

## Stage 0b — frozen-feature probe (`research/src/feature_depth_probe.py`)

Ridge regression from frozen P3/P4/P5 features, bilinearly sampled at each labelled joint, to (a) log root depth,
(b) metric Δz, (c) scale-free Δz = log(z_j) − log(z_root). Split by sequence (3DPW test) or image (COCO val,
teacher labels); alpha chosen inside the training half. It measures how much person depth a checkpoint's
features already expose to a linear read-out — a prior on the warm start, not its outcome.

**Backbone taps (layers 4/6/10 — the part every candidate transfers), 640:**

| checkpoint | 3DPW root AbsRel | 3DPW Δz MAE mm | 3DPW ratio-Δz MAE | COCO root AbsRel | COCO Δz MAE mm | COCO ratio-Δz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random init | 0.566 | 181.0 | 0.0395 | 0.636 | 149.2 | 0.0414 |
| baseline-yolo26n-depth | 0.197 | 172.1 | 0.0371 | 0.276 | 138.7 | 0.0373 |
| YOLO27n-depth (shipped) | 0.162 | 170.5 | 0.0366 | 0.239 | 136.7 | 0.0366 |
| YOLO27n-depth, O365 arm | 0.181 | 168.1 | 0.0361 | 0.254 | 136.9 | 0.0365 |
| yolo26n detect | 0.131 | 164.0 | 0.0350 | 0.164 | 134.9 | 0.0358 |
| **yolo26n-pose** | **0.073** | **159.8** | **0.0329** | **0.129** | **128.2** | **0.0340** |
| F_clip (trained pose3d, reference) | 0.082 | 161.6 | 0.0345 | 0.130 | 126.7 | 0.0330 |

(Predicting the training median: Δz 186.1 / 149.9 mm, ratio 0.0403 / 0.0399.) Neck taps give the same order
(depth models 0.103–0.186 root AbsRel on 3DPW vs 0.086 for both detect and pose). Probing the depth models at
their native 768 changes no ranking (YOLO27n-depth 3DPW backbone 0.162 -> 0.172).

**Reading.** On every target, tap and size, depth-pretrained features encode *person* depth worse than a plain
detector's, which in turn is worse than the 2D pose model's — order depth < detect < pose. The scale-free target
was added because the depth models are median-aligned (scale-invariant) by training; they lose there too. The detect
control matters: it has no pose head and never saw keypoints, so pose's lead is not only "trained on these pixels".
Plausible mechanism: a dense scene-depth model spends its capacity on walls and roads, and a person is a thin,
small, mostly front-facing object in that loss; detector and pose features are person-centric.

Caveats: linear read-out of frozen features; the depth head is excluded; fine-tuning re-shapes features. The probe
is a prior with a bad sign, not a verdict.

## Stage 1 — training A/B (proposed, matched to F_clip's recipe)

Recipe exactly F_clip's: `coco-pose3d-refocal`, 100 epochs, batch 64, imgsz 640, seed 0, default aug, one RTX 6000D
per arm (~3.5 h). F_clip is the pose-init arm, already trained.

| arm | init | transfers |
| --- | --- | --- |
| `F_clip` (exists) | yolo26n-pose | backbone + neck + most of head |
| `D27_depthinit` | YOLO27n-depth shipped | backbone 98% only; neck + head random |
| `D26_depthinit` | baseline-yolo26n-depth | backbone + neck 100%; head random |
| `O_detinit` | yolo26n detect | backbone + neck 100%; head mostly random |
| `S_scratch_full` | none | — (A_scratch was trained on the half-loaded set; not comparable) |

`D27` loads by building `yolo26n-pose3d`, `load_state_dict(strict=False)` with the shape-matched tensors from the
dumped sni state dict, and saving that as the init checkpoint — no sni code enters the pose3d repo. `O_detinit`
is the control that makes the result interpretable: without it, "depth init loses" would only show the pose head
mattered. If D beats O, decomposition arms follow (depth backbone + pose neck/head, and the reverse).

**Rulers.** 3DPW test MPJPE / PA-MPJPE / AbsRel(Z) / δ1(Z); COCO val framed + truncated splits (H9); COCO-GT 2D
pose and box mAP (`coco-pose-gt-18x4`, rect=False).

**Locked predictions** (from Stage 0b): (P1) both D arms worse than F_clip on 3DPW MPJPE and COCO-GT pose mAP;
(P2) D ≤ O on root AbsRel and δ1(Z); (P3) D, O both above S on everything; (P4) the open cell is relative depth —
δ1(Z) is where a depth init could still win after fine-tuning.

**Decision rule.** Adopt a depth init only if it beats F_clip by ≥ 3 mm 3DPW MPJPE **or** ≥ 10% relative on root
AbsRel, without losing > 0.01 COCO-GT pose mAP. D beats O but not F_clip -> "depth pretext beats detect pretext, pose
pretext still best": recorded, recipe unchanged.

**Budget.** 4 new arms × ~3.5 h = ~14 GPU-h, ~3.5 h wall-clock on ultra15's free GPUs; evals ~0.5 GPU-h.
A cheaper pilot (`fraction=0.5`, 50 epochs, ~1 h each) answers P1 if the full run is not wanted.

## Stage 2 — the other reading: depth weights as a teacher, not an init (not run)

Init is one use of the weights; the other is supervision. Run YOLO27n/l-depth over COCO train, and add a dense
depth auxiliary head (or feature distillation on the neck) to pose3d training. This survives a negative Stage 1,
because it gives the student scene context (ground plane, occluders) without asking its trunk to be a depth model.
Costs a pseudo-labelling pass (~2.19M-image model at 2.98 ms T4 -> COCO's 118k in minutes) and a second head.

## Risks

- The probe is linear and frozen; a full fine-tune can overturn it (it predicted direction correctly for H10's prior,
  which is the only calibration it has).
- Persons are in-distribution for detect and pose features and out-of-distribution emphasis for a scene-depth loss;
  that is the effect being measured, but it means the probe cannot say whether a depth model trained with a
  person-weighted loss would behave differently.
- YOLO27n-depth's neck does not transfer, so D27 tests the backbone only; D26 tests backbone+neck but from the
  weaker baseline model.
- The depth models were trained at 768 and median-aligned; pose3d trains at 640 in metres.
