# Analysis — H11, YOLO-depth checkpoints as a pose3d warm start

_2026-10-24, task ft-syjg. Protocol `protocol.md` (Stage 0 probe + locked predictions written before Stage 1).
Stage 1: 4 arms × 100 epochs at F_clip's exact recipe, ultra11 GPUs 3-6, ~3.3 h wall / ~13 GPU-h; evals on ultra15._

## Answer

**No. A YOLO-depth init is no better than training from scratch, and worse than the current yolo26n-pose init.**
On 3DPW test both depth inits land at 117.3 mm MPJPE — the same as random init (117.3) — against 116.0 for a plain
detector init and **109.9 for yolo26n-pose (F_clip)**. They also lose 0.047-0.053 COCO-GT pose mAP. The one cell a
depth init touches is 3DPW root AbsRel (D26 0.0434 vs F_clip 0.0456, −4.7%), under the 10% bar and reversed on COCO.

## Results (single seed each, identical recipe; only the initialisation differs)

| arm | init (layers transferred) | 3DPW MPJPE | 3DPW PA | 3DPW AbsRel | 3DPW δ1(Z) | COCO MPJPE | framed AbsRel | trunc AbsRel | COCO-GT pose | COCO-GT box |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **F_clip** | yolo26n-pose (backbone+neck+head) | **109.85** | **82.96** | 0.0456 | **0.657** | **110.01** | **0.0562** | **0.109** | **0.473** | 0.689 |
| O_detinit | yolo26n detect (0-22) | 115.95 | 86.86 | 0.0447 | 0.632 | 115.02 | 0.0585 | 0.118 | 0.442 | **0.691** |
| D26_depthinit | baseline-yolo26n-depth (0-22) | 117.27 | 88.33 | **0.0434** | 0.633 | 120.91 | 0.0615 | 0.124 | 0.420 | 0.667 |
| D27_depthinit | **YOLO27n-depth** shipped (backbone, 98%) | 117.34 | 88.29 | 0.0453 | 0.628 | 118.93 | 0.0610 | 0.125 | 0.426 | 0.667 |
| S_scratch_full | none | 117.29 | 88.01 | 0.0480 | 0.637 | 119.62 | 0.0627 | 0.124 | 0.419 | 0.669 |

## Against the locked predictions

| # | prediction | outcome |
| --- | --- | --- |
| P1 | both D worse than F_clip on 3DPW MPJPE and COCO-GT pose | **confirmed** — +7.4 / +7.5 mm, −0.053 / −0.047 mAP |
| P2 | D ≤ O on root AbsRel and δ1(Z) | **mixed** — D26 beats O on 3DPW AbsRel (0.0434 vs 0.0447), loses on COCO framed/trunc; δ1 ties |
| P3 | D and O above scratch on everything | **refuted for D** — D ≈ scratch on 3DPW MPJPE (117.3 vs 117.3) and COCO-GT pose; O beats both |
| P4 | relative depth (δ1) is where depth init could win | **refuted** — δ1(Z) 0.628-0.633 vs scratch 0.637, F_clip 0.657 |

Decision rule (≥3 mm MPJPE or ≥10% AbsRel over F_clip, ≤0.01 pose mAP loss): **not met** by either depth arm.

## Reading

- **The frozen-feature probe called it.** Stage 0 ranked person-depth information depth < detect < pose; after 100
  epochs of fine-tuning the order on 3DPW MPJPE is depth ≈ scratch < detect < pose. A dense scene-depth pretext does
  not give the trunk anything a person-centric task can use that random init does not also reach in 100 epochs.
- **What the pose init buys is mostly the head and the person prior, not depth.** F_clip vs O (both pretrained
  trunks) is 6.1 mm and 0.031 pose mAP; O vs scratch is 1.3 mm. The 2D-pose pretext is the useful one.
- **Root depth is the only place a depth init registers**, and only on 3DPW (D26 −4.7% vs F_clip, −9.6% vs
  scratch); on COCO framed and truncated it is worse than F_clip. Single seed, so this is within what H10 measured
  as seed spread for AbsRel (≈0.003-0.009).
- Side result: a **from-scratch** pose3d on the full 56k images reaches 117.3 mm — the old `A_scratch` 126.9 was
  mostly the half-loaded dataset, not the missing pretrain.

## Remaining risks / not tested

- One seed per arm. The MPJPE gaps that decide the answer (F_clip vs D: 7.4 mm) are 3-5× the seed spread seen in
  H10; the AbsRel cell is not.
- **Stage 2 untested:** depth weights as a teacher (dense depth pseudo-labels as an auxiliary head, or feature
  distillation) instead of an init. It is the other reading of "use the depth weights" and this result does not
  rule it out.
- Larger depth models (s/m) and a combined init (yolo26n-pose head + depth backbone) not tried; the probe and P1 make
  both low-prior.
- YOLO27n-depth's neck does not transfer (different design), so D27 tested its backbone only; D26 tested
  backbone+neck from the older baseline.
