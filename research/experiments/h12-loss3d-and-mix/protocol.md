# Protocol — H12, two levers on the relative-depth error: a 3D-space loss (#3) and COCO + BEDLAM mixing (#1)

_Locked 2026-10-24, before any run. Follows H10/H11: per-joint relative-depth magnitude is ~55% of 3DPW MPJPE; the
body-model prior and the YOLO-depth init both failed. Single seed per arm; ultra11 GPUs 2-6._

## #3 — loss on the lifted skeleton

`Pose3DLoss` supervises depth as an L1 per joint in the encoded [0, 1] space, which is not what MPJPE scores: MPJPE
is the Euclidean error of the root-relative skeleton after lifting each joint with its own absolute depth, so 2D
error, relative depth and root depth all interact through `x = (u - cx) z / f`. `POSE3D_MPJPE_GAIN` adds exactly that
quantity in metres (shared `_lift` with the H10 bone prior; same canvas-centre / 1.2·max focal convention for
prediction and target, so convention errors cancel), on joints with visibility 2. The existing terms are kept.

| arm | init | data | ep | MPJPE gain |
| --- | --- | --- | ---: | ---: |
| X_ftctrl | F_clip | COCO | 20 | 0 |
| X_ftmpjpe1 | F_clip | COCO | 20 | 1 |
| X_ftmpjpe4 | F_clip | COCO | 20 | 4 |

Predictions: (A1) gain 1 beats X_ftctrl by 2–6 mm 3DPW MPJPE and by ≥ 3 mm PA-MPJPE; (A2) gain 4 helps 3D more but
costs > 0.01 COCO-GT pose mAP (the term also pulls on 2D); (A3) root AbsRel within ±5%.

## #1 — COCO + BEDLAM, 1:1

`coco-bedlam-pose3d.yaml`: COCO pseudo-labels (56,599) + every 5th BEDLAM non-stadium frame (56,514, SMPL-X GT,
real Δz). H10 found the two complementary — F_clip is 18.5 mm worse than BEDLAM-only on BEDLAM's held-out scene,
BEDLAM-only is 30 mm worse on 3DPW. Both arms are step-matched to a COCO-only reference:

| arm | init | data | ep | step-matched to |
| --- | --- | --- | ---: | --- |
| M_mix50 | yolo26n-pose | COCO+BEDLAM (113k/ep) | 50 | F_clip (100 ep COCO) |
| M_mixft10 | F_clip | COCO+BEDLAM | 10 | X_ftctrl (20 ep COCO) |

Predictions: (B1) M_mix50 beats F_clip on 3DPW MPJPE by ≥ 3 mm and on BEDLAM-val by ≥ 10 mm; (B2) COCO-GT pose mAP
within 0.01 of F_clip (half the COCO steps, but H8 says data beats steps); (B3) M_mixft10 beats X_ftctrl on 3DPW.

## Rulers and decision rule

3DPW test, COCO val (teacher), COCO framed/trunc, COCO-GT 2D pose/box, BEDLAM stadium val. A lever is adopted if it
beats its step-matched control by ≥ 3 mm 3DPW MPJPE without losing > 0.01 COCO-GT pose mAP or getting worse on the
truncated split by > 10% AbsRel.
