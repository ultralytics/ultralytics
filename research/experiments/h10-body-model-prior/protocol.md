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

## Addendum (2026-10-22) — real-GT training, the standard 3DPW protocol

Confound in the first round: every COCO training label was read off SAM 3D Body's MHR rig, so the labels already
carry a body model's skeleton, and a bone/symmetry prior on top of them is partly redundant. Real-GT 3D training
removes that. Only 3DPW is on the cluster (H36M / MPI-INF-3DHP / EMDB are licence-gated and Shenzhen egress is
blocked), so: **train on 3DPW train (24 seq, 8,828 frames), select on 3DPW validation (12 seq), report on 3DPW
test** — all SMPL-derived GT, `focal = 1.2·max(w,h)`, 12 body joints with depth.

| arm | init | epochs | POSE3D_BONE_GAIN |
| --- | --- | ---: | ---: |
| P_pw_ctrl | yolo26n-pose (no teacher labels anywhere) | 60 | 0 |
| Q_pw_bone1 | yolo26n-pose | 60 | 1 |
| R_pw_bone4 | yolo26n-pose | 60 | 4 |
| S_pwft_ctrl | F_clip (pretrain on teacher, fine-tune on real GT) | 30 | 0 |
| T_pwft_bone1 | F_clip | 30 | 1 |

Locked predictions: (A1) on real GT the prior is no longer harmful — Q within ±3 mm of P on 3DPW test MPJPE;
(A2) R (gain 4) still worse than P; (A3) S beats F_clip's 109.8 by > 10 mm (in-domain fine-tune); T within ±3 mm
of S. Decision rule unchanged: the prior earns a place only if it beats its control by ≥ 3 mm.

## Addendum B (2026-10-22) — BEDLAM with SMPL-X-derived relative depth

**Labels.** `convert_bedlam.py --smplx-dir` poses the neutral SMPL-X (11 betas, `pose_cam`, translation
`trans_cam + cam_ext[:3,3]`); this reprojects onto BEDLAM's shipped `gtkps` at a median **0.0 px** on four
sequences checked (incl. AGORA, stadium, a 30fps set), so the 3D joints are BEDLAM's GT exactly. All 17 joints
get visibility 2 with real Δz; root depth is the true mid-hip. Joint definitions are SMPL-X internal joints —
the same convention as 3DPW's SMPL GT, unlike COCO surface keypoints — so a BEDLAM-trained model does not pay the
definition tax F_clip pays on 3DPW (and pays it instead on COCO-GT 2D). SMPL-X files come from a third-party HF
mirror (camenduru/SMPLer-X), on Rick's decision; MPI licence not accepted — same provenance caveat as BEDLAM itself.

**Interpretation frame, stated before the result.** BEDLAM's GT is *generated* by SMPL-X, so its labels are
rig-consistent by construction, like the teacher's MHR labels on COCO and like 3DPW's SMPL fits. Every 3D-human GT
in the field is body-model-derived. This round therefore does not remove that property; it asks whether the
bone/symmetry prior helps **as a regulariser on the largest real-GT set available**, with 30x 3DPW's frames.

**Split.** By scene: both `20221018_*_stadium_6fps` sequences (only place `stadium` appears) are BEDLAM-val;
everything else trains. 3DPW test remains the real-image ruler.

**Arms** (all from `yolo26n-pose`, no teacher labels; epochs set from the measured first-epoch speed):
`U_bed_ctrl_s0`, `U_bed_ctrl_s1` (gain 0) · `V_bed_bone1_s0`, `V_bed_bone1_s1` (gain 1) · `W_bed_bone4_s0` (gain 4).

**Locked predictions.** (B1) gain 1 within ±2 mm of ctrl (seed-mean) on BEDLAM-val and 3DPW test; (B2) gain 4
worse than ctrl on both; (B3) BEDLAM-only ctrl on 3DPW test lands at 110–140 mm (synthetic->real gap, offset partly
by matching joint definitions). Decision rule unchanged: the prior must beat its control by ≥ 3 mm.
