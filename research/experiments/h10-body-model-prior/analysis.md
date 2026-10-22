# Analysis — H10, does a parametric body model buy accuracy?

_2026-10-22, task ft-mnvw. Protocol locked first (`protocol.md`). Subject `F_clip` (3DPW MPJPE 109.85 / PA 82.96 /
AbsRel 0.04556, reproduced to the digit by `dump_matched_pairs.py`). Scripts: `research/src/dump_matched_pairs.py`,
`body_prior_probe.py`, `score_teacher_3dpw.py`. GPU spent on ultra15: ~0.2 h teacher (6 GPUs × 10 min) + 3 × 0.6 h fine-tunes + evals, ~2.5 GPU-h total._

## Answer

**No — not for accuracy, not now.** The student's error is not the kind a body model removes. It is per-joint
relative depth, and the student's skeleton is already as plausible as the error allows: constraining it to the
exact bone lengths of the person (an oracle no deployed model has) recovers at most 5.5 mm of 110 on 3DPW, tuned on
the test set, and **0.8 mm** on COCO with the teacher's own rig skeleton. Population-mean bones — what a body model
gives a model that does not also estimate shape — make it **worse** on both datasets. A train-time body-structure
prior was then tested directly by fine-tuning, and it made every 3D metric worse, monotonically in its weight (§6).

## 1. Where the error is (3DPW, 7,003 matched persons)

| swap | student MPJPE | teacher MPJPE |
| --- | ---: | ---: |
| as predicted | 109.85 | 66.69 |
| per-person scale fit | 107.68 | 64.96 |
| PA-MPJPE | 82.96 | 47.21 |
| GT 2D, predicted depth | 93.26 | 54.24 |
| **GT depth, predicted 2D** | **48.97** | 35.32 |
| GT root depth only | 109.43 | 66.47 |

Perfect depth removes 55% of the student's error; perfect 2D removes 15%; perfect root depth removes 0.4%. **The
error is the root-relative depth channel (Δz)** — which way and how far each limb points into the scene. COCO val
says the same with both halves larger (GT depth 70.2, GT 2D 84.3 of 110.0).

## 2. Is the predicted skeleton implausible?

Somewhat, and it does not matter. On COCO the student's left/right limbs disagree far more than the teacher's
rig-generated labels (forearm 12.5% vs 1.1%, upper arm 5.9% vs 0.8%), and forearms run ~13% short. But symmetrising
them (`sym` refit) changes MPJPE by **+0.0 to +0.2 mm** on both datasets. On 3DPW the large bone "errors" — shoulder
width 0.33 vs 0.40 m, hip width 0.15 vs 0.12 m, shin 0.35 vs 0.42 m — are **joint-definition**, not student error:
the teacher shows the same offsets (0.36 / 0.17 / 0.40), because 3DPW's GT is SMPL-regressor joints and ours are
COCO surface keypoints.

## 3. Refit under body-model constraints (best of a 3×3 trust sweep, i.e. tuned on test — an upper bound)

| constraint | 3DPW MPJPE | 3DPW PA | COCO MPJPE | COCO PA |
| --- | ---: | ---: | ---: | ---: |
| none (F_clip) | 109.85 | 82.96 | 110.01 | 63.89 |
| **oracle**: this person's GT bone lengths | 104.39 | 79.38 | 109.18 | 63.57 |
| **mean**: population bone lengths | 109.72 | 80.06 | 112.15 | 64.83 |
| **prop**: mean proportions at predicted size | 109.39 | 79.84 | 110.81 | 63.47 |
| **sym**: own lengths, L/R averaged | 109.32 | 81.65 | 110.04 | 63.78 |

Only the oracle clears the protocol's bar, and only on 3DPW (−5.5 mm < 10 mm). On COCO the oracle **is** a body
model's skeleton — the labels were read off SAM 3D Body's MHR rig — and it buys 0.8 mm. On the teacher, the same
oracle buys 8.6 mm (66.7 -> 58.1), almost all of it the SMPL-vs-COCO joint definition; mean bones cost it 10 mm.

Why so little: given 2D and a bone length, a bone-length constraint fixes |Δz| of the bone only when the 2D is
exact, and near-fronto-parallel limbs make |Δz| ill-conditioned on 2D noise. The scope of this test is lengths and
symmetry; the articulation half (joint limits, sign) is bounded separately below.

**Sign vs magnitude — the half of a body model the refit cannot test.** Joint limits and pose priors (elbows do not
bend backwards) resolve which way a limb points, and a bone-length refit started from the prediction never flips a
sign. So the oracle for "an articulation prior that got every direction right" was scored directly: keep the
predicted |Δz|, take each joint's Δz sign (in front of or behind the root) from GT.

| Δz swap | 3DPW MPJPE | COCO MPJPE |
| --- | ---: | ---: |
| as predicted | 109.85 | 110.01 |
| GT sign, predicted magnitude | 100.68 | 102.57 |
| predicted sign, GT magnitude | 70.29 | 83.11 |
| GT Δz | 49.72 | 68.28 |

14.5% of 3DPW joints have the wrong depth sign (10.9% where |Δz| > 5 cm). A *perfect* sign prior is worth 9.2 of
the 60 mm depth error; correct magnitudes are worth 39.6. **The error is depth magnitude, not direction**, and
neither half of a body model — lengths or joint limits — constrains magnitude for a limb whose length is already
about right. A real articulation prior would capture only part of the 9 mm sign headroom.

## 4. Root depth from body size

| root AbsRel(Z) | 3DPW | COCO |
| --- | ---: | ---: |
| learned (F_clip) | **0.0456** | **0.0619** |
| rescaled to oracle body size | 0.0572 | 0.1263 |
| geometric blend with oracle | 0.0357 | 0.0873 |
| rescaled to mean body size | 0.0504 | 0.1358 |
| blend with mean | 0.0455 | 0.0919 |

Metric body size helps root depth only if the person's own size is known (oracle blend, 3DPW). Population size
is no better than what the network already learned from apparent size, and much worse on COCO's truncated people.

## 5. Teacher ceiling

SAM 3D Body on 3DPW test from GT boxes, depth re-keyed from its diagonal focal: **MPJPE 66.69 / PA-MPJPE 47.21 /
AbsRel 0.0350**, 7,097/7,097 persons. The student is 43 mm behind. Sections 1–3 say that gap is not the rig: it is
per-joint depth from a 3.4M-parameter shared trunk at 640 versus a DINOv3 ViT on a per-person crop.

## 6. Train-time body-structure prior (fine-tune)

`Pose3DLoss` gained an env-gated term (`POSE3D_BONE_GAIN`, off by default): relative bone-length error of the
lifted predicted skeleton against the label skeleton — on COCO, SAM 3D Body's MHR-rig skeleton — plus left/right
limb symmetry. This is the train-time-only part of a body model, with the deployed head unchanged. Three arms from
`F_clip/weights/best.pt`, 20 epochs each, identical except the gain; `I_ftctrl` is the same-compute control.

| arm | gain | 3DPW MPJPE | 3DPW PA | 3DPW AbsRel | 3DPW δ1(Z) | COCO MPJPE | COCO PA | framed AbsRel | trunc AbsRel | COCO-GT pose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_clip (reference) | — | 109.85 | 82.96 | 0.0456 | 0.657 | 110.01 | 63.89 | 0.0562 | 0.109 | 0.4731 |
| I_ftctrl | 0 | **111.08** | **83.43** | 0.0434 | **0.652** | **111.82** | **64.72** | 0.0596 | **0.115** | **0.4596** |
| J_ftbone1 | 1 | 116.39 | 89.30 | **0.0421** | 0.633 | 114.91 | 68.63 | **0.0581** | 0.134 | 0.4570 |
| K_ftbone4 | 4 | 120.84 | 92.47 | 0.0440 | 0.616 | 117.53 | 70.76 | 0.0594 | 0.162 | 0.4474 |

**Refuted, monotonically.** Against the same-compute control, the prior costs **+5.3 / +9.8 mm MPJPE** and
**+5.9 / +9.0 mm PA-MPJPE** on 3DPW at gains 1 / 4, the same direction on COCO (+3.1 / +5.7 mm), and 0.3–1.2 points
of COCO-GT 2D pose mAP. The one cell it wins is 3DPW root AbsRel (0.0434 -> 0.0421, gain 1 only, gone at gain 4),
paid for on truncated persons (0.115 -> 0.134 -> 0.162) — the H9 trade again, and invisible on 3DPW. PA-MPJPE
rising is the mechanism: the term is satisfiable by moving joints along the bone rather than to the right place,
and it pulls the head's capacity from the per-joint depth that §1 says is the actual error.

(The control itself is ~1 mm below F_clip: re-warming the LR for 20 epochs on a converged model costs a little.
Every comparison here is against the control, not F_clip.)

## Against the locked predictions

| # | prediction | outcome |
| --- | --- | --- |
| P1 | depth residual > 2D residual; scale fit < 10 mm | **confirmed** — 93.3 vs 49.0 mm; scale fit 2.2 mm |
| P2 | bones within 10% of GT; asymmetry within 2x of GT | **partly refuted** — median errors 5–22% on 3DPW (mostly joint definition); COCO forearm asymmetry 11x the teacher's |
| P3 | oracle 3–8 mm, mean 0–3 mm, sym ≈ 0 | **confirmed** — 5.5 / 0.1–0.5 / 0.5 mm on 3DPW (mean is negative on COCO) |
| P4 | mean-size root depth worse than learned | **confirmed** — 0.0504 vs 0.0456 (3DPW), 0.136 vs 0.062 (COCO) |
| P5 | teacher 80–95 mm on 3DPW | **refuted, better** — 66.7 mm |

Decision rule outcome: R-mean < 3 mm and R-oracle < 10 mm -> **no body model for accuracy**. The training arm,
run anyway because a post-hoc refit is not a training signal, agrees.

## Recommendation

1. **Do not add MHR/SMPL to the student for accuracy** — neither as output (H4) nor as a train-time prior. Keep H4
   deferred; its remaining case is product-side (a mesh output), and it still carries the gated-rig/AGPL and
   edge-export costs recorded in the design doc.
2. **Spend on the relative-depth channel**: it is 55% of the error and the 43 mm teacher gap. Candidates, in
   prior order: a larger student (s/m), higher input resolution for small people, more 3D-labelled data
   (Objects365 pseudo-labels already on disk, T5), and a relative-depth loss that weights by limb rather than joint.
3. If 3DPW numbers matter for publication, add a fixed COCO->SMPL joint-regressor offset at evaluation — part of
   the measured "error" is joint definition, and the teacher pays it too.

## 7. Real-GT training — the standard 3DPW protocol (addendum)

The first round had a confound: every COCO label was read off SAM 3D Body's MHR rig, so the training data already
carried a body model's skeleton. Here no teacher label is used: train on **3DPW train** (24 sequences, 8,828
frames, SMPL-derived GT), select `best.pt` on **3DPW validation**, report on **3DPW test**. H36M / MPI-INF-3DHP /
EMDB are not on the cluster (licence-gated, Shenzhen egress blocked), so 3DPW is the only real-GT set available.

| arm | init | ep | gain | **test MPJPE** | test PA | test AbsRel | val MPJPE | val PA | test pose mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_clip (no 3DPW training) | — | — | — | **109.85** | **82.96** | **0.0456** | — | — | **0.758** |
| P_pw_ctrl | yolo26n-pose | 60 | 0 | 145.21 | 102.99 | 0.0823 | 103.57 | 80.06 | 0.540 |
| Q_pw_bone1 | yolo26n-pose | 60 | 1 | 151.34 | 112.45 | 0.0797 | 108.64 | 88.08 | 0.532 |
| R_pw_bone4 | yolo26n-pose | 60 | 4 | 192.05 | 141.52 | 0.0809 | 147.91 | 117.66 | 0.496 |
| S_pwft_ctrl | F_clip | 30 | 0 | 135.03 | 96.07 | 0.0781 | 97.84 | 76.90 | 0.612 |
| T_pwft_bone1 | F_clip | 30 | 1 | 128.90 | 93.98 | 0.0651 | 100.66 | 81.05 | 0.651 |

**3DPW train alone is too small to train on.** Both pure-real-GT and fine-tuned models score far better on
validation (98–104 mm) than on test (135–145 mm), and all are worse on test than F_clip, which never saw 3DPW.
Prediction A3 (fine-tune beats F_clip by > 10 mm) is **refuted** — it loses 25 mm. Twenty-four sequences of a few
actors are memorised; the teacher-labelled COCO volume generalises better than in-domain real GT at this scale.

**From scratch on real GT, the prior is still harmful**, monotonically: +6.1 mm (gain 1), +46.8 mm (gain 4) on test,
and +5.1 / +44.3 mm on validation. A1 (no longer harmful) **refuted**; A2 confirmed.

**From F_clip, the splits disagree.** T beats S on test by 6.1 mm MPJPE (and on 2D pose mAP, 0.651 vs 0.612) but
loses on validation by 2.8 mm. That pattern reads as regularisation against memorising 3DPW train, not as better
3D — and T is still 19 mm worse than not fine-tuning at all. Seed replicates below.

**The error structure is the same on real GT.** On P/S/T the GT-depth swap removes 49–54% of MPJPE (P: 145.2 ->
74.4), the GT-2D swap 19–20%, root depth ~1.5%, and the best oracle bone-length refit recovers **0.1–0.9 mm**.
The prior does what it says — T's forearm asymmetry falls 0.136 -> 0.058 — and that is not where the error is.

**Seed replicates (S vs T, 3 seeds each).**

| arm | test MPJPE s0 / s1 / s2 | mean | test PA mean | val MPJPE s0 / s1 / s2 | mean |
| --- | --- | ---: | ---: | --- | ---: |
| S_pwft_ctrl | 135.03 / 134.31 / 133.09 | 134.14 | **95.90** | 97.84 / 97.47 / 99.04 | **98.12** |
| T_pwft_bone1 | 128.90 / 132.13 / 134.48 | **131.84** | 96.38 | 100.66 / 99.88 / 102.43 | 100.99 |

Seed 0's 6.1 mm test win was mostly seed luck: over three seeds the prior is **−2.3 mm on test MPJPE** (under the
protocol's 3 mm bar, inside T's own 5.6 mm seed spread), **+0.5 mm worse on test PA-MPJPE**, and **+2.9 mm worse on
validation in every seed** — the split that chose the checkpoints. Not a win.

**Real-GT verdict: unchanged.** Training on real 3D GT removes the teacher-rig confound and the conclusion holds:
the body-structure prior hurts from scratch, is noise-level at best on top of F_clip, and the error is per-joint
depth magnitude on real GT exactly as on pseudo-labels. Separately, and more useful: **3DPW train is too small to
help** — every 3DPW-trained arm is 21–35 mm worse on test than F_clip, so real-GT fine-tuning at this scale is not a
lever either. A larger real-GT set (BEDLAM with SMPL-X-derived relative depth, which needs the gated body model for
*labels*, not in the student) is the one place a body model would enter this project.
