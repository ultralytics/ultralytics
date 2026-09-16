# Literature survey — monocular 3D human pose / mesh, and what a YOLO task would have to beat

_Bootstrap survey, 2026-09-16. One section per paper: what it does, and what it implies for YOLOPose-3D._

## The teacher

### SAM 3D Body (3DB) — Meta Superintelligence Labs, arXiv:2602.15989
Promptable single-image **full-body human mesh recovery** (body + hands + feet). Encoder–decoder
(DINOv3 backbone), accepts optional 2D-keypoint / mask prompts SAM-style. Introduces **MHR
(Momentum Human Rig)**, a parametric mesh representation that *decouples skeletal structure from
surface shape*. Trained on annotations from a multi-stage engine (manual keypoints + differentiable
optimization + multi-view geometry + dense keypoint detection), deliberately mined for rare poses
and imaging conditions. Weights gated on HF (`facebook/sam-3d-body-dinov3`), open-source code.
https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/

**Measured locally** (from `~/snowboard/docs/design.md`, real snowboard-cross frames, 2026-09-07):
`process_one_image` returns per person `pred_keypoints_3d` (70,3) metres camera-space root-relative,
`pred_keypoints_2d` (70,2) px, `pred_vertices` (18439,3), `pred_cam_t` (3,) root translation,
predicted `focal_length`, `pred_global_rots` (127,3,3), and MHR `body_pose_params`/`shape_params`.
Sanity on a carving rider: nose→ankle 1.18 m, ankle separation 0.38 m.
**It is a per-crop, one-person-at-a-time model**, and on our own footage a 37 s clip cost ≈1 h.
That cost is the whole reason this project exists.

### Fast SAM 3D Body — arXiv:2603.15603
**Training-free** acceleration of 3DB: decouples serial spatial dependencies, architecture-aware
pruning, and replaces iterative MHR→SMPL mesh fitting with a feedforward map (10^4× on that step).
Up to **10.9× end-to-end**, parity with 3DB (beats it on LSPET). Used for vision-only humanoid
teleoperation. **Still crop-based multi-crop ViT inference** — cost still scales with the number of
people, and there is no single-shot detector path.
→ *This is the closest prior art to our question, and it defines the gap: it makes the teacher
faster, it does not make it a detector.*

### Investigating Anthropometric Fidelity in SAM 3D Body — arXiv:2601.06035
Independent audit of how faithful 3DB's body measurements are. Relevant as a **bound on the
teacher**: whatever bias it carries becomes our pseudo-label bias.

## One-stage / real-time multi-person 3D

- **SAT-HMR** (CVPR 2025, arXiv:2411.19824) — real-time multi-person 3D mesh via scale-adaptive
  tokens; DETR-style one-stage.
- **Multi-HMR 2** (TMLR 2026, arXiv:2606.14841) — multi-person **camera-centric** detection + mesh +
  tracking. Camera-centric (metric root translation per person) is exactly the output contract we
  need. Reported in the literature alongside a **YOLO11m-Pose baseline at 99.2 % recall / 58.49
  MPJPE**, i.e. someone has already measured a YOLO pose model as a *2D front-end*, not as a 3D head.
- **EMO-X** (arXiv:2504.08718) — efficient one-stage multi-person pose+shape.
- **RTMW** (arXiv:2407.08634) — real-time multi-person 2D **and 3D** whole-body pose; the standing
  real-time-throughput reference outside the mesh literature.
- **CondiMen** (arXiv:2412.13058) — conditional (distributional) multi-person mesh recovery.

## Reading of the gap

Every real-time line above is either (a) DETR/ViT one-stage at server cost, or (b) an acceleration
of a crop model. None of them is a **single-shot, anchor-free, export-to-edge detector head** in the
YOLO sense, and none reuses a detector the deployment already runs. The open question this project
takes is therefore not "can 3D pose be done" but **how much of 3DB's capability survives compression
into one YOLO forward pass**, and *where* it breaks (root depth? limb foreshortening? crowding?).

## Benchmarks used by this literature

3DPW (in-the-wild, MPJPE / PA-MPJPE / PVE), AGORA, BEDLAM, EMDB, Human3.6M, LSPET.
**None are on disk here** (`~/datasets` has KITTI/COCO/NYU/cityscapes/Market-1501 but no 3D-human
set) — acquiring one is a bootstrap task, see research-state.yaml `open_tasks`.
