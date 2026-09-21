# Protocol — H7, 2D parity against the pose2d baseline

_Locked 2026-09-21, before any model is run. Evaluation only: no training, no label edits._

## Question

Every `pose mAP` in this project is a comparison between pose3d arms. None of them has ever been put beside a
2D pose model, and the numbers as reported cannot be: a pose3d model is scored over **18** keypoints, the 18th
being the root, whose sigma the dataset YAMLs set to 0.107 — the most forgiving value in the COCO table, on a
joint that is the midpoint of two joints the model already predicts. That is an OKS term a 17-joint model has
no way to earn.

**Does the pose3d student match a 2D pose model on 2D, once both are scored by the same ruler?**

## Hypothesis

**H7: adding the depth task costs little or nothing in 2D accuracy, but the student's 2D is capped by the
annotator it was distilled from, so it loses to `yolo26n-pose` on human ground truth and wins on the teacher's.**

## Design — a 2x2, because one ruler cannot separate the two effects

|  | COCO GT ruler | teacher ruler |
| --- | --- | --- |
| `yolo26n-pose` (2D baseline) | A | B |
| pose3d arms | C | D |

`A vs C` answers the question on the standard ruler. `B vs D` answers it on the ruler the student was trained
against. If C loses to A while D beats B, the gap is annotator convention, not capability; if C loses to A
*and* D loses to B, the depth task cost real 2D accuracy.

## Arms

- **`pose2d`** — `~/assets/yolo26n-pose.pt`, the checkpoint arms B–E were warm-started from. Same scale, same
  family; this is the fairest available baseline.
- **`A_scratch`**, **`B_warmstart`**, **`E_refocal`** — `best.pt` of each, matching every number reported so
  far. A is included as the floor: it inherited no 2D weights, so A-vs-B bounds how much of the student's 2D is
  inherited rather than learned.

## Rulers

- **COCO GT** — `research/experiments/h7-2d-parity/coco-pose-gt.yaml`, COCO's human keypoint annotation,
  2,346 val2017 images, absolute path (these boxes cannot reach GitHub, and a failed resolve hangs).
- **Teacher** — `ultralytics/cfg/datasets/coco-pose3d.yaml`, the SAM 3D Body pseudo-labels, the same 2,346
  images (that dataset's `images/val2017` is a symlink into the GT one).

Both are scored by `research/src/eval_2d_parity.py`, which drops the root joint and the depth channel from
predictions *and* labels and forces `OKS_SIGMA` on both sides. Every model goes through that one class, so a
harness bug lands on all of them equally. `imgsz=640`, default `conf`/`iou`, `best.pt` throughout.

Two things that follow from the setup and should not be read as results:

- The refocal relabel only touched the depth channel, so on the teacher ruler the original and re-keyed label
  sets are **identical once z is dropped**. One converted view serves every arm.
- The pseudo-labelling dropped the persons the teacher could not process, so the teacher ruler may carry fewer
  GT persons than the COCO ruler on the same images. **mAP is comparable within a column only.**

## Amendments

_2026-09-21, after the first pass and before the numbers were read._

- **The person sets are identical, not merely similar**: 6,352 rows in both label trees, and the boxes are
  byte-identical (the pseudo-labeller copies COCO's box verbatim). Only the keypoint coordinates and the
  visibility convention differ. The "within a column only" rule stands, for the reason below instead.
- **One evaluator, forced.** Both dataset paths contain `coco` and end in `val2017.txt`, so
  `DetectionValidator.init_metrics` sets `is_coco` and forces `save_json` on for both — but only the COCO tree
  ships an `annotations/` dir, so the first pass scored that column with pycocotools and the other with
  Ultralytics' own metric. Two evaluators, two columns, one table: not readable across. `eval_2d_parity.py`
  now clears `save_json` after `init_metrics`, and both rulers were re-run.

## Predictions

- **A > C on the COCO GT ruler.** The teacher's own reprojection agrees with COCO annotation at OKS mean 0.805,
  and the student cannot exceed the annotator it was fit to. A gap is expected and is not evidence the depth
  task hurt.
- **D > B on the teacher ruler**, for the mirror-image reason.
- **C(E_refocal) ≈ C(B_warmstart)**, within ~0.005 pose mAP50-95. Every depth intervention so far left 2D mAP
  at 0.780 ± 0.0002, so the depth channel should be invisible here too.
- **C(A_scratch) well below both**, by more than the 0.075 pose mAP that warm-starting bought on 3DPW.

**The number that answers the question is the size of the A–C gap.** If E lands within a few points of
`yolo26n-pose` on human ground truth, the depth task was close to free even on the ruler that favours the
baseline. If the gap is large, the honest claim is that pose3d buys depth at a real 2D cost — and the B-vs-D
column says whether that cost is the task or the teacher.

## Failure modes worth naming

- If `pose2d` scores far below its published COCO number, the harness is wrong, not the model — stop and fix
  the harness before reading anything else.
- If all four models score identically on a ruler, suspect the truncation silently disabled OKS matching.
