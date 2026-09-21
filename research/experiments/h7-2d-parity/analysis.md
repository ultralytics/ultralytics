# Analysis — H7, 2D parity against the pose2d baseline

_2026-09-21. Evaluation only, no training. `research/src/eval_2d_parity.py` on ultra15 GPUs 2 and 3, ~4 min per
model per ruler. Every number below comes from the same class with the same truncation and the same sigmas._

## Results

**COCO GT ruler** — human annotation, 2,346 val2017 images, 6,352 persons, all of them loaded.

| model | box mAP50-95 | pose mAP50 | pose mAP50-95 |
| --- | ---: | ---: | ---: |
| pose2d (`yolo26n-pose`) | **0.6645** | **0.8250** | **0.5676** |
| A_scratch | 0.5808 | 0.6598 | 0.3486 |
| B_warmstart | 0.6288 | 0.7347 | 0.4363 |
| E_refocal | 0.6298 | 0.7336 | 0.4360 |

**Teacher ruler** — SAM 3D Body pseudo-labels on the same 2,346 images, of which **1,168 load** (see below).

| model | box mAP50-95 | pose mAP50 | pose mAP50-95 |
| --- | ---: | ---: | ---: |
| pose2d (`yolo26n-pose`) | 0.6852 | 0.8230 | 0.5463 |
| A_scratch | 0.6666 | 0.7827 | 0.4806 |
| B_warmstart | **0.6990** | **0.8355** | 0.5718 |
| E_refocal | 0.6963 | 0.8325 | **0.5722** |

## Against the locked predictions

| prediction | outcome |
| --- | --- |
| A > C on COCO GT | **confirmed** — 0.5676 vs 0.4360, a gap of 0.1316 |
| D > B on the teacher ruler | **confirmed** — 0.5722 vs 0.5463, +0.0259 |
| E ≈ B within 0.005 | **confirmed** — largest gap 0.0043 across all three columns |
| A_scratch well below both, by more than 0.075 | confirmed — 0.0877 below B on GT |

## The answer

**On the human ruler, no: the pose3d student does not beat pose2d, and is not close.** `yolo26n-pose` scores
0.5676 pose mAP50-95 against COCO's own annotation; the best pose3d arm scores 0.4360, or 77% of it. The
ranking inverts on the teacher's ruler, where the student edges ahead by 0.026 — so a large part of the gap is
the annotator, not the model. But the deployment ruler is the human one, and on that ruler the gap is real.

**The depth task is not what costs the 2D.** E_refocal and B_warmstart are within 0.0003 of each other on both
rulers, and the entire H2/H2.1/H6 family of depth interventions moved 2D mAP by less than 0.0005. Whatever the
student is losing, it is not losing to the fourth channel.

**The teacher column is a weaker piece of evidence than the GT column** and should not be quoted on its own:
the images that survive label validation are exactly the ones whose people are entirely inside the frame,
which is the easy half. "The student wins on its own ruler" is true on an easier subset.

## What the setup turned up: half the dataset never reached training

Building the second ruler meant reading the label caches, and they say this:

| tree | images found | images loaded | instances |
| --- | ---: | ---: | ---: |
| `coco-pose3d` train2017 | 56,599 | **29,344** | 65,857 |
| `coco-pose3d-refocal` train2017 | 56,599 | **29,344** | 65,857 |
| `coco-pose3d` val2017 | 2,346 | **1,168** | 2,609 |
| `3dpw-pose3d` / `-conv` / `-r12` | 4,907 | 4,907 | 7,097 |

**27,255 training images — 48% — were silently rejected**, and the same in val. The cause is in the cache's own
message: `ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.24015 1.27461 ...]`.
`ultralytics/data/utils.py:358` asserts every keypoint x/y is `<= 1.01`, and the pseudo-labeller writes the
teacher's reconstructed coordinates unclipped — SAM 3D Body reconstructs joints outside the frame rather than
omitting them, and `pseudo_label_sam3d.py` only uses that fact to set visibility, never to clip the value. One
out-of-frame joint on one person rejects the whole image.

**3DPW is unaffected** — all three converted copies load 4,907 of 4,907 images with zero corrupt, so every
outer-loop number (MPJPE 106.5 mm, AbsRel 0.0398) is scored on the full benchmark. The loss is training-side and
COCO-val-side only. Every result in this project so far — R0, H2, H2.1, H6 — was trained on 29,344 images and
validated in-domain on 1,168, not the 56,599 and 2,346 the protocols claim. Nothing is invalidated by this, since every arm was handicapped
identically, but every absolute number is from a half-sized dataset, and the images lost are the ones with
people at the frame edge.

The fix is one line in the pseudo-labeller: clip `xy` into [0, 1] before writing. Visibility is already 0 for
those joints and `Pose3DLoss` recomputes the mask from it, so a clipped joint is ignored by the loss — the clip
changes nothing the model learns from, it only stops the loader throwing the image away. That is the first
thing to test against the 0.1316 gap.

## The follow-up probe: the same 1,168 images, both annotators

The teacher ruler only ever scored the 1,168 images whose pseudo-labels survive validation, so the first two
columns were not on the same pixels. `coco-pose-gt-surviving.yaml` fixes that — COCO's own annotation
restricted to exactly those images and those 2,609 persons.

**COCO GT ruler, 1,168 surviving images.**

| model | box mAP50-95 | pose mAP50 | pose mAP50-95 |
| --- | ---: | ---: | ---: |
| pose2d (`yolo26n-pose`) | 0.6852 | **0.8628** | **0.6169** |
| A_scratch | 0.6666 | 0.7931 | 0.4587 |
| B_warmstart | **0.6990** | 0.8400 | 0.5453 |
| E_refocal | 0.6963 | 0.8395 | 0.5410 |

Box mAP here is **identical to the teacher column** to four decimals for every model (0.6852 / 0.6666 / 0.6990 /
0.6963), which is the check that the subset is exactly right: the two trees carry the same boxes, so once the
image sets match, detection cannot differ. Only the keypoints do.

Two readings, both clean:

**The gap nearly halves on the images the student was trained to expect.** pose2d's lead over E_refocal goes
from **0.1316** on the full val set to **0.0759** on the surviving 1,168 — 42% of it lives in the frame-edge
images, which are exactly the ones the loader threw out of training. E gains +0.105 moving to the subset
(0.4360 -> 0.5410) against pose2d's +0.049. That is the strongest evidence yet for candidate 2, though it does
not separate "never trained on them" from "intrinsically harder".

**The annotator is worth about 0.10 of swing.** On identical pixels and identical people, pose2d leads by
0.0759 when COCO defines truth and trails by 0.0259 when the teacher does. Same models, same images — only the
annotation changes.

**And the student never wins on human ground truth.** Even on its friendliest subset it is 0.076 behind.

## Reading the gap

Three candidates, the first two now both measured:

1. **Annotator convention.** The teacher agrees with COCO at OKS mean 0.805; a student fit to it inherits that
   disagreement. Worth ~0.10 of swing on matched images, by the probe above.
2. **Half the training data**, per above — worth ~0.056 of the 0.1316, by the subset probe. `yolo26n-pose`
   trained on all of COCO-pose.
3. **Visibility means "in frame", not "unoccluded"**, deliberately (`pseudo_label_sam3d.py`), so the student is
   supervised on joints COCO leaves unlabelled. This changes what the model puts on screen but cannot by itself
   lower OKS, which only scores labelled joints.

Candidate 2 now has a number against it, and the clipped-label relabel plus retrain is the experiment that
would collect it: it should close part of the 0.0557 that the frame-edge images cost, and leave the 0.0759
residual on the easy subset as the annotator's share.

## Harness notes

- The first pass scored the two columns with two different evaluators. Both dataset paths contain `coco` and
  end in `val2017.txt`, so `DetectionValidator.init_metrics` sets `is_coco` and forces `save_json`; only the
  COCO tree ships `annotations/`, so that column came back from pycocotools and the other from Ultralytics'
  own metric. `eval_2d_parity.py` now clears `save_json` after `init_metrics`. Both columns above are the
  internal metric.
- As a cross-check, the discarded pycocotools pass put `yolo26n-pose` at keypoint AP 0.567 against the internal
  metric's 0.5676 — two independent evaluators agreeing on the truncated 17-joint predictions.
- Boxes are byte-identical between the two label trees (the pseudo-labeller copies COCO's box verbatim), so the
  box mAP difference across columns is the loaded-image difference, nothing else.
