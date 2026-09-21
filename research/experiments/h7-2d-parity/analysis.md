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
| E ≈ B within 0.005 | **confirmed exactly** — 0.4360 vs 0.4363 on GT, 0.5722 vs 0.5718 on the teacher |
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

**27,255 training images — 48% — were silently rejected**, and the same in val. The cause is in the cache's own
message: `ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.24015 1.27461 ...]`.
`ultralytics/data/utils.py:358` asserts every keypoint x/y is `<= 1.01`, and the pseudo-labeller writes the
teacher's reconstructed coordinates unclipped — SAM 3D Body reconstructs joints outside the frame rather than
omitting them, and `pseudo_label_sam3d.py` only uses that fact to set visibility, never to clip the value. One
out-of-frame joint on one person rejects the whole image.

Every result in this project so far — R0, H2, H2.1, H6 — was trained on 29,344 images and validated on 1,168,
not the 56,599 and 2,346 the protocols claim. Nothing is invalidated by this, since every arm was handicapped
identically, but every absolute number is from a half-sized dataset, and the images lost are the ones with
people at the frame edge.

The fix is one line in the pseudo-labeller: clip `xy` into [0, 1] before writing. Visibility is already 0 for
those joints and `Pose3DLoss` recomputes the mask from it, so a clipped joint is ignored by the loss — the clip
changes nothing the model learns from, it only stops the loader throwing the image away. That is the first
thing to test against the 0.1316 gap.

## Reading the gap

Three candidates, not separable from these runs alone:

1. **Annotator convention.** The teacher agrees with COCO at OKS mean 0.805; a student fit to it inherits that
   disagreement. The ruler flip is direct evidence this is a real component.
2. **Half the training data**, per above — and `yolo26n-pose` trained on all of COCO-pose.
3. **Visibility means "in frame", not "unoccluded"**, deliberately (`pseudo_label_sam3d.py`), so the student is
   supervised on joints COCO leaves unlabelled. This changes what the model puts on screen but cannot by itself
   lower OKS, which only scores labelled joints.

Candidate 2 is cheap to test and is the obvious next run.

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
