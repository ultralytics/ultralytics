# Mask-IoU-aware label assignment for YOLO26-seg (experiment, ft-ee84)

Follow-up to the pose OKS-aware assigner (`pose-oks-assign`, +1.8 to +2.2 keypoint AP on all sizes): give the
segmentation assigner the quality it is evaluated on (mask IoU) instead of box CIoU alone.

## Idea

`TaskAlignedAssigner` ranks candidate anchors by `score^0.5 * overlap^6` and uses the normalized overlap as the soft
classification target, with `overlap = CIoU(box)`. Mask AP is matched on mask IoU, so the score learns to rank by box
quality, not mask quality. `seg_assign=mask` switches to

    overlap = (1 - w) * CIoU + w * maskIoU,   w = seg_assign_w = 0.5

where each predicted mask is decoded from prototypes, cropped to its predicted box and binarized (as validation does), at
prototype resolution (160x160 for 640 input). Mask IoU is computed only for each gt's top `3 * topk` candidates by the box
alignment metric (only these can become positives); on one coco8-seg batch with the pretrained model the positives and
gt indices are identical to computing it for all candidates, soft targets differ by <= 0.01 (fp16 boundary pixels).
Equivalence early in training (weak masks) was not checked.

Code: branch `seg-assign-maskiou`, `SegTaskAlignedAssigner` in `ultralytics/utils/tal.py`, wiring in
`v8SegmentationLoss` (`ultralytics/utils/loss.py`), knobs `seg_assign` / `seg_assign_w` in `cfg/default.yaml`.

## Which tasks were triaged

| task | assigner overlap | val matching metric | gap? |
|---|---|---|---|
| segment | box CIoU | mask IoU | **yes -> this experiment** |
| pose | box CIoU | OKS | yes -> done (pose-oks-assign) |
| obb | ProbIoU | ProbIoU (`batch_probiou`) | no, already aligned |
| detect | CIoU | IoU | negligible |
| depth / semantic / classify | no assigner (dense or global loss) | - | not applicable |

Pre-training diagnostic (`align_diag.py`, Spearman over matched predictions with box IoU >= 0.5, COCO val):

| model | score~boxIoU | score~task metric |
|---|---|---|
| yolo26n-pose released | 0.725 | 0.604 (OKS) |
| pose OKS-assign run (+2.2 AP) | 0.769 | 0.774 |
| yolo26n-seg released | 0.601 | 0.520 (mask IoU) |
| seg box_s0 (this study, baseline) | 0.592 | 0.517 |
| seg mask_s0 (this study, seg_assign=mask) | 0.612 | **0.569** |

The assigner moves score~maskIoU by +0.05 (pose moved score~OKS by +0.15), matching the smaller AP gain below.

## Setup

- Model `yolo26n-seg.yaml`, init from `yolo26n.pt` (detect), COCO train2017 (118k) / val2017 (5000).
- Short recipe identical for all arms: epochs=100, batch=64, imgsz=640, default hyperparameters, `deterministic=True`.
- Only `seg_assign` and `seed` differ. Arms: `box_s0`, `box_s1` (baseline, 2 seeds), `mask_s0`, `mask_s1`.
- Hardware: RTX PRO 6000 Blackwell, 1 GPU per arm (ultra5 GPUs 0/6/7, ultra4 GPU 0 for box_s1), torch 2.14.1+cu130.

## Results

Official COCO AP (faster-coco-eval on each run's final `predictions.json` from `best.pt`), `results/coco_official_ap.txt`:

| run | segm AP | bbox AP |
|---|---|---|
| box_s0 | 33.21 | 39.82 |
| box_s1 | 33.22 | 39.84 |
| mask_s0 | **33.75** | 39.80 |
| mask_s1 | **33.67** | 39.79 |
| **mean delta mask - box** | **+0.49** | **-0.03** |

Paired random-half-subset resampling of val (n=200):

| pair | full-set segm delta | half-subset mean | 95% interval | positive |
|---|---|---|---|---|
| mask_s0 - box_s0 | +0.54 | +0.56 | [+0.29, +0.81] | 99.5% |
| mask_s1 - box_s1 | +0.45 | +0.43 | [+0.09, +0.75] | 99.0% |
| box_s1 - box_s0 (noise) | +0.02 | +0.05 | [-0.20, +0.38] | 60.0% |

Internal validator mask mAP50-95 per epoch (`results/*_results.csv`), delta = mean(mask) - mean(box):

| epoch | box_s0 | box_s1 | mask_s0 | mask_s1 | delta M | delta B |
|---|---|---|---|---|---|---|
| 10 | .2909 | .2923 | .2938 | .2949 | +0.27 | -0.03 |
| 30 | .3046 | .3044 | .3074 | .3082 | +0.33 | -0.06 |
| 50 | .3126 | .3117 | .3164 | .3172 | +0.47 | -0.08 |
| 70 | .3201 | .3198 | .3249 | .3238 | +0.44 | +0.06 |
| 90 | .3272 | .3282 | .3334 | .3327 | +0.53 | +0.08 |
| 100 | .3325 | .3324 | .3374 | .3370 | **+0.47** | -0.01 |

mask mAP50: box .5265/.5247 -> mask .5318/.5336 (+0.71).

Cost: training wall time 9.7/9.8 h (box) vs 12.5/12.3 h (mask), about +28% (assigner ~18 ms/call vs 2 ms, two
branches per step). Inference/export unchanged (training-only change).

## Conclusion

Real but small: +0.5 mask AP, confidence interval excludes zero on both seeds, box AP unchanged, at +28% training time.
About a quarter of the pose gain, because box IoU already tracks mask IoU much better than it tracks OKS.

## Reproduce

```bash
# code
git clone -b seg-assign-maskiou https://github.com/ultralytics/ultralytics.git ult-segassign
uv venv ~/venv-segassign --python 3.12
VIRTUAL_ENV=~/venv-segassign uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV=~/venv-segassign uv pip install -e ult-segassign faster-coco-eval
# data: coco-seg.yaml points at /data/shared-datasets/coco (Ultralytics COCO with polygon labels); yolo26n.pt from assets v8.4.0
~/venv-segassign/bin/python train.py box_s0 box 0 0   # <name> <seg_assign> <seed> <device>
~/venv-segassign/bin/python train.py box_s1 box 1 1
~/venv-segassign/bin/python train.py mask_s0 mask 0 2
~/venv-segassign/bin/python train.py mask_s1 mask 1 3
~/venv-segassign/bin/python coco_pair_seg.py /data/rick/segassign/runs          # official segm/bbox AP + paired CI
~/venv-segassign/bin/python align_diag.py segment runs/mask_s0/weights/best.pt coco-seg.yaml out.npz 0   # score~maskIoU diagnostic
# val.py <name> <device>: only needed if a run lacks predictions.json (training's final COCO val already writes it)
```

Runs and weights: ultra5 `/data/rick/segassign/runs/{box_s0,mask_s0,mask_s1,box_s1(predictions only)}`, box_s1 full
run on ultra4 `/data/rick/segassign/runs/box_s1`.

## Open risks / next steps

- Only yolo26n, short recipe (100 ep). Pose held across n..x with official recipes; seg is untested at other sizes and at
  the official epoch count, and a +0.5 gain at n could shrink.
- +28% train time is mostly the per-image Python loop over prototype decoding; needs vectorizing before a PR.
- `seg_assign_w` was not swept (0.5 copied from pose); w=0.75/1.0 might give more.
- Not tested: YOLOE-seg (`TVPSegmentLoss`), `overlap_mask=False` training (path implemented and spot-checked only), DDP.
- Any PR must come from a fresh branch off main (this branch carries the experiment knobs and this record).
- For a PR, pose and seg `get_box_metrics` overrides should merge into one per-pair quality hook on
  `TaskAlignedAssigner` instead of two near-identical subclasses.
