# YOLO-FastMoving

YOLO detection of small fast-moving objects (a tennis ball) from temporal input, the job TrackNet does with heatmaps.
`ultralytics/data/temporal.py` adds `TemporalYOLODataset`: when a data YAML has a `temporal` section, each image is the
current frame stacked with motion channels built from the preceding frames, inside `load_image`, so mosaic/affine/flip
apply identically to every channel and the model, loss and trainer are unchanged (only `channels` differs).

| `temporal.mode` | channels (`frames: 3`) | input                                      |
| --------------- | ---------------------- | ------------------------------------------ |
| `rgb`           | 3                      | current frame only (ablation)              |
| `diff`          | 5                      | BGR(t), \|g(t)−g(t−1)\|, \|g(t−1)−g(t−2)\| |
| `stack`         | 9                      | BGR(t), BGR(t−1), BGR(t−2), TrackNet-style |

## Reproduce

```bash
# images extracted to <out>/images from tennis_dataset:dataset/flyingball_2026-03-15/flyingball_images_2026-03-15.zip
python prepare_flyingball.py --ann <out>/images/annotations.json --out <out>
yolo detect train model=yolo26n.yaml pretrained=yolo26n.pt data=<out>/flyingball-diff.yaml imgsz=1280 epochs=50 batch=32 hsv_h=0 hsv_s=0 hsv_v=0
python eval_fastmoving.py --weights runs/detect/train/weights/best.pt --data <out>/flyingball-diff.yaml   # add --gpu-prep for the deployment path
python ../../../tracknet/train.py --dataset_root <out>/tracknet --model tracknet --seq_len 3 --sigma 10 --epochs 50  # baseline, same split
yolo export model=best.pt format=engine imgsz=736,1280 quantize=fp16 batch=1
python bench_speed.py --yolo best.pt:diff:1280 best.engine:diff:1280 --tracknet-repo ../../../tracknet
```

## Data (flyingball_2026-03-15)

53,248 labelled 1920×1080 frames, 136 clips, two sources (Wimbledon 2022 broadcast, ZQW 2025). Ball median ~8 px.
Only labelled frames are in the zip, so diff history is the neighbouring labelled frame and is zero at a track start.
Video-level split stratified by source, seed 0: train 107 videos / 42,881 frames, val 26 videos / 8,933 frames. Every
val frame contains a ball. TrackNet was retrained on the identical split and labels (`<out>/tracknet`).

**Label alignment differs per source.** Label Studio frame k shows the ball at image frame k−2 for ZQW and k−3 for
Wimbledon: the median in-box motion ratio is 148 at −2 vs ≤4 at neighbouring offsets for ZQW, and 94 at −3 vs ≤3 for
Wimbledon (`measure_offsets`). The TrackNet repo applied −3 to both, i.e. a third of its labels were one frame off, so
its published F1@10 0.952 is not comparable to the table below.

## Accuracy (val, TrackNet protocol)

One prediction per frame (top-confidence box center / heatmap centroid), scaled to 512×288; TP if within tol px, else
FP+FN. `conf 0.25` is fixed; `best conf` sweeps 0.05–0.95 on val by F1@10 (always picked 0.05, the lowest, since val has
no negatives). TrackNet's numbers are its best epoch by val F1@10. Both sides are therefore selected on val; there is no
held-out test split.

| model                | input         | F1@4 / F1@10 conf 0.25 | F1@4 / F1@10 best conf |
| -------------------- | ------------- | ---------------------- | ---------------------- |
| TrackNet (11.3M)     | 3×RGB 288×512 | —                      | 0.900 / 0.958          |
| TrackNetDeep (46.7M) | 3×RGB 288×512 | —                      | 0.911 / 0.960          |
| YOLO26n rgb (2.4M)   | 1280          | 0.914 / 0.941          | 0.916 / 0.953          |
| YOLO26n diff         | 1280          | 0.915 / 0.943          | 0.925 / 0.962          |
| YOLO26n stack        | 1280          | 0.916 / 0.940          | **0.933 / 0.964**      |
| YOLO26n-p2 diff      | 1280          | 0.914 / 0.948          | 0.925 / 0.962          |
| YOLO26s diff (9.5M)  | 1280          | **0.917 / 0.950**      | 0.929 / **0.966**      |
| YOLO26n diff         | 640           | 0.784 / 0.823          | 0.851 / 0.892          |

YOLO rows are `best.pt`; `last.pt` is within ±0.004 everywhere. `--gpu-prep` (torch resize/gray/diff on the GPU) matches
the cv2 path within 0.0002 F1 (YOLO26n-p2 diff).

- YOLO at 1280 beats TrackNet at the tight tolerance (F1@4 +0.01–0.03) and ties/edges it at F1@10 (best conf).
- Motion input helps, but modestly: rgb → diff/stack is +0.01–0.02 at best conf, ~0 at conf 0.25. Raw frame stacking is
  as good as or better than explicit differences.
- Resolution matters most: at 640 the ball is ~2.7 px and F1@4 drops by ~0.07–0.13.

## Speed (RTX PRO 6000 Blackwell, idle GPU, batch 1, FP16, median)

End-to-end per frame from a raw 1080p frame: resize, temporal channels, forward, top-1. YOLO is one call per frame.
TrackNet consumes 3 new frames per call and emits 3 heatmaps, so its per-frame cost is per-call/3 (at the price of up to
2 frames of extra latency).

| model                | TRT forward        | TRT e2e, GPU prep            | PyTorch e2e, cv2 prep |
| -------------------- | ------------------ | ---------------------------- | --------------------- |
| TrackNet             | 0.87 ms / 3 frames | **0.45 ms/frame (2213 FPS)** | 1.30 ms/frame         |
| TrackNetDeep         | 1.15 ms / 3 frames | 0.55 ms/frame (1832 FPS)     | 1.56 ms/frame         |
| YOLO26n diff 1280    | 0.53 ms            | 0.79 ms (1266 FPS)           | 10.4 ms               |
| YOLO26n-p2 diff 1280 | 0.76 ms            | 1.04 ms (964 FPS)            | 10.8 ms               |
| YOLO26s diff 1280    | 0.90 ms            | 1.24 ms (806 FPS)            | 10.3 ms               |
| YOLO26n diff 640     | 0.46 ms            | 0.63 ms (1591 FPS)           | 4.7 ms                |

The CPU (cv2) preprocessing of a 1080p frame (~6–8 ms) dominates the PyTorch column; deploy with GPU preprocessing.
`rgb`/`stack` GPU-prep paths are not implemented in the bench; their TRT forwards are 0.55/0.57 ms.
