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
python <tracknet-repo>/train.py --dataset_root <out>/tracknet --model tracknet --seq_len 3 --sigma 10 --epochs 50  # baseline, same split
yolo export model=best.pt format=engine imgsz=736,1280 quantize=fp16 batch=1
python bench_speed.py --yolo best.pt:diff:1280 best.engine:diff:1280 --tracknet-repo <tracknet-repo>
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
FP+FN. **Fixed threshold** is the fair comparison: YOLO conf 0.25, TrackNet its built-in heatmap threshold 0.3.
`Tuned conf` additionally sweeps YOLO's threshold 0.05–0.95 on val by F1@10; it always picked 0.05 (the lowest, since val
has no negatives), so that column is optimistic for YOLO. TrackNet's epoch is chosen by val F1@10 and YOLO's `best.pt`
by val mAP; there is no held-out test split. TrackNet scores 8,910 of the 8,933 val frames (non-overlapping 3-frame
windows), YOLO all of them.

| model                | input         | F1@4 / F1@10 fixed thr | F1@4 / F1@10 tuned conf |
| -------------------- | ------------- | ---------------------- | ----------------------- |
| TrackNet (11.3M)     | 3×RGB 288×512 | 0.900 / 0.958          | —                       |
| TrackNetDeep (46.7M) | 3×RGB 288×512 | 0.911 / **0.960**      | —                       |
| YOLO26n rgb (2.4M)   | 1280          | 0.914 / 0.941          | 0.916 / 0.953           |
| YOLO26n diff         | 1280          | 0.915 / 0.943          | 0.925 / 0.962           |
| YOLO26n stack        | 1280          | 0.916 / 0.940          | **0.933 / 0.964**       |
| YOLO26n-p2 diff      | 1280          | 0.914 / 0.948          | 0.925 / 0.962           |
| YOLO26s diff (9.5M)  | 1280          | **0.917 / 0.950**      | 0.929 / **0.966**       |
| YOLO26n diff         | 640           | 0.784 / 0.823          | 0.851 / 0.892           |

YOLO rows are `best.pt`; `last.pt` is within ±0.004 everywhere. `--gpu-prep` (torch resize/gray/diff on the GPU) matches
the cv2 path within 0.0002 F1 (YOLO26n-p2 diff).

- At a fixed threshold YOLO at 1280 wins the tight tolerance (F1@4 0.915–0.917 vs 0.900/0.911) and loses the loose one
  by ~0.01 (F1@10 0.940–0.950 vs 0.958/0.960): it localises more precisely but drops more frames below threshold.
  With a val-tuned threshold YOLO reaches F1@10 0.962–0.966.
- Motion input barely matters for this metric: rgb/diff/stack are 0.914/0.915/0.916 F1@4 at the fixed threshold, inside
  single-seed noise; at the tuned threshold diff/stack gain +0.01–0.02 F1@10 over rgb. Raw frame stacking is as good as
  explicit differences.
- Resolution matters most: at 640 the ball is ~2.7 px and F1@4 drops by 0.13 (fixed) / 0.07 (tuned).

## Speed (RTX PRO 6000 Blackwell, idle GPU, batch 1, FP16, median)

End-to-end per frame from a raw 1080p frame: resize, temporal channels, forward, top-1. YOLO is one call per frame.
TrackNet consumes 3 new frames per call and emits 3 heatmaps, so its per-frame cost is per-call/3 (at the price of up to
2 frames of extra latency). Run one call per frame (sliding window, no added latency) it costs the full call: 0.87 ms
TRT forward vs 0.53 ms for YOLO26n at 1280, on 6.4× fewer pixels.

| model                | TRT forward        | TRT e2e, GPU prep            | PyTorch e2e, cv2 prep |
| -------------------- | ------------------ | ---------------------------- | --------------------- |
| TrackNet             | 0.87 ms / 3 frames | **0.45 ms/frame (2213 FPS)** | 1.30 ms/frame         |
| TrackNetDeep         | 1.15 ms / 3 frames | 0.55 ms/frame (1832 FPS)     | 1.56 ms/frame         |
| YOLO26n diff 1280    | 0.53 ms            | 0.79 ms (1266 FPS)           | 10.4 ms               |
| YOLO26n-p2 diff 1280 | 0.76 ms            | 1.04 ms (964 FPS)            | 10.8 ms               |
| YOLO26s diff 1280    | 0.90 ms            | 1.24 ms (806 FPS)            | 10.3 ms               |
| YOLO26n diff 640     | 0.46 ms            | 0.63 ms (1591 FPS)           | 4.7 ms                |
| YOLO26n stack 1280   | 0.57 ms            | —                            | 11.8 ms (TRT: 9.8 ms) |

The CPU (cv2) preprocessing of a 1080p frame (~6–8 ms) dominates the PyTorch column; deploy with GPU preprocessing.
`rgb`/`stack` GPU-prep paths are not implemented in the bench (`rgb` TRT forward 0.55 ms).

## Caveats

- No held-out test split; single seed per config, so sub-0.01 differences (rgb vs diff vs stack) are not significant.
- 1,449 labels whose shifted frame is not in the zip (track ends) are excluded from train and val for both models; 46
  frames with two balls keep one.
- HSV augmentation was disabled for every run (it only applies to 3-channel images), so no model saw colour
  augmentation; expect a gap on new courts/lighting. Only two sources, val clips come from the same matches as train.

## TrackNet paper dataset (10 games, 19,835 frames)

The tennis dataset released with TrackNet (`Dataset.zip`, Google Drive id `1DQ3ZbvokTsgOq6x-ay6O8U2W4a8e3LFw`, also on
`tennis_dataset:dataset/tracknet/`): 10 broadcast games, 95 clips, 1280×720, per-frame ball center + visibility (0 none,
1 easy, 2 hard, 3 occluded). Frames are contiguous within each clip, 729 frames have no ball (negatives). Labels have no
ball extent, so YOLO boxes are a fixed 10 px square (`prepare_tracknet_paper.py --box`).

This is **not** the data behind the TrackNet paper's headline table (P 99.8 / R 96.6 / F1 98.2): that was a single
20,844-frame match with a random 70/30 frame split. The comparable published numbers on this dataset are WASB (BMVC
2023): games 1–7 train, 8–10 test (5,675 frames), 288×512 grid, τ = 4 px.

```bash
python prepare_tracknet_paper.py --src Dataset --out tennis_tracknet
yolo detect train model=yolo26n.yaml pretrained=yolo26n.pt data=tennis_tracknet/tennis-wasb-diff.yaml imgsz=1280 epochs=100 batch=32 hsv_h=0 hsv_s=0 hsv_v=0
python eval_fastmoving.py --weights last.pt --data tennis_tracknet/tennis-wasb-diff.yaml
python eval_fastmoving.py --tracknet <tracknet-repo>:TrackNet_cur.pt --conf 0.3 --data tennis_tracknet/tennis-wasb-diff.yaml
```

WASB has no validation split (games 8–10 are the test set), so the primary numbers are `last.pt` / last epoch with a
fixed threshold (YOLO conf 0.25, TrackNet heatmap 0.3) and threshold-free AP; nothing is selected on the test set.
TrackNet was retrained here on the same split (100 epochs, seq 3, σ 10) and scored by the same script (causal window
[t−2, t−1, t], heatmap of t). All YOLO runs: YOLO26n, imgsz 1280, 100 epochs.

| model (games 8–10, τ = 4 px @ 512×288)       | F1        | P     | R     | Acc   | AP        | recall easy / hard |
| -------------------------------------------- | --------- | ----- | ----- | ----- | --------- | ------------------ |
| TrackNetV2 (WASB paper)                      | 0.894     |       |       | 0.814 | 0.806     |                    |
| MonoTrack (WASB paper)                       | 0.921     |       |       | 0.859 | 0.873     |                    |
| WASB step 1 (WASB paper)                     | **0.956** |       |       | 0.918 | 0.942     |                    |
| TrackNet, retrained here                     | 0.897     | 0.931 | 0.865 | 0.863 | 0.837     | 0.889 / 0.644      |
| YOLO26n rgb (current frame only)             | 0.731     | 0.868 | 0.631 | 0.641 | 0.670     | 0.662 / 0.335      |
| YOLO26n stack (3×RGB)                        | 0.920     | 0.990 | 0.859 | 0.862 | 0.944     | 0.887 / 0.607      |
| **YOLO26n diff (RGB + 2 frame differences)** | 0.936     | 0.984 | 0.892 | 0.890 | **0.947** | 0.927 / 0.573      |

- The retrained TrackNet lands on the published TrackNetV2 numbers (F1 0.897 vs 0.894, AP 0.837 vs 0.806), so this
  protocol matches WASB's.
- YOLO26n-diff beats TrackNet by +0.04 F1 / +0.11 AP and MonoTrack on both; it trails WASB's F1 by 0.02 but edges its AP.
  Its errors are mostly misses at conf 0.25 (P 0.98, R 0.89), concentrated on hard-visibility frames.
- **Motion input matters across games**: rgb → diff is +0.20 F1 / +0.28 AP on unseen courts, whereas on flyingball
  (val clips from the training matches) it was within noise. Explicit differences ≥ raw frame stacking.
- Thresholds are fragile on unseen games: the stack run's mAP-selected `best.pt` (epoch ~60) scores F1 0.43 at conf 0.25
  with AP 0.949 — its confidences sit below 0.25. Prefer AP, or calibrate the threshold on held-out games. YOLO's
  in-training mAP (box IoU against fixed 10 px boxes) also declines after ~epoch 35 while F1/AP keep improving, so it is
  a poor checkpoint selector here.

**Paper-style random 70/30 frame split** (`tennis-paper-diff.yaml`, 5,980 test frames): YOLO26n-diff F1 0.969, P 0.995,
R 0.943, AP 0.984 (identical at 640×360/5 px and 512×288/4 px, the same tolerance geometrically). The same model scores
0.936 on unseen games, so ~0.03 F1 comes from adjacent frames leaking between train and test; the paper's 98.2 is from
that kind of split on a different, single-match dataset.

Speed is the same as above (same YOLO26n, 1280×736): 0.79 ms/frame end-to-end with TensorRT FP16 and GPU preprocessing.
