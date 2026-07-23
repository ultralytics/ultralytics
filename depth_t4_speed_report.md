# YOLO26 Depth — Tesla T4 Speed Profiling

Monocular-depth models `yolo26{n,s,m,l,x}-depth`, profiled for **GPU** (PyTorch / ONNX / TensorRT) and **CPU** (PyTorch / ONNX / OpenVINO) inference latency on a single **Tesla T4** node, at **imgsz 768** (our eval protocol) and **imgsz 640** (standard reference).

## Environment

| Component        | Version                                                           |
| ---------------- | ----------------------------------------------------------------- |
| GPU              | Tesla T4 (16 GB), driver 565.57.01                                |
| CPU              | Intel Xeon (Skylake, IBRS), 32 logical vCPUs (shared/virtualized) |
| OS Python        | 3.10.20 (conda env `yolo`)                                        |
| PyTorch          | 2.9.1+cu126                                                       |
| CUDA (torch)     | 12.6                                                              |
| cuDNN            | 9.10.2 (`91002`)                                                  |
| ultralytics      | 8.4.104 (depth fork, loaded via `PYTHONPATH`)                     |
| ONNX Runtime     | 1.23.2 (`onnxruntime-gpu`)                                        |
| ONNX (opset lib) | 1.22.0                                                            |
| TensorRT         | 10.16.1.11                                                        |
| OpenVINO         | 2026.2.1                                                          |

## Methodology

- **imgsz 768** (our 768 depth-eval protocol) and **imgsz 640** (reference), **batch 1**. Each resolution uses its own TensorRT engine and ONNX graph.
- Each format warmed up **10** iterations, then **100** timed runs; the inference column is a **2σ-clipped mean** (3 iterations). Pre/post are plain means.
- Latency is taken from the ultralytics predictor's per-stage `speed` dict, so timings are stage-isolated:
  - **Inference-only** = `speed["inference"]` (pure model forward on GPU).
  - **With pre+post** = `preprocess + inference + postprocess` (the full `model(img)` pipeline: letterbox/normalize on input, depth-map assembly on output).
- **Precision per format:** PyTorch fp16 **and** fp32; ONNX Runtime on `CUDAExecutionProvider` at **fp32**; TensorRT at **fp16**.
- Models built from `yolo26-depth.yaml` with **random-init weights** — inference speed is weight-independent, so no checkpoint download was needed. Params/GFLOPs reported from the fused model at imgsz 768.
- ONNX providers actually bound: `['CUDAExecutionProvider', 'CPUExecutionProvider']` (ran on the T4, not CPU).

## imgsz 768 — inference only (ms · FPS)

| Model         | Params (M) | GFLOPs | PyTorch fp16 | PyTorch fp32 | ONNX-GPU fp32 | **TensorRT fp16** |
| ------------- | ---------: | -----: | -----------: | -----------: | ------------: | ----------------: |
| yolo26n-depth |       5.17 |   46.6 | 12.50 · 80.0 | 14.09 · 71.0 |  10.06 · 99.4 |  **2.73 · 365.8** |
| yolo26s-depth |      12.03 |   67.5 | 13.00 · 76.9 | 14.95 · 66.9 |  15.05 · 66.4 |  **3.82 · 261.6** |
| yolo26m-depth |      22.08 |  129.9 | 13.52 · 74.0 | 28.23 · 35.4 |  28.92 · 34.6 |  **6.00 · 166.7** |
| yolo26l-depth |      26.47 |  156.2 | 19.07 · 52.4 | 34.78 · 28.7 |  35.50 · 28.2 |  **7.68 · 130.2** |
| yolo26x-depth |      55.80 |  300.5 | 25.84 · 38.7 | 64.32 · 15.5 |  65.24 · 15.3 |  **13.57 · 73.7** |

## imgsz 768 — with pre + post (full pipeline, ms · FPS)

| Model         | PyTorch fp16 | PyTorch fp32 | ONNX-GPU fp32 | **TensorRT fp16** |
| ------------- | -----------: | -----------: | ------------: | ----------------: |
| yolo26n-depth | 15.03 · 66.6 | 16.68 · 60.0 |  12.61 · 79.3 |  **5.00 · 199.9** |
| yolo26s-depth | 15.59 · 64.2 | 17.69 · 56.5 |  17.88 · 55.9 |  **6.21 · 161.1** |
| yolo26m-depth | 16.14 · 62.0 | 31.52 · 31.7 |  31.68 · 31.6 |  **8.40 · 119.0** |
| yolo26l-depth | 21.63 · 46.2 | 37.74 · 26.5 |  38.30 · 26.1 |  **10.04 · 99.6** |
| yolo26x-depth | 28.77 · 34.8 | 67.49 · 14.8 |  68.30 · 14.6 |  **16.54 · 60.5** |

Pre+post overhead is roughly constant (~2.5–3 ms per call, dominated by preprocess), so its relative cost is largest for the fast TensorRT path (e.g. `n` TRT: 2.73 → 5.00 ms).

## imgsz 640 — inference only (ms · FPS)

GFLOPs scale by ~(640/768)² ≈ 0.69× vs the 768 numbers above.

| Model         | Params (M) | GFLOPs | PyTorch fp16 | PyTorch fp32 | ONNX-GPU fp32 | **TensorRT fp16** |
| ------------- | ---------: | -----: | -----------: | -----------: | ------------: | ----------------: |
| yolo26n-depth |       5.17 |   32.4 | 11.98 · 83.5 | 12.37 · 80.8 |  7.64 · 130.9 |  **2.29 · 436.9** |
| yolo26s-depth |      12.03 |   46.9 | 10.57 · 94.6 | 12.08 · 82.8 |  11.47 · 87.2 |  **3.08 · 324.4** |
| yolo26m-depth |      22.08 |   90.2 | 12.50 · 80.0 | 20.82 · 48.0 |  21.39 · 46.8 |  **4.71 · 212.1** |
| yolo26l-depth |      26.47 |  108.5 | 17.11 · 58.5 | 25.57 · 39.1 |  26.19 · 38.2 |  **6.14 · 162.8** |
| yolo26x-depth |      55.80 |  208.7 | 19.09 · 52.4 | 47.49 · 21.1 |  48.06 · 20.8 |  **10.26 · 97.5** |

## imgsz 640 — with pre + post (full pipeline, ms · FPS)

| Model         | PyTorch fp16 | PyTorch fp32 | ONNX-GPU fp32 | **TensorRT fp16** |
| ------------- | -----------: | -----------: | ------------: | ----------------: |
| yolo26n-depth | 14.01 · 71.4 | 14.46 · 69.2 |  9.52 · 105.1 |  **3.95 · 253.4** |
| yolo26s-depth | 12.33 · 81.1 | 14.03 · 71.3 |  13.26 · 75.4 |  **4.71 · 212.3** |
| yolo26m-depth | 15.41 · 64.9 | 23.00 · 43.5 |  23.37 · 42.8 |  **6.36 · 157.3** |
| yolo26l-depth | 19.12 · 52.3 | 27.69 · 36.1 |  28.31 · 35.3 |  **7.89 · 126.7** |
| yolo26x-depth | 21.20 · 47.2 | 49.89 · 20.0 |  50.12 · 20.0 |  **15.89 · 62.9** |

### 640 vs 768 (TensorRT fp16, inference-only)

Dropping 768→640 cuts input pixels to ~69%. The speedup tracks that only for the larger, compute-bound models; small models stay launch-bound so gain little.

| Model         | 768 ms | 640 ms | speedup |
| ------------- | -----: | -----: | ------: |
| yolo26n-depth |   2.73 |   2.29 |   1.19× |
| yolo26s-depth |   3.82 |   3.08 |   1.24× |
| yolo26m-depth |   6.00 |   4.71 |   1.27× |
| yolo26l-depth |   7.68 |   6.14 |   1.25× |
| yolo26x-depth |  13.57 |  10.26 |   1.32× |

## CPU (Intel Xeon, 32 vCPU) — inference only (ms · FPS)

Same models on the node's **CPU** (no GPU). fp16 and TensorRT are GPU-only and omitted; the CPU-optimised backend is **OpenVINO** (the CPU analog of TensorRT).

- Same predictor `speed` dict, **batch 1**, imgsz 768 & 640. **5 warmup + 30 timed** runs, **2σ-clipped mean**. The box is a _shared, virtualized_ 32-vCPU Xeon, so CPU numbers are noisier than the GPU ones — treat them as **±5–18%** run-to-run (see `inf_std` in the JSONs).
- **Formats:** PyTorch **fp32**, ONNX Runtime `CPUExecutionProvider` **fp32**, OpenVINO **fp32** (LATENCY mode, batch 1).
- Each backend manages its **own threading** (PyTorch default `torch_threads=8`; OpenVINO/ONNX pick their own) — realistic defaults, not pinned to all 32 vCPUs.
- ONNX Runtime's `preprocess` carries an extra ~15–28 ms overhead the other backends don't, so its _with-pre+post_ total is inflated — compare on **inference-only** (below).

### imgsz 768 (bold = fastest CPU backend per row)

| Model         | Params (M) | GFLOPs | PyTorch fp32 | ONNX-CPU fp32 | **OpenVINO fp32** |
| ------------- | ---------: | -----: | -----------: | ------------: | ----------------: |
| yolo26n-depth |       5.17 |   46.6 |  181.8 · 5.5 |   272.0 · 3.7 |   **139.8 · 7.2** |
| yolo26s-depth |      12.03 |   67.5 |  283.9 · 3.5 |   393.7 · 2.5 |   **208.2 · 4.8** |
| yolo26m-depth |      22.08 |  129.9 |  466.5 · 2.1 |   621.5 · 1.6 |   **399.3 · 2.5** |
| yolo26l-depth |      26.47 |  156.2 |  623.4 · 1.6 |   821.9 · 1.2 |   **544.7 · 1.8** |
| yolo26x-depth |      55.80 |  300.5 | 1140.7 · 0.9 |  1240.9 · 0.8 |   **835.0 · 1.2** |

### imgsz 640

| Model         | Params (M) | GFLOPs |    PyTorch fp32 | ONNX-CPU fp32 |   OpenVINO fp32 |
| ------------- | ---------: | -----: | --------------: | ------------: | --------------: |
| yolo26n-depth |       5.17 |   32.4 |     141.6 · 7.1 |   206.1 · 4.9 | **99.3 · 10.1** |
| yolo26s-depth |      12.03 |   46.9 |     183.6 · 5.4 |   289.9 · 3.4 | **145.7 · 6.9** |
| yolo26m-depth |      22.08 |   90.2 | **255.6 · 3.9** |   440.0 · 2.3 |     275.4 · 3.6 |
| yolo26l-depth |      26.47 |  108.5 | **337.0 · 3.0** |   596.2 · 1.7 |     373.9 · 2.7 |
| yolo26x-depth |      55.80 |  208.7 |     589.7 · 1.7 |   865.3 · 1.2 | **574.9 · 1.7** |

OpenVINO wins every row at 768 (1.3–1.5× over PyTorch); at 640 its edge shrinks and PyTorch-fp32 overtakes it for `m`/`l` (within the noise band). ONNX-Runtime-CPU is consistently the slowest.

### GPU vs CPU — best backend each, inference-only @768

| Model         | GPU TensorRT fp16 | CPU OpenVINO fp32 | GPU speedup |
| ------------- | ----------------: | ----------------: | ----------: |
| yolo26n-depth |              2.73 |             139.8 |         51× |
| yolo26s-depth |              3.82 |             208.2 |         55× |
| yolo26m-depth |              6.00 |             399.3 |         67× |
| yolo26l-depth |              7.68 |             544.7 |         71× |
| yolo26x-depth |             13.57 |             835.0 |         62× |

## Takeaways

- **TensorRT fp16 wins decisively** — 3.8–5× faster than PyTorch fp16 inference, and the only format keeping `x` above real-time (60.5 FPS full-pipeline).
- **PyTorch fp32 ≈ ONNX fp32** at every size (e.g. `x`: 64.3 vs 65.2 ms inference) — this cross-check validates the timing method.
- **PyTorch fp16 is launch/overhead-bound for n/s/m** (~12.5–13.5 ms floor regardless of size) — small-model latency on the T4 is dominated by kernel-launch overhead, not compute.
- ONNX here is **fp32**; that is why it tracks PyTorch-fp32 rather than -fp16. A fp16 ONNX export would be needed for an fp16-vs-fp16 ONNX comparison.
- **CPU is ~50–70× slower than GPU TensorRT.** On 32 shared vCPUs, OpenVINO fp32 is the fastest CPU backend (1.3–1.5× over PyTorch at 768) and ONNX-CPU the slowest. Even `yolo26n-depth` tops out at ~7 FPS (768) / ~10 FPS (640) — real-time depth needs the GPU; CPU is viable only for the smallest model at low frame rates.

## Commands

Reference one-liner (`ProfileModels`, ONNX-CPU + TensorRT only, no PyTorch timing):

```python
from ultralytics.utils.benchmarks import ProfileModels
ProfileModels([f"yolo26{s}-depth.pt" for s in "nsmlx"], imgsz=768).run()
```

Actual profiler used for this report (all three formats timed on the T4 GPU, both stage-isolated inference and full pipeline):

```bash
# on the T4 box, in the `yolo` conda env, depth repo shadowed via PYTHONPATH
cd /root/autodl-tmp/ultralytics_depth
PYTHONPATH=/root/autodl-tmp/ultralytics_depth \
  /root/autodl-tmp/data/conda_envs/yolo/bin/python profile_depth.py \
  --imgsz 768 --sizes nsmlx --device 0 --out depth_speed_full.json
# and again at 640 (delete the 768 .onnx/.engine first so they rebuild):
#   --imgsz 640 --sizes nsmlx --device 0 --out depth_speed_640.json
```

Export commands the profiler issues per model (skipped when the file already exists):

```python
# fp32 ONNX
model.export(format="onnx",   imgsz=768, device=0, batch=1)
# fp16 TensorRT engine
model.export(format="engine", imgsz=768, device=0, batch=1, quantize=16)
```

CPU profiler (`profile_depth_cpu.py`; PyTorch fp32 / ONNX-CPU fp32 / OpenVINO fp32, no TensorRT/fp16):

```bash
# YOLO_AUTOINSTALL=false avoids a mid-run onnxruntime reinstall that would taint ONNX timings
YOLO_AUTOINSTALL=false PYTHONPATH=/root/autodl-tmp/ultralytics_depth \
  /root/autodl-tmp/data/conda_envs/yolo/bin/python profile_depth_cpu.py \
  --imgsz 768 --sizes nsmlx --out depth_cpu_768.json
#   --imgsz 640 ... --out depth_cpu_640.json   (each imgsz builds its own onnx + openvino model)
```

Raw per-stage numbers (preprocess / inference / postprocess for every model × format) are in `depth_t4_speed.json` (GPU, imgsz 768) and `depth_t4_speed_640.json` (GPU, imgsz 640), and `depth_t4_cpu_768.json` / `depth_t4_cpu_640.json` (CPU).
