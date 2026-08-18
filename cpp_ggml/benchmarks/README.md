# ggml inference benchmark and parity report

This directory is the reproducible evidence store for the C++ integration. It compares the same source image and model
input geometry across PyTorch CUDA, ggml CUDA, ggml Vulkan, and ggml CPU.

## Artifacts

- [speed_by_model.png](speed_by_model.png): grouped end-to-end latency by model family and scale.
- [latency_by_backend.png](latency_by_backend.png): full detection-model backend latency comparison using GGML F16.
- [speed_by_dtype.png](speed_by_dtype.png): F32, F16, and Q8_0 comparison for the n/s deployment models.
- [depth_latency.png](depth_latency.png): YOLO26n-depth latency by backend and precision at the 768 input size.
- [speedup_table.md](speedup_table.md): complete model, precision, backend, latency, and PyTorch speedup matrix.
- [parity_grid_bus.png](parity_grid_bus.png) and [parity_grid_zidane.png](parity_grid_zidane.png): rendered detection
  output comparisons.
- [depth_parity_bus.png](depth_parity_bus.png): PyTorch and three-backend absolute-depth output comparison.
- [bench.jsonl](bench.jsonl) and [pytorch.jsonl](pytorch.jsonl): raw measurements. Plot generation keeps the latest
  entry for each model/backend/precision key and ignores legacy C++ rows without a structured `e2e_ms` field.

## Current status

| Integration goal                               | Evidence-based status                                                                                    |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| YOLOv8 n/s/m/l/x detection                     | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLO26 n/s/m/l/x detection                     | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLO26n absolute depth                         | F32/F16/Q8_0 conversion, CPU/CUDA/Vulkan execution, source-size restoration, and focused parity verified |
| CUDA faster than PyTorch CUDA for every model  | Achieved for the F16 deployment path: 11/11 models, x1.04 to x2.01                                       |
| Vulkan close to PyTorch CUDA for every model   | Achieved against the declared x0.70 throughput floor: 11/11 models, x0.70 to x1.32                       |
| Focused numerical parity with official PyTorch | All 10 detectors and absolute depth pass the stated single-image tolerances on CUDA and Vulkan           |
| Dataset-level accuracy for every GGML format   | Not established; focused output parity is not a substitute for COCO and NYU Depth V2 validation          |
| Runtime stability                              | 500-iteration, restart, alternate-resolution, and malformed-model focused checks pass                    |
| Stable production embedding API                | CLI and internal session API are unified, but a versioned installed public C API is still outstanding    |

Performance and numerical equivalence are measurements, not properties that can be guaranteed by documentation. F32
is the parity reference. F16 and Q8_0 deliberately change weight representation and must use declared task-level
tolerances. A production accuracy claim requires COCO validation for detection and NYU Depth V2 validation for depth on
every format/backend combination.

## Measured result on this machine

The evidence store contains 99 unique backend/model/precision keys: all 10 detection models and YOLO26n-depth on CPU,
CUDA, and Vulkan in F32, F16, and Q8_0. The release performance target is F16, the default GPU deployment format. Values
above `x1.00` are faster than PyTorch. Vulkan "close" is explicitly defined here as at least x0.70 PyTorch-CUDA
throughput, equivalent to no more than 1.43 times its latency. The complete per-precision matrix is in
[speedup_table.md](speedup_table.md).

| Model         | PyTorch CUDA | Best GGML CUDA | Speedup | Best GGML Vulkan | Speedup |
| ------------- | -----------: | -------------: | ------: | ---------------: | ------: |
| yolov8n       |      5.70 ms |  4.11 ms (F16) |   x1.39 |    6.44 ms (F16) |   x0.88 |
| yolov8s       |      5.93 ms |  5.37 ms (F16) |   x1.10 |    8.33 ms (F16) |   x0.71 |
| yolov8m       |     10.48 ms |  9.29 ms (F16) |   x1.13 |   14.75 ms (F16) |   x0.71 |
| yolov8l       |     15.16 ms | 13.87 ms (F16) |   x1.09 |   21.60 ms (F16) |   x0.70 |
| yolov8x       |     24.16 ms | 18.33 ms (F16) |   x1.32 |   31.04 ms (F16) |   x0.78 |
| yolo26n       |      7.86 ms |  3.91 ms (F16) |   x2.01 |    5.96 ms (F16) |   x1.32 |
| yolo26s       |      7.96 ms |  5.30 ms (F16) |   x1.50 |    8.36 ms (F16) |   x0.95 |
| yolo26m       |      9.28 ms |  8.38 ms (F16) |   x1.11 |   11.75 ms (F16) |   x0.79 |
| yolo26l       |     12.40 ms | 11.51 ms (F16) |   x1.08 |   15.64 ms (F16) |   x0.79 |
| yolo26x       |     19.98 ms | 19.16 ms (F16) |   x1.04 |   24.65 ms (F16) |   x0.81 |
| yolo26n-depth |     11.75 ms |  6.17 ms (F16) |   x1.90 |   10.43 ms (F16) |   x1.13 |

CUDA has a lower measured mean on all 11 models. The narrowest CUDA margin is YOLO26x at x1.04, while the largest is
YOLO26n at x2.01. Vulkan meets the declared proximity floor on all 11 models; YOLOv8l is the boundary case at x0.702.
These results establish the target on this machine and protocol, not a portable guarantee for other drivers or GPUs.

The checked-in matrix was collected sequentially without concurrent compiler or inference jobs. JSONL rows include min,
p50, p90, and max so residual jitter remains visible. For release decisions, rerun on the target machine rather than
treating this hardware snapshot as a portable performance guarantee.

Focused detection parity uses the exact PyTorch letterbox tensor and compares raw pre-postprocess output for every
detector. F16 mean absolute error is 0.0031-0.0219 on CUDA and 0.0062-0.0313 on Vulkan; all tensor shapes match. Focused
absolute-depth parity on `bus.jpg` compares restored per-pixel meters against official PyTorch output. F16 mean relative
error is 0.104% on CUDA and 0.130% on Vulkan, with p99 relative errors of 0.578% and 0.542%. The focused acceptance limits
are raw detection mean absolute error below 0.04 and depth mean/p99 relative error below 0.2%/0.7%. These checks cover
all integrated models on one image; they are not COCO mAP or NYU Depth V2 accuracy results.

## Methodology

- Input: `ultralytics/assets/bus.jpg`, decoded once before timing.
- Detection geometry: a 480x640 stride-aligned canvas on both engines.
- Depth geometry: the checkpoint's 768 default, with aspect-preserving stride alignment.
- C++ end-to-end latency: letterbox preprocessing + graph compute/readback + NMS or depth restoration.
- PyTorch end-to-end latency: `model.predict()` on the already decoded image, including preprocessing, forward, and
  postprocessing.
- GPU protocol: 20 warmups, 50 timed iterations, and a 3 second cooldown between entries.
- CPU protocol: 8 threads; 10 to 30 timed iterations depending on model size.
- Reported comparison: arithmetic mean in milliseconds per frame; raw min/p50/p90/max remain in JSONL.

File I/O, model loading, GGUF parsing, graph construction, and first-run shader/JIT compilation are excluded from both
steady-state measurements. Use p90 and max, not only the mean, when evaluating latency stability.

## Focused stability checks

The narrow-margin CUDA model, YOLO26x F16, completed 500 consecutive iterations with a 19.33 ms mean, 20.34 ms p90, and
23.31 ms maximum. YOLOv8l F16 completed 500 Vulkan iterations with a 22.01 ms mean and 23.44 ms p90. YOLO26n-depth F16
completed 500 Vulkan iterations with a 9.86 ms mean, 11.42 ms p90, and a stable 2.22-17.84 m range. Alternate-resolution,
fresh-process, and malformed-GGUF checks also pass. These checks cover observed runtime paths; they do not replace
sanitizer, out-of-memory, truncated-model, or multi-device release testing.

## Environment for the checked-in snapshot

| Component    | Value                                                 |
| ------------ | ----------------------------------------------------- |
| CPU          | AMD Ryzen 9 5950X, 16 cores / 32 threads              |
| GPU          | NVIDIA GeForce RTX 3060 12 GB, compute capability 8.6 |
| Driver       | 550.144.03                                            |
| CUDA toolkit | 11.8                                                  |
| cuDNN        | 9.1.0 with cudnn-frontend `936021b`                   |
| ggml         | v0.18.1 (`90951f99`) plus repository patches          |

Results are hardware and software specific. Regenerate them after driver, toolkit, compiler, ggml revision, model,
input-size, or timing-protocol changes.

## Reproduce

Run from `cpp_ggml/`. Build parallelism is intentionally capped at 6.

```bash
cmake -S . -B build-cuda -DYOLO_GGML_CUDA=ON
cmake --build build-cuda --parallel 6
cmake -S . -B build-vulkan -DYOLO_GGML_VULKAN=ON
cmake --build build-vulkan --parallel 6
cmake -S . -B build-cpu
cmake --build build-cpu --parallel 6

bash scripts/convert_all.sh
for backend in cuda vulkan cpu; do
  bash scripts/bench_all.sh "build-$backend" "$backend"
done
python3 scripts/bench_pytorch.py > benchmarks/pytorch.jsonl
python3 scripts/plot_benchmarks.py
python3 scripts/render_parity.py
```

`bench_all.sh` appends so interrupted runs retain completed measurements. Remove or archive the JSONL inputs before a
formal release run when a single clean snapshot is required.

## Optimization boundary

The shipped integration keeps F16 GPU activations resident, reuses preprocessing/output buffers, uses CUDA graph replay,
and performs postprocessing in the logit domain. CUDA convolution owns one complete shape key and caches its measured
choice among NHWC cuDNN Graph Conv+Bias+SiLU, CUTLASS Ampere implicit-GEMM, and legacy cuDNN. This avoids repeating plan
searches while retaining fallback coverage. Mixed F32/F16 transpose convolution deliberately bypasses cuDNN because its
descriptors require one storage type.

Vulkan executes F16 convolution directly from the image tensor without an im2col allocation, fuses Conv+Bias+SiLU,
uses cooperative-matrix tiles selected by output shape, and batches submissions according to graph FLOPs. F16 pool,
scale, upscale, depthwise convolution, and raw contiguous copies remain on device. The measured target is therefore met
without changing warmup counts, excluding preprocessing/readback/postprocess, or selecting only favorable models.

## Remaining integration work

1. Run COCO and NYU Depth V2 validation across F32/F16/Q8_0 and all supported backends with explicit task tolerances.
2. Automate the focused stability checks and extend them with sanitizers, truncated models, OOM, and multi-device cases.
3. Validate the x0.70 Vulkan portability floor on AMD and Intel hardware; the checked-in result covers NVIDIA only.
4. Promote the internal C++ session types to a versioned installed C API only when an embedding consumer requires it.
5. Extend YOLO26 absolute-depth support from n to s/m/l/x when those checkpoints are placed in the canonical model set.
