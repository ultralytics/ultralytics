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

| Integration goal                                        | Evidence-based status                                                                                    |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| YOLOv8 n/s/m/l/x detection                              | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLO26 n/s/m/l/x detection                              | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLO26n absolute depth                                  | F32/F16/Q8_0 conversion, CPU/CUDA/Vulkan execution, source-size restoration, and focused parity verified |
| CUDA faster than PyTorch CUDA for every model           | Partially achieved: 6/11 measured means win; the other 5 are within 10% at their best precision          |
| Vulkan close to PyTorch CUDA for every model            | Not achieved; best per-model speedups are x0.31 to x0.72 on this NVIDIA GPU                              |
| Same accuracy as official PyTorch for every GGML format | Not established; focused output parity is not a substitute for full dataset validation                   |
| Runtime stability                                       | 500-iteration, restart, alternate-resolution, and malformed-model focused checks pass                    |
| Stable production embedding API                         | CLI and internal session API are unified, but a versioned installed public C API is still outstanding    |

Performance and numerical equivalence are measurements, not properties that can be guaranteed by documentation. F32
is the parity reference. F16 and Q8_0 deliberately change weight representation and must use declared task-level
tolerances. A production accuracy claim requires COCO validation for detection and NYU Depth V2 validation for depth on
every format/backend combination.

## Measured result on this machine

The refreshed matrix contains 99 unique backend/model/precision keys: all 10 detection models and YOLO26n-depth on CPU,
CUDA, and Vulkan in F32, F16, and Q8_0. Values above `x1.00` are faster than PyTorch. The best precision for each GPU
backend is shown below; the complete per-precision matrix is in [speedup_table.md](speedup_table.md).

| Model         | PyTorch CUDA | Best GGML CUDA | Speedup | Best GGML Vulkan | Speedup |
| ------------- | -----------: | -------------: | ------: | ---------------: | ------: |
| yolov8n       |      5.90 ms |  4.36 ms (F32) |   x1.35 |    9.90 ms (F32) |   x0.60 |
| yolov8s       |      6.68 ms |  6.71 ms (F32) |  x0.995 |   16.70 ms (F32) |   x0.40 |
| yolov8m       |     11.09 ms | 11.24 ms (F16) |   x0.99 |   32.67 ms (F32) |   x0.34 |
| yolov8l       |     16.19 ms | 17.27 ms (F16) |   x0.94 |   48.88 ms (F32) |   x0.33 |
| yolov8x       |     23.89 ms | 23.25 ms (F16) |   x1.03 |   60.52 ms (F32) |   x0.39 |
| yolo26n       |      8.01 ms |  4.48 ms (F32) |   x1.79 |   11.17 ms (F32) |   x0.72 |
| yolo26s       |      8.00 ms |  6.84 ms (F32) |   x1.17 |   16.75 ms (F32) |   x0.48 |
| yolo26m       |     10.05 ms | 10.91 ms (F16) |   x0.92 |   32.00 ms (F32) |   x0.31 |
| yolo26l       |     12.33 ms | 13.57 ms (F16) |   x0.91 |   38.52 ms (F32) |   x0.32 |
| yolo26x       |     21.21 ms | 21.13 ms (F16) |  x1.004 |   53.09 ms (F32) |   x0.40 |
| yolo26n-depth |     11.27 ms |  9.12 ms (F16) |   x1.24 |   21.06 ms (F32) |   x0.53 |

CUDA has a lower measured mean on 6 of 11 models; the YOLO26x margin is only 0.4% and should be treated as run-to-run
noise rather than a robust advantage. All five misses are within 10% using the best measured precision. This still does
not satisfy an all-model pass criterion. Vulkan remains outside a reasonable "close" band on this GPU and is therefore
also marked not achieved.

The checked-in matrix was collected sequentially without concurrent compiler or inference jobs. JSONL rows include min,
p50, p90, and max so residual jitter remains visible. For release decisions, rerun on the target machine rather than
treating this hardware snapshot as a portable performance guarantee.

Focused absolute-depth parity on `bus.jpg` compares restored per-pixel meters against official PyTorch output. F32
mean relative error was 0.089% on CPU, 0.110% on CUDA, and 0.127% on Vulkan; F16 remained below 0.13%, while Q8_0 was
0.23% to 0.32%. These are single-image numerical checks, not NYU Depth V2 accuracy results.

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

YOLO26n-depth F16 completed 500 consecutive CUDA iterations with a 9.49 ms mean, 10.24 ms p90, and 11.06 ms maximum.
Three fresh-process 50-iteration runs completed with 9.23-9.80 ms means and consistent 2.22-17.88 m output ranges. A
second image geometry exercised a 768x448 graph for 100 iterations without failure, and `/dev/null` was rejected as an
invalid GGUF with a nonzero exit code. These checks cover the observed runtime paths; they do not replace sanitizer,
out-of-memory, truncated-model, or multi-device release testing.

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

The shipped integration amortizes the CPU thread pool, uses CUDA graph replay, performs postprocessing in the logit
domain, parallelizes preprocessing and CPU im2col, and fixes quantized Vulkan im2col scheduling. F32/F16 CUDA convolution
now uses timed cuDNN algorithms with custom direct-kernel fallback; supported cuDNN frontend plans also fuse
Conv+Bias+SiLU. Mixed F32/F16 transpose convolution deliberately bypasses cuDNN because its descriptors require a single
storage type.

The remaining CUDA gap is architectural: NCHW limits available fused cuDNN plans, while several attention and layout
operations still require standalone kernels. The next credible path is an end-to-end NHWC activation layout or tuned
CUTLASS implicit-GEMM convolution, selected by measured shape rather than a blanket replacement. Vulkan still lowers
most convolution through im2col plus matrix multiplication and needs a native convolution implementation. Tuning
warmups, excluding preprocessing, or selecting only favorable models would not satisfy the target.

## Remaining integration work

1. Prototype NHWC/cuDNN and CUTLASS CUDA paths, retain only plans that improve the five remaining models, and rerun all 11.
2. Implement or upstream direct Vulkan convolution and demonstrate a declared proximity threshold on the full matrix.
3. Run COCO and NYU Depth V2 validation across F32/F16/Q8_0 and all supported backends with explicit tolerances.
4. Automate the focused stability checks and extend them with sanitizers, truncated models, OOM, and multi-device cases.
5. Promote the internal C++ session types to a versioned installed C API only when an embedding consumer requires it.
6. Extend YOLO26 absolute-depth support from n to s/m/l/x when those checkpoints are placed in the canonical model set.
