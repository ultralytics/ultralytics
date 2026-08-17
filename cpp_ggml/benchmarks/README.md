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

| Integration goal                                         | Evidence-based status                                                                                    |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| YOLOv8 n/s/m/l/x detection                               | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLO26 n/s/m/l/x detection                               | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLO26n absolute depth                                   | F32/F16/Q8_0 conversion, CPU/CUDA/Vulkan execution, source-size restoration, and focused parity verified |
| CUDA and Vulkan faster than PyTorch CUDA for every model | Not achieved; larger models remain slower because ggml convolution lowers through im2col plus GEMM       |
| Same accuracy as official PyTorch for every GGML format  | Not established; focused output parity is not a substitute for full dataset validation                   |
| Stable production embedding API                          | CLI and internal session API are unified, but a versioned installed public C API is still outstanding    |

Performance and numerical equivalence are measurements, not properties that can be guaranteed by documentation. F32
is the parity reference. F16 and Q8_0 deliberately change weight representation and must use declared task-level
tolerances. A production accuracy claim requires COCO validation for detection and NYU Depth V2 validation for depth on
every format/backend combination.

## Measured result on this machine

The refreshed matrix contains 99 unique backend/model/precision keys: all 10 detection models and YOLO26n-depth on CPU,
CUDA, and Vulkan in F32, F16, and Q8_0. It does not meet the requested universal speed target: every measured
GGML/PyTorch mean ratio is below `x1.00`, where values above one would be faster. Representative end-to-end means are:

| Model                  | PyTorch CUDA F32 | GGML CUDA F16 | GGML Vulkan F16 | GGML CPU F16, 8T |
| ---------------------- | ---------------: | ------------: | --------------: | ---------------: |
| YOLOv8n detection      |          6.09 ms |      12.34 ms |        16.69 ms |         72.22 ms |
| YOLOv8x detection      |         27.19 ms |      56.87 ms |        97.49 ms |        920.85 ms |
| YOLO26n detection      |          8.49 ms |      16.72 ms |        20.36 ms |         69.61 ms |
| YOLO26x detection      |         21.97 ms |      79.61 ms |        92.97 ms |        907.17 ms |
| YOLO26n absolute depth |         16.71 ms |      51.86 ms |        35.29 ms |        225.62 ms |

The workstation was shared with other compiler and inference jobs during development. Rows collected under clear
contention were discarded or rerun; the checked-in JSONL still includes min, p50, p90, and max so residual jitter is
visible. For release decisions, rerun on an otherwise idle target machine rather than treating this hardware snapshot
as a portable performance guarantee.

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

## Environment for the checked-in snapshot

| Component    | Value                                                 |
| ------------ | ----------------------------------------------------- |
| CPU          | AMD Ryzen 9 5950X, 16 cores / 32 threads              |
| GPU          | NVIDIA GeForce RTX 3060 12 GB, compute capability 8.6 |
| Driver       | 550.144.03                                            |
| CUDA toolkit | 11.8                                                  |
| ggml         | v0.18.1 plus repository patches                       |

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

The shipped integration already amortizes the CPU thread pool, uses CUDA graph replay, performs postprocessing in the
logit domain, parallelizes CPU im2col over output rows, and fixes quantized Vulkan im2col scheduling. These improve the
current graph without changing model semantics.

The remaining large-model gap is architectural: generic ggml convolution materializes im2col before GEMM while PyTorch
CUDA uses optimized convolution kernels. Meeting the universal "faster than PyTorch CUDA" target requires a backend
convolution path that avoids that materialization, followed by a full matrix rerun. Tuning warmups, excluding
preprocessing, or selecting only favorable models would not satisfy the target.

## Remaining integration work

1. Implement or upstream direct CUDA and Vulkan convolution kernels, then prove speed on the entire 10-model matrix.
2. Run COCO and NYU Depth V2 validation across F32/F16/Q8_0 and all supported backends with explicit tolerances.
3. Add repeated-process, malformed-model, multi-resolution, and long-running stability coverage.
4. Promote the internal C++ session types to a versioned installed C API only when an embedding consumer requires it.
5. Extend YOLO26 absolute-depth support from n to s/m/l/x when those checkpoints are placed in the canonical model set.
