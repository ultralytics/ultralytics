# YOLO inference with ggml

`cpp_ggml` is a standalone C++17 inference runtime for YOLOv8 and YOLO26 detection plus YOLO26 absolute depth. It
loads metadata-driven GGUF graphs without Python or PyTorch at runtime. CPU, CUDA, and Vulkan are validated here;
Metal and HIP have build wiring but remain experimental until they pass the same benchmark and parity matrix.
Conversion supports F32, F16, and Q8_0 weights.

The [model card](models/MODEL_CARD.md) defines every supported checkpoint and the canonical model layout. The
[benchmark report](benchmarks/README.md) contains raw measurements, latency charts, visual comparisons, parity scope,
and remaining performance gaps.

## Repository layout

```text
cpp_ggml/
├── CMakeLists.txt
├── src/                       # loader, graph builder, backends, preprocessing, postprocessing, CLI
├── examples/cli/              # yolo-cli target
├── models/
│   ├── MODEL_CARD.md
│   ├── pytorch/               # canonical .pt conversion inputs (ignored)
│   └── gguf/                  # generated GGUF files (ignored)
├── benchmarks/                # report, raw JSONL, plots, and parity images
├── scripts/                   # conversion, benchmark, parity, and plot tools
└── third_party/
    ├── ggml/                  # pinned v0.18.1 submodule
    ├── cudnn-frontend/        # pinned cuDNN Graph API headers
    └── ggml-patches/          # idempotent local performance/correctness patches
```

## Clone and build

Prerequisites are a C++17 compiler, CMake 3.14 or newer, Git, and Python dependencies from the repository for model
conversion. CUDA builds need the CUDA toolkit and cuDNN development headers/libraries; Vulkan builds need a discoverable
Vulkan SDK. Each build directory contains exactly one GPU backend. CPU fallback is included in every build.

```bash
git clone --recurse-submodules <repository-url>
cd ultralytics-ggml
git submodule update --init --recursive

cmake -S cpp_ggml -B cpp_ggml/build-cpu
cmake --build cpp_ggml/build-cpu --parallel 6

cmake -S cpp_ggml -B cpp_ggml/build-cuda -DYOLO_GGML_CUDA=ON
cmake --build cpp_ggml/build-cuda --parallel 6

cmake -S cpp_ggml -B cpp_ggml/build-vulkan -DYOLO_GGML_VULKAN=ON
cmake --build cpp_ggml/build-vulkan --parallel 6
```

Export `CUDNN_ROOT` when cuDNN is outside the system search path. A CUDA build can use
`-DYOLO_GGML_CUDNN=OFF` when cuDNN is unavailable, but that falls back to generic convolution kernels and is intended
for compatibility rather than peak performance.

Do not raise build parallelism above 6 on memory-constrained hosts. With CMake older than 3.24, the CUDA configure step
detects the attached NVIDIA GPU compute capability. For cross-compilation or a headless builder, set it explicitly,
for example `-DCMAKE_CUDA_ARCHITECTURES=86` for an RTX 30-series GPU.

CUDA and Vulkan are deliberately separate builds because the runtime currently selects one compiled GPU backend. CMake
rejects ambiguous multi-GPU-backend configurations instead of silently picking one.

## Convert models

Source checkpoints live under `cpp_ggml/models/pytorch/`. Supported missing release checkpoints are downloaded there by
Ultralytics during conversion. Old copies under `cpp_ggml/` or the repository root are not used. From `cpp_ggml/`:

```bash
# 10 detection variants plus YOLO26n-depth, each as F32/F16/Q8_0.
bash scripts/convert_all.sh

# Single-model examples; output defaults to models/gguf/<model>-<dtype>.gguf.
python3 scripts/convert_yolo_to_gguf.py --model yolo26n --dtype f16
python3 scripts/convert_yolo_to_gguf.py --model yolo26n-depth --dtype f16
```

The converter loads and fuses the local Ultralytics model, then serializes a small operation vocabulary into GGUF.
The C++ graph builder therefore has one execution path for model scale, weight precision, task, and backend.

## Run inference

```bash
# Detection: prints detections and optionally writes an annotated image.
cpp_ggml/build-cuda/bin/yolo-cli detect \
    --model cpp_ggml/models/gguf/yolo26n-f16.gguf \
    --source ultralytics/assets/bus.jpg --out detection.png --conf 0.25

# Absolute depth: writes metric float data plus an optional display-only PNG.
cpp_ggml/build-cuda/bin/yolo-cli depth \
    --model cpp_ggml/models/gguf/yolo26n-depth-f16.gguf \
    --source ultralytics/assets/bus.jpg --raw depth.bin --out depth.png

# Full preprocess + graph/readback + postprocess latency.
cpp_ggml/build-cuda/bin/yolo-cli bench \
    --model cpp_ggml/models/gguf/yolo26n-f16.gguf \
    --source ultralytics/assets/bus.jpg --warmup 20 --iters 50
```

The depth `--raw` file starts with the `YDEP0001` magic, two little-endian int32 dimensions `(height, width)`, then
row-major float32 meters. The PNG applies a color map and must not be used as metric data. Monocular depth is useful as
an approximate spatial prior, but it is not a replacement for a calibrated safety sensor; see the model card for
failure modes.

## Verification and benchmarks

```bash
cd cpp_ggml
for backend in cuda vulkan cpu; do
    bash scripts/bench_all.sh "build-$backend" "$backend"
done
python3 scripts/bench_pytorch.py > benchmarks/pytorch.jsonl
python3 scripts/plot_benchmarks.py
python3 scripts/render_parity.py
```

Speed depends on the GPU, toolkit, driver, thermal state, model scale, and precision. The current evidence does not show
all CUDA and Vulkan combinations beating PyTorch CUDA. F32 task-output parity has been checked on focused samples;
quantized formats require tolerance-based and dataset-level validation. The benchmark report keeps these constraints
visible rather than promoting an unverified universal claim.

## Architecture and extension points

- `convert_yolo_to_gguf.py` owns PyTorch-module-to-operation lowering and GGUF metadata.
- `gguf_loader.cpp` owns format validation; graph format version 2 adds depth operations while retaining version 1
  detection models.
- `yolo_graph.cpp` owns task-independent graph construction and task-specific output reads.
- `backend.cpp` owns CPU plus one optional GPU scheduler and CPU fallback.
- `image_io.cpp` and `postprocess.cpp` own preprocessing and task output restoration.
- `cli.cpp` is a thin command adapter for `detect`, `depth`, `info`, and `bench`.

To add an architecture, extend converter lowering only for genuinely new modules and implement the matching generic op
in the graph builder. To add a task, keep output decoding and source-size restoration in the task owner rather than
forking the model execution path.
