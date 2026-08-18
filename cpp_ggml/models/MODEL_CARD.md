# YOLO ggml model card

This directory is the single model store for the C++ integration. PyTorch checkpoints are conversion inputs; GGUF
files are runtime artifacts. Both are ignored by Git and can be regenerated.

```text
models/
├── MODEL_CARD.md
├── pytorch/                    # canonical .pt conversion inputs
│   ├── yolov8{n,s,m,l,x}.pt
│   ├── yolo26{n,s,m,l,x}.pt
│   └── yolo26n-depth.pt
└── gguf/                       # generated runtime models
    ├── <detect-model>-{f32,f16,q8_0}.gguf
    └── yolo26n-depth-{f32,f16,q8_0}.gguf
```

Do not put checkpoints in `cpp_ggml/` or the repository root. The converter resolves model aliases against
`models/pytorch/` and writes to `models/gguf/` by default.

## Layout migration

Older checkouts may contain the 11 source checkpoints directly under `cpp_ggml/` (or a duplicate `yolo26n.pt` in the
repository root). Those paths are retired. Move any locally retained files once, then remove the old copies:

```bash
mkdir -p cpp_ggml/models/pytorch
for name in yolov8n yolov8s yolov8m yolov8l yolov8x yolo26n yolo26s yolo26m yolo26l yolo26x yolo26n-depth; do
    test -f "cpp_ggml/$name.pt" && mv "cpp_ggml/$name.pt" "cpp_ggml/models/pytorch/$name.pt"
done
```

All conversion, benchmark, parity, and rendering scripts resolve this canonical directory; no script should reference
`cpp_ggml/<model>.pt` or a root-level checkpoint.

## Supported models

| Model         | Task           | Default input | Recommended use                                      |
| ------------- | -------------- | ------------: | ---------------------------------------------------- |
| YOLOv8n       | detect         |           640 | Lowest detection latency and memory use              |
| YOLOv8s       | detect         |           640 | Small edge deployments needing more capacity than n  |
| YOLOv8m       | detect         |           640 | Balanced accuracy and compute                        |
| YOLOv8l       | detect         |           640 | Accuracy-oriented GPU deployment                     |
| YOLOv8x       | detect         |           640 | Highest-capacity YOLOv8 integration target           |
| YOLO26n       | detect         |           640 | Lowest-latency end-to-end YOLO26 detector            |
| YOLO26s       | detect         |           640 | Compact end-to-end detector                          |
| YOLO26m       | detect         |           640 | Balanced end-to-end detector                         |
| YOLO26l       | detect         |           640 | Accuracy-oriented end-to-end detector                |
| YOLO26x       | detect         |           640 | Highest-capacity YOLO26 detection target             |
| YOLO26n-depth | absolute depth |           768 | Monocular metric-depth preview and spatial reasoning |

Detection models use COCO's 80 classes. YOLO26 detection checkpoints use the end-to-end head exported by the local
Ultralytics checkout. The depth model produces one floating-point distance in meters per source pixel.

### YOLOv8 detector family

- **YOLOv8n** is the default for latency-sensitive applications and constrained GPUs.
- **YOLOv8s** trades a small latency increase for more capacity while remaining suitable for edge deployment.
- **YOLOv8m** is the balanced choice when throughput and detection quality have similar weight.
- **YOLOv8l** targets accuracy-oriented GPU services where a larger memory and latency budget is available.
- **YOLOv8x** is the highest-capacity YOLOv8 checkpoint in this integration and the runtime/memory stress case.

All five use the same graph and postprocessing owners; model scale changes tensor shapes, not the public CLI or GGUF
contract.

### YOLO26 detector family

- **YOLO26n** is the lowest-latency end-to-end model and the strongest throughput choice in the measured matrix.
- **YOLO26s** is the compact accuracy/latency step above n.
- **YOLO26m** is the general balanced deployment model.
- **YOLO26l** uses a larger capacity budget for accuracy-oriented inference.
- **YOLO26x** is the largest YOLO26 integration target and the narrowest measured CUDA speedup case.

The exported end-to-end head is part of the model contract. Do not substitute YOLOv8 head decoding or assume that raw
tensor layouts are interchangeable across the two families.

### YOLO26n absolute depth

**YOLO26n-depth** predicts a dense metric-depth map rather than COCO detections. It has a 768 default input and a
task-specific restoration step that removes letterbox padding and resizes values to source resolution. Use the `depth`
CLI command; the `detect` command intentionally rejects this checkpoint.

## Formats

| Format | Precision                                                           | Intended use                                 |
| ------ | ------------------------------------------------------------------- | -------------------------------------------- |
| F32    | Float32 weights and activations                                     | Reference parity and debugging               |
| F16    | Float16 matrix/conv weights; backend-dependent activations          | Default GPU deployment format                |
| Q8_0   | Block-quantized eligible conv weights; other tensors remain F16/F32 | Smaller files; validate accuracy per dataset |

F16 and Q8_0 are not expected to be bit-identical to PyTorch F32. Integration parity means equivalent task output
within declared tolerances, not identical intermediate activations. Dataset accuracy must be validated on the target
dataset before production deployment.

## Runtime support

| Backend | F32 | F16 | Q8_0 | Status                                                                  |
| ------- | --- | --- | ---- | ----------------------------------------------------------------------- |
| CPU     | yes | yes | yes  | Functional reference/fallback; 8-thread measurements are checked in     |
| CUDA    | yes | yes | yes  | F16 release path; all 11 models beat the measured PyTorch-CUDA baseline |
| Vulkan  | yes | yes | yes  | F16 release path; all 11 models meet the declared x0.70 proximity floor |

See the [benchmark and parity report](../benchmarks/README.md) for hardware, protocol, per-model values, raw numerical
error, and the limits of the current accuracy evidence.

## Rebuild

From `cpp_ggml/`:

```bash
# Missing supported release checkpoints are downloaded into models/pytorch/.
bash scripts/convert_all.sh

# Or convert one model. Both commands resolve the canonical directory.
python3 scripts/convert_yolo_to_gguf.py --model yolo26n --dtype f16
python3 scripts/convert_yolo_to_gguf.py --model yolo26n-depth --dtype f16
```

## Absolute-depth notes

YOLO26n-depth is monocular absolute-depth estimation. It is useful for approximate scene layout, obstacle-distance
priors, measurement assistance, and robotics perception when a calibrated depth sensor is unavailable. The raw
`YDEP0001` output is the machine-readable result in meters; the PNG is only a colorized visualization.

Depth from one RGB image remains sensitive to camera intrinsics, domain shift, reflective or transparent surfaces,
low texture, occlusion boundaries, and objects outside the training distribution. Do not use it as the sole input for
safety-critical distance decisions. Preserve aspect ratio and use the default 768 input for comparisons with the
released checkpoint.
