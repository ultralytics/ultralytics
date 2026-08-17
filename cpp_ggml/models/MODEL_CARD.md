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

## Formats

| Format | Precision                                                           | Intended use                                 |
| ------ | ------------------------------------------------------------------- | -------------------------------------------- |
| F32    | Float32 weights and activations                                     | Reference parity and debugging               |
| F16    | Float16 matrix/conv weights with float accumulations                | Default GPU deployment format                |
| Q8_0   | Block-quantized eligible conv weights; other tensors remain F16/F32 | Smaller files; validate accuracy per dataset |

F16 and Q8_0 are not expected to be bit-identical to PyTorch F32. Integration parity means equivalent task output
within declared tolerances, not identical intermediate activations. Dataset accuracy must be validated on the target
dataset before production deployment.

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
